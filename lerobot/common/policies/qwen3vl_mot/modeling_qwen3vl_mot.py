"""Qwen3-VL understanding plus Cosmos-style multimodal generation."""

from __future__ import annotations

import json
import logging
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist

from lerobot.common.policies.ace.configuration_robo_contrast import RoboContrastConfig
from lerobot.common.policies.ace.modeling_robo_contrast import RoboContrast
from lerobot.common.policies.pretrained import PreTrainedPolicy

from .configuration_qwen3vl_mot import Qwen3VLMoTConfig
from .modeling_generation import GenerationExpert, GenerationStream
from .modeling_understanding import Qwen3VLUnderstandingExpert, _base_qwen
from .stage1_transfer import merge_peft_module, transfer_stage1_perception
from .tasks import TASK_SPECS, ModalityRole, TaskSpec, resolve_task_specs

logger = logging.getLogger(__name__)


def _distributed_world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _all_reduce_detached(value: torch.Tensor) -> torch.Tensor:
    result = value.detach().clone()
    if _distributed_world_size() > 1:
        dist.all_reduce(result)
    return result


def _strip_distributed_prefix(name: str) -> str:
    for prefix in ("module.", "_forward_module."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _resolve_deepspeed_model_file(path: Path) -> Path:
    if path.is_file():
        return path
    latest = path / "latest"
    if latest.is_file():
        tag = latest.read_text().strip()
        candidate = path / tag / "mp_rank_00_model_states.pt"
        if candidate.is_file():
            return candidate
    direct = path / "mp_rank_00_model_states.pt"
    if direct.is_file():
        return direct
    candidates = sorted(path.glob("*/mp_rank_00_model_states.pt"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(
        f"No model.safetensors or DeepSpeed mp_rank_00_model_states.pt found under {path}."
    )


def _load_stage1_config(config_source: Path) -> RoboContrastConfig:
    config_file = (
        config_source
        if config_source.is_file() and config_source.suffix == ".json"
        else config_source / "config.json"
    )
    if not config_file.is_file():
        raise FileNotFoundError(
            "A DeepSpeed stage-one checkpoint also needs its RoboContrast config. Put "
            "config.json in the checkpoint root or pass `stage1_config=/path/to/config.json`."
        )
    return _decode_stage1_config(json.loads(config_file.read_text(encoding="utf-8")))


def _override_stage1_model_paths(
    stage1_config: RoboContrastConfig,
    runtime_config: Qwen3VLMoTConfig,
) -> RoboContrastConfig:
    for field_name in ("qwen3vl_dir", "cosmos3_dir"):
        saved_path = getattr(stage1_config, field_name)
        runtime_path = getattr(runtime_config, field_name)
        if runtime_path and runtime_path != saved_path:
            logger.info(
                "Overriding Stage 1 %s from checkpoint path %s to runtime path %s",
                field_name,
                saved_path,
                runtime_path,
            )
            setattr(stage1_config, field_name, runtime_path)
    return stage1_config


def _load_stage1_policy(
    checkpoint_path: Path,
    config_path: str,
    runtime_config: Qwen3VLMoTConfig,
) -> tuple[RoboContrast, tuple[str, ...]]:
    config_source = Path(config_path) if config_path else checkpoint_path
    stage1_config = _override_stage1_model_paths(
        _load_stage1_config(config_source),
        runtime_config,
    )

    exported_model = checkpoint_path / "model.safetensors" if checkpoint_path.is_dir() else None
    if exported_model is not None and exported_model.is_file():
        from safetensors import safe_open

        with safe_open(exported_model, framework="pt", device="cpu") as checkpoint:
            checkpoint_keys = tuple(checkpoint.keys())
        teacher = RoboContrast.from_pretrained(
            checkpoint_path,
            config=stage1_config,
            map_location="cpu",
            strict=False,
        )
        return teacher, checkpoint_keys

    model_file = _resolve_deepspeed_model_file(checkpoint_path)
    payload = torch.load(model_file, map_location="cpu", weights_only=False)
    state_dict = payload.get("module", payload)
    state_dict = {
        _strip_distributed_prefix(name): value
        for name, value in state_dict.items()
    }
    teacher = RoboContrast(stage1_config)
    teacher.load_state_dict(state_dict, strict=False)
    return teacher, tuple(state_dict)


def _encode_stage1_config(config: RoboContrastConfig) -> dict:
    import draccus

    payload = draccus.encode(config)
    payload.pop("type", None)
    return payload


def _decode_stage1_config(payload: dict) -> RoboContrastConfig:
    import draccus

    encoded = dict(payload)
    encoded.pop("type", None)
    return draccus.decode(RoboContrastConfig, encoded)


class Qwen3VLMoTPolicy(PreTrainedPolicy):
    """Stage-two world model distilled from a frozen RoboContrast checkpoint."""

    config_class = Qwen3VLMoTConfig
    name = "qwen3vl_mot"

    def __init__(self, config: Qwen3VLMoTConfig, dataset_stats=None):
        super().__init__(config)
        self.config = config
        config.validate_features()

        restoring_embedded = not config.initialize_from_stage1
        if restoring_embedded:
            if config.stage1_policy_config is None:
                raise ValueError(
                    "A stage-two restore requires the embedded `stage1_policy_config`; "
                    "this checkpoint predates self-contained stage-two restoration."
                )
            stage1_config = _override_stage1_model_paths(
                _decode_stage1_config(config.stage1_policy_config),
                config,
            )
            # These modules are removed from the stage-two policy, so constructing them only
            # wastes memory and can unnecessarily load the stage-one VAE.
            stage1_config.num_predictor_layers = 0
            stage1_config.perception_recon_weight = 0.0
            teacher = RoboContrast(stage1_config)
            teacher.perception_encoder.vision_backbone = merge_peft_module(
                teacher.perception_encoder.vision_backbone
            )
            teacher.perception_encoder.text_backbone = merge_peft_module(
                teacher.perception_encoder.text_backbone
            )
        else:
            if not config.stage1_checkpoint:
                raise ValueError(
                    "`stage1_checkpoint` must point to an exported RoboContrast policy when "
                    "constructing a fresh qwen3vl_mot model."
                )
            stage1_path = Path(config.stage1_checkpoint)
            if not stage1_path.exists():
                raise FileNotFoundError(
                    f"The configured stage-one checkpoint does not exist: {stage1_path}."
                )
            teacher, checkpoint_keys = _load_stage1_policy(
                stage1_path,
                config.stage1_config,
                config,
            )
            required_prefixes = (
                "perception_encoder.vision_backbone.",
                "physical_encoder.state_proj.",
                "physical_encoder.action_proj.",
                "physical_encoder.blocks.",
            )
            missing_prefixes = [
                prefix
                for prefix in required_prefixes
                if not any(key.startswith(prefix) for key in checkpoint_keys)
            ]
            if missing_prefixes:
                raise ValueError(
                    "The stage-one checkpoint does not contain the required trained modules: "
                    f"{missing_prefixes}."
                )
            config.stage1_policy_config = _encode_stage1_config(teacher.config)
        if teacher.physical_encoder is None:
            raise ValueError(
                "The stage-one checkpoint was saved with perception_only=True and has no "
                "physical transformer to transfer into stage two."
            )
        self._validate_stage1_contract(teacher.config)

        # Load the stock Qwen checkpoint first, overwrite every compatible stage-one tensor,
        # and only then attach new LoRA adapters. This preserves the actual stage-one weights
        # instead of loading them into PEFT base-layer names by accident.
        requested_tuning = config.understanding_tuning_mode
        self.understanding = Qwen3VLUnderstandingExpert(
            config.qwen3vl_dir,
            num_queries=config.num_latent_queries,
            max_text_tokens=config.understanding_max_text_tokens,
            kv_layers=config.understanding_kv_layers,
            tuning_mode="frozen",
            lora_rank=config.understanding_lora_rank,
            lora_alpha=config.understanding_lora_alpha,
            lora_dropout=config.understanding_lora_dropout,
            text_lora_layers=config.understanding_text_lora_layers,
            vision_lora_layers=config.understanding_vision_lora_layers,
            gradient_checkpointing=config.understanding_gradient_checkpointing,
        )
        qwen_text_config = _base_qwen(self.understanding.model).config.text_config
        qwen_kv_heads = int(qwen_text_config.num_key_value_heads)
        qwen_head_dim = int(qwen_text_config.head_dim)
        if (
            config.generation_num_kv_heads != qwen_kv_heads
            or config.generation_hidden_dim // config.generation_num_heads != qwen_head_dim
        ):
            raise ValueError(
                "Native Qwen K/V reuse requires generation attention geometry "
                f"{qwen_kv_heads} KV heads x {qwen_head_dim} dims; got "
                f"{config.generation_num_kv_heads} KV heads and "
                f"{config.generation_hidden_dim // config.generation_num_heads}-dim heads."
            )
        if not restoring_embedded:
            vision_report, text_report = transfer_stage1_perception(
                teacher.perception_encoder,
                _base_qwen(self.understanding.model),
            )
            logger.info(
                "Stage-one -> Qwen transfer: vision %.1f%% (%d tensors), text %.1f%% (%d tensors).",
                100.0 * vision_report.coverage,
                vision_report.copied_tensors,
                100.0 * text_report.coverage,
                text_report.copied_tensors,
            )
            if config.require_stage1_vision_transfer and vision_report.coverage < 0.90:
                raise RuntimeError(
                    "The stage-one vision tower is not Qwen3-VL-compatible: only "
                    f"{vision_report.coverage:.1%} of the reusable Qwen vision parameters matched."
                )
            if config.require_stage1_text_transfer and text_report.coverage < 0.90:
                raise RuntimeError(
                    "The stage-one text tower is not Qwen3-VL-compatible: only "
                    f"{text_report.coverage:.1%} of Qwen text parameters matched. The current "
                    "RoboContrast checkpoints normally use SigLIP2 text, so leave "
                    "`require_stage1_text_transfer=false` unless the checkpoint truly contains "
                    "a Qwen text tower."
                )
            if text_report.coverage < 0.90:
                logger.warning(
                    "Stage-one text weights are not Qwen3-VL-compatible (coverage %.1f%%); "
                    "the understanding text tower keeps its Qwen pretrained initialization.",
                    100.0 * text_report.coverage,
                )
        if requested_tuning != "frozen":
            self.understanding._configure_tuning(
                tuning_mode=requested_tuning,
                rank=config.understanding_lora_rank,
                alpha=config.understanding_lora_alpha,
                dropout=config.understanding_lora_dropout,
                text_layers=config.understanding_text_lora_layers,
                vision_layers=config.understanding_vision_lora_layers,
            )
            self.understanding.tuning_mode = requested_tuning

        # The physical module itself becomes the stage-two conditioner; it is not copied.
        # Moving it out of the teacher avoids holding two 200M-parameter physical towers.
        self.physical_encoder = teacher.physical_encoder
        teacher.physical_encoder = None
        for parameter in self.physical_encoder.parameters():
            parameter.requires_grad_(config.physical_tuning_mode == "full")

        self.teacher_perception = teacher.perception_encoder
        self.teacher_perception.predictor = None
        self.teacher_perception.vae = None
        for parameter in self.teacher_perception.parameters():
            parameter.requires_grad_(False)
        self.teacher_perception.eval()
        del teacher

        from lerobot.common.policies.ace.cosmos3_encoders import build_cosmos3_vae

        self.video_vae, latent_dim, self.temporal_compression, latent_mean, latent_std = (
            build_cosmos3_vae(
                config.cosmos3_dir,
                encoder_only=not config.load_vae_decoder,
            )
        )
        if latent_dim != config.video_latent_dim:
            raise ValueError(
                f"Configured video_latent_dim={config.video_latent_dim}, but the VAE emits "
                f"{latent_dim} channels."
            )
        for parameter in self.video_vae.parameters():
            parameter.requires_grad_(False)
        self.video_vae.eval()
        self.register_buffer(
            "video_latent_mean",
            latent_mean.view(1, latent_dim, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "video_latent_std",
            latent_std.view(1, latent_dim, 1, 1, 1),
            persistent=False,
        )

        self.latent_action_projection = nn.Sequential(
            nn.LayerNorm(self.understanding.hidden_dim),
            nn.Linear(self.understanding.hidden_dim, config.latent_action_dim),
            nn.LayerNorm(config.latent_action_dim),
        )
        if config.tactile_generation_target == "spatial_patches":
            if (
                self.physical_encoder.use_ftp1_tactile
                or self.physical_encoder.use_anytouch_tactile
                or self.physical_encoder.tactile_recon is None
            ):
                raise ValueError(
                    "Decodable tactile generation requires a stage-one checkpoint trained "
                    "with tactile_backbone='resnet18' and tactile_recon_weight > 0."
                )
            for module in (
                self.physical_encoder.tactile_cnn,
                self.physical_encoder.tactile_recon,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
            tactile_generation_dim = self.physical_encoder.tactile_feat_dim
        else:
            tactile_generation_dim = config.physical_hidden_dim
        input_dims = {
            "video": config.video_latent_dim * config.video_latent_patch_size**2,
            "state": config.physical_hidden_dim,
            "action": config.physical_hidden_dim,
            "tactile": tactile_generation_dim,
        }
        output_dims = {
            "video": config.video_latent_dim * config.video_latent_patch_size**2,
            "state": config.group_size * config.max_state_dim,
            "action": config.group_size * config.max_action_dim,
            "tactile": tactile_generation_dim,
        }
        self.task_specs, self.task_weights = resolve_task_specs(
            config.task_names,
            config.task_weights,
        )
        self.generation = GenerationExpert(
            understanding_dim=self.understanding.hidden_dim,
            hidden_dim=config.generation_hidden_dim,
            depth=config.generation_depth,
            num_heads=config.generation_num_heads,
            num_kv_heads=config.generation_num_kv_heads,
            intermediate_dim=config.generation_intermediate_dim,
            hidden_act=config.generation_hidden_act,
            dropout=config.generation_dropout,
            input_dims=input_dims,
            output_dims=output_dims,
            task_names=config.task_names,
            gradient_checkpointing=config.generation_gradient_checkpointing,
            rotary_config=qwen_text_config,
        )
        config.initialize_from_stage1 = False

    def _validate_stage1_contract(self, stage1_config) -> None:
        if getattr(stage1_config, "vision_backbone", None) != "qwen3vl":
            raise ValueError(
                "Stage two can inherit its understanding vision tower only from a "
                "RoboContrast checkpoint trained with vision_backbone='qwen3vl'."
            )
        expected = {
            "chunk_size": self.config.chunk_size,
            "group_size": self.config.group_size,
            "hidden_dim": self.config.physical_hidden_dim,
            "max_action_dim": self.config.max_action_dim,
            "max_state_dim": self.config.max_state_dim,
            "max_tactile_signal_dim": self.config.max_tactile_signal_dim,
            "max_tactile_views": self.config.max_tactile_views,
            "tactile_tokens_per_pad": self.config.tactile_tokens_per_pad,
            "tactile_frames": self.config.tactile_frames,
            "tactile_img_size": self.config.tactile_img_size,
            "num_change_queries": self.config.num_latent_queries,
        }
        mismatches = {
            name: (getattr(stage1_config, name), value)
            for name, value in expected.items()
            if getattr(stage1_config, name) != value
        }
        if stage1_config.hidden_dim != self.config.latent_action_dim:
            mismatches["latent_action_dim"] = (
                stage1_config.hidden_dim,
                self.config.latent_action_dim,
            )
        if mismatches:
            details = ", ".join(
                f"{name}: stage1={old}, stage2={new}"
                for name, (old, new) in mismatches.items()
            )
            raise ValueError(
                "Stage-two dimensions must match the checkpoint whose projections and "
                f"physical transformer are reused ({details})."
            )

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher_perception.eval()
        self.video_vae.eval()
        if self.config.physical_tuning_mode == "frozen":
            self.physical_encoder.eval()
        if self.config.tactile_generation_target == "spatial_patches":
            self.physical_encoder.tactile_cnn.eval()
            self.physical_encoder.tactile_recon.eval()
        return self

    def reset(self):
        return None

    def _inference_physical_batch(
        self,
        batch: dict,
        action: torch.Tensor,
    ) -> dict:
        device = action.device
        batch_size = action.shape[0]
        state = batch.get("observation.state")
        if state is None:
            raise KeyError(
                "`sample_canonical_action` requires canonical normalized "
                "`observation.state`."
            )
        state = state.to(device)
        if state.ndim not in (2, 3):
            raise ValueError(
                "`sample_canonical_action` expects state shaped (B,D) or (B,T,D), got "
                f"{tuple(state.shape)}."
            )
        current_state = state if state.ndim == 2 else state[:, 0]
        if current_state.shape != (batch_size, self.config.max_state_dim):
            raise ValueError(
                "`sample_canonical_action` expects canonical state width "
                f"{self.config.max_state_dim}, got {tuple(current_state.shape)}."
            )
        state = current_state.unsqueeze(1).expand(-1, self.config.chunk_size, -1)
        state_mask = batch.get("state_mask")
        if state_mask is None or state_mask.shape != (
            batch_size,
            self.config.max_state_dim,
        ):
            raise ValueError(
                "`sample_canonical_action` requires the canonical `state_mask` shaped "
                f"(B,{self.config.max_state_dim})."
            )
        state_mask = state_mask.to(device)
        action_mask = batch.get("action_mask")
        if action_mask is None or action_mask.shape != (
            batch_size,
            self.config.max_action_dim,
        ):
            raise ValueError(
                "`sample_canonical_action` requires the embodiment's canonical "
                f"`action_mask` shaped (B,{self.config.max_action_dim})."
            )
        action_mask = action_mask.to(device)

        tactile_signal = torch.zeros(
            batch_size,
            self.config.chunk_size,
            self.config.max_tactile_signal_dim,
            device=device,
        )
        tactile_image = torch.zeros(
            batch_size,
            self.config.max_tactile_views,
            self.config.tactile_frames,
            3,
            self.config.tactile_img_size,
            self.config.tactile_img_size,
            device=device,
            dtype=torch.uint8,
        )
        return {
            **batch,
            "observation.state": state,
            "state_mask": state_mask,
            "action": action,
            "action_mask": action_mask,
            "tactile_signal": tactile_signal,
            "tactile_signal_mask": torch.zeros(batch_size, device=device),
            "tactile_image": tactile_image,
            "tactile_image_mask": torch.zeros(
                batch_size,
                self.config.max_tactile_views,
                device=device,
            ),
            "tactile_sensor_id": torch.zeros(
                batch_size,
                self.config.max_tactile_views,
                device=device,
                dtype=torch.long,
            ),
            "tactile_img_mean": torch.zeros(
                batch_size,
                self.config.max_tactile_views,
                3,
                device=device,
            ),
            "tactile_img_std": torch.ones(
                batch_size,
                self.config.max_tactile_views,
                3,
                device=device,
            ),
            "sample_rate": batch.get(
                "sample_rate",
                torch.full((batch_size,), 10, device=device, dtype=torch.long),
            ),
        }

    def _current_image(self, batch: dict) -> torch.Tensor:
        image = batch.get("image_t0")
        if image is None:
            image = batch.get("observation.image")
        if image is None:
            for key in sorted(self.config.image_features):
                value = batch.get(key)
                if isinstance(value, torch.Tensor):
                    image = value
                    break
        if image is None:
            raise KeyError(
                "Action sampling requires `image_t0`, `observation.image`, or one of the "
                "configured visual feature keys."
            )
        return image

    def _sample_canonical_action(
        self,
        batch: dict,
        understanding,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        image = self._current_image(batch)
        device = next(self.generation.parameters()).device
        action = torch.randn(
            image.shape[0],
            self.config.chunk_size,
            self.config.max_action_dim,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        action_mask = batch["action_mask"].to(device).unsqueeze(1)
        action = action * action_mask
        schedule = torch.linspace(
            1.0,
            0.0,
            self.config.inference_steps + 1,
            device=device,
        )
        for sigma, next_sigma in zip(schedule[:-1], schedule[1:], strict=True):
            physical_batch = self._inference_physical_batch(batch, action)
            contextual, _, _ = self._physical_context(
                physical_batch,
                include_tactile=False,
                force_no_grad=True,
            )
            start = self.physical_encoder.num_cls_tokens
            state_tokens = contextual[:, start : start + self.config.num_groups]
            action_tokens = contextual[
                :,
                start + self.config.num_groups : start + 2 * self.config.num_groups,
            ]
            state_keep = physical_batch["state_mask"].sum(dim=-1, keepdim=True) > 0
            state_keep = state_keep.expand(-1, self.config.num_groups)
            action_keep = torch.ones(
                image.shape[0],
                self.config.num_groups,
                device=device,
                dtype=torch.bool,
            )
            sigma_batch = sigma.expand(image.shape[0])
            predictions = self.generation(
                [
                    GenerationStream(
                        "state",
                        state_tokens,
                        torch.zeros_like(sigma_batch),
                        state_keep,
                    ),
                    GenerationStream(
                        "action",
                        action_tokens,
                        sigma_batch,
                        action_keep,
                    ),
                ],
                understanding.hidden_states,
                understanding.attention_mask,
                "action_prediction",
                understanding_key_values=understanding.key_values,
            )
            velocity = predictions["action"].reshape_as(action).float()
            action = action + (next_sigma - sigma) * velocity
            action = action * action_mask
        return action[:, : self.config.n_action_steps]

    @torch.no_grad()
    def sample_canonical_action(self, batch: dict) -> torch.Tensor:
        """Sample a normalized canonical action chunk.

        This deliberately does not masquerade as `select_action`: mapping the canonical
        40-dimensional result back to one robot's action order and units belongs to the
        embodiment adapter that produced `action_mask`.
        """
        image = self._current_image(batch)
        device = next(self.generation.parameters()).device
        image = image.to(device)
        texts = batch.get("task", [""] * image.shape[0])
        if isinstance(texts, str):
            texts = [texts] * image.shape[0]
        understanding = self.understanding(image, texts)
        return self._sample_canonical_action(batch, understanding)

    def select_action(self, batch):
        raise NotImplementedError(
            "The stage-two model predicts normalized canonical 40D actions. Deployment needs "
            "an embodiment-specific canonical mapping and inverse normalization; call "
            "`sample_canonical_action` only when the batch already follows that contract."
        )

    def get_optim_params(self):
        understanding, physical, main = [], [], []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("understanding.model."):
                understanding.append(parameter)
            elif name.startswith("physical_encoder."):
                physical.append(parameter)
            else:
                main.append(parameter)
        groups = [{"params": main, "group_name": "main"}]
        if understanding:
            groups.append(
                {
                    "params": understanding,
                    "lr": self.config.optimizer_lr * self.config.understanding_lr_scale,
                    "group_name": "understanding",
                }
            )
        if physical:
            groups.append(
                {
                    "params": physical,
                    "lr": self.config.optimizer_lr * self.config.physical_lr_scale,
                    "group_name": "physical",
                }
            )
        return groups

    def _task_valid_rows(self, task: TaskSpec, batch: dict, device: torch.device) -> torch.Tensor:
        batch_size = batch["image_t0"].shape[0]
        pair = batch.get("pair_is_valid")
        pair = (
            torch.ones(batch_size, device=device, dtype=torch.bool)
            if pair is None
            else pair.to(device).reshape(-1) > 0.5
        )
        has_text = batch.get("has_text")
        has_text = (
            torch.ones(batch_size, device=device, dtype=torch.bool)
            if has_text is None
            else has_text.to(device).reshape(-1) > 0.5
        )
        has_state = batch["state_mask"].to(device).sum(dim=-1) > 0
        has_action = batch["action_mask"].to(device).sum(dim=-1) > 0
        has_tactile = (
            batch["tactile_signal_mask"].to(device).reshape(-1) > 0
        ) | (batch["tactile_image_mask"].to(device).sum(dim=-1) > 0)
        if self.config.tactile_generation_target == "spatial_patches":
            has_tactile = (
                batch["tactile_image_mask"].to(device).sum(dim=-1) > 0
            )

        if task.name == "t2v":
            return pair & has_text
        if task.name == "i2v":
            return pair
        if task.name in (
            "action_video_prediction",
            "forward_dynamics",
            "inverse_dynamics",
            "state_prediction",
        ):
            return pair & has_state & has_action
        if task.name == "action_prediction":
            return has_state & has_action
        if task.name == "tactile_prediction":
            return pair & has_state & has_action & has_tactile
        raise KeyError(task.name)

    def _select_task(self, batch: dict, forced_task: str | None) -> tuple[TaskSpec, torch.Tensor]:
        device = batch["image_t0"].device
        local_counts = torch.stack(
            [
                self._task_valid_rows(task, batch, device).sum()
                for task in self.task_specs
            ]
        ).to(torch.float32)
        global_counts = _all_reduce_detached(local_counts)
        available = global_counts > 0
        if forced_task is not None:
            if forced_task not in TASK_SPECS:
                raise ValueError(
                    f"Unknown forced stage-two task {forced_task!r}; choose from {tuple(TASK_SPECS)}."
                )
            index = next(
                (i for i, task in enumerate(self.task_specs) if task.name == forced_task),
                None,
            )
            if index is None:
                raise ValueError(f"Task {forced_task!r} is not enabled by `task_names`.")
            if not available[index]:
                raise ValueError(f"The global batch has no valid rows for task {forced_task!r}.")
            return self.task_specs[index], self._task_valid_rows(self.task_specs[index], batch, device)

        probabilities = torch.tensor(self.task_weights, device=device) * available
        if probabilities.sum() <= 0:
            raise RuntimeError("The global batch has no valid rows for any configured stage-two task.")
        probabilities = probabilities / probabilities.sum()
        task_index = torch.zeros(1, device=device, dtype=torch.long)
        if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
            task_index = torch.multinomial(probabilities, 1)
        if _distributed_world_size() > 1:
            dist.broadcast(task_index, src=0)
        task = self.task_specs[int(task_index.item())]
        return task, self._task_valid_rows(task, batch, device)

    def _sample_sigma(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.empty(batch_size, device=device).uniform_(
            self.config.sigma_min,
            self.config.sigma_max,
        )

    @staticmethod
    def _flow_corrupt(
        clean: torch.Tensor,
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(clean)
        shape = (clean.shape[0],) + (1,) * (clean.ndim - 1)
        sigma_view = sigma.view(shape).to(clean.dtype)
        noisy = (1.0 - sigma_view) * clean + sigma_view * noise
        return noisy, noise - clean

    def _encode_video(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if video.ndim != 5:
            raise ValueError(f"`video` must have shape (B,T,C,H,W), got {tuple(video.shape)}.")
        if video.shape[1] < 1:
            raise ValueError("Video encoding needs at least one frame.")
        video = video.to(torch.float32)
        if video.shape[-2:] != (
            self.config.video_image_size,
            self.config.video_image_size,
        ):
            batch_size, frames = video.shape[:2]
            video = F.interpolate(
                video.reshape(batch_size * frames, *video.shape[2:]),
                size=(self.config.video_image_size, self.config.video_image_size),
                mode="bilinear",
                align_corners=False,
            ).reshape(batch_size, frames, 3, self.config.video_image_size, self.config.video_image_size)
        video = (video / 255.0 - 0.5) / 0.5
        video = video.permute(0, 2, 1, 3, 4).contiguous()
        vae_dtype = next(self.video_vae.parameters()).dtype
        with torch.no_grad():
            latent = self.video_vae.encode(video.to(vae_dtype)).latent_dist.mean
        latent = (
            latent.float()
            - self.video_latent_mean.to(device=latent.device)
        ) / self.video_latent_std.to(device=latent.device)
        batch, channels, frames, height, width = latent.shape
        patch = self.config.video_latent_patch_size
        if height % patch or width % patch:
            raise ValueError(
                f"Wan latent grid {height}x{width} must be divisible by "
                f"video_latent_patch_size={patch}."
            )
        patch_height, patch_width = height // patch, width // patch
        tokens = latent.view(
            batch,
            channels,
            frames,
            patch_height,
            patch,
            patch_width,
            patch,
        )
        tokens = tokens.permute(0, 2, 3, 5, 4, 6, 1).reshape(
            batch,
            frames * patch_height * patch_width,
            channels * patch * patch,
        )
        position_ids = self._video_position_ids(
            frames,
            patch_height,
            patch_width,
            latent.device,
        )
        return tokens, position_ids

    @staticmethod
    def _video_position_ids(
        frames: int,
        patch_height: int,
        patch_width: int,
        device: torch.device,
    ) -> torch.Tensor:
        temporal = torch.arange(frames, device=device)
        rows = torch.arange(patch_height, device=device)
        columns = torch.arange(patch_width, device=device)
        return torch.cartesian_prod(temporal, rows, columns)

    def _video_geometry(self) -> tuple[int, int, int, int]:
        latent_frames = 1 + (
            self.config.world_video_frames - 1
        ) // self.temporal_compression
        latent_side = self.config.video_image_size // 16
        patch = self.config.video_latent_patch_size
        if latent_side % patch:
            raise ValueError(
                f"Video latent side {latent_side} is not divisible by patch size {patch}."
            )
        patch_side = latent_side // patch
        return latent_frames, latent_side, patch_side, patch_side * patch_side

    def _video_tokens_to_latents(
        self,
        tokens: torch.Tensor,
        *,
        denormalize: bool,
        latent_frames: int | None = None,
        patch_height: int | None = None,
        patch_width: int | None = None,
    ) -> torch.Tensor:
        if latent_frames is None or patch_height is None or patch_width is None:
            default_frames, _, default_height, _ = self._video_geometry()
            latent_frames = (
                default_frames if latent_frames is None else int(latent_frames)
            )
            patch_height = (
                default_height if patch_height is None else int(patch_height)
            )
            patch_width = (
                default_height if patch_width is None else int(patch_width)
            )
        else:
            latent_frames = int(latent_frames)
            patch_height = int(patch_height)
            patch_width = int(patch_width)
        tokens_per_frame = patch_height * patch_width
        expected_tokens = latent_frames * tokens_per_frame
        expected_width = (
            self.config.video_latent_dim
            * self.config.video_latent_patch_size**2
        )
        if tokens.shape[1:] != (expected_tokens, expected_width):
            raise ValueError(
                "Video tokens have incompatible shape: expected "
                f"(B,{expected_tokens},{expected_width}), got {tuple(tokens.shape)}."
            )
        batch = tokens.shape[0]
        patch = self.config.video_latent_patch_size
        latent = tokens.view(
            batch,
            latent_frames,
            patch_height,
            patch_width,
            patch,
            patch,
            self.config.video_latent_dim,
        )
        latent = latent.permute(0, 6, 1, 2, 4, 3, 5).reshape(
            batch,
            self.config.video_latent_dim,
            latent_frames,
            patch_height * patch,
            patch_width * patch,
        )
        if denormalize:
            latent = (
                latent.float()
                * self.video_latent_std.to(device=latent.device)
                + self.video_latent_mean.to(device=latent.device)
            )
        return latent

    def _prepare_video_stream(
        self,
        role: ModalityRole,
        batch: dict,
        valid_rows: torch.Tensor,
    ):
        if role is ModalityRole.ABSENT:
            return None, None, None, None
        if "video" not in batch:
            raise KeyError(
                "The stage-two batch is missing `video`; construct the dataset with a "
                "Qwen3VLMoTConfig so `world_video_frames` are decoded."
            )
        clean, position_ids = self._encode_video(batch["video"])
        keep = valid_rows.unsqueeze(1).expand(-1, clean.shape[1])
        if role is ModalityRole.CLEAN:
            sigma = torch.zeros(clean.shape[0], device=clean.device)
            return GenerationStream(
                "video",
                clean,
                sigma,
                keep,
                position_ids=position_ids,
            ), None, None, sigma
        sigma = self._sample_sigma(clean.shape[0], clean.device)
        noisy, target = self._flow_corrupt(clean, sigma)
        loss_mask = keep.unsqueeze(-1).expand_as(target).clone()
        if role is ModalityRole.FUTURE_NOISY:
            tokens_per_frame = int((position_ids[:, 0] == 0).sum().item())
            noisy[:, :tokens_per_frame] = clean[:, :tokens_per_frame]
            target[:, :tokens_per_frame] = 0
            loss_mask[:, :tokens_per_frame] = False
        return GenerationStream(
            "video",
            noisy,
            sigma,
            keep,
            position_ids=position_ids,
        ), target, loss_mask, sigma

    def _sample_video_latents(
        self,
        batch: dict,
        understanding,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if "i2v" not in self.generation.task_to_id:
            raise ValueError("Generation validation requires the enabled task 'i2v'.")
        current_tokens, current_positions = self._encode_video(
            batch["video"][:, :1]
        )
        first_frame = current_positions[:, 0] == 0
        current_tokens = current_tokens[:, first_frame]
        current_positions = current_positions[first_frame]
        patch_height = int(current_positions[:, 1].max().item()) + 1
        patch_width = int(current_positions[:, 2].max().item()) + 1
        tokens_per_frame = patch_height * patch_width
        latent_frames = 1 + (
            self.config.world_video_frames - 1
        ) // self.temporal_compression
        total_tokens = latent_frames * tokens_per_frame
        values = torch.randn(
            current_tokens.shape[0],
            total_tokens,
            current_tokens.shape[-1],
            device=current_tokens.device,
            dtype=torch.float32,
            generator=generator,
        )
        values[:, :tokens_per_frame] = current_tokens.float()
        keep = torch.ones(
            values.shape[:2],
            device=values.device,
            dtype=torch.bool,
        )
        position_ids = self._video_position_ids(
            latent_frames,
            patch_height,
            patch_width,
            values.device,
        )
        schedule = torch.linspace(
            1.0,
            0.0,
            self.config.inference_steps + 1,
            device=values.device,
        )
        for sigma, next_sigma in zip(schedule[:-1], schedule[1:], strict=True):
            sigma_batch = sigma.expand(values.shape[0])
            prediction = self.generation(
                [
                    GenerationStream(
                        "video",
                        values,
                        sigma_batch,
                        keep,
                        position_ids=position_ids,
                    )
                ],
                understanding.hidden_states,
                understanding.attention_mask,
                "i2v",
                understanding_key_values=understanding.key_values,
            )["video"].float()
            prediction[:, :tokens_per_frame] = 0
            values = values + (next_sigma - sigma) * prediction
            values[:, :tokens_per_frame] = current_tokens.float()
        return self._video_tokens_to_latents(
            values,
            denormalize=True,
            latent_frames=latent_frames,
            patch_height=patch_height,
            patch_width=patch_width,
        )

    def _prepare_grouped_modality(
        self,
        name: str,
        role: ModalityRole,
        values: torch.Tensor,
        dimension_mask: torch.Tensor,
        valid_rows: torch.Tensor,
    ):
        dimension = values.shape[-1]
        if values.shape[1] != self.config.chunk_size:
            raise ValueError(
                f"{name} must contain {self.config.chunk_size} timesteps, got {values.shape[1]}."
            )
        clean = values.float().view(
            values.shape[0],
            self.config.num_groups,
            self.config.group_size,
            dimension,
        )
        element_mask = dimension_mask.to(clean.device).to(torch.bool)
        element_mask = element_mask[:, None, None, :].expand_as(clean)
        element_mask = element_mask & valid_rows[:, None, None, None]
        grouped_mask = element_mask.reshape(clean.shape[0], self.config.num_groups, -1)
        group_keep = grouped_mask.any(dim=-1)

        if role is ModalityRole.ABSENT:
            return (
                torch.zeros_like(values),
                torch.zeros_like(dimension_mask),
                None,
                None,
                None,
                None,
            )
        if role is ModalityRole.CLEAN:
            return values.float(), dimension_mask.float(), None, None, group_keep, torch.zeros(
                values.shape[0],
                device=values.device,
            )
        if role is ModalityRole.CURRENT_ONLY:
            current = values[:, :1].expand(-1, self.config.chunk_size, -1).float()
            return current, dimension_mask.float(), None, None, group_keep, torch.zeros(
                values.shape[0],
                device=values.device,
            )

        sigma = self._sample_sigma(values.shape[0], values.device)
        noisy, target = self._flow_corrupt(clean, sigma)
        noisy = torch.where(element_mask, noisy, torch.zeros_like(noisy))
        target = torch.where(element_mask, target, torch.zeros_like(target))
        if role is ModalityRole.FUTURE_NOISY:
            noisy[:, 0, 0] = clean[:, 0, 0]
            grouped_mask[:, 0, :dimension] = False
            target[:, 0, 0] = 0
        return (
            noisy.reshape_as(values),
            (dimension_mask.to(values.device) * valid_rows.unsqueeze(-1)).float(),
            target.reshape(values.shape[0], self.config.num_groups, -1),
            grouped_mask,
            group_keep,
            sigma,
        )

    def _physical_context(
        self,
        physical_batch: dict,
        *,
        include_tactile: bool,
        force_no_grad: bool = False,
    ):
        frozen = self.config.physical_tuning_mode == "frozen" or force_no_grad
        context = torch.no_grad() if frozen else nullcontext()
        with context:
            return self.physical_encoder(
                physical_batch,
                return_tokens=True,
                apply_modality_dropout=False,
                include_tactile=include_tactile,
                compute_reconstruction=False,
            )

    def _prepare_physical_streams(
        self,
        task: TaskSpec,
        batch: dict,
        valid_rows: torch.Tensor,
    ):
        streams: list[GenerationStream] = []
        targets: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        sigmas: dict[str, torch.Tensor] = {}
        if task.state is ModalityRole.ABSENT and task.action is ModalityRole.ABSENT:
            return streams, targets, sigmas

        state = batch["observation.state"]
        action = batch["action"]
        state_input, state_mask, state_target, state_loss_mask, state_keep, state_sigma = (
            self._prepare_grouped_modality(
                "state",
                task.state,
                state,
                batch["state_mask"],
                valid_rows,
            )
        )
        action_input, action_mask, action_target, action_loss_mask, action_keep, action_sigma = (
            self._prepare_grouped_modality(
                "action",
                task.action,
                action,
                batch["action_mask"],
                valid_rows,
            )
        )
        physical_batch = dict(batch)
        physical_batch["observation.state"] = state_input
        physical_batch["state_mask"] = state_mask
        physical_batch["action"] = action_input
        physical_batch["action_mask"] = action_mask
        contextual, _, _ = self._physical_context(
            physical_batch,
            include_tactile=(
                self.config.use_tactile_conditioning
                and task.tactile is ModalityRole.CLEAN
            ),
        )
        start = self.physical_encoder.num_cls_tokens
        state_tokens = contextual[:, start : start + self.config.num_groups]
        action_tokens = contextual[
            :,
            start + self.config.num_groups : start + 2 * self.config.num_groups,
        ]
        if task.state.is_present:
            streams.append(
                GenerationStream("state", state_tokens, state_sigma, state_keep)
            )
            sigmas["state"] = state_sigma
            if state_target is not None:
                targets["state"] = (state_target, state_loss_mask)
        if task.action.is_present:
            streams.append(
                GenerationStream("action", action_tokens, action_sigma, action_keep)
            )
            sigmas["action"] = action_sigma
            if action_target is not None:
                targets["action"] = (action_target, action_loss_mask)
        return streams, targets, sigmas

    def _prepare_tactile_stream(
        self,
        task: TaskSpec,
        batch: dict,
        valid_rows: torch.Tensor,
    ):
        if task.tactile is ModalityRole.ABSENT:
            return None, None, None, None
        if self.config.tactile_generation_target == "spatial_patches":
            clean, keep, position_ids, _ = self._tactile_patch_tokens(
                batch,
                frame_index=-1,
                valid_rows=valid_rows,
            )
            if task.tactile is ModalityRole.CLEAN:
                sigma = torch.zeros(clean.shape[0], device=clean.device)
                return (
                    GenerationStream(
                        "tactile",
                        clean,
                        sigma,
                        keep,
                        position_ids=position_ids,
                    ),
                    None,
                    None,
                    sigma,
                )
            sigma = self._sample_sigma(clean.shape[0], clean.device)
            noisy, target = self._flow_corrupt(clean, sigma)
            noisy = torch.where(keep.unsqueeze(-1), noisy, torch.zeros_like(noisy))
            target = torch.where(keep.unsqueeze(-1), target, torch.zeros_like(target))
            loss_mask = keep.unsqueeze(-1).expand_as(target)
            return (
                GenerationStream(
                    "tactile",
                    noisy,
                    sigma,
                    keep,
                    position_ids=position_ids,
                ),
                target,
                loss_mask,
                sigma,
            )
        clean_tokens, token_keep, _ = self._physical_context(
            batch,
            include_tactile=True,
            force_no_grad=True,
        )
        start = self.physical_encoder.num_cls_tokens + 2 * self.config.num_groups
        clean = clean_tokens[:, start:].detach()
        keep = token_keep[:, start:] & valid_rows.unsqueeze(1)
        if task.tactile is ModalityRole.CLEAN:
            sigma = torch.zeros(clean.shape[0], device=clean.device)
            return GenerationStream("tactile", clean, sigma, keep), None, None, sigma
        sigma = self._sample_sigma(clean.shape[0], clean.device)
        noisy, target = self._flow_corrupt(clean, sigma)
        loss_mask = keep.unsqueeze(-1).expand_as(target)
        return GenerationStream("tactile", noisy, sigma, keep), target, loss_mask, sigma

    def _tactile_patch_tokens(
        self,
        batch: dict,
        *,
        frame_index: int,
        valid_rows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int, int]]:
        images = batch["tactile_image"][:, :, frame_index]
        image_mask = batch["tactile_image_mask"].to(images.device) > 0
        batch_size, views = images.shape[:2]
        selected = image_mask.reshape(-1).nonzero(as_tuple=True)[0]
        grid_height = (images.shape[-2] + 31) // 32
        grid_width = (images.shape[-1] + 31) // 32
        feature_dim = int(self.physical_encoder.tactile_feat_dim)
        if selected.numel():
            selected_images = images.reshape(
                batch_size * views,
                *images.shape[2:],
            )[selected].unsqueeze(1)
            with torch.no_grad():
                selected_patches = self.physical_encoder.encode_tactile_patches(
                    selected_images
                )[:, 0]
            grid_height, grid_width = selected_patches.shape[-2:]
            patch_maps = torch.zeros(
                batch_size * views,
                feature_dim,
                grid_height,
                grid_width,
                device=selected_patches.device,
                dtype=selected_patches.dtype,
            )
            patch_maps = patch_maps.index_put((selected,), selected_patches)
        else:
            dtype = self.generation.input_projections["tactile"].weight.dtype
            patch_maps = torch.zeros(
                batch_size * views,
                feature_dim,
                grid_height,
                grid_width,
                device=images.device,
                dtype=dtype,
            )
        patch_maps = patch_maps.view(
            batch_size,
            views,
            feature_dim,
            grid_height,
            grid_width,
        )
        values = patch_maps.permute(0, 1, 3, 4, 2).reshape(
            batch_size,
            views * grid_height * grid_width,
            feature_dim,
        )
        keep = image_mask.repeat_interleave(
            grid_height * grid_width,
            dim=1,
        )
        keep = keep & valid_rows.to(keep.device).unsqueeze(1)
        view_ids = torch.arange(views, device=values.device)
        rows = torch.arange(grid_height, device=values.device)
        columns = torch.arange(grid_width, device=values.device)
        position_ids = torch.cartesian_prod(view_ids, rows, columns)
        return (
            values.detach(),
            keep,
            position_ids,
            (views, grid_height, grid_width),
        )

    def _sample_tactile_images(
        self,
        batch: dict,
        understanding,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = batch["image_t0"].shape[0]
        image_mask = batch["tactile_image_mask"].to(
            batch["image_t0"].device
        ) > 0
        if self.config.tactile_generation_target != "spatial_patches":
            return (
                torch.empty(batch_size, 0, device=image_mask.device),
                image_mask,
            )
        if "tactile_prediction" not in self.generation.task_to_id:
            raise ValueError(
                "Generation validation requires the enabled task 'tactile_prediction'."
            )
        valid_rows = (
            self._task_valid_rows(
                TASK_SPECS["tactile_prediction"],
                batch,
                batch["image_t0"].device,
            )
        )
        _, keep, position_ids, geometry = self._tactile_patch_tokens(
            batch,
            frame_index=0,
            valid_rows=valid_rows,
        )
        views, grid_height, grid_width = geometry
        values = torch.randn(
            batch_size,
            views * grid_height * grid_width,
            int(self.physical_encoder.tactile_feat_dim),
            device=keep.device,
            dtype=torch.float32,
            generator=generator,
        )
        values = torch.where(
            keep.unsqueeze(-1),
            values,
            torch.zeros_like(values),
        )
        task = TASK_SPECS["tactile_prediction"]
        video_stream, _, _, _ = self._prepare_video_stream(
            task.video,
            batch,
            valid_rows,
        )
        physical_streams, _, _ = self._prepare_physical_streams(
            task,
            batch,
            valid_rows,
        )
        conditioning_streams = [video_stream, *physical_streams]
        schedule = torch.linspace(
            1.0,
            0.0,
            self.config.inference_steps + 1,
            device=values.device,
        )
        for sigma, next_sigma in zip(schedule[:-1], schedule[1:], strict=True):
            sigma_batch = sigma.expand(batch_size)
            prediction = self.generation(
                [
                    *conditioning_streams,
                    GenerationStream(
                        "tactile",
                        values,
                        sigma_batch,
                        keep,
                        position_ids=position_ids,
                    ),
                ],
                understanding.hidden_states,
                understanding.attention_mask,
                "tactile_prediction",
                understanding_key_values=understanding.key_values,
            )["tactile"].float()
            values = values + (next_sigma - sigma) * prediction
            values = torch.where(
                keep.unsqueeze(-1),
                values,
                torch.zeros_like(values),
            )
        patches = values.view(
            batch_size,
            views,
            grid_height,
            grid_width,
            int(self.physical_encoder.tactile_feat_dim),
        ).permute(0, 1, 4, 2, 3)
        decoded = self.physical_encoder.decode_tactile_patches(patches)
        return decoded.float(), image_mask

    def _latent_action_target(self, batch: dict, enabled: bool) -> tuple[torch.Tensor | None, torch.Tensor]:
        device = batch["image_t0"].device
        pair_valid = batch["pair_is_valid"].to(device).reshape(-1) > 0.5
        if not enabled:
            return None, pair_valid & False
        target = torch.zeros(
            batch["image_t0"].shape[0],
            self.config.num_latent_queries,
            self.config.latent_action_dim,
            device=device,
            dtype=self.latent_action_projection[1].weight.dtype,
        )
        if not pair_valid.any():
            return target, pair_valid
        indices = pair_valid.nonzero(as_tuple=True)[0]
        texts = [batch["task"][int(index)] for index in indices.tolist()]
        has_text = batch.get("has_text")
        has_text = has_text[indices] if has_text is not None else None
        with torch.no_grad():
            _, _, aux = self.teacher_perception(
                batch["image_t0"][indices],
                batch["image_t1"][indices],
                texts,
                has_text,
                return_latent_action=True,
            )
        target[indices] = aux["latent_action"].to(target.dtype)
        return target, pair_valid

    def _distributed_mse(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float]:
        error = (prediction.float() - target.float()).square()
        while mask.ndim < error.ndim:
            mask = mask.unsqueeze(-1)
        mask = mask.to(device=error.device, dtype=error.dtype).expand_as(error)
        local_sum = (error * mask).sum()
        local_count = mask.sum()
        global_count = _all_reduce_detached(local_count)
        if global_count.item() == 0:
            return prediction.sum() * 0.0, 0.0, 0.0
        loss = local_sum * (_distributed_world_size() / global_count)
        global_sum = _all_reduce_detached(local_sum)
        return loss, (global_sum / global_count).item(), global_count.item()

    @staticmethod
    def _generation_rng(device: torch.device, seed: int) -> torch.Generator:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        return generator

    @torch.no_grad()
    def generate_validation(
        self,
        batch: dict,
        *,
        seed: int,
    ) -> dict[str, torch.Tensor]:
        """Generate fixed-noise action, video, and decodable tactile predictions."""
        image = self._current_image(batch)
        device = next(self.generation.parameters()).device
        image = image.to(device)
        texts = batch.get("task", [""] * image.shape[0])
        if isinstance(texts, str):
            texts = [texts] * image.shape[0]
        understanding = self.understanding(image, texts)
        action = self._sample_canonical_action(
            batch,
            understanding,
            generator=self._generation_rng(device, seed),
        )
        video_latents = self._sample_video_latents(
            batch,
            understanding,
            generator=self._generation_rng(device, seed + 1),
        )
        tactile_decoded_z, tactile_mask = self._sample_tactile_images(
            batch,
            understanding,
            generator=self._generation_rng(device, seed + 2),
        )
        return {
            "action_normalized": action,
            "video_latents": video_latents,
            "tactile_decoded_z": tactile_decoded_z,
            "tactile_mask": tactile_mask,
        }

    def forward(
        self,
        batch: dict,
        task_type: str = "train_stage2",
        step: int = 0,
    ):
        if task_type == "generate_validation":
            return self.generate_validation(batch, seed=step)
        del step
        forced_task = task_type if task_type in TASK_SPECS else None
        if task_type not in ("train_stage2", "qwen3vl_mot") and forced_task is None:
            raise ValueError(
                f"Unsupported qwen3vl_mot task_type {task_type!r}; use train_stage2 or a "
                f"specific task from {tuple(TASK_SPECS)}."
            )
        task, valid_rows = self._select_task(batch, forced_task)
        understanding = self.understanding(
            batch["image_t0"] if task.understanding_image else None,
            batch["task"],
        )
        projection_dtype = next(self.latent_action_projection.parameters()).dtype
        predicted_latent_action = self.latent_action_projection(
            understanding.latent_queries.to(projection_dtype)
        )
        latent_target, latent_rows = self._latent_action_target(
            batch,
            enabled=task.understanding_image,
        )

        streams: list[GenerationStream] = []
        targets: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        sigmas: dict[str, torch.Tensor] = {}
        video_stream, video_target, video_mask, video_sigma = self._prepare_video_stream(
            task.video,
            batch,
            valid_rows,
        )
        if video_stream is not None:
            streams.append(video_stream)
            sigmas["video"] = video_sigma
            if video_target is not None:
                targets["video"] = (video_target, video_mask)

        physical_streams, physical_targets, physical_sigmas = self._prepare_physical_streams(
            task,
            batch,
            valid_rows,
        )
        streams.extend(physical_streams)
        targets.update(physical_targets)
        sigmas.update(physical_sigmas)

        tactile_stream, tactile_target, tactile_mask, tactile_sigma = self._prepare_tactile_stream(
            task,
            batch,
            valid_rows,
        )
        if tactile_stream is not None:
            streams.append(tactile_stream)
            sigmas["tactile"] = tactile_sigma
            if tactile_target is not None:
                targets["tactile"] = (tactile_target, tactile_mask)

        predictions = self.generation(
            streams,
            understanding.hidden_states,
            understanding.attention_mask,
            task.name,
            understanding_key_values=understanding.key_values,
        )
        weights = {
            "video": self.config.video_loss_weight,
            "action": self.config.action_loss_weight,
            "state": self.config.state_loss_weight,
            "tactile": self.config.tactile_loss_weight,
        }
        loss = predicted_latent_action.sum() * 0.0
        loss_dict = {f"task_{name}": float(name == task.name) for name in self.config.task_names}
        for name, (target, mask) in targets.items():
            modality_loss, metric, count = self._distributed_mse(
                predictions[name],
                target,
                mask,
            )
            loss = loss + weights[name] * modality_loss
            loss_dict[f"{name}_flow_loss"] = metric
            loss_dict[f"{name}_target_elements"] = count
            loss_dict[f"{name}_sigma"] = sigmas[name].float().mean().item()

        if task.understanding_image:
            latent_loss, metric, count = self._distributed_mse(
                predicted_latent_action,
                latent_target,
                latent_rows[:, None, None],
            )
            loss = loss + self.config.latent_action_loss_weight * latent_loss
            loss_dict["latent_action_loss"] = metric
            loss_dict["latent_action_elements"] = count
        else:
            loss_dict["latent_action_loss"] = 0.0
            loss_dict["latent_action_elements"] = 0.0
        loss_dict["valid_rows"] = _all_reduce_detached(valid_rows.sum().float()).item()
        return loss, loss_dict
