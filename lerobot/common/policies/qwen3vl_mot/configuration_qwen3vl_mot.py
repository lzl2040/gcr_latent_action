"""Configuration for the Qwen3-VL stage-two mixture-of-transformers model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lerobot.common.optim.optimizers import AdamW8bitConfig
from lerobot.common.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.configs.policies import PreTrainedConfig

from .tasks import DEFAULT_TASK_NAMES, DEFAULT_TASK_WEIGHTS, resolve_task_specs


@PreTrainedConfig.register_subclass("qwen3vl_mot")
@dataclass
class Qwen3VLMoTConfig(PreTrainedConfig):
    """Stage two: Qwen understanding, stage-one physical tokens, and flow generation."""

    pretrained_path: str | None = None
    stage1_checkpoint: str = ""
    stage1_config: str = ""
    stage1_policy_config: dict[str, Any] | None = None
    initialize_from_stage1: bool = True
    require_stage1_vision_transfer: bool = True
    require_stage1_text_transfer: bool = False

    qwen3vl_dir: str = "/Data/lzl/huggingface/Qwen3-VL-4B-Instruct"
    cosmos3_dir: str = "/Data/lzl/huggingface/Cosmos3-Edge"
    # "full" means full tuning of Qwen's multimodal language transformer layers.
    # The visual backbone still uses LoRA, while token embeddings and the final language
    # norm remain frozen.
    understanding_tuning_mode: str = "lora"
    understanding_lora_rank: int = 16
    understanding_lora_alpha: int = 16
    understanding_lora_dropout: float = 0.0
    understanding_text_lora_layers: int = 0
    understanding_vision_lora_layers: int = 4
    understanding_lr_scale: float = 0.1
    understanding_gradient_checkpointing: bool = True
    understanding_max_text_tokens: int = 64
    understanding_kv_layers: int = 8

    num_latent_queries: int = 16
    latent_action_dim: int = 1024
    latent_action_loss_weight: float = 1.0

    # Cosmos3-Edge-sized generation path. Qwen3-VL-4B supplies 8 native KV heads
    # of width 128; the generator expands only its query pathway to 16 heads.
    generation_hidden_dim: int = 2048
    generation_depth: int = 28
    generation_num_heads: int = 16
    generation_num_kv_heads: int = 8
    generation_intermediate_dim: int = 9216
    generation_hidden_act: str = "relu2"
    generation_dropout: float = 0.0
    generation_gradient_checkpointing: bool = True
    video_latent_dim: int = 48
    video_latent_patch_size: int = 2
    world_video_frames: int = 9
    video_image_size: int = 256
    load_vae_decoder: bool = False

    chunk_size: int = 32
    group_size: int = 4
    n_action_steps: int = 32
    max_action_dim: int = 40
    max_state_dim: int = 40
    max_tactile_signal_dim: int = 32
    physical_hidden_dim: int = 1024
    physical_tuning_mode: str = "frozen"
    physical_lr_scale: float = 0.3
    max_tactile_views: int = 6
    tactile_tokens_per_pad: int = 2
    use_tactile_conditioning: bool = False

    task_names: tuple[str, ...] = DEFAULT_TASK_NAMES
    task_weights: tuple[float, ...] = DEFAULT_TASK_WEIGHTS
    sigma_min: float = 1e-3
    sigma_max: float = 0.999
    video_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    state_loss_weight: float = 1.0
    tactile_loss_weight: float = 0.25
    inference_steps: int = 20

    # Stage two predicts executable action chunks, so these must be consecutive source
    # commands rather than the duration-resampled trajectories used by stage-one contrastive
    # pre-training. Video is sampled sparsely inside the same [t, t + chunk_size - 1] window.
    window_mode: str = "frames"
    chunk_seconds: float = 1.6
    chunk_frames_min: int = 8
    chunk_frames_max: int = 48
    frame_horizon: int | None = None
    tactile_img_size: int = 112
    tactile_frames: int = 4
    tactile_dead_std: float = 0.002
    use_wrist_image: bool = False
    same_dataset_frac: float = 0.75
    episode_group_frac: float = 0.75
    episode_group_size: int = 8
    min_frame_gap: int = 32

    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    scheduler_warmup_steps: int = 500
    scheduler_plateau_steps: int = 2_000
    scheduler_decay_steps: int = 100_000
    scheduler_decay_lr: float = 1.5e-6

    def __post_init__(self):
        super().__post_init__()
        if self.understanding_tuning_mode not in ("frozen", "lora", "full"):
            raise ValueError(
                "`understanding_tuning_mode` must be 'frozen', 'lora', or 'full', got "
                f"{self.understanding_tuning_mode!r}."
            )
        if self.understanding_text_lora_layers < 0 or self.understanding_vision_lora_layers < 0:
            raise ValueError("Understanding LoRA layer counts must be non-negative.")
        if (
            self.understanding_tuning_mode == "lora"
            and self.understanding_text_lora_layers == 0
            and self.understanding_vision_lora_layers == 0
        ):
            raise ValueError(
                "LoRA tuning requires at least one text or vision layer."
            )
        if (
            self.understanding_tuning_mode == "full"
            and self.understanding_text_lora_layers != 0
        ):
            raise ValueError(
                "Full VLM-transformer tuning keeps the text embedding path free of LoRA; "
                "set `understanding_text_lora_layers=0`."
            )
        if (
            self.understanding_tuning_mode == "full"
            and self.understanding_vision_lora_layers == 0
        ):
            raise ValueError(
                "Full VLM-transformer tuning still adapts the visual backbone with LoRA; "
                "set `understanding_vision_lora_layers` to a positive value."
            )
        if self.physical_tuning_mode not in ("frozen", "full"):
            raise ValueError(
                "`physical_tuning_mode` must be 'frozen' or 'full', got "
                f"{self.physical_tuning_mode!r}."
            )
        if self.chunk_size <= 0 or self.group_size <= 0 or self.chunk_size % self.group_size:
            raise ValueError(
                f"`chunk_size` ({self.chunk_size}) must be divisible by positive "
                f"`group_size` ({self.group_size})."
            )
        if not 1 <= self.n_action_steps <= self.chunk_size:
            raise ValueError(
                f"`n_action_steps` must be in [1, chunk_size], got {self.n_action_steps} "
                f"for chunk_size={self.chunk_size}."
            )
        if self.window_mode != "frames":
            raise ValueError(
                "qwen3vl_mot requires `window_mode='frames'` so action/state chunks are "
                "consecutive executable commands."
            )
        if self.frame_horizon not in (None, self.chunk_size - 1):
            raise ValueError(
                "`frame_horizon` must be unset or equal to `chunk_size - 1` in frame mode; "
                "otherwise video and action would end at different timesteps."
            )
        if self.world_video_frames < 2:
            raise ValueError("`world_video_frames` must contain the current and at least one future frame.")
        if (self.world_video_frames - 1) % 4:
            raise ValueError(
                "Wan's causal temporal stride requires `world_video_frames = 1 + 4*k`; "
                f"got {self.world_video_frames}."
            )
        if self.video_image_size <= 0 or self.video_image_size % 16:
            raise ValueError("`video_image_size` must be positive and divisible by the VAE stride 16.")
        if self.generation_hidden_dim % self.generation_num_heads:
            raise ValueError(
                f"`generation_hidden_dim` ({self.generation_hidden_dim}) must be divisible by "
                f"`generation_num_heads` ({self.generation_num_heads})."
            )
        if not 1 <= self.generation_num_kv_heads <= self.generation_num_heads:
            raise ValueError(
                "`generation_num_kv_heads` must be positive and no larger than "
                "`generation_num_heads`."
            )
        if self.generation_num_heads % self.generation_num_kv_heads:
            raise ValueError(
                "`generation_num_heads` must be divisible by `generation_num_kv_heads`."
            )
        if self.generation_intermediate_dim <= 0:
            raise ValueError("`generation_intermediate_dim` must be positive.")
        if self.generation_hidden_act not in ("silu", "relu2"):
            raise ValueError("`generation_hidden_act` must be 'silu' or 'relu2'.")
        if self.video_latent_patch_size <= 0:
            raise ValueError("`video_latent_patch_size` must be positive.")
        if self.understanding_kv_layers < 1:
            raise ValueError("`understanding_kv_layers` must be positive.")
        if not 0 <= self.sigma_min < self.sigma_max <= 1:
            raise ValueError(
                f"Expected 0 <= sigma_min < sigma_max <= 1, got "
                f"{self.sigma_min} and {self.sigma_max}."
            )
        resolve_task_specs(self.task_names, self.task_weights)

    @property
    def num_groups(self) -> int:
        return self.chunk_size // self.group_size

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def validate_features(self) -> None:
        return None

    def get_optimizer_preset(self) -> AdamW8bitConfig:
        return AdamW8bitConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_platform_steps=self.scheduler_plateau_steps,
            num_decay_steps=self.scheduler_decay_steps,
            decay_from_platform_end=True,
        )
