"""Qwen3-VL understanding expert with appended latent-action queries."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class UnderstandingOutput:
    hidden_states: list[torch.Tensor]
    key_values: list[tuple[torch.Tensor, torch.Tensor]]
    attention_mask: torch.Tensor
    latent_queries: torch.Tensor


def _base_qwen(model: nn.Module) -> nn.Module:
    get_base_model = getattr(model, "get_base_model", None)
    return get_base_model() if callable(get_base_model) else model


def _selected_layers(total_layers: int, requested: int) -> tuple[int, ...]:
    count = min(total_layers, requested)
    if count == 1:
        return (total_layers - 1,)
    positions = torch.linspace(0, total_layers - 1, steps=count).round().to(torch.long).tolist()
    return tuple(dict.fromkeys(int(position) for position in positions))


def _forward_decoder_layer_with_kv(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    *,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    cache_position: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a decoder block normally while retaining its native projected K/V."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

    wrapped_layer = getattr(layer, "_fsdp_wrapped_module", layer)
    attention = wrapped_layer.self_attn
    captured: dict[str, torch.Tensor] = {}

    def capture_key(_module, _inputs, output):
        captured["key"] = output

    def capture_value(_module, _inputs, output):
        captured["value"] = output

    handles = (
        attention.k_norm.register_forward_hook(capture_key),
        attention.v_proj.register_forward_hook(capture_value),
    )
    try:
        output = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
    finally:
        for handle in handles:
            handle.remove()

    if "key" not in captured or "value" not in captured:
        raise RuntimeError("Qwen decoder layer did not expose its projected key/value states.")
    key = captured["key"].transpose(1, 2)
    value_shape = (*captured["value"].shape[:-1], -1, attention.head_dim)
    value = captured["value"].view(value_shape).transpose(1, 2)
    _, key = apply_rotary_pos_emb(key, key, *position_embeddings)
    return output, key, value


class Qwen3VLUnderstandingExpert(nn.Module):
    """Runs Qwen3-VL on ``[image_t, instruction, learnable queries]``."""

    def __init__(
        self,
        model_dir: str,
        *,
        num_queries: int,
        max_text_tokens: int,
        kv_layers: int,
        tuning_mode: str,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        text_lora_layers: int,
        vision_lora_layers: int,
        gradient_checkpointing: bool,
    ):
        super().__init__()
        from transformers import AutoTokenizer
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

        full_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        self.model = full_model.model
        del full_model
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.num_queries = num_queries
        self.max_text_tokens = max_text_tokens
        hidden_dim = int(self.model.config.text_config.hidden_size)
        self.query_tokens = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        self.selected_layer_indices = _selected_layers(
            int(self.model.config.text_config.num_hidden_layers),
            kv_layers,
        )
        self.gradient_checkpointing = gradient_checkpointing
        self.tuning_mode = tuning_mode
        self._configure_tuning(
            tuning_mode=tuning_mode,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            text_layers=text_lora_layers,
            vision_layers=vision_lora_layers,
        )
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    @property
    def hidden_dim(self) -> int:
        return int(_base_qwen(self.model).config.text_config.hidden_size)

    def _configure_tuning(
        self,
        *,
        tuning_mode: str,
        rank: int,
        alpha: int,
        dropout: float,
        text_layers: int,
        vision_layers: int,
    ) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if tuning_mode == "frozen":
            return
        if tuning_mode not in ("lora", "full"):
            raise ValueError(f"Unsupported understanding tuning mode {tuning_mode!r}.")

        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "Qwen3-VL LoRA requires `peft`; install the repository's contrast extras."
            ) from exc

        base = _base_qwen(self.model)
        text_depth = len(base.language_model.layers)
        vision_depth = len(base.visual.blocks)
        if text_layers > text_depth or vision_layers > vision_depth:
            raise ValueError(
                "Requested more Qwen LoRA layers than the checkpoint contains: "
                f"text {text_layers}/{text_depth}, vision {vision_layers}/{vision_depth}."
            )
        effective_text_layers = text_layers if tuning_mode == "lora" else 0
        text_start = text_depth - effective_text_layers
        vision_start = vision_depth - vision_layers
        targets = []
        for name, module in base.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            text_prefix = "language_model.layers."
            vision_prefix = "visual.blocks."
            if name.startswith(text_prefix):
                layer = int(name[len(text_prefix) :].split(".", 1)[0])
                if layer >= text_start and name.rsplit(".", 1)[-1] in {
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                }:
                    targets.append(name)
            elif name.startswith(vision_prefix):
                layer = int(name[len(vision_prefix) :].split(".", 1)[0])
                if layer >= vision_start and name.rsplit(".", 1)[-1] in {"qkv", "proj"}:
                    targets.append(name)
        if not targets:
            raise RuntimeError("No Qwen3-VL attention projections were found for LoRA.")
        self.model = get_peft_model(
            self.model,
            LoraConfig(
                r=rank,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=targets,
                bias="none",
                init_lora_weights=True,
            ),
        )
        if tuning_mode == "full":
            # Qwen3-VL performs multimodal fusion inside its decoder transformer. Train those
            # blocks fully, but keep token embeddings/final norm frozen and retain LoRA-only
            # adaptation for the visual backbone.
            for parameter in _base_qwen(self.model).language_model.layers.parameters():
                parameter.requires_grad_(True)

    def _pack_images(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        base = _base_qwen(self.model)
        visual = base.visual
        config = visual.config
        size = 256
        input_is_float = images.is_floating_point()
        images = images.to(dtype=torch.float32)
        if images.shape[-2:] != (size, size):
            images = F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)
        if input_is_float and images.detach().amin().item() >= 0.0 and images.detach().amax().item() <= 1.0:
            images = images * 2.0 - 1.0
        else:
            images = images / 127.5 - 1.0

        batch, channels, height, width = images.shape
        patch = int(config.patch_size)
        merge = int(config.spatial_merge_size)
        temporal = int(config.temporal_patch_size)
        if height % (patch * merge) or width % (patch * merge):
            raise ValueError(
                f"Qwen3-VL image size must be divisible by {patch * merge}, got "
                f"{height}x{width}."
            )
        grid_h, grid_w = height // patch, width // patch
        block_h, block_w = grid_h // merge, grid_w // merge
        packed = images.view(
            batch,
            channels,
            block_h,
            merge,
            patch,
            block_w,
            merge,
            patch,
        )
        packed = packed.permute(0, 2, 5, 3, 6, 1, 4, 7)
        packed = packed.unsqueeze(6).expand(
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            temporal,
            -1,
            -1,
        )
        packed = packed.reshape(
            batch * grid_h * grid_w,
            channels * temporal * patch * patch,
        )
        visual_dtype = next(visual.parameters()).dtype
        packed = packed.to(device=images.device, dtype=visual_dtype)
        grid_thw = torch.tensor(
            [[1, grid_h, grid_w]] * batch,
            dtype=torch.long,
        )

        visual_context = (
            torch.enable_grad()
            if any(parameter.requires_grad for parameter in visual.parameters())
            else torch.no_grad()
        )
        with visual_context:
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
                and any(parameter.requires_grad for parameter in visual.parameters())
            ):
                image_features, deepstack_features = checkpoint(
                    lambda pixels: visual(pixels, grid_thw=grid_thw),
                    packed,
                    use_reentrant=False,
                )
            else:
                image_features, deepstack_features = visual(packed, grid_thw=grid_thw)
        merged_tokens = grid_h * grid_w // (merge * merge)
        image_features = image_features.view(batch, merged_tokens, -1)
        deepstack_features = [
            feature.view(batch, merged_tokens, -1)
            for feature in deepstack_features
        ]
        return image_features, deepstack_features, grid_thw

    def _tokenize(self, texts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_tokens,
            return_tensors="pt",
        )
        return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)

    def forward(
        self,
        images: torch.Tensor | None,
        texts: list[str],
    ) -> UnderstandingOutput:
        from transformers.masking_utils import create_causal_mask

        base = _base_qwen(self.model)
        language_model = base.language_model
        embedding = language_model.embed_tokens
        device = embedding.weight.device
        dtype = embedding.weight.dtype
        input_ids, text_keep = self._tokenize(texts, device)
        batch = input_ids.shape[0]
        text_embeddings = embedding(input_ids)
        query_embeddings = self.query_tokens.to(dtype).unsqueeze(0).expand(batch, -1, -1)
        query_ids = torch.full(
            (batch, self.num_queries),
            self.tokenizer.pad_token_id,
            device=device,
            dtype=torch.long,
        )
        query_keep = torch.ones(batch, self.num_queries, device=device, dtype=text_keep.dtype)

        image_slice = None
        deepstack_features: list[torch.Tensor] = []
        image_grid_thw = None
        if images is not None:
            image_features, deepstack_features, image_grid_thw = self._pack_images(images.to(device))
            image_features = image_features.to(dtype)
            num_image_tokens = image_features.shape[1]
            vision_start = torch.full(
                (batch, 1),
                base.config.vision_start_token_id,
                device=device,
                dtype=torch.long,
            )
            image_ids = torch.full(
                (batch, num_image_tokens),
                base.config.image_token_id,
                device=device,
                dtype=torch.long,
            )
            vision_end = torch.full(
                (batch, 1),
                base.config.vision_end_token_id,
                device=device,
                dtype=torch.long,
            )
            prefix_ids = torch.cat([vision_start, image_ids, vision_end], dim=1)
            prefix_embeddings = embedding(prefix_ids)
            image_slice = slice(1, 1 + num_image_tokens)
            prefix_embeddings = torch.cat(
                [
                    prefix_embeddings[:, :1],
                    image_features,
                    prefix_embeddings[:, 1 + num_image_tokens :],
                ],
                dim=1,
            )
            input_ids = torch.cat([prefix_ids, input_ids, query_ids], dim=1)
            attention_keep = torch.cat(
                [
                    torch.ones_like(prefix_ids),
                    text_keep,
                    query_keep,
                ],
                dim=1,
            )
            hidden_states = torch.cat(
                [prefix_embeddings, text_embeddings, query_embeddings],
                dim=1,
            )
        else:
            input_ids = torch.cat([input_ids, query_ids], dim=1)
            attention_keep = torch.cat([text_keep, query_keep], dim=1)
            hidden_states = torch.cat([text_embeddings, query_embeddings], dim=1)

        position_ids, _ = base.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_keep,
        )
        cache_position = torch.arange(hidden_states.shape[1], device=device)
        causal_mask = create_causal_mask(
            config=language_model.config,
            input_embeds=hidden_states,
            attention_mask=attention_keep,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids[0],
        )
        position_embeddings = language_model.rotary_emb(hidden_states, position_ids)

        selected_key_values: list[tuple[torch.Tensor, torch.Tensor]] = []
        selected_set = set(self.selected_layer_indices)
        for layer_index, layer in enumerate(language_model.layers):
            capture_key_values = layer_index in selected_set
            if (
                getattr(self, "gradient_checkpointing", False)
                and self.training
                and torch.is_grad_enabled()
            ):
                if capture_key_values:
                    hidden_states, key, value = checkpoint(
                        lambda states, cos, sin, _layer=layer: _forward_decoder_layer_with_kv(
                            _layer,
                            states,
                            attention_mask=causal_mask,
                            position_ids=position_ids[0],
                            cache_position=cache_position,
                            position_embeddings=(cos, sin),
                        ),
                        hidden_states,
                        position_embeddings[0],
                        position_embeddings[1],
                        use_reentrant=False,
                    )
                else:
                    hidden_states = checkpoint(
                        lambda states, cos, sin, _layer=layer: _layer(
                            states,
                            attention_mask=causal_mask,
                            position_ids=position_ids[0],
                            past_key_values=None,
                            cache_position=cache_position,
                            position_embeddings=(cos, sin),
                        ),
                        hidden_states,
                        position_embeddings[0],
                        position_embeddings[1],
                        use_reentrant=False,
                    )
            elif capture_key_values:
                hidden_states, key, value = _forward_decoder_layer_with_kv(
                    layer,
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids[0],
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids[0],
                    past_key_values=None,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            if capture_key_values:
                selected_key_values.append((key, value))
            if image_slice is not None and layer_index < len(deepstack_features):
                hidden_states = hidden_states.clone()
                hidden_states[:, image_slice] = (
                    hidden_states[:, image_slice]
                    + deepstack_features[layer_index].to(device=device, dtype=dtype)
                )
        hidden_states = language_model.norm(hidden_states)
        return UnderstandingOutput(
            hidden_states=[],
            key_values=selected_key_values,
            attention_mask=attention_keep.to(torch.bool),
            latent_queries=hidden_states[:, -self.num_queries :],
        )
