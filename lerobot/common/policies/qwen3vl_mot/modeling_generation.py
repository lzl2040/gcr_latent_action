"""Cosmos-style one-way understanding-to-generation transformer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .cosmos3_pretrained import (
    CosmosRMSNorm,
    CosmosRotaryEmbedding,
    generation_checkpoint_mapping,
    rotate_half,
)


def _sinusoidal_embedding(
    values: torch.Tensor,
    dim: int,
    *,
    max_period: float = 10_000.0,
) -> torch.Tensor:
    """Continuous sinusoidal embedding with a stable fp32 phase computation."""
    half = dim // 2
    if half == 0:
        return values.unsqueeze(-1)
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=values.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    phase = values.float().unsqueeze(-1) * frequencies
    embedding = torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


@dataclass
class GenerationStream:
    """One fixed-width token stream handed to the generation expert."""

    name: str
    values: torch.Tensor
    sigma: torch.Tensor
    keep: torch.Tensor
    position_ids: torch.Tensor | None = None
    view_ids: torch.Tensor | None = None
    noise_mask: torch.Tensor | None = None
    fps: torch.Tensor | None = None

    def validate(self, input_dim: int) -> None:
        if self.values.ndim != 3 or self.values.shape[-1] != input_dim:
            raise ValueError(
                f"{self.name} values must have shape (B,N,{input_dim}), got "
                f"{tuple(self.values.shape)}."
            )
        if self.sigma.shape != (self.values.shape[0],):
            raise ValueError(
                f"{self.name} sigma must have shape ({self.values.shape[0]},), got "
                f"{tuple(self.sigma.shape)}."
            )
        if self.keep.shape != self.values.shape[:2]:
            raise ValueError(
                f"{self.name} keep mask must have shape {tuple(self.values.shape[:2])}, got "
                f"{tuple(self.keep.shape)}."
            )
        if self.position_ids is not None and self.position_ids.shape not in (
            (self.values.shape[1], 3),
            (self.values.shape[0], self.values.shape[1], 3),
        ):
            raise ValueError(
                f"{self.name} position_ids must have shape (N,3) or (B,N,3), got "
                f"{tuple(self.position_ids.shape)}."
            )
        if self.view_ids is not None and self.view_ids.shape != (self.values.shape[1],):
            raise ValueError(f"{self.name} view_ids must have shape (N,).")
        if self.noise_mask is not None and self.noise_mask.shape != self.keep.shape:
            raise ValueError(f"{self.name} noise_mask must have shape (B,N).")
        if self.fps is not None and self.fps.shape != self.sigma.shape:
            raise ValueError(f"{self.name} fps must have shape (B,).")


class CosmosTimestepLinear(nn.Linear):
    """Keep native time-conditioning GEMMs in FP32, including under BF16/FP8 training."""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=values.device.type, enabled=False):
            bias = None if self.bias is None else self.bias.float()
            return F.linear(values.float(), self.weight.float(), bias)


class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, *, native_cosmos: bool = False):
        super().__init__()
        self.native_cosmos = native_cosmos
        self.frequency_dim = 256 if native_cosmos else min(256, hidden_dim)
        linear = CosmosTimestepLinear if native_cosmos else nn.Linear
        self.mlp = nn.Sequential(
            linear(self.frequency_dim, hidden_dim),
            nn.SiLU(),
            linear(hidden_dim, hidden_dim),
        )

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        if self.native_cosmos:
            # Cosmos applies timestep_scale=.001 to scheduler timesteps (1000*sigma).
            half = self.frequency_dim // 2
            frequencies = torch.exp(
                -math.log(10_000) * torch.arange(half, device=sigma.device).float() / half
            )
            phase = sigma.float().unsqueeze(-1) * frequencies
            embedding = torch.cat([phase.cos(), phase.sin()], dim=-1)
            return self.mlp(embedding)
        return self.mlp(_sinusoidal_embedding(sigma * 1_000.0, self.frequency_dim).to(self.mlp[0].weight.dtype))


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int, dropout: float):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class ReLUSquaredMLP(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int, dropout: float):
        super().__init__()
        self.up = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.relu(self.up(x)).square()))


class AsymmetricAttention(nn.Module):
    """Generation queries attend to understanding KV and generation KV.

    There is deliberately no understanding-query path in this module. Understanding states
    are computed before this block and are only projected into keys and values, so generation
    can consume understanding while understanding can never consume generation.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        num_kv_heads: int | None = None,
        cosmos_config: dict | None = None,
    ):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("Generation hidden size must be divisible by query heads.")
        num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if num_heads % num_kv_heads:
            raise ValueError("Generation query heads must be divisible by KV heads.")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_kv_heads = num_kv_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        kv_dim = self.num_kv_heads * self.head_dim
        self.k_proj = nn.Linear(hidden_dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, kv_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.q_norm = self.k_norm = self.understanding_k_norm = nn.Identity()
        self.key_adapter = self.value_adapter = None
        if cosmos_config is not None:
            def norm():
                return CosmosRMSNorm(
                    self.head_dim, cosmos_config["rms_norm_eps"],
                    nemotron=cosmos_config["hidden_act"] == "relu2",
                )
            self.q_norm, self.k_norm, self.understanding_k_norm = norm(), norm(), norm()
            self.key_adapter = nn.Linear(kv_dim, kv_dim, bias=False)
            self.value_adapter = nn.Linear(kv_dim, kv_dim, bias=False)

    def _heads(self, x: torch.Tensor, num_heads: int | None = None) -> torch.Tensor:
        num_heads = x.shape[-1] // self.head_dim if num_heads is None else num_heads
        batch, length, _ = x.shape
        return x.view(batch, length, num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        generation: torch.Tensor,
        understanding: torch.Tensor | None,
        understanding_keep: torch.Tensor,
        generation_keep: torch.Tensor,
        understanding_key: torch.Tensor | None = None,
        understanding_value: torch.Tensor | None = None,
        generation_cos: torch.Tensor | None = None,
        generation_sin: torch.Tensor | None = None,
        understanding_cos: torch.Tensor | None = None,
        understanding_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.q_norm(self._heads(self.q_proj(generation), self.num_heads))
        generation_key = self.k_norm(self._heads(self.k_proj(generation), self.num_kv_heads))
        generation_value = self._heads(self.v_proj(generation), self.num_kv_heads)
        if generation_cos is not None and generation_sin is not None:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

            q, generation_key = apply_rotary_pos_emb(
                q,
                generation_key,
                generation_cos,
                generation_sin,
            )
        if understanding_key is None or understanding_value is None:
            if understanding is None:
                raise ValueError("Understanding hidden states or native key/value tensors are required.")
            understanding_key = self._heads(
                self.k_proj(understanding),
                self.num_kv_heads,
            )
            understanding_value = self._heads(
                self.v_proj(understanding),
                self.num_kv_heads,
            )
        expected = (self.num_kv_heads, self.head_dim)
        if understanding_key.shape[1:] != (
            expected[0],
            understanding_keep.shape[1],
            expected[1],
        ):
            raise ValueError(
                "Understanding K/V geometry must match generation attention: expected "
                f"(B,{expected[0]},{understanding_keep.shape[1]},{expected[1]}), got "
                f"{tuple(understanding_key.shape)}."
            )
        if self.key_adapter is not None:
            context_shape = understanding_key.shape
            key = understanding_key.transpose(1, 2).flatten(2).to(self.key_adapter.weight.dtype)
            value = understanding_value.transpose(1, 2).flatten(2).to(self.value_adapter.weight.dtype)
            understanding_key = self.understanding_k_norm(
                self._heads(self.key_adapter(key), self.num_kv_heads)
            )
            understanding_value = self._heads(self.value_adapter(value), self.num_kv_heads)
            if understanding_cos is None or understanding_sin is None:
                raise ValueError("Native Cosmos conditioning requires its own rotary embeddings.")
            understanding_key = (
                understanding_key * understanding_cos[:, None]
                + rotate_half(understanding_key) * understanding_sin[:, None]
            )
            if understanding_value.shape != context_shape:
                raise ValueError("Understanding key and value shapes must match.")
        k = torch.cat([understanding_key.to(q.dtype), generation_key], dim=2)
        v = torch.cat([understanding_value.to(q.dtype), generation_value], dim=2)
        context_keep = torch.cat([understanding_keep, generation_keep], dim=1).to(torch.bool)

        attention_bias = torch.zeros(
            generation.shape[0],
            1,
            1,
            k.shape[2],
            device=generation.device,
            dtype=q.dtype,
        )
        attention_bias.masked_fill_(~context_keep[:, None, None, :], torch.finfo(q.dtype).min)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=self.num_heads != self.num_kv_heads,
        )
        out = out.transpose(1, 2).reshape(generation.shape)
        return self.out_proj(out) * generation_keep.unsqueeze(-1).to(out.dtype)


class GenerationBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_dim: int,
        hidden_act: str,
        dropout: float,
        cosmos_config: dict | None = None,
    ):
        super().__init__()
        if cosmos_config is None:
            self.generation_norm = nn.LayerNorm(hidden_dim)
            self.understanding_norm = nn.LayerNorm(hidden_dim)
        else:
            norm_args = {
                "eps": cosmos_config["rms_norm_eps"],
                "nemotron": hidden_act == "relu2",
            }
            self.generation_norm = CosmosRMSNorm(hidden_dim, **norm_args)
            self.understanding_norm = nn.Identity()
        self.attention = AsymmetricAttention(
            hidden_dim,
            num_heads,
            dropout,
            num_kv_heads,
            cosmos_config,
        )
        self.mlp_norm = (
            nn.LayerNorm(hidden_dim) if cosmos_config is None
            else CosmosRMSNorm(hidden_dim, **norm_args)
        )
        if hidden_act == "silu":
            self.mlp = SwiGLU(hidden_dim, intermediate_dim, dropout)
        elif hidden_act == "relu2":
            self.mlp = ReLUSquaredMLP(hidden_dim, intermediate_dim, dropout)
        else:
            raise ValueError(f"Unsupported generation activation {hidden_act!r}.")

    def forward(
        self,
        generation: torch.Tensor,
        understanding: torch.Tensor | None,
        understanding_keep: torch.Tensor,
        generation_keep: torch.Tensor,
        understanding_key: torch.Tensor | None = None,
        understanding_value: torch.Tensor | None = None,
        generation_cos: torch.Tensor | None = None,
        generation_sin: torch.Tensor | None = None,
        understanding_cos: torch.Tensor | None = None,
        understanding_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn = self.attention(
            self.generation_norm(generation),
            self.understanding_norm(understanding) if understanding is not None else None,
            understanding_keep,
            generation_keep,
            understanding_key,
            understanding_value,
            generation_cos,
            generation_sin,
            understanding_cos,
            understanding_sin,
        )
        generation = generation + attn
        generation = generation + self.mlp(self.mlp_norm(generation))
        return generation * generation_keep.unsqueeze(-1).to(generation.dtype)


class GenerationExpert(nn.Module):
    """Flow transformer over video, action, state, and tactile token streams."""

    def __init__(
        self,
        *,
        understanding_dim: int,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float,
        input_dims: dict[str, int],
        output_dims: dict[str, int],
        task_names: tuple[str, ...],
        gradient_checkpointing: bool,
        rotary_config=None,
        num_kv_heads: int | None = None,
        intermediate_dim: int | None = None,
        hidden_act: str = "silu",
        num_video_views: int = 1,
        cosmos_config: dict | None = None,
        video_fps: float = 24.0,
    ):
        super().__init__()
        if set(input_dims) != set(output_dims):
            raise ValueError("Generation input and output modality names must match.")
        self.hidden_dim = hidden_dim
        self.cosmos_config = cosmos_config
        self.video_fps = video_fps
        num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        intermediate_dim = (
            int(hidden_dim * mlp_ratio)
            if intermediate_dim is None
            else intermediate_dim
        )
        self.gradient_checkpointing = gradient_checkpointing
        self.input_dims = dict(input_dims)
        self.input_projections = nn.ModuleDict(
            {name: nn.Linear(width, hidden_dim) for name, width in input_dims.items()}
        )
        self.output_heads = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, output_dims[name]) for name in input_dims}
        )
        for head in self.output_heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        self.modality_to_id = {name: index for index, name in enumerate(input_dims)}
        self.modality_embedding = nn.Embedding(len(input_dims), hidden_dim)
        self.video_view_embedding = (
            nn.Embedding(num_video_views, hidden_dim) if num_video_views > 1 else None
        )
        self.task_to_id = {name: index for index, name in enumerate(task_names)}
        self.task_embedding = nn.Embedding(len(task_names), hidden_dim)
        self.timestep_embedding = TimestepEmbedding(hidden_dim, native_cosmos=cosmos_config is not None)
        self.understanding_projection = (
            nn.Sequential(nn.LayerNorm(understanding_dim), nn.Linear(understanding_dim, hidden_dim))
            if cosmos_config is None else None
        )
        if cosmos_config is not None:
            self.rotary_embedding = CosmosRotaryEmbedding(
                cosmos_config["head_dim"], cosmos_config["rope_theta"], cosmos_config["rope_axes_dim"]
            )
        elif rotary_config is None:
            self.rotary_embedding = None
        else:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

            self.rotary_embedding = Qwen3VLTextRotaryEmbedding(config=rotary_config)
        self.blocks = nn.ModuleList(
            [
                GenerationBlock(
                    hidden_dim,
                    num_heads,
                    num_kv_heads,
                    intermediate_dim,
                    hidden_act,
                    dropout,
                    cosmos_config,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = (
            nn.LayerNorm(hidden_dim) if cosmos_config is None else CosmosRMSNorm(
                hidden_dim, cosmos_config["rms_norm_eps"], nemotron=hidden_act == "relu2"
            )
        )

    def initialize_cosmos_adapters(self) -> None:
        """Initialize only new parameters after meta allocation; all native weights are loaded."""
        if self.cosmos_config is None:
            raise ValueError("Adapter initialization requires a native Cosmos generation expert.")
        with torch.no_grad():
            for name, projection in self.input_projections.items():
                if name != "video":
                    projection.reset_parameters()
            for name, head in self.output_heads.items():
                if name != "video":
                    nn.init.zeros_(head.weight)
                    nn.init.zeros_(head.bias)
            nn.init.zeros_(self.modality_embedding.weight)
            nn.init.zeros_(self.task_embedding.weight)
            if self.video_view_embedding is not None:
                nn.init.zeros_(self.video_view_embedding.weight)
            for block in self.blocks:
                nn.init.eye_(block.attention.key_adapter.weight)
                nn.init.eye_(block.attention.value_adapter.weight)

    def configure_cosmos_tuning(self, mode: str) -> None:
        if mode not in ("full", "frozen"):
            raise ValueError("Generation tuning mode must be full or frozen.")
        parameters = dict(self.named_parameters())
        for name in generation_checkpoint_mapping(self):
            parameters[name].requires_grad_(mode == "full")

    def _embed_stream(
        self,
        stream: GenerationStream,
        task_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_dim = self.input_dims[stream.name]
        stream.validate(input_dim)
        values = stream.values.to(self.input_projections[stream.name].weight.dtype)
        hidden = self.input_projections[stream.name](values)
        if stream.view_ids is not None:
            if stream.name != "video" or self.video_view_embedding is None:
                raise ValueError("Camera view_ids require a multi-view video Generation Expert.")
            hidden = hidden + self.video_view_embedding(stream.view_ids.to(hidden.device)).to(hidden.dtype)
        if self.cosmos_config is not None:
            if stream.position_ids is None:
                positions = torch.arange(hidden.shape[1], device=hidden.device)
                position_ids = positions.view(1, -1, 1).expand(hidden.shape[0], -1, 3)
            else:
                position_ids = stream.position_ids.to(hidden.device)
                if position_ids.ndim == 2:
                    position_ids = position_ids.unsqueeze(0).expand(hidden.shape[0], -1, -1)
            position_ids = position_ids.float().clone()
            if stream.name == "video" and self.cosmos_config["enable_fps_modulation"]:
                fps = self.video_fps if stream.fps is None else stream.fps.to(hidden.device)[:, None]
                position_ids[..., 0] *= self.cosmos_config["base_fps"] / fps
            time = self.timestep_embedding(stream.sigma.to(hidden.device))
            noise_mask = (
                stream.noise_mask.to(hidden.device).bool() if stream.noise_mask is not None
                else (stream.sigma[:, None].to(hidden.device) > 0).expand_as(stream.keep)
            )
            hidden = hidden + time[:, None].to(hidden.dtype) * noise_mask[..., None]
            hidden = hidden + task_embedding[:, None].to(hidden.dtype)
            hidden = hidden + self.modality_embedding.weight[self.modality_to_id[stream.name]].to(hidden.dtype)
            keep = stream.keep.to(device=hidden.device, dtype=torch.bool)
            return hidden * keep[..., None].to(hidden.dtype), position_ids
        if stream.position_ids is None:
            position_ids = torch.arange(
                hidden.shape[1],
                device=hidden.device,
            ).view(1, -1, 1).expand(hidden.shape[0], -1, 3)
            positions = _sinusoidal_embedding(
                position_ids[..., 0],
                self.hidden_dim,
            ).to(hidden.dtype)
        else:
            position_ids = stream.position_ids.to(hidden.device)
            if position_ids.ndim == 2:
                position_ids = position_ids.unsqueeze(0).expand(hidden.shape[0], -1, -1)
            positions = sum(
                _sinusoidal_embedding(position_ids[..., axis], self.hidden_dim)
                for axis in range(3)
            ).to(hidden.dtype)
        modality_id = self.modality_to_id[stream.name]
        hidden = (
            hidden
            + positions
            + self.modality_embedding.weight[modality_id].to(hidden.dtype)
        )
        time = self.timestep_embedding(stream.sigma.to(hidden.device))
        hidden = hidden + (time + task_embedding).unsqueeze(1).to(hidden.dtype)
        keep = stream.keep.to(device=hidden.device, dtype=torch.bool)
        return hidden * keep.unsqueeze(-1).to(hidden.dtype), position_ids

    def forward(
        self,
        streams: list[GenerationStream],
        understanding_states: list[torch.Tensor],
        understanding_keep: torch.Tensor,
        task_name: str,
        understanding_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        understanding_rotary: tuple[torch.Tensor, torch.Tensor] | None = None,
        understanding_position_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if not streams:
            raise ValueError("The generation expert needs at least one present modality stream.")
        if not understanding_states and not understanding_key_values:
            raise ValueError("At least one understanding hidden state is required.")
        if task_name not in self.task_to_id:
            raise KeyError(f"Task {task_name!r} is not configured for this generation expert.")

        batch = streams[0].values.shape[0]
        task_id = torch.full(
            (batch,),
            self.task_to_id[task_name],
            device=streams[0].values.device,
            dtype=torch.long,
        )
        task_embedding = self.task_embedding(task_id)

        hidden_parts: list[torch.Tensor] = []
        keep_parts: list[torch.Tensor] = []
        position_parts: list[torch.Tensor] = []
        slices: dict[str, slice] = {}
        offset = 0
        for stream in streams:
            if stream.name in slices:
                raise ValueError(f"Generation stream {stream.name!r} was supplied twice.")
            hidden, position_ids = self._embed_stream(stream, task_embedding)
            hidden_parts.append(hidden)
            keep_parts.append(stream.keep.to(hidden.device, dtype=torch.bool))
            position_parts.append(position_ids)
            slices[stream.name] = slice(offset, offset + hidden.shape[1])
            offset += hidden.shape[1]

        generation = torch.cat(hidden_parts, dim=1)
        generation_keep = torch.cat(keep_parts, dim=1)
        generation_cos = generation_sin = None
        understanding_cos = understanding_sin = None
        if self.cosmos_config is not None:
            if not understanding_key_values:
                raise ValueError("Pretrained Cosmos generation requires Qwen key/value conditions.")
            if understanding_position_ids is None:
                understanding_position_ids = torch.arange(
                    understanding_keep.shape[1], device=generation.device
                ).view(1, 1, -1).expand(3, batch, -1)
            else:
                understanding_position_ids = understanding_position_ids.to(generation.device)
            understanding_cos, understanding_sin = self.rotary_embedding(
                generation, understanding_position_ids
            )
            if understanding_rotary is not None:
                cos, sin = (part[:, None].float() for part in understanding_rotary)
                denominator = cos.square() + sin.square()
                understanding_key_values = [
                    (
                        ((key.float() * cos - rotate_half(key.float()) * sin) / denominator).to(key.dtype),
                        value,
                    )
                    for key, value in understanding_key_values
                ]
            context_positions = understanding_position_ids.masked_fill(
                ~understanding_keep[None].to(generation.device).bool(), -1
            )
            temporal_offset = context_positions.amax(dim=(0, 2)) + 1
            temporal_offset += self.cosmos_config["unified_3d_mrope_temporal_modality_margin"]
            for stream, positions in zip(streams, position_parts, strict=True):
                positions[..., 0] += temporal_offset[:, None]
                positions[..., 0] += (
                    self.modality_to_id[stream.name]
                    * self.cosmos_config["unified_3d_mrope_temporal_modality_margin"]
                )
        if self.rotary_embedding is not None:
            generation_position_ids = torch.cat(position_parts, dim=1).permute(2, 0, 1)
            generation_cos, generation_sin = self.rotary_embedding(
                generation,
                generation_position_ids,
            )
        projected_understanding = None
        if understanding_key_values is None:
            projected_understanding = [
                self.understanding_projection(state.to(generation.dtype))
                for state in understanding_states
            ]
        elif understanding_states and len(understanding_key_values) != len(understanding_states):
            raise ValueError(
                "Understanding hidden-state and key/value lists must have equal length."
            )
        understanding_keep = understanding_keep.to(device=generation.device, dtype=torch.bool)

        depth = len(self.blocks)
        num_contexts = (
            len(understanding_key_values)
            if understanding_key_values is not None
            else len(understanding_states)
        )
        for index, block in enumerate(self.blocks):
            context_index = min(num_contexts - 1, index * num_contexts // depth)
            context = (
                projected_understanding[context_index]
                if projected_understanding is not None
                else None
            )
            native_key = native_value = None
            if understanding_key_values is not None:
                native_key, native_value = understanding_key_values[context_index]
            if self.gradient_checkpointing and self.training:
                if self.cosmos_config is not None:
                    generation = checkpoint(
                        block, generation, None, understanding_keep, generation_keep,
                        native_key, native_value, generation_cos, generation_sin,
                        understanding_cos, understanding_sin, use_reentrant=False,
                    )
                elif native_key is None:
                    generation = checkpoint(
                        lambda gen, context_hidden, keep_u, keep_g, cos, sin, _block=block: _block(
                            gen,
                            context_hidden,
                            keep_u,
                            keep_g,
                            None,
                            None,
                            cos,
                            sin,
                        ),
                        generation,
                        context,
                        understanding_keep,
                        generation_keep,
                        generation_cos,
                        generation_sin,
                        use_reentrant=False,
                    )
                else:
                    generation = checkpoint(
                        lambda gen, keep_u, keep_g, key, value, cos, sin, _block=block: _block(
                            gen,
                            None,
                            keep_u,
                            keep_g,
                            key,
                            value,
                            cos,
                            sin,
                        ),
                        generation,
                        understanding_keep,
                        generation_keep,
                        native_key,
                        native_value,
                        generation_cos,
                        generation_sin,
                        use_reentrant=False,
                    )
            else:
                generation = block(
                    generation,
                    context,
                    understanding_keep,
                    generation_keep,
                    native_key,
                    native_value,
                    generation_cos,
                    generation_sin,
                    understanding_cos,
                    understanding_sin,
                )

        generation = self.final_norm(generation)
        return {
            name: self.output_heads[name](generation[:, token_slice])
            for name, token_slice in slices.items()
        }
