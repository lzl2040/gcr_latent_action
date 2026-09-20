"""Cosmos-style one-way understanding-to-generation transformer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


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


class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.frequency_dim = min(256, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
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


class AsymmetricAttention(nn.Module):
    """Generation queries attend to understanding KV and generation KV.

    There is deliberately no understanding-query path in this module. Understanding states
    are computed before this block and are only projected into keys and values, so generation
    can consume understanding while understanding can never consume generation.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

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
    ) -> torch.Tensor:
        q = self._heads(self.q_proj(generation))
        generation_key = self._heads(self.k_proj(generation))
        generation_value = self._heads(self.v_proj(generation))
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
            understanding_key = self._heads(self.k_proj(understanding))
            understanding_value = self._heads(self.v_proj(understanding))
        expected = (self.num_heads, self.head_dim)
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
        )
        out = out.transpose(1, 2).reshape(generation.shape)
        return self.out_proj(out) * generation_keep.unsqueeze(-1).to(out.dtype)


class GenerationBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.generation_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.understanding_norm = nn.LayerNorm(hidden_dim)
        self.attention = AsymmetricAttention(hidden_dim, num_heads, dropout)
        self.mlp_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = SwiGLU(hidden_dim, int(hidden_dim * mlp_ratio), dropout)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self,
        generation: torch.Tensor,
        understanding: torch.Tensor | None,
        understanding_keep: torch.Tensor,
        generation_keep: torch.Tensor,
        conditioning: torch.Tensor,
        understanding_key: torch.Tensor | None = None,
        understanding_value: torch.Tensor | None = None,
        generation_cos: torch.Tensor | None = None,
        generation_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.modulation(
            conditioning
        ).chunk(6, dim=-1)
        attn_input = self.generation_norm(generation) * (1 + scale_attn) + shift_attn
        attn = self.attention(
            attn_input,
            self.understanding_norm(understanding) if understanding is not None else None,
            understanding_keep,
            generation_keep,
            understanding_key,
            understanding_value,
            generation_cos,
            generation_sin,
        )
        generation = generation + torch.tanh(gate_attn) * attn
        mlp_input = self.mlp_norm(generation) * (1 + scale_mlp) + shift_mlp
        generation = generation + torch.tanh(gate_mlp) * self.mlp(mlp_input)
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
        mlp_ratio: float,
        dropout: float,
        input_dims: dict[str, int],
        output_dims: dict[str, int],
        task_names: tuple[str, ...],
        gradient_checkpointing: bool,
        rotary_config=None,
    ):
        super().__init__()
        if set(input_dims) != set(output_dims):
            raise ValueError("Generation input and output modality names must match.")
        self.hidden_dim = hidden_dim
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
        self.task_to_id = {name: index for index, name in enumerate(task_names)}
        self.task_embedding = nn.Embedding(len(task_names), hidden_dim)
        self.timestep_embedding = TimestepEmbedding(hidden_dim)
        self.understanding_projection = nn.Sequential(
            nn.LayerNorm(understanding_dim),
            nn.Linear(understanding_dim, hidden_dim),
        )
        if rotary_config is None:
            self.rotary_embedding = None
        else:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

            self.rotary_embedding = Qwen3VLTextRotaryEmbedding(config=rotary_config)
        self.blocks = nn.ModuleList(
            [
                GenerationBlock(hidden_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

    def _embed_stream(
        self,
        stream: GenerationStream,
        task_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_dim = self.input_dims[stream.name]
        stream.validate(input_dim)
        values = stream.values.to(self.input_projections[stream.name].weight.dtype)
        hidden = self.input_projections[stream.name](values)
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
        hidden = hidden + positions + self.modality_embedding.weight[modality_id].to(hidden.dtype)
        time = self.timestep_embedding(stream.sigma.to(hidden.device))
        conditioning = time + task_embedding
        conditioning = conditioning.unsqueeze(1).expand(-1, hidden.shape[1], -1)
        keep = stream.keep.to(device=hidden.device, dtype=torch.bool)
        return hidden * keep.unsqueeze(-1).to(hidden.dtype), conditioning, position_ids

    def forward(
        self,
        streams: list[GenerationStream],
        understanding_states: list[torch.Tensor],
        understanding_keep: torch.Tensor,
        task_name: str,
        understanding_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
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
        condition_parts: list[torch.Tensor] = []
        keep_parts: list[torch.Tensor] = []
        position_parts: list[torch.Tensor] = []
        slices: dict[str, slice] = {}
        offset = 0
        for stream in streams:
            if stream.name in slices:
                raise ValueError(f"Generation stream {stream.name!r} was supplied twice.")
            hidden, conditioning, position_ids = self._embed_stream(stream, task_embedding)
            hidden_parts.append(hidden)
            condition_parts.append(conditioning)
            keep_parts.append(stream.keep.to(hidden.device, dtype=torch.bool))
            position_parts.append(position_ids)
            slices[stream.name] = slice(offset, offset + hidden.shape[1])
            offset += hidden.shape[1]

        generation = torch.cat(hidden_parts, dim=1)
        conditioning = torch.cat(condition_parts, dim=1)
        generation_keep = torch.cat(keep_parts, dim=1)
        generation_cos = generation_sin = None
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
                if native_key is None:
                    generation = checkpoint(
                        lambda gen, context_hidden, keep_u, keep_g, cond, cos, sin, _block=block: _block(
                            gen,
                            context_hidden,
                            keep_u,
                            keep_g,
                            cond,
                            None,
                            None,
                            cos,
                            sin,
                        ),
                        generation,
                        context,
                        understanding_keep,
                        generation_keep,
                        conditioning,
                        generation_cos,
                        generation_sin,
                        use_reentrant=False,
                    )
                else:
                    generation = checkpoint(
                        lambda gen, keep_u, keep_g, cond, key, value, cos, sin, _block=block: _block(
                            gen,
                            None,
                            keep_u,
                            keep_g,
                            cond,
                            key,
                            value,
                            cos,
                            sin,
                        ),
                        generation,
                        understanding_keep,
                        generation_keep,
                        conditioning,
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
                    conditioning,
                    native_key,
                    native_value,
                    generation_cos,
                    generation_sin,
                )

        generation = self.final_norm(generation)
        return {
            name: self.output_heads[name](generation[:, token_slice])
            for name, token_slice in slices.items()
        }
