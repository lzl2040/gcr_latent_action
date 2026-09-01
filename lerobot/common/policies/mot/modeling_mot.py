"""Mixture-of-Transformers world model with a frozen Phi-4-mini understanding expert.

Architecture follows NVIDIA Cosmos3's ``Cosmos3PackedMoTAttention`` layout: every layer
carries two parameter sets ("experts") that share a single attention *stage* but run two
separate attention *calls*:

    und tokens : causal self-attention over und only        (never sees gen)
    gen tokens : full attention over cat([k_und, k_gen])    (conditioned on und)

Because the understanding stream is causal-self-only it is completely independent of the
generation stream.  With Phi frozen, the whole und stack can therefore run once under
``no_grad`` and export only its per-layer key/value tensors, instead of being interleaved
layer-by-layer with the gen stack.  That is what :meth:`MoTModel.forward_und` does.

The und expert reuses Phi-4-mini's parameter names verbatim (``qkv_proj``, ``gate_up_proj``
...) so a Phi checkpoint loads into it directly.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

# Phi-4-mini: head_dim 128, partial_rotary_factor 0.75 -> 96 rotary dims -> 48 frequencies.
# An mrope section therefore has to sum to 48 rather than head_dim // 2.
DEFAULT_MROPE_SECTION = (16, 16, 16)


@dataclass
class MoTConfig:
    """Shapes for the two experts.

    ``head_dim`` and ``num_key_value_heads`` are *shared*: the gen stream concatenates its
    keys/values onto the und stream's, so those two axes must agree.  Everything else about
    the gen expert -- width, query-head count, MLP size -- is free, which is what keeps the
    from-scratch half affordable.
    """

    phi_dir: str = "/Data/lzl/huggingface/Phi-4-mini-instruct"

    # --- und expert (read from the Phi config, kept here for reference/validation) ---
    und_hidden_size: int = 3072
    num_hidden_layers: int = 32
    und_num_attention_heads: int = 24
    und_intermediate_size: int = 8192
    vocab_size: int = 200064

    # --- shared attention geometry ---
    head_dim: int = 128
    num_key_value_heads: int = 8
    rope_theta: float = 10000.0
    partial_rotary_factor: float = 0.75
    mrope_section: tuple[int, int, int] = DEFAULT_MROPE_SECTION
    rms_norm_eps: float = 1e-5

    # --- gen expert (from scratch) ---
    # gen_hidden_size is independent of gen_num_attention_heads * head_dim: add_q_proj and
    # to_add_out bridge the two widths.  Only head_dim and num_key_value_heads are shared with
    # the und expert, because the two streams concatenate keys/values.  gen query heads must
    # stay a multiple of num_key_value_heads for GQA.
    #
    # Defaults size the gen expert at 1.423B, matching Cosmos3-Edge's generation branch
    # exactly (it runs 28 layers of 50.34M; we run 32 of 44.5M).  Cosmos keeps gen and und at
    # 1:1 per layer, which for Phi's fatter layers would cost 3.22B -- see doc/phi4_mot.md.
    gen_hidden_size: int = 2048
    gen_num_attention_heads: int = 16
    gen_intermediate_size: int = 7680

    # --- generation heads ---
    latent_channels: int = 48
    latent_patch_size: int = 2
    time_embed_dim: int = 256
    action_dim: int = 64
    num_embodiment_domains: int = 32
    enable_action_gen: bool = True

    @property
    def patch_latent_dim(self) -> int:
        return self.latent_channels * self.latent_patch_size**2

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    def validate(self) -> None:
        if sum(self.mrope_section) * 2 != self.rotary_dim:
            raise ValueError(
                f"mrope_section {self.mrope_section} sums to {sum(self.mrope_section)}, but "
                f"rotary_dim // 2 is {self.rotary_dim // 2}"
            )
        if self.und_hidden_size != self.und_num_attention_heads * self.head_dim:
            raise ValueError("und_hidden_size must equal und_num_attention_heads * head_dim")
        for name, heads in (
            ("und", self.und_num_attention_heads),
            ("gen", self.gen_num_attention_heads),
        ):
            if heads % self.num_key_value_heads:
                raise ValueError(f"{name} query heads {heads} not divisible by kv heads")

    @classmethod
    def from_phi_dir(cls, phi_dir: str, **overrides) -> "MoTConfig":
        cfg = json.loads((Path(phi_dir) / "config.json").read_text())
        head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
        out = cls(
            phi_dir=phi_dir,
            und_hidden_size=cfg["hidden_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            und_num_attention_heads=cfg["num_attention_heads"],
            und_intermediate_size=cfg["intermediate_size"],
            vocab_size=cfg["vocab_size"],
            head_dim=head_dim,
            num_key_value_heads=cfg["num_key_value_heads"],
            rope_theta=cfg["rope_theta"],
            partial_rotary_factor=cfg.get("partial_rotary_factor", 1.0),
            rms_norm_eps=cfg["rms_norm_eps"],
            **overrides,
        )
        out.validate()
        return out


class RMSNorm(nn.Module):
    """Matches ``Phi3RMSNorm`` exactly, including the cast-back-then-scale ordering."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the leading ``cos.shape[-1]`` channels and pass the tail through untouched.

    Phi applies RoPE to only 96 of 128 head channels; the remaining 32 carry no positional
    signal.  ``cos``/``sin`` are ``(B, L, rotary_dim)`` and broadcast over the head axis.
    """
    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    x_rot = x_rot * cos + _rotate_half(x_rot) * sin
    return torch.cat((x_rot, x_pass), dim=-1)


class MRotaryEmbedding(nn.Module):
    """Unified 3-D mRoPE that degenerates to Phi's native 1-D RoPE.

    ``inv_freq`` is split into three contiguous sections; section ``a`` is driven by
    ``position_ids[a]``.  When all three axes carry the same index (as they do for text) every
    frequency sees the same position and the result is bit-identical to standard RoPE -- which
    is what lets a text-only pretrained LLM sit in this stack without losing its positional
    behaviour.

    ``attention_scaling`` is taken from the checkpoint's rope config.  Phi-4-mini uses LongRoPE
    with ``short_factor`` all-ones, so below ``original_max_position_embeddings`` the scaling
    is an identity and no interpolation is applied.
    """

    def __init__(self, config: MoTConfig, attention_scaling: float = 1.0):
        super().__init__()
        self.mrope_section = tuple(config.mrope_section)
        self.attention_scaling = attention_scaling
        half = config.rotary_dim // 2
        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, half, dtype=torch.float32) / half)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``position_ids``: ``(3, B, L)`` long -> two ``(B, L, rotary_dim)`` float tensors."""
        if position_ids.ndim != 3 or position_ids.shape[0] != 3:
            raise ValueError(f"expected position_ids of shape (3, B, L), got {tuple(position_ids.shape)}")
        inv_freq = self.inv_freq.to(position_ids.device)
        sections = []
        offset = 0
        for axis, width in enumerate(self.mrope_section):
            freq = inv_freq[offset : offset + width]
            sections.append(position_ids[axis].float().unsqueeze(-1) * freq)
            offset += width
        freqs = torch.cat(sections, dim=-1)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos() * self.attention_scaling, emb.sin() * self.attention_scaling


class MoTLayer(nn.Module):
    """One layer holding both experts.

    Und-side parameter names mirror Phi-4-mini so its checkpoint loads unchanged.  Gen-side
    names mirror Cosmos3 (``add_q_proj``/``mlp_moe_gen``/...) so the layout stays recognisable.
    """

    def __init__(self, config: MoTConfig):
        super().__init__()
        self.config = config
        h_und, h_gen, hd = config.und_hidden_size, config.gen_hidden_size, config.head_dim
        self.n_und = config.und_num_attention_heads
        self.n_gen = config.gen_num_attention_heads
        self.n_kv = config.num_key_value_heads

        # --- understanding expert (Phi-4-mini) ---
        self.input_layernorm = RMSNorm(h_und, config.rms_norm_eps)
        self.self_attn = nn.Module()
        self.self_attn.qkv_proj = nn.Linear(h_und, (self.n_und + 2 * self.n_kv) * hd, bias=False)
        self.self_attn.o_proj = nn.Linear(self.n_und * hd, h_und, bias=False)
        self.post_attention_layernorm = RMSNorm(h_und, config.rms_norm_eps)
        self.mlp = nn.Module()
        self.mlp.gate_up_proj = nn.Linear(h_und, 2 * config.und_intermediate_size, bias=False)
        self.mlp.down_proj = nn.Linear(config.und_intermediate_size, h_und, bias=False)

        # --- generation expert (from scratch) ---
        self.input_layernorm_moe_gen = RMSNorm(h_gen, config.rms_norm_eps)
        self.add_q_proj = nn.Linear(h_gen, self.n_gen * hd, bias=False)
        self.add_k_proj = nn.Linear(h_gen, self.n_kv * hd, bias=False)
        self.add_v_proj = nn.Linear(h_gen, self.n_kv * hd, bias=False)
        self.to_add_out = nn.Linear(self.n_gen * hd, h_gen, bias=False)
        self.norm_added_q = RMSNorm(hd, config.rms_norm_eps)
        self.norm_added_k = RMSNorm(hd, config.rms_norm_eps)
        self.k_norm_und_for_gen = RMSNorm(hd, config.rms_norm_eps)
        self.post_attention_layernorm_moe_gen = RMSNorm(h_gen, config.rms_norm_eps)
        self.mlp_moe_gen = nn.Module()
        self.mlp_moe_gen.up_proj = nn.Linear(h_gen, config.gen_intermediate_size, bias=False)
        self.mlp_moe_gen.down_proj = nn.Linear(config.gen_intermediate_size, h_gen, bias=False)

    def _split_qkv(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, length, _ = hidden.shape
        hd = self.config.head_dim
        qkv = self.self_attn.qkv_proj(hidden)
        q_end = self.n_und * hd
        k_end = q_end + self.n_kv * hd
        q = qkv[..., :q_end].view(b, length, self.n_und, hd).transpose(1, 2)
        k = qkv[..., q_end:k_end].view(b, length, self.n_kv, hd).transpose(1, 2)
        v = qkv[..., k_end:].view(b, length, self.n_kv, hd).transpose(1, 2)
        return q, k, v

    def und_forward(
        self, hidden: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Causal self-attention. Returns the new hidden state plus *pre-RoPE* k/v for the gen stack."""
        residual = hidden
        x = self.input_layernorm(hidden)
        q, k, v = self._split_qkv(x)
        cos, sin = rope
        attn = F.scaled_dot_product_attention(
            apply_partial_rope(q, cos, sin),
            apply_partial_rope(k, cos, sin),
            v,
            is_causal=True,
            enable_gqa=True,
        )
        b, _, length, _ = attn.shape
        hidden = residual + self.self_attn.o_proj(attn.transpose(1, 2).reshape(b, length, -1))

        residual = hidden
        x = self.post_attention_layernorm(hidden)
        gate, up = self.mlp.gate_up_proj(x).chunk(2, dim=-1)
        hidden = residual + self.mlp.down_proj(up * F.silu(gate))
        return hidden, k, v

    def gen_forward(
        self,
        hidden: torch.Tensor,
        rope_gen: tuple[torch.Tensor, torch.Tensor],
        rope_und: tuple[torch.Tensor, torch.Tensor],
        k_und: torch.Tensor,
        v_und: torch.Tensor,
    ) -> torch.Tensor:
        """Full attention over ``cat([k_und, k_gen])`` -- the only channel by which und conditions gen."""
        b, length, _ = hidden.shape
        hd = self.config.head_dim
        residual = hidden
        x = self.input_layernorm_moe_gen(hidden)
        q = self.norm_added_q(self.add_q_proj(x).view(b, length, self.n_gen, hd)).transpose(1, 2)
        k = self.norm_added_k(self.add_k_proj(x).view(b, length, self.n_kv, hd)).transpose(1, 2)
        v = self.add_v_proj(x).view(b, length, self.n_kv, hd).transpose(1, 2)

        k_und_for_gen = self.k_norm_und_for_gen(k_und.transpose(1, 2)).transpose(1, 2)
        cos_u, sin_u = rope_und
        cos_g, sin_g = rope_gen
        all_k = torch.cat([apply_partial_rope(k_und_for_gen, cos_u, sin_u), apply_partial_rope(k, cos_g, sin_g)], dim=2)
        all_v = torch.cat([v_und, v], dim=2)

        attn = F.scaled_dot_product_attention(
            apply_partial_rope(q, cos_g, sin_g), all_k, all_v, is_causal=False, enable_gqa=True
        )
        hidden = residual + self.to_add_out(attn.transpose(1, 2).reshape(b, length, -1))

        residual = hidden
        x = self.post_attention_layernorm_moe_gen(hidden)
        # relu^2 (Nemotron/Cosmos style): two matrices instead of three, no gate.
        hidden = residual + self.mlp_moe_gen.down_proj(F.relu(self.mlp_moe_gen.up_proj(x)) ** 2)
        return hidden


class PerDomainLinear(nn.Module):
    """A separate ``Linear(in, out)`` per embodiment domain, stored as one stacked tensor.

    Mirrors Cosmos3's ``action_proj_in``/``action_proj_out``, whose weights ship as
    ``[num_domains, in*out]`` with a ``[num_domains, out]`` bias.
    """

    def __init__(self, num_domains: int, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Parameter(torch.empty(num_domains, out_features * in_features))
        self.bias = nn.Parameter(torch.zeros(num_domains, out_features))
        nn.init.normal_(self.fc, std=in_features**-0.5)

    def forward(self, x: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, L, in)``; ``domain_id``: ``(B,)`` long."""
        w = self.fc[domain_id].view(-1, self.out_features, self.in_features)
        return torch.bmm(x, w.transpose(1, 2)) + self.bias[domain_id].unsqueeze(1)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float().unsqueeze(-1) * freqs
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class MoTModel(nn.Module):
    """Phi-4-mini understanding expert + a from-scratch generation expert."""

    def __init__(self, config: MoTConfig, attention_scaling: float = 1.0):
        super().__init__()
        config.validate()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.und_hidden_size)
        self.layers = nn.ModuleList([MoTLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.und_hidden_size, config.rms_norm_eps)
        self.norm_moe_gen = RMSNorm(config.gen_hidden_size, config.rms_norm_eps)
        self.rotary_emb = MRotaryEmbedding(config, attention_scaling)
        # Set by the training wrapper; trades gen-expert recompute for activation memory.
        self.gradient_checkpointing = False

        h_gen = config.gen_hidden_size
        self.proj_in = nn.Linear(config.patch_latent_dim, h_gen, bias=True)
        self.proj_out = nn.Linear(h_gen, config.patch_latent_dim, bias=True)
        self.time_embedder = nn.Sequential(
            nn.Linear(config.time_embed_dim, h_gen),
            nn.SiLU(),
            nn.Linear(h_gen, h_gen),
        )
        if config.enable_action_gen:
            self.action_modality_embed = nn.Parameter(torch.zeros(h_gen))
            self.action_proj_in = PerDomainLinear(config.num_embodiment_domains, config.action_dim, h_gen)
            self.action_proj_out = PerDomainLinear(config.num_embodiment_domains, h_gen, config.action_dim)

    # ------------------------------------------------------------------ und

    def forward_und(
        self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]]:
        """Run the whole understanding stack, exporting per-layer pre-RoPE k/v.

        Safe to call under ``no_grad`` when Phi is frozen: the und stream never reads gen
        tokens, so nothing downstream needs its activations.
        """
        rope = self.rotary_emb(position_ids)
        rope = tuple(r.to(inputs_embeds.dtype) for r in rope)
        hidden = inputs_embeds
        kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.layers:
            hidden, k, v = layer.und_forward(hidden, rope)
            kv.append((k, v))
        return self.norm(hidden), kv, rope

    # ------------------------------------------------------------------ gen

    def forward_gen(
        self,
        gen_hidden: torch.Tensor,
        gen_position_ids: torch.Tensor,
        kv: list[tuple[torch.Tensor, torch.Tensor]],
        rope_und: tuple[torch.Tensor, torch.Tensor],
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        rope_gen = self.rotary_emb(gen_position_ids)
        rope_gen = tuple(r.to(gen_hidden.dtype) for r in rope_gen)
        t_emb = self.time_embedder(
            timestep_embedding(timestep, self.config.time_embed_dim).to(gen_hidden.dtype)
        )
        hidden = gen_hidden + t_emb.unsqueeze(1)
        use_ckpt = self.gradient_checkpointing and self.training
        for layer, (k_und, v_und) in zip(self.layers, kv, strict=True):
            if use_ckpt:
                hidden = torch.utils.checkpoint.checkpoint(
                    layer.gen_forward, hidden, rope_gen, rope_und, k_und, v_und, use_reentrant=False
                )
            else:
                hidden = layer.gen_forward(hidden, rope_gen, rope_und, k_und, v_und)
        return self.norm_moe_gen(hidden)

    # ------------------------------------------------------------------ loading

    def load_phi_weights(self, phi_dir: str | None = None) -> None:
        """Load Phi-4-mini into the und expert.

        Every und parameter must be covered and every Phi tensor consumed; anything left over
        on either side is raised rather than warned about, so a checkpoint whose layout drifts
        cannot silently initialise half the tower randomly.
        """
        from safetensors.torch import load_file

        phi_dir = Path(phi_dir or self.config.phi_dir)
        index = json.loads((phi_dir / "model.safetensors.index.json").read_text())["weight_map"]
        shards: dict[str, dict] = {}
        for shard in sorted(set(index.values())):
            shards[shard] = load_file(str(phi_dir / shard))

        def take(name: str) -> torch.Tensor:
            return shards[index[name]][name]

        with torch.no_grad():
            self.embed_tokens.weight.copy_(take("model.embed_tokens.weight"))
            self.norm.weight.copy_(take("model.norm.weight"))
            for i, layer in enumerate(self.layers):
                p = f"model.layers.{i}."
                layer.input_layernorm.weight.copy_(take(p + "input_layernorm.weight"))
                layer.post_attention_layernorm.weight.copy_(take(p + "post_attention_layernorm.weight"))
                layer.self_attn.qkv_proj.weight.copy_(take(p + "self_attn.qkv_proj.weight"))
                layer.self_attn.o_proj.weight.copy_(take(p + "self_attn.o_proj.weight"))
                layer.mlp.gate_up_proj.weight.copy_(take(p + "mlp.gate_up_proj.weight"))
                layer.mlp.down_proj.weight.copy_(take(p + "mlp.down_proj.weight"))

        expected = {"model.embed_tokens.weight", "model.norm.weight"}
        for i in range(len(self.layers)):
            p = f"model.layers.{i}."
            expected |= {
                p + s
                for s in (
                    "input_layernorm.weight",
                    "post_attention_layernorm.weight",
                    "self_attn.qkv_proj.weight",
                    "self_attn.o_proj.weight",
                    "mlp.gate_up_proj.weight",
                    "mlp.down_proj.weight",
                )
            }
        # lm_head is tied to embed_tokens in this checkpoint and unused here (no text decoding).
        unused = set(index) - expected - {"lm_head.weight"}
        if unused:
            raise RuntimeError(f"unconsumed Phi tensors: {sorted(unused)[:8]} ({len(unused)} total)")

    def freeze_und(self) -> None:
        """Freeze every understanding-expert parameter, leaving the gen expert trainable."""
        self.embed_tokens.requires_grad_(False)
        self.norm.requires_grad_(False)
        for layer in self.layers:
            layer.input_layernorm.requires_grad_(False)
            layer.post_attention_layernorm.requires_grad_(False)
            layer.self_attn.requires_grad_(False)
            layer.mlp.requires_grad_(False)

    def param_report(self) -> dict[str, int]:
        und, gen = 0, 0
        und_mods = ("input_layernorm.", "post_attention_layernorm.", "self_attn.", "mlp.")
        for name, p in self.named_parameters():
            tail = name.split(".", 2)[-1] if name.startswith("layers.") else name
            is_und = name in ("embed_tokens.weight", "norm.weight") or (
                name.startswith("layers.") and tail.startswith(und_mods)
            )
            if is_und:
                und += p.numel()
            else:
                gen += p.numel()
        return {
            "und": und,
            "gen": gen,
            "total": und + gen,
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
