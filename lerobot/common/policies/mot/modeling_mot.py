"""Mixture-of-Transformers world model with a Phi-4-Multimodal understanding expert.

Architecture follows NVIDIA Cosmos3's ``Cosmos3PackedMoTAttention`` layout: every layer
carries two parameter sets ("experts") that share a single attention *stage* but run two
separate attention *calls*:

    und tokens : causal self-attention over und only        (never sees gen)
    gen tokens : full attention over cat([k_und, k_gen])    (conditioned on und)

Because the understanding stream is causal-self-only it is completely independent of the
generation stream. Training follows Cosmos3 and advances both streams layer-by-layer so each
layer's und K/V can be consumed and released immediately. Inference may instead run the whole
und stack once under ``no_grad`` and reuse its per-layer K/V across denoising steps.

Phi-4-Multimodal adds a rank-256 vision LoRA to every attention/MLP projection. The und
expert keeps those adapters: image-conditioned tasks enable them, while text-only tasks
disable them exactly like the official ``InputMode.LANGUAGE`` path.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

# Phi-4-MM: head_dim 128, partial_rotary_factor 0.75 -> 96 rotary dims -> 48 frequencies.
# An mrope section therefore has to sum to 48 rather than head_dim // 2.
DEFAULT_MROPE_SECTION = (16, 16, 16)


def copy_safetensor_tensors(root: str | Path, targets: dict[str, torch.Tensor]) -> None:
    """Copy selected checkpoint tensors without materialising unrelated audio weights."""
    root = Path(root)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    missing = sorted(set(targets) - set(index))
    if missing:
        raise RuntimeError(f"checkpoint is missing required tensors: {missing[:8]}")

    by_shard: dict[str, list[str]] = {}
    for name in targets:
        by_shard.setdefault(index[name], []).append(name)
    with torch.no_grad():
        for shard, names in sorted(by_shard.items()):
            with safe_open(root / shard, framework="pt", device="cpu") as handle:
                for name in names:
                    targets[name].copy_(handle.get_tensor(name))


@dataclass
class MoTConfig:
    """Shapes for the two experts.

    ``head_dim`` and ``num_key_value_heads`` are *shared*: the gen stream concatenates its
    keys/values onto the und stream's, so those two axes must agree.  Everything else about
    the gen expert -- width, query-head count, MLP size -- is free, which is what keeps the
    from-scratch half affordable.
    """

    phi_dir: str = "/Data/lzl/huggingface/Phi-4-multimodal-instruct"

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
    attention_scaling: float = 1.0

    # --- Phi-4-Multimodal vision adapter --------------------------------------------------
    vision_lora_rank: int = 256
    vision_lora_alpha: float = 512.0

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
        if self.vision_lora_rank <= 0:
            raise ValueError("vision_lora_rank must be positive")
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
        rope_scale = cfg["max_position_embeddings"] / cfg["original_max_position_embeddings"]
        attention_scaling = math.sqrt(
            1.0 + math.log(rope_scale) / math.log(cfg["original_max_position_embeddings"])
        )
        vision_lora = cfg["vision_lora"]
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
            attention_scaling=attention_scaling,
            vision_lora_rank=vision_lora["r"],
            vision_lora_alpha=vision_lora["lora_alpha"],
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


class VisionLoRALinear(nn.Module):
    """Phi-4-MM linear layer with only the vision adapter retained."""

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float):
        super().__init__()
        self.base_layer = nn.Linear(in_features, out_features, bias=False)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor, use_vision_lora: bool) -> torch.Tensor:
        out = self.base_layer(x)
        if use_vision_lora:
            out = out + self.lora_B(self.lora_A(x)) * self.scaling
        return out


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

    ``attention_scaling`` is taken from the checkpoint's LongRoPE config. Phi-4-MM uses
    all-one short factors below ``original_max_position_embeddings`` but still applies the
    checkpoint's global amplitude scale.
    """

    def __init__(self, config: MoTConfig):
        super().__init__()
        self.mrope_section = tuple(config.mrope_section)
        self.attention_scaling = config.attention_scaling
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

    Und-side parameter names mirror Phi-4-MM. Gen-side
    names mirror Cosmos3 (``add_q_proj``/``mlp_moe_gen``/...) so the layout stays recognisable.
    """

    def __init__(self, config: MoTConfig):
        super().__init__()
        self.config = config
        h_und, h_gen, hd = config.und_hidden_size, config.gen_hidden_size, config.head_dim
        self.n_und = config.und_num_attention_heads
        self.n_gen = config.gen_num_attention_heads
        self.n_kv = config.num_key_value_heads

        # --- understanding expert (Phi-4-Multimodal base + vision LoRA) ---
        lora = (config.vision_lora_rank, config.vision_lora_alpha)
        self.input_layernorm = RMSNorm(h_und, config.rms_norm_eps)
        self.self_attn = nn.Module()
        self.self_attn.qkv_proj = VisionLoRALinear(
            h_und, (self.n_und + 2 * self.n_kv) * hd, *lora
        )
        self.self_attn.o_proj = VisionLoRALinear(self.n_und * hd, h_und, *lora)
        self.post_attention_layernorm = RMSNorm(h_und, config.rms_norm_eps)
        self.mlp = nn.Module()
        self.mlp.gate_up_proj = VisionLoRALinear(
            h_und, 2 * config.und_intermediate_size, *lora
        )
        self.mlp.down_proj = VisionLoRALinear(config.und_intermediate_size, h_und, *lora)

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

    def _split_qkv(
        self, hidden: torch.Tensor, use_vision_lora: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, length, _ = hidden.shape
        hd = self.config.head_dim
        qkv = self.self_attn.qkv_proj(hidden, use_vision_lora)
        q_end = self.n_und * hd
        k_end = q_end + self.n_kv * hd
        q = qkv[..., :q_end].view(b, length, self.n_und, hd).transpose(1, 2)
        k = qkv[..., q_end:k_end].view(b, length, self.n_kv, hd).transpose(1, 2)
        v = qkv[..., k_end:].view(b, length, self.n_kv, hd).transpose(1, 2)
        return q, k, v

    def und_forward(
        self,
        hidden: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        use_vision_lora: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Causal self-attention. Returns the new hidden state plus *pre-RoPE* k/v for the gen stack."""
        residual = hidden
        x = self.input_layernorm(hidden)
        q, k, v = self._split_qkv(x, use_vision_lora)
        cos, sin = rope
        attn = F.scaled_dot_product_attention(
            apply_partial_rope(q, cos, sin),
            apply_partial_rope(k, cos, sin),
            v,
            is_causal=True,
            enable_gqa=True,
        )
        b, _, length, _ = attn.shape
        hidden = residual + self.self_attn.o_proj(
            attn.transpose(1, 2).reshape(b, length, -1), use_vision_lora
        )

        residual = hidden
        x = self.post_attention_layernorm(hidden)
        gate, up = self.mlp.gate_up_proj(x, use_vision_lora).chunk(2, dim=-1)
        hidden = residual + self.mlp.down_proj(up * F.silu(gate), use_vision_lora)
        return hidden, k, v

    def und_kv(
        self,
        hidden: torch.Tensor,
        use_vision_lora: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Export this layer's K/V without computing an unused final attention/MLP output."""
        x = self.input_layernorm(hidden)
        _, k, v = self._split_qkv(x, use_vision_lora)
        return k, v

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

    def interleaved_forward(
        self,
        und_hidden: torch.Tensor,
        gen_hidden: torch.Tensor,
        rope_und: tuple[torch.Tensor, torch.Tensor],
        rope_gen: tuple[torch.Tensor, torch.Tensor],
        use_vision_lora: bool,
        und_requires_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance both Cosmos3 pathways through one layer.

        UND never consumes GEN. Its same-layer pre-RoPE K/V are therefore the only temporary
        tensors crossing from the causal pathway to the full-attention pathway.
        """
        with torch.set_grad_enabled(torch.is_grad_enabled() and und_requires_grad):
            und_hidden, k_und, v_und = self.und_forward(
                und_hidden,
                rope_und,
                use_vision_lora,
            )
        gen_hidden = self.gen_forward(
            gen_hidden,
            rope_gen,
            rope_und,
            k_und,
            v_und,
        )
        return und_hidden, gen_hidden

    def interleaved_last_forward(
        self,
        und_hidden: torch.Tensor,
        gen_hidden: torch.Tensor,
        rope_und: tuple[torch.Tensor, torch.Tensor],
        rope_gen: tuple[torch.Tensor, torch.Tensor],
        use_vision_lora: bool,
        und_requires_grad: bool,
    ) -> torch.Tensor:
        """Run the final MoT layer without computing an unused final UND state."""
        with torch.set_grad_enabled(torch.is_grad_enabled() and und_requires_grad):
            k_und, v_und = self.und_kv(und_hidden, use_vision_lora)
        return self.gen_forward(
            gen_hidden,
            rope_gen,
            rope_und,
            k_und,
            v_und,
        )


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
    """Phi-4-Multimodal understanding expert + a from-scratch generation expert."""

    def __init__(self, config: MoTConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.und_hidden_size)
        self.layers = nn.ModuleList([MoTLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.und_hidden_size, config.rms_norm_eps)
        self.norm_moe_gen = RMSNorm(config.gen_hidden_size, config.rms_norm_eps)
        self.rotary_emb = MRotaryEmbedding(config)
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
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        use_vision_lora: bool = False,
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
            hidden, k, v = layer.und_forward(hidden, rope, use_vision_lora)
            kv.append((k, v))
        return self.norm(hidden), kv, rope

    def forward_und_kv(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        use_vision_lora: bool = False,
    ):
        """Run only the UND computation required by the GEN expert."""
        rope = tuple(r.to(inputs_embeds.dtype) for r in self.rotary_emb(position_ids))
        hidden = inputs_embeds
        kv = []
        for layer in self.layers[:-1]:
            hidden, k, v = layer.und_forward(hidden, rope, use_vision_lora)
            kv.append((k, v))
        k, v = self.layers[-1].und_kv(hidden, use_vision_lora)
        kv.append((k, v))
        return kv, rope

    # ------------------------------------------------------------------ gen

    def _add_timestep(self, gen_hidden: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Add a per-sample or per-token diffusion timestep embedding."""
        # ``timestep`` is (B,) when every token shares a noise level, or (B, L) when it does
        # not -- which is what the task family needs, since context frames stay clean at
        # sigma=0 while the frames being predicted carry real noise.
        if timestep.ndim == 1:
            t_emb = self.time_embedder(
                timestep_embedding(timestep, self.config.time_embed_dim).to(gen_hidden.dtype)
            ).unsqueeze(1)
        else:
            bsz, seq = timestep.shape
            t_emb = self.time_embedder(
                timestep_embedding(timestep.reshape(-1), self.config.time_embed_dim).to(
                    gen_hidden.dtype
                )
            ).view(bsz, seq, -1)
        return gen_hidden + t_emb

    def forward_interleaved(
        self,
        und_hidden: torch.Tensor,
        und_position_ids: torch.Tensor,
        gen_hidden: torch.Tensor,
        gen_position_ids: torch.Tensor,
        timestep: torch.Tensor,
        use_vision_lora: bool = False,
        und_requires_grad: bool = True,
        checkpoint_layers: bool | None = None,
        checkpoint_segment_size: int = 4,
    ) -> torch.Tensor:
        """Cosmos3-style training path with layer-local UND K/V.

        Checkpointing wraps short runs of complete dual-pathway layers. Backward recomputes
        each run's UND K/V instead of retaining every layer's cache or every layer's UND
        hidden state for the lifetime of the forward.
        """
        if checkpoint_segment_size <= 0:
            raise ValueError("checkpoint_segment_size must be positive")
        rope_und = tuple(r.to(und_hidden.dtype) for r in self.rotary_emb(und_position_ids))
        rope_gen = tuple(r.to(gen_hidden.dtype) for r in self.rotary_emb(gen_position_ids))
        gen_hidden = self._add_timestep(gen_hidden, timestep)
        if checkpoint_layers is None:
            checkpoint_layers = self.gradient_checkpointing and self.training
        use_ckpt = checkpoint_layers and torch.is_grad_enabled()

        normal_layers = len(self.layers) - 1
        if use_ckpt:
            for start in range(0, normal_layers, checkpoint_segment_size):
                end = min(start + checkpoint_segment_size, normal_layers)
                und_hidden, gen_hidden = torch.utils.checkpoint.checkpoint(
                    self._forward_interleaved_segment,
                    und_hidden,
                    gen_hidden,
                    rope_und,
                    rope_gen,
                    use_vision_lora,
                    und_requires_grad,
                    start,
                    end,
                    use_reentrant=False,
                )
        else:
            for layer in self.layers[:-1]:
                und_hidden, gen_hidden = layer.interleaved_forward(
                    und_hidden,
                    gen_hidden,
                    rope_und,
                    rope_gen,
                    use_vision_lora,
                    und_requires_grad,
                )

        last = self.layers[-1]
        if use_ckpt:
            gen_hidden = torch.utils.checkpoint.checkpoint(
                last.interleaved_last_forward,
                und_hidden,
                gen_hidden,
                rope_und,
                rope_gen,
                use_vision_lora,
                und_requires_grad,
                use_reentrant=False,
            )
        else:
            gen_hidden = last.interleaved_last_forward(
                und_hidden,
                gen_hidden,
                rope_und,
                rope_gen,
                use_vision_lora,
                und_requires_grad,
            )
        return self.norm_moe_gen(gen_hidden)

    def _forward_interleaved_segment(
        self,
        und_hidden: torch.Tensor,
        gen_hidden: torch.Tensor,
        rope_und: tuple[torch.Tensor, torch.Tensor],
        rope_gen: tuple[torch.Tensor, torch.Tensor],
        use_vision_lora: bool,
        und_requires_grad: bool,
        start: int,
        end: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a checkpoint segment while keeping every layer Cosmos3-interleaved."""
        for layer in self.layers[start:end]:
            und_hidden, gen_hidden = layer.interleaved_forward(
                und_hidden,
                gen_hidden,
                rope_und,
                rope_gen,
                use_vision_lora,
                und_requires_grad,
            )
        return und_hidden, gen_hidden

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
        hidden = self._add_timestep(gen_hidden, timestep)
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
        """Load the Phi-4-MM language backbone and vision LoRA into the und expert."""
        phi_dir = Path(phi_dir or self.config.phi_dir)
        targets = {
            "model.embed_tokens.weight": self.embed_tokens.weight,
            "model.norm.weight": self.norm.weight,
        }
        for i, layer in enumerate(self.layers):
            p = f"model.layers.{i}."
            targets[p + "input_layernorm.weight"] = layer.input_layernorm.weight
            targets[p + "post_attention_layernorm.weight"] = (
                layer.post_attention_layernorm.weight
            )
            for name, module in (
                ("self_attn.qkv_proj", layer.self_attn.qkv_proj),
                ("self_attn.o_proj", layer.self_attn.o_proj),
                ("mlp.gate_up_proj", layer.mlp.gate_up_proj),
                ("mlp.down_proj", layer.mlp.down_proj),
            ):
                targets[p + name + ".base_layer.weight"] = module.base_layer.weight
                targets[p + name + ".lora_A.vision.weight"] = module.lora_A.weight
                targets[p + name + ".lora_B.vision.weight"] = module.lora_B.weight
        copy_safetensor_tensors(phi_dir, targets)

    def set_und_trainable(self, flag: bool, kv_only: bool = False) -> None:
        """Toggle the Phi-4-MM base, vision LoRA, norms and token embedding.

        In the world model only each layer's pre-RoPE K/V is consumed. The final UND norm and
        the last layer's post-K/V attention/MLP path therefore cannot influence the loss and
        are frozen under ``kv_only`` rather than being sent to an optimizer with permanent
        ``grad=None``.
        """
        self.embed_tokens.requires_grad_(flag)
        self.norm.requires_grad_(flag)
        for layer in self.layers:
            layer.input_layernorm.requires_grad_(flag)
            layer.post_attention_layernorm.requires_grad_(flag)
            layer.self_attn.requires_grad_(flag)
            layer.mlp.requires_grad_(flag)
        if flag and kv_only:
            self.norm.requires_grad_(False)
            last = self.layers[-1]
            last.self_attn.o_proj.requires_grad_(False)
            last.post_attention_layernorm.requires_grad_(False)
            last.mlp.requires_grad_(False)

    def freeze_und(self) -> None:
        """Freeze every understanding-expert parameter, leaving the gen expert trainable."""
        self.set_und_trainable(False)

    def set_gen_trainable(self, flag: bool) -> None:
        """Toggle the generation expert, its in/out heads and the und->gen key norm."""
        for layer in self.layers:
            layer.input_layernorm_moe_gen.requires_grad_(flag)
            layer.post_attention_layernorm_moe_gen.requires_grad_(flag)
            layer.add_q_proj.requires_grad_(flag)
            layer.add_k_proj.requires_grad_(flag)
            layer.add_v_proj.requires_grad_(flag)
            layer.to_add_out.requires_grad_(flag)
            layer.norm_added_q.requires_grad_(flag)
            layer.norm_added_k.requires_grad_(flag)
            layer.k_norm_und_for_gen.requires_grad_(flag)
            layer.mlp_moe_gen.requires_grad_(flag)
        self.norm_moe_gen.requires_grad_(flag)
        self.proj_in.requires_grad_(flag)
        self.proj_out.requires_grad_(flag)
        self.time_embedder.requires_grad_(flag)
        if self.config.enable_action_gen:
            self.action_proj_in.requires_grad_(flag)
            self.action_proj_out.requires_grad_(flag)
            self.action_modality_embed.requires_grad_(flag)

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
