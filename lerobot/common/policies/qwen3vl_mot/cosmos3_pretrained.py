"""Native Cosmos3 generation contracts and selective checkpoint loading.

Reference: diffusers/models/transformers/transformer_cosmos3.py. The understanding
tower and the original domain-specific action/audio heads are deliberately not loaded.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

COSMOS_GENERATION_CONFIGS = {
    "edge": {
        "hidden_size": 2048,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 9216,
        "hidden_act": "relu2",
        "rms_norm_eps": 1e-5,
        "rope_theta": 100_000_000,
        "rope_axes_dim": [24, 20, 20],
        "qk_norm_for_text": False,
        "use_und_k_norm_for_gen": True,
        "latent_channel": 48,
        "latent_patch_size": 2,
        "patch_latent_dim": 192,
        "timestep_scale": 0.001,
        "base_fps": 24,
        "enable_fps_modulation": True,
        "unified_3d_mrope_reset_spatial_ids": True,
        "unified_3d_mrope_temporal_modality_margin": 15000,
    },
    "nano": {
        "hidden_size": 4096,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 12288,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "rope_theta": 5_000_000,
        "rope_axes_dim": [24, 20, 20],
        "qk_norm_for_text": True,
        "use_und_k_norm_for_gen": False,
        "latent_channel": 48,
        "latent_patch_size": 2,
        "patch_latent_dim": 192,
        "timestep_scale": 0.001,
        "base_fps": 24,
        "enable_fps_modulation": True,
        "unified_3d_mrope_reset_spatial_ids": True,
        "unified_3d_mrope_temporal_modality_margin": 15000,
    },
}


class CosmosRMSNorm(nn.Module):
    def __init__(self, width: int, eps: float, *, nemotron: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps
        self.nemotron = nemotron

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        variance = values.float().square().mean(dim=-1, keepdim=True)
        normalized = values.float() * torch.rsqrt(variance + self.eps)
        if self.nemotron:
            return (normalized * self.weight.float()).to(values.dtype)
        return normalized.to(self.weight.dtype) * self.weight


def rotate_half(values: torch.Tensor) -> torch.Tensor:
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class CosmosRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float, axes: list[int]):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self.axes = tuple(axes)
        if head_dim % 2 or len(axes) != 3 or sum(axes) != head_dim // 2:
            raise ValueError("Cosmos mRoPE requires three axis sections summing to head_dim / 2.")

    def _frequencies(self, device=None) -> torch.Tensor:
        return 1.0 / (self.theta ** (
            torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim
        ))

    def forward(self, values: torch.Tensor, positions: torch.Tensor):
        if positions.ndim != 3 or positions.shape[:2] != (3, values.shape[0]):
            raise ValueError("Cosmos position_ids must have shape (3,B,N).")
        with torch.autocast(device_type=values.device.type, enabled=False):
            # FSDP casts buffers to BF16 without invoking Module._apply.
            phases = positions.float().unsqueeze(-1) * self._frequencies(values.device)
            combined = phases[0].clone()
            for axis in (1, 2):
                section = slice(axis, self.axes[axis] * 3, 3)
                combined[..., section] = phases[axis, ..., section]
            phases = torch.cat([combined, combined], dim=-1)
        return phases.cos().to(values.dtype), phases.sin().to(values.dtype)


@dataclass(frozen=True)
class GenerationLoadReport:
    copied_tensors: int
    copied_parameters: int
    ignored_tensors: int


def generation_checkpoint_mapping(model: nn.Module) -> dict[str, str]:
    """Map every reused generation parameter, including the condition-key norm."""
    mapping = {
        "final_norm.weight": "norm_moe_gen.weight",
        "input_projections.video.weight": "proj_in.weight",
        "input_projections.video.bias": "proj_in.bias",
        "output_heads.video.weight": "proj_out.weight",
        "output_heads.video.bias": "proj_out.bias",
        "timestep_embedding.mlp.0.weight": "time_embedder.linear_1.weight",
        "timestep_embedding.mlp.0.bias": "time_embedder.linear_1.bias",
        "timestep_embedding.mlp.2.weight": "time_embedder.linear_2.weight",
        "timestep_embedding.mlp.2.bias": "time_embedder.linear_2.bias",
    }
    suffixes = {
        "generation_norm.weight": "input_layernorm_moe_gen.weight",
        "mlp_norm.weight": "post_attention_layernorm_moe_gen.weight",
        "mlp.up.weight": "mlp_moe_gen.up_proj.weight",
        "mlp.down.weight": "mlp_moe_gen.down_proj.weight",
        "attention.q_proj.weight": "self_attn.add_q_proj.weight",
        "attention.k_proj.weight": "self_attn.add_k_proj.weight",
        "attention.v_proj.weight": "self_attn.add_v_proj.weight",
        "attention.out_proj.weight": "self_attn.to_add_out.weight",
        "attention.q_norm.weight": "self_attn.norm_added_q.weight",
        "attention.k_norm.weight": "self_attn.norm_added_k.weight",
        "attention.understanding_k_norm.weight": (
            "self_attn.k_norm_und_for_gen.weight"
            if model.cosmos_config["use_und_k_norm_for_gen"]
            else "self_attn.norm_k.weight"
        ),
    }
    if model.cosmos_config["hidden_act"] == "silu":
        suffixes["mlp.gate.weight"] = "mlp_moe_gen.gate_proj.weight"
    for index in range(len(model.blocks)):
        mapping.update({
            f"blocks.{index}.{target}": f"layers.{index}.{source}"
            for target, source in suffixes.items()
        })
    return mapping


def _is_generation_checkpoint_key(name: str) -> bool:
    if name.startswith(("proj_in.", "proj_out.", "time_embedder.", "norm_moe_gen.")):
        return True
    return name.startswith("layers.") and any(
        component in name for component in (
            "_moe_gen.", ".self_attn.add_", ".self_attn.to_add_out.",
            ".self_attn.norm_added_", ".self_attn.k_norm_und_for_gen.",
        )
    )


def inspect_generation_checkpoint(
    model: nn.Module, model_dir: str | Path
) -> tuple[dict[str, str], dict[str, str], GenerationLoadReport]:
    root = Path(model_dir) / "transformer"
    config_path = root / "config.json"
    source_config = json.loads(config_path.read_text())
    source_config.setdefault("use_und_k_norm_for_gen", False)
    source_axes = source_config.get(
        "rope_axes_dim", source_config.get("rope_scaling", {}).get("mrope_section")
    )
    source_config["rope_axes_dim"] = source_axes
    mismatches = {
        name: (source_config.get(name), value)
        for name, value in model.cosmos_config.items()
        if source_config.get(name) != value
    }
    if mismatches:
        raise ValueError(f"Cosmos generation checkpoint contract mismatch in {config_path}: {mismatches}.")

    index_path = root / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
    else:
        filename = "diffusion_pytorch_model.safetensors"
        with safe_open(root / filename, framework="pt", device="cpu") as source:
            weight_map = dict.fromkeys(source.keys(), filename)
    mapping = generation_checkpoint_mapping(model)
    missing = set(mapping.values()) - set(weight_map)
    unhandled = {
        name for name in weight_map
        if _is_generation_checkpoint_key(name) and name not in mapping.values()
    }
    if missing or unhandled:
        raise RuntimeError(
            f"Incomplete Cosmos generation mapping: missing={sorted(missing)}, "
            f"unhandled generation weights={sorted(unhandled)}."
        )
    parameters = dict(model.named_parameters())
    count = 0
    for filename in sorted({weight_map[name] for name in mapping.values()}):
        shard_path = root / filename
        if shard_path.parent != root or not shard_path.is_file():
            raise FileNotFoundError(f"Missing or invalid Cosmos weight shard: {shard_path}.")
        with safe_open(shard_path, framework="pt", device="cpu") as source:
            for target, name in mapping.items():
                if weight_map[name] != filename:
                    continue
                shape = tuple(source.get_slice(name).get_shape())
                if shape != tuple(parameters[target].shape):
                    raise ValueError(
                        f"Cosmos weight shape mismatch: {name}={shape}, "
                        f"{target}={tuple(parameters[target].shape)}."
                    )
                count += parameters[target].numel()
    return mapping, weight_map, GenerationLoadReport(
        copied_tensors=len(mapping),
        copied_parameters=count,
        ignored_tensors=len(weight_map) - len(mapping),
    )


def load_generation_checkpoint(model: nn.Module, model_dir: str | Path) -> GenerationLoadReport:
    mapping, weight_map, report = inspect_generation_checkpoint(model, model_dir)
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for filename in sorted({weight_map[name] for name in mapping.values()}):
            with safe_open(Path(model_dir) / "transformer" / filename, framework="pt", device="cpu") as source:
                for target, name in mapping.items():
                    if weight_map[name] == filename:
                        parameters[target].copy_(source.get_tensor(name))
    logging.info(
        "Cosmos pretrained generation loaded: %d tensors, %s parameters, 100%% of required "
        "generation weights; %d original understanding/action/audio tensors excluded.",
        report.copied_tensors, f"{report.copied_parameters:,}", report.ignored_tensors,
    )
    return report


def validate_generation_vae(model_dir: str | Path, vae_dir: str | Path) -> None:
    source = json.loads((Path(model_dir) / "vae/config.json").read_text())
    actual = json.loads((Path(vae_dir) / "vae/config.json").read_text())
    fields = (
        "z_dim", "scale_factor_spatial", "scale_factor_temporal",
        "latents_mean", "latents_std", "base_dim", "dim_mult",
    )
    mismatches = [name for name in fields if source.get(name) != actual.get(name)]
    if mismatches:
        raise ValueError(f"Pretrained generation and configured video VAE are incompatible: {mismatches}.")
