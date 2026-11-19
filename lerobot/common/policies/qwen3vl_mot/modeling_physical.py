"""Independent grouped MLPs with an optional Stage-1 tactile image codec."""

from __future__ import annotations

import torch
from torch import nn

from .configuration_qwen3vl_mot import Qwen3VLMoTConfig


class PhysicalMLP(nn.Module):
    def __init__(self, config: Qwen3VLMoTConfig):
        super().__init__()
        self.config = config
        self.num_cls_tokens = 0
        for name in ("state", "action"):
            width = getattr(config, f"max_{name}_dim")
            setattr(
                self,
                f"{name}_mlp",
                nn.Sequential(
                    nn.Linear(config.group_size * width * 2, config.physical_hidden_dim),
                    nn.GELU(),
                    nn.Linear(config.physical_hidden_dim, config.physical_hidden_dim),
                    nn.LayerNorm(config.physical_hidden_dim),
                ),
            )

    def forward(
        self,
        batch: dict,
        *,
        return_tokens: bool,
        apply_modality_dropout: bool,
        include_tactile: bool,
        compute_reconstruction: bool,
    ):
        if not return_tokens or apply_modality_dropout or include_tactile or compute_reconstruction:
            raise ValueError("PhysicalMLP provides only deterministic action/state tokens.")
        tokens, masks = [], []
        for name, key in (("state", "observation.state"), ("action", "action")):
            mlp = getattr(self, f"{name}_mlp")
            values = batch[key].to(device=mlp[0].weight.device, dtype=mlp[0].weight.dtype)
            width = getattr(self.config, f"max_{name}_dim")
            if values.ndim != 3 or values.shape[1:] != (self.config.chunk_size, width):
                raise ValueError(f"{name} must be a complete canonical chunk.")
            mask = batch[f"{name}_mask"].to(device=values.device).bool()
            if mask.shape != (values.shape[0], width):
                raise ValueError(f"{name}_mask must have shape (batch, {width}).")
            mask = mask[:, None].expand_as(values)
            values = torch.where(mask, values, torch.zeros_like(values))
            features = torch.cat((values, mask.to(values.dtype)), dim=-1).reshape(
                values.shape[0], self.config.num_groups, -1
            )
            keep = mask.reshape(values.shape[0], self.config.num_groups, -1).any(dim=-1)
            encoded = mlp(features)
            tokens.append(torch.where(keep.unsqueeze(-1), encoded, torch.zeros_like(encoded)))
            masks.append(keep)
        return torch.cat(tokens, dim=1), torch.cat(masks, dim=1), {}

    def encode_history(self, name: str, values: torch.Tensor, mask: torch.Tensor):
        mlp = getattr(self, f"{name}_mlp")
        values = values.to(device=mlp[0].weight.device, dtype=mlp[0].weight.dtype)
        mask = mask.to(device=values.device, dtype=torch.bool)
        width = getattr(self.config, f"max_{name}_dim")
        if values.ndim != 3 or values.shape[-1] != width or mask.shape != values.shape:
            raise ValueError(f"History {name} and its mask must have shape (B,N,{width}).")
        values = torch.where(mask, values, torch.zeros_like(values))
        if not torch.isfinite(values).all():
            raise ValueError(f"Valid history {name} values must be finite.")
        padding = -values.shape[1] % self.config.group_size
        values = torch.nn.functional.pad(values, (0, 0, padding, 0))
        mask = torch.nn.functional.pad(mask, (0, 0, padding, 0))
        features = torch.cat((values, mask.to(values.dtype)), dim=-1).reshape(
            values.shape[0], -1, self.config.group_size * width * 2
        )
        keep = mask.reshape(values.shape[0], -1, self.config.group_size * width).any(-1)
        encoded = mlp(features)
        return torch.where(keep[..., None], encoded, torch.zeros_like(encoded)), keep


class PhysicalMLPWithTactile(PhysicalMLP):
    """Independent MLPs plus only the reusable Stage-1 tactile image codec."""

    def __init__(self, config: Qwen3VLMoTConfig):
        super().__init__(config)
        from lerobot.common.policies.ace.modeling_robo_contrast import (
            TactileImageEncoder,
            TactilePatchDecoder,
        )

        self.tactile_feat_dim = config.stage1_tactile_config["tactile_feat_dim"]
        self.use_ftp1_tactile = self.use_anytouch_tactile = False
        self.tactile_cnn = TactileImageEncoder(self.tactile_feat_dim, pretrained=False)
        self.tactile_recon = TactilePatchDecoder(
            self.tactile_feat_dim, config.stage1_tactile_config["tactile_recon_size"]
        )

    def encode_tactile_patches(self, images: torch.Tensor) -> torch.Tensor:
        return self.tactile_cnn(images)[1]

    def decode_tactile_patches(self, patches: torch.Tensor) -> torch.Tensor:
        return self.tactile_recon(patches.to(self.tactile_recon.input_projection[0].weight.dtype))
