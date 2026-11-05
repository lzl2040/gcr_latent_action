"""Independent grouped MLP encoders for the no-stage-one baseline."""

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
