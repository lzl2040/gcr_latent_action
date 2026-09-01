"""Frozen Wan VAE encoder turning uint8 clips into the latents the gen expert trains on.

Kept separate from the world model because it is a *data* stage, not a model stage: nothing
here is trained, and in a real run these latents would be cached to disk rather than recomputed
every epoch. Holding it apart keeps that option open.

Two details are easy to get wrong and silent when wrong:

* **Latent normalisation.** The checkpoint ships ``latents_mean``/``latents_std`` per channel,
  and the diffusion objective assumes roughly unit variance. Skipping the scaling leaves
  channels whose std ranges over 0.35-1.17 in this checkpoint, so the loss would be dominated
  by a handful of them.
* **Temporal layout.** Wan compresses time 4x but maps the first frame to its own latent, so a
  clip must have ``T = 1 + 4k`` frames to come back with ``1 + k`` latent frames. Any other
  length silently truncates.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class WanLatentEncoder(nn.Module):
    """``(B, T, 3, H, W)`` uint8 -> ``(B, C, T', H', W')`` normalised latents."""

    def __init__(self, vae_dir: str, resolution: int = 256, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        from diffusers import AutoencoderKLWan

        self.vae = AutoencoderKLWan.from_pretrained(vae_dir, torch_dtype=dtype)
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.resolution = resolution

        cfg = json.loads((Path(vae_dir) / "config.json").read_text())
        mean = torch.tensor(cfg["latents_mean"], dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = torch.tensor(cfg["latents_std"], dtype=torch.float32).view(1, -1, 1, 1, 1)
        self.register_buffer("latents_mean", mean, persistent=False)
        self.register_buffer("latents_std", std, persistent=False)

    @property
    def latent_channels(self) -> int:
        return self.latents_mean.shape[1]

    @staticmethod
    def required_frames(latent_frames: int) -> int:
        """Pixel frames needed for ``latent_frames`` latents: Wan's ``T = 1 + 4k``."""
        return 1 + 4 * (latent_frames - 1)

    @torch.no_grad()
    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        if clip.dtype == torch.uint8:
            clip = clip.float().div_(255.0)
        b, t = clip.shape[:2]
        x = clip.reshape(b * t, *clip.shape[2:])
        if x.shape[-1] != self.resolution or x.shape[-2] != self.resolution:
            x = F.interpolate(
                x.float(), size=(self.resolution, self.resolution), mode="bilinear", align_corners=False
            )
        # The VAE was trained on [-1, 1], not [0, 1].
        x = x.mul(2.0).sub(1.0)
        x = x.view(b, t, *x.shape[1:]).permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

        dist = self.vae.encode(x.to(self.vae.dtype)).latent_dist
        # mode() rather than sample(): the VAE's own sampling noise is not a useful
        # augmentation here and only adds variance to the flow-matching target.
        latents = dist.mode()
        return (latents.float() - self.latents_mean) / self.latents_std

    @torch.no_grad()
    def denormalise(self, latents: torch.Tensor) -> torch.Tensor:
        return latents * self.latents_std + self.latents_mean
