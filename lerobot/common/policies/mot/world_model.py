"""World model built on the Phi-4-mini MoT: video-latent flow matching conditioned on a VLM.

Data flow::

    frame_0 --[Qwen3-VL ViT, frozen]--> 256 tok --[2x2 merge, trained]--> 64 tok --.
    instruction --[Phi embed]--> text tok --------------------------------------.  |
                                                                                v  v
                                                       und stream (Phi-4-mini, frozen)
                                                                    | per-layer k/v
                                                                    v
    clip --[Wan VAE, frozen]--> latents --> noise --> patches --> gen stream --> velocity

The und stream is a frozen feature extractor; only the vision merge/projector and the whole
generation expert are trained.  Because gradients still have to reach the projector *through*
Phi, the und stack is gradient-checkpointed when the projector is trainable, and switches to a
no_grad fast path when it is not.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from lerobot.common.policies.ace.qwen3vl_encoder import build_qwen3vl_vision
from lerobot.common.policies.mot.modeling_mot import MoTConfig, MoTModel

# Qwen3-VL preprocessing, taken from the checkpoint config (rescale 1/255, mean/std 0.5).
QWEN3VL_MEAN = 0.5
QWEN3VL_STD = 0.5


@dataclass
class WorldModelConfig:
    mot: MoTConfig
    qwen3vl_dir: str = "/Data/lzl/huggingface/Qwen3-VL-4B-Instruct"
    vision_merge_size: int = 2
    freeze_vision_projector: bool = False
    und_grad_checkpointing: bool = True
    latent_grid: int = 16
    action_loss_weight: float = 1.0


def build_mrope_positions(
    segments: list[tuple[int, int, int]], device: torch.device
) -> torch.Tensor:
    """Assign 3-D positions to a concatenation of ``(grid_t, grid_h, grid_w)`` segments.

    Follows Qwen2-VL's scheme: segments are laid out sequentially and each one advances a
    shared counter by its largest extent, so different modalities never collide in position
    space while a plain text run (``1 x 1 x n`` handled as ``n`` unit segments) reproduces
    ordinary 1-D RoPE.  A ``(1, 1, 1)`` segment is a single text token.

    Returns ``(3, L)``.
    """
    t_ids, h_ids, w_ids = [], [], []
    start = 0
    for grid_t, grid_h, grid_w in segments:
        t = torch.arange(grid_t, device=device).view(-1, 1, 1).expand(grid_t, grid_h, grid_w)
        h = torch.arange(grid_h, device=device).view(1, -1, 1).expand(grid_t, grid_h, grid_w)
        w = torch.arange(grid_w, device=device).view(1, 1, -1).expand(grid_t, grid_h, grid_w)
        t_ids.append(t.reshape(-1) + start)
        h_ids.append(h.reshape(-1) + start)
        w_ids.append(w.reshape(-1) + start)
        start += max(grid_t, grid_h, grid_w)
    return torch.stack([torch.cat(t_ids), torch.cat(h_ids), torch.cat(w_ids)])


class VisionMerger(nn.Module):
    """Learned 2x2 spatial merge, mirroring Qwen3-VL's own merger.

    The ViT trunk keeps all 256 tokens (its merger is stripped upstream); pooling here cuts the
    und sequence 4x, which is what makes gradient-checkpointed backprop through the frozen LLM
    affordable.
    """

    def __init__(self, in_dim: int, out_dim: int, merge_size: int = 2):
        super().__init__()
        self.merge_size = merge_size
        self.norm = nn.LayerNorm(in_dim * merge_size**2)
        self.proj = nn.Sequential(
            nn.Linear(in_dim * merge_size**2, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, n, d = tokens.shape
        side = int(math.isqrt(n))
        if side * side != n:
            raise ValueError(f"expected a square token grid, got {n}")
        m = self.merge_size
        x = tokens.view(b, side // m, m, side // m, m, d).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(b, (side // m) ** 2, d * m * m)
        return self.proj(self.norm(x))


class MoTWorldModel(nn.Module):
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        self.mot = MoTModel(config.mot)

        vision, vision_dim, vision_image_size = build_qwen3vl_vision(config.qwen3vl_dir)
        self.vision = vision
        self.vision.requires_grad_(False)
        self.vision_image_size = vision_image_size
        self.vision_merger = VisionMerger(
            vision_dim, config.mot.und_hidden_size, config.vision_merge_size
        )
        if config.freeze_vision_projector:
            self.vision_merger.requires_grad_(False)

        patch_size = json.loads((Path(config.qwen3vl_dir) / "config.json").read_text())["vision_config"][
            "patch_size"
        ]
        self.vision_grid = vision_image_size // int(patch_size) // config.vision_merge_size
        self.latent_side = config.latent_grid // config.mot.latent_patch_size

    # ------------------------------------------------------------------ setup

    def load_pretrained(self) -> None:
        self.mot.load_phi_weights()
        self.mot.freeze_und()

    @property
    def und_needs_grad(self) -> bool:
        return any(p.requires_grad for p in self.vision_merger.parameters())

    # ------------------------------------------------------------------ und

    def encode_und(self, pixel_values: torch.Tensor, text_ids: torch.Tensor):
        """Build and run the understanding stream. Returns ``(kv, rope_und)``."""
        with torch.no_grad():
            vision_tokens = self.vision(self._to_pixel_values(pixel_values)).last_hidden_state
        image_embeds = self.vision_merger(vision_tokens.to(self.vision_merger.norm.weight.dtype))
        text_embeds = self.mot.embed_tokens(text_ids)
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)

        n_text = text_ids.shape[1]
        segments = [(1, self.vision_grid, self.vision_grid)] + [(1, 1, 1)] * n_text
        pos = build_mrope_positions(segments, pixel_values.device)
        pos = pos.unsqueeze(1).expand(3, pixel_values.shape[0], -1)

        run = self._forward_und_checkpointed if self.und_needs_grad else self._forward_und_nograd
        _, kv, rope = run(inputs_embeds, pos)
        return kv, rope

    def _forward_und_nograd(self, inputs_embeds, pos):
        with torch.no_grad():
            return self.mot.forward_und(inputs_embeds, pos)

    def _forward_und_checkpointed(self, inputs_embeds, pos):
        if not (self.config.und_grad_checkpointing and self.training):
            return self.mot.forward_und(inputs_embeds, pos)
        rope = tuple(r.to(inputs_embeds.dtype) for r in self.mot.rotary_emb(pos))
        hidden = inputs_embeds
        kv = []
        for layer in self.mot.layers:
            hidden, k, v = checkpoint(layer.und_forward, hidden, rope, use_reentrant=False)
            kv.append((k, v))
        return self.mot.norm(hidden), kv, rope

    def _to_pixel_values(self, images: torch.Tensor) -> torch.Tensor:
        size = self.vision_image_size
        if images.shape[-1] != size or images.shape[-2] != size:
            images = F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)
        return (images - QWEN3VL_MEAN) / QWEN3VL_STD

    # ------------------------------------------------------------------ gen

    def patchify(self, latents: torch.Tensor) -> torch.Tensor:
        """``(B, C, T, H, W)`` -> ``(B, T*(H/p)*(W/p), C*p*p)``."""
        b, c, t, h, w = latents.shape
        p = self.config.mot.latent_patch_size
        x = latents.permute(0, 2, 3, 4, 1).reshape(b, t, h // p, p, w // p, p, c)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).reshape(b, t * (h // p) * (w // p), c * p * p)
        return x

    def unpatchify(self, tokens: torch.Tensor, t: int) -> torch.Tensor:
        b, n, _ = tokens.shape
        p = self.config.mot.latent_patch_size
        c = self.config.mot.latent_channels
        side = self.latent_side
        x = tokens.view(b, t, side, side, p, p, c).permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(b, t, side * p, side * p, c).permute(0, 4, 1, 2, 3)
        return x

    def gen_positions(self, num_latent_frames: int, n_action: int, device) -> torch.Tensor:
        """Video latents keep a real 3-D grid; action tokens follow as a 1-D run."""
        scale = self.vision_grid / self.latent_side
        segments = [(num_latent_frames, self.latent_side, self.latent_side)]
        pos = build_mrope_positions(segments, device).float()
        # Rescale the spatial axes so gen tokens span the same coordinate range as the und
        # image grid; otherwise an 8x8 latent would only cover half of a 16x16 image.
        pos[1:] = pos[1:] * scale
        pos = pos.long()
        if n_action:
            start = int(pos.max().item()) + 1
            act = torch.arange(n_action, device=device) + start
            pos = torch.cat([pos, act.unsqueeze(0).expand(3, -1)], dim=1)
        return pos

    # ------------------------------------------------------------------ loss

    def forward(
        self,
        latents: torch.Tensor,
        pixel_values: torch.Tensor,
        text_ids: torch.Tensor,
        actions: torch.Tensor | None = None,
        domain_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Rectified-flow training step. ``latents``: ``(B, C, T, H, W)`` clean VAE latents."""
        b, _, t_lat, _, _ = latents.shape
        device = latents.device

        noise = torch.randn_like(latents)
        sigma = torch.rand(b, device=device, dtype=latents.dtype)
        s = sigma.view(b, 1, 1, 1, 1)
        noisy = (1.0 - s) * latents + s * noise
        target = noise - latents

        gen_tokens = self.mot.proj_in(self.patchify(noisy))
        n_video = gen_tokens.shape[1]

        n_action = 0
        if actions is not None and self.config.mot.enable_action_gen:
            if domain_id is None:
                domain_id = torch.zeros(b, dtype=torch.long, device=device)
            action_noise = torch.randn_like(actions)
            sa = sigma.view(b, 1, 1)
            noisy_actions = (1.0 - sa) * actions + sa * action_noise
            action_target = action_noise - actions
            action_tokens = self.mot.action_proj_in(noisy_actions, domain_id)
            action_tokens = action_tokens + self.mot.action_modality_embed
            gen_tokens = torch.cat([gen_tokens, action_tokens], dim=1)
            n_action = action_tokens.shape[1]

        kv, rope_und = self.encode_und(pixel_values, text_ids)
        pos = self.gen_positions(t_lat, n_action, device).unsqueeze(1).expand(3, b, -1)
        hidden = self.mot.forward_gen(gen_tokens, pos, kv, rope_und, sigma * 1000.0)

        video_pred = self.mot.proj_out(hidden[:, :n_video])
        loss_video = F.mse_loss(video_pred.float(), self.patchify(target).float())
        out = {"loss_video": loss_video, "loss": loss_video}

        if n_action:
            action_pred = self.mot.action_proj_out(hidden[:, n_video:], domain_id)
            loss_action = F.mse_loss(action_pred.float(), action_target.float())
            out["loss_action"] = loss_action
            out["loss"] = loss_video + self.config.action_loss_weight * loss_action
        return out

    # ------------------------------------------------------------------ reporting

    def param_report(self) -> dict[str, float]:
        def count(module):
            return sum(p.numel() for p in module.parameters())

        mot = self.mot.param_report()
        vision = count(self.vision)
        merger = count(self.vision_merger)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "vision_frozen": vision,
            "vision_merger": merger,
            "und_frozen": mot["und"],
            "gen_trainable": mot["gen"],
            "total": vision + merger + mot["total"],
            "trainable": trainable,
        }
