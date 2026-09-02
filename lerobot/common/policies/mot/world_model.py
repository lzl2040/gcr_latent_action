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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from lerobot.common.policies.ace.qwen3vl_encoder import build_qwen3vl_vision
from lerobot.common.policies.mot.modeling_mot import MoTConfig, MoTModel

# Qwen3-VL preprocessing, taken from the checkpoint config (rescale 1/255, mean/std 0.5).
QWEN3VL_MEAN = 0.5
QWEN3VL_STD = 0.5


Role = Literal["noisy", "clean", "absent"]


@dataclass
class TaskSpec:
    """One entry of the Cosmos stage-2/stage-3 task family.

    Every task is the *same* rectified-flow objective. What separates them is the **role** each
    of the two streams plays, and there are only three possibilities per stream:

    ``noisy``   sigma ~ U(0,1), enters the transformer noised, **is** the prediction target.
    ``clean``   sigma = 0, enters the transformer as ground truth, excluded from the loss.
    ``absent``  not in the sequence at all -- the model cannot see it and does not pay for it.

    Crossing those roles over the two streams is what generates the whole family, including
    forward and inverse dynamics, from one code path::

        video \\ action   absent        clean          noisy
        noisy            t2v/i2v/v2v   fwd_dyn        joint_action
        clean            (no target)   --             inv_dyn
        absent           --            --             policy

    ``context`` counts *latent* frames, not pixel frames. Wan's VAE maps frame 0 to latent 0
    on its own, so ``context=1`` is exactly "condition on the first frame" -- image-to-video.
    Context frames are always clean and always excluded from the loss; ``video`` describes the
    frames *after* the context. ``latent_frames=None`` means "use whatever the clip provides".
    """

    context: int
    image: bool
    video: Role = "noisy"
    action: Role = "absent"
    latent_frames: int | None = None

    def __post_init__(self) -> None:
        if self.video == "absent" and self.context == 0:
            raise ValueError("a task with no video target needs at least one context frame")
        if self.video != "noisy" and self.action != "noisy":
            raise ValueError(
                "at least one stream must be 'noisy', otherwise the task has no target"
            )


TASK_SPECS: dict[str, TaskSpec] = {
    # --- stage 2: generation only -----------------------------------------------------------
    "t2i": TaskSpec(context=0, image=False, latent_frames=1),
    "t2v": TaskSpec(context=0, image=False),
    "i2v": TaskSpec(context=1, image=True),
    "v2v": TaskSpec(context=2, image=True),
    # --- stage 3: action ---------------------------------------------------------------------
    # Joint denoising of future frames and action. The two sigmas are drawn *independently*,
    # so the model sees clean-future/noisy-action and noisy-future/clean-action as well as the
    # diagonal. Sharing one sigma, as this used to, trains only the diagonal and leaves
    # deployment -- where the future does not exist, i.e. sigma_video = 1 -- undertrained.
    "joint_action": TaskSpec(context=1, image=True, video="noisy", action="noisy"),
    # Forward dynamics: given the true action, predict the frames it produces. The action is
    # conditioning here, not a target, so it enters clean and takes no loss.
    "fwd_dyn": TaskSpec(context=1, image=True, video="noisy", action="clean"),
    # Inverse dynamics: given clean before-and-after frames, recover the action between them.
    "inv_dyn": TaskSpec(context=1, image=True, video="clean", action="noisy"),
    # Deployment condition: only the current observation exists. Future frames are dropped
    # from the sequence entirely, which is also what makes this the cheap inference path --
    # video tokens are ~6x the action tokens, so not generating them is most of the cost.
    "policy": TaskSpec(context=1, image=True, video="absent", action="noisy"),
}

TASK_MIXES: dict[str, dict[str, float]] = {
    "stage2": {"t2i": 0.25, "t2v": 0.25, "i2v": 0.3, "v2v": 0.2},
    # Stage 3: action is the objective, but the stage-2 tasks stay in to stop the generative
    # branch drifting while the action head trains. The action budget is split across the four
    # action tasks rather than spent entirely on the joint one, because `policy` is the only
    # one that matches deployment and `inv_dyn`/`fwd_dyn` are the two halves of the dynamics
    # the joint task has to learn implicitly.
    "stage3": {
        "policy": 0.2,
        "joint_action": 0.15,
        "inv_dyn": 0.1,
        "fwd_dyn": 0.05,
        "i2v": 0.2,
        "v2v": 0.15,
        "t2v": 0.1,
        "t2i": 0.05,
    },
    # The old stage-3 mix, kept so measurements taken before the task family grew stay
    # reproducible.
    "stage3_joint_only": {
        "joint_action": 0.5,
        "i2v": 0.2,
        "v2v": 0.15,
        "t2v": 0.1,
        "t2i": 0.05,
    },
    # Everything action, for debugging the action path without generative noise in the loss.
    "action_only": {"policy": 0.5, "joint_action": 0.2, "inv_dyn": 0.2, "fwd_dyn": 0.1},
}
STAGE2_MIX = TASK_MIXES["stage2"]
STAGE3_MIX = TASK_MIXES["stage3"]


def parse_mix(spec: str) -> dict[str, float]:
    """Resolve a preset name or an explicit ``"policy=0.4,i2v=0.6"`` string into a mix.

    Weights are renormalised, so they can be given as counts or percentages.
    """
    if spec in TASK_MIXES:
        return dict(TASK_MIXES[spec])
    mix: dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, weight = part.partition("=")
        name = name.strip()
        if name not in TASK_SPECS:
            raise ValueError(f"unknown task {name!r}; expected one of {sorted(TASK_SPECS)}")
        mix[name] = float(weight) if weight else 1.0
    if not mix:
        raise ValueError(f"empty task mix {spec!r}")
    total = sum(mix.values())
    if total <= 0:
        raise ValueError(f"task mix {spec!r} sums to {total}")
    return {k: v / total for k, v in mix.items()}


def sample_task(mix: dict[str, float], generator: torch.Generator | None = None) -> str:
    names = list(mix)
    weights = torch.tensor([mix[n] for n in names], dtype=torch.float32)
    idx = int(torch.multinomial(weights, 1, generator=generator).item())
    return names[idx]


@dataclass
class TrainableScope:
    """Which of the four weight groups take gradients.

    The interesting choice is the understanding branch. Cosmos3 keeps it frozen, which is what
    makes a from-scratch gen expert stable: gen attends over und's per-layer K/V at every
    layer, so a moving und changes the input distribution of all 1.42B gen weights at once.
    pi-0.5 takes the opposite view and trains everything except the vision encoder, on the
    argument that the language model has to adapt to embodied data. Both are defensible; they
    differ by ~3.8B optimizer parameters, so the choice is as much a memory decision as a
    modelling one.
    """

    vision: bool  # Qwen3-VL ViT
    merger: bool  # 2x2 vision merger into Phi's width
    und: bool  # Phi-4-mini
    gen: bool  # generation expert


TRAINABLE_SCOPES: dict[str, TrainableScope] = {
    # Cosmos3-style: only the generation side moves. Cheapest and the safest for a
    # from-scratch gen expert.
    "gen_only": TrainableScope(vision=False, merger=True, und=False, gen=True),
    # pi-0.5-style: freeze the vision encoder, train the rest.
    "freeze_vision": TrainableScope(vision=False, merger=True, und=True, gen=True),
    "all": TrainableScope(vision=True, merger=True, und=True, gen=True),
}


@dataclass
class WorldModelConfig:
    mot: MoTConfig
    qwen3vl_dir: str = "/Data/lzl/huggingface/Qwen3-VL-4B-Instruct"
    vision_merge_size: int = 2
    trainable_scope: str = "gen_only"
    # Applied on top of the scope. Freezing the merger lets the und stack run under no_grad
    # in the gen_only scope, which measured 1.58x faster; it does nothing once und is trained.
    freeze_vision_projector: bool = False
    und_grad_checkpointing: bool = True
    latent_grid: int = 16
    action_loss_weight: float = 1.0

    def scope(self) -> TrainableScope:
        if self.trainable_scope not in TRAINABLE_SCOPES:
            raise ValueError(
                f"unknown trainable_scope {self.trainable_scope!r}; "
                f"expected one of {sorted(TRAINABLE_SCOPES)}"
            )
        s = TRAINABLE_SCOPES[self.trainable_scope]
        if self.freeze_vision_projector:
            s = replace(s, merger=False)
        return s


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
        self.vision_image_size = vision_image_size
        self.vision_merger = VisionMerger(
            vision_dim, config.mot.und_hidden_size, config.vision_merge_size
        )

        patch_size = json.loads((Path(config.qwen3vl_dir) / "config.json").read_text())["vision_config"][
            "patch_size"
        ]
        self.vision_grid = vision_image_size // int(patch_size) // config.vision_merge_size
        self.latent_side = config.latent_grid // config.mot.latent_patch_size
        self.apply_trainable_scope()

    # ------------------------------------------------------------------ setup

    def apply_trainable_scope(self) -> None:
        """Set ``requires_grad`` across the four weight groups from the configured scope."""
        s = self.config.scope()
        self.vision.requires_grad_(s.vision)
        self.vision_merger.requires_grad_(s.merger)
        self.mot.set_und_trainable(s.und)
        self.mot.set_gen_trainable(s.gen)

    def load_pretrained(self) -> None:
        self.mot.load_phi_weights()
        self.apply_trainable_scope()

    @property
    def und_needs_grad(self) -> bool:
        """Whether the und stack must build a graph.

        True if und itself trains, or if anything feeding it does -- a trainable merger or
        vision tower sits upstream, so their gradients only exist if und is differentiated
        through. Getting this wrong is silent: no_grad here would leave those weights with
        ``grad = None`` while the loss still fell on the gen expert alone.
        """
        s = self.config.scope()
        return s.und or s.merger or s.vision

    # ------------------------------------------------------------------ und

    def encode_und(self, pixel_values: torch.Tensor | None, text_ids: torch.Tensor):
        """Build and run the understanding stream. Returns ``(kv, rope_und)``.

        ``pixel_values`` is optional: the text-to-image and text-to-video tasks have no input
        frame, and feeding a blank one would spend a full vision-tower pass teaching the model
        that "no image" looks like a particular grey rectangle.
        """
        device = text_ids.device
        text_embeds = self.mot.embed_tokens(text_ids)
        n_text = text_ids.shape[1]

        if pixel_values is None:
            inputs_embeds = text_embeds
            segments = [(1, 1, 1)] * n_text
        else:
            # no_grad only when the tower is actually frozen; wrapping a trainable tower would
            # silently starve it of gradients.
            vision_trains = self.config.scope().vision
            with torch.set_grad_enabled(vision_trains and torch.is_grad_enabled()):
                vision_tokens = self.vision(self._to_pixel_values(pixel_values)).last_hidden_state
            image_embeds = self.vision_merger(vision_tokens.to(self.vision_merger.norm.weight.dtype))
            inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)
            segments = [(1, self.vision_grid, self.vision_grid)] + [(1, 1, 1)] * n_text

        pos = build_mrope_positions(segments, device)
        pos = pos.unsqueeze(1).expand(3, text_ids.shape[0], -1)

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
        pixel_values: torch.Tensor | None,
        text_ids: torch.Tensor,
        actions: torch.Tensor | None = None,
        domain_id: torch.Tensor | None = None,
        task: str = "i2v",
    ) -> dict[str, torch.Tensor]:
        """Rectified-flow training step. ``latents``: ``(B, C, T, H, W)`` clean VAE latents.

        The noise level is per *frame*, not per sample: the first ``spec.context`` latent
        frames stay at sigma=0, so they enter the transformer clean and are excluded from the
        loss. What happens to the rest, and to the action, is ``spec.video`` / ``spec.action``.

        The two sigmas are drawn **independently**. Tying them, as an earlier version did, only
        ever trains the diagonal sigma_video == sigma_action, so the model never learns to
        produce an action when the future is unavailable -- which is precisely the deployment
        condition (sigma_video = 1, or the frames simply absent).
        """
        if task not in TASK_SPECS:
            raise ValueError(f"unknown task {task!r}; expected one of {sorted(TASK_SPECS)}")
        spec = TASK_SPECS[task]

        if spec.latent_frames is not None:
            latents = latents[:, :, : spec.latent_frames]
        if spec.video == "absent":
            latents = latents[:, :, : spec.context]
        b, _, t_lat, _, _ = latents.shape
        device = latents.device
        if spec.video != "absent" and spec.context >= t_lat:
            raise ValueError(
                f"task {task!r} wants {spec.context} context frames but the clip has {t_lat}"
            )
        if not spec.image:
            pixel_values = None

        # (B, T): 0 on context frames and on frames the task keeps clean, one draw elsewhere.
        sigma_video = torch.rand(b, device=device, dtype=latents.dtype)
        frame_is_target = torch.zeros(b, t_lat, device=device, dtype=latents.dtype)
        if spec.video == "noisy":
            frame_is_target[:, spec.context :] = 1.0
        sigma = sigma_video.unsqueeze(1) * frame_is_target

        noise = torch.randn_like(latents)
        s = sigma.view(b, 1, t_lat, 1, 1)
        noisy = (1.0 - s) * latents + s * noise
        target = noise - latents

        gen_tokens = self.mot.proj_in(self.patchify(noisy))
        n_video = gen_tokens.shape[1]
        # patchify is frame-major, so each latent frame owns a contiguous run of tokens.
        tokens_per_frame = n_video // t_lat
        token_sigma = sigma.repeat_interleave(tokens_per_frame, dim=1)

        n_action = 0
        want_action = (
            spec.action != "absent"
            and actions is not None
            and self.config.mot.enable_action_gen
        )
        if want_action:
            if domain_id is None:
                domain_id = torch.zeros(b, dtype=torch.long, device=device)
            action_noise = torch.randn_like(actions)
            # "clean" means the action is conditioning, not a target: sigma = 0 and no loss.
            sigma_action = (
                torch.rand(b, device=device, dtype=latents.dtype)
                if spec.action == "noisy"
                else torch.zeros(b, device=device, dtype=latents.dtype)
            )
            sa = sigma_action.view(b, 1, 1)
            noisy_actions = (1.0 - sa) * actions + sa * action_noise
            action_target = action_noise - actions
            action_tokens = self.mot.action_proj_in(noisy_actions, domain_id)
            action_tokens = action_tokens + self.mot.action_modality_embed
            gen_tokens = torch.cat([gen_tokens, action_tokens], dim=1)
            n_action = action_tokens.shape[1]
            token_sigma = torch.cat(
                [token_sigma, sigma_action.unsqueeze(1).expand(b, n_action)], dim=1
            )

        kv, rope_und = self.encode_und(pixel_values, text_ids)
        pos = self.gen_positions(t_lat, n_action, device).unsqueeze(1).expand(3, b, -1)
        hidden = self.mot.forward_gen(gen_tokens, pos, kv, rope_und, token_sigma * 1000.0)

        out: dict[str, torch.Tensor] = {}
        total = None
        if spec.video == "noisy":
            video_pred = self.mot.proj_out(hidden[:, :n_video]).float()
            video_target = self.patchify(target).float()
            # Mean over the predicted frames only. Dividing by the target count rather than by
            # the full token count keeps the loss scale comparable across tasks with different
            # amounts of context, so t2v and v2v numbers can be read side by side.
            loss_mask = (
                frame_is_target.repeat_interleave(tokens_per_frame, dim=1).unsqueeze(-1).float()
            )
            loss_video = ((video_pred - video_target).pow(2) * loss_mask).sum() / loss_mask.sum(
            ).clamp(min=1.0) / video_pred.shape[-1]
            out["loss_video"] = loss_video
            total = loss_video

        if n_action and spec.action == "noisy":
            action_pred = self.mot.action_proj_out(hidden[:, n_video:], domain_id)
            loss_action = F.mse_loss(action_pred.float(), action_target.float())
            out["loss_action"] = loss_action
            weighted = self.config.action_loss_weight * loss_action
            total = weighted if total is None else total + weighted

        if total is None:  # __post_init__ rules this out; a guard beats a silent zero loss.
            raise RuntimeError(f"task {task!r} produced no loss term")
        out["loss"] = total
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
