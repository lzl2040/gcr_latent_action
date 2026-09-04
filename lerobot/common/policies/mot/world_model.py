"""World model built on the Phi-4-Multimodal MoT.

Data flow::

    frame_0 --[Phi-4-MM SigLIP + projector]--> 545 image tok --.
    instruction --[Phi-4-MM embed]-----------> text tok --------.  |
                                                               v  v
                             und stream (Phi-4-MM + vision LoRA) ----.
                                      | same-layer K/V               |
                                      v                              v
    clip --[Wan VAE, frozen]--> latents --> noise --> patches --> gen stream --> velocity

Training advances UND and GEN together layer-by-layer, matching Cosmos3. Inference keeps the
separable UND-cache path so fixed image/text conditioning can be encoded once and reused.

The official multimodal checkpoint contributes the SigLIP NaViT, image projector, language
backbone and rank-256 vision LoRA. Audio and speech-LoRA weights are deliberately not loaded.
For square robot frames, the global and sub crops are identical; the ViT is evaluated once and
its feature grid is reused to construct the checkpoint's exact 545-token ``sub_glb`` layout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import Phi4MultimodalVisionConfig
from transformers.models.phi4_multimodal.modeling_phi4_multimodal import (
    Phi4MultimodalVisionModel,
)

from lerobot.common.policies.mot.modeling_mot import (
    MoTConfig,
    MoTModel,
    copy_safetensor_tensors,
)


Role = Literal["noisy", "clean", "absent"]
TrainingExecution = Literal["interleaved", "cached"]


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

    vision: bool  # Phi-4-MM SigLIP NaViT
    projector: bool  # official image projection + learned row/global separators
    und: bool  # Phi-4-MM language base + vision LoRA
    gen: bool  # generation expert


TRAINABLE_SCOPES: dict[str, TrainableScope] = {
    # Cosmos3-style: only the generation side moves. Cheapest and the safest for a
    # from-scratch gen expert.
    "gen_only": TrainableScope(vision=False, projector=False, und=False, gen=True),
    # pi-0.5-style: freeze the vision encoder, train the rest.
    "freeze_vision": TrainableScope(vision=False, projector=True, und=True, gen=True),
    "all": TrainableScope(vision=True, projector=True, und=True, gen=True),
}


@dataclass
class WorldModelConfig:
    mot: MoTConfig
    trainable_scope: str = "gen_only"
    # Applied on top of the scope; useful for isolating the pretrained projector cost.
    freeze_vision_projector: bool = False
    und_grad_checkpointing: bool = True
    # Cosmos3-style layer-interleaved execution is the default during training. ``cached`` is
    # retained as an A/B and compatibility path; eval always uses the reusable cache path.
    training_execution: TrainingExecution = "interleaved"
    # Checkpoint several complete dual-pathway layers as one segment. Per-layer checkpointing
    # retains one long UND hidden state at every layer; segments retain only their boundaries.
    mot_checkpoint_segment_size: int = 4
    # Image-conditioned UND sequences are 545 image tokens plus text. During interleaved
    # training the complete UND+GEN model is sliced along the batch dimension; during cached
    # execution only UND is sliced. Both preserve the external/global batch semantics.
    und_microbatch_size: int | None = 32
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
            s = replace(s, projector=False)
        if self.training_execution not in ("interleaved", "cached"):
            raise ValueError(
                f"unknown training_execution {self.training_execution!r}; "
                "expected 'interleaved' or 'cached'"
            )
        if self.mot_checkpoint_segment_size <= 0:
            raise ValueError("mot_checkpoint_segment_size must be positive")
        if self.und_microbatch_size is not None and self.und_microbatch_size <= 0:
            raise ValueError("und_microbatch_size must be positive or None")
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


class Phi4MMVisionEmbedding(nn.Module):
    """Official Phi-4-MM vision tower/projector with a square-frame fast path."""

    image_size = 448
    patch_size = 14
    feature_grid = 16
    hidden_size = 1152
    num_tokens = 545

    def __init__(self, out_dim: int):
        super().__init__()
        vision_config = Phi4MultimodalVisionConfig(
            hidden_size=self.hidden_size,
            intermediate_size=4304,
            num_hidden_layers=27,
            num_attention_heads=16,
            image_size=self.image_size,
            patch_size=self.patch_size,
            feature_layer=-2,
            _attn_implementation="sdpa",
        )
        self.img_processor = Phi4MultimodalVisionModel(vision_config)
        # Transformers 4.57 marks this bidirectional SigLIP attention as causal when routing
        # through the generic SDPA backend. The checkpoint's original implementation is
        # non-causal; correct the backend hint while retaining fused SDPA performance.
        for layer in self.img_processor.encoder.layers:
            layer.self_attn.is_causal = False
        self.image_token_compression = nn.AvgPool2d(kernel_size=2, stride=2)
        self.glb_GN = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.sub_GN = nn.Parameter(torch.zeros(1, 1, 1, self.hidden_size))
        self.img_projection = nn.Sequential(
            nn.Linear(self.hidden_size, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def set_trainable(self, encoder: bool, projector: bool) -> None:
        self.img_processor.requires_grad_(encoder)
        if encoder:
            self.img_processor.gradient_checkpointing_enable()
        else:
            self.img_processor.gradient_checkpointing_disable()
        # Phi-4-MM takes the second-to-last encoder output. The final layer, post norm and
        # pooling head are checkpoint residents but not part of the image-embedding path.
        self.img_processor.encoder.layers[-1].requires_grad_(False)
        self.img_processor.post_layernorm.requires_grad_(False)
        self.img_processor.head.requires_grad_(False)
        self.glb_GN.requires_grad_(projector)
        self.sub_GN.requires_grad_(projector)
        self.img_projection.requires_grad_(projector)

    def load_pretrained(self, phi_dir) -> None:
        prefix = "model.embed_tokens_extend.image_embed."
        targets = {prefix + name: tensor for name, tensor in self.state_dict().items()}
        copy_safetensor_tensors(phi_dir, targets)

    def forward(self, images: torch.Tensor, encoder_grad: bool) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"expected images shaped (B,3,H,W), got {tuple(images.shape)}")

        # The official processor uses (x/255 - .5)/.5. Inputs to this module are already [0,1].
        x = F.interpolate(
            images.float(),
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
        )
        x = (x - 0.5) / 0.5
        x = x.to(self.img_processor.embeddings.patch_embedding.weight.dtype)

        with torch.set_grad_enabled(encoder_grad and torch.is_grad_enabled()):
            patch_mask = torch.ones(
                x.shape[0],
                32,
                32,
                dtype=torch.bool,
                device=x.device,
            )
            features = self.img_processor.embeddings(
                pixel_values=x,
                patch_attention_mask=patch_mask,
            )
            # feature_layer=-2 is the output after layer 25. Stop there instead of running
            # the unused 27th layer, post-layernorm and pooling head.
            for layer in self.img_processor.encoder.layers[:-1]:
                features = layer(features, attention_mask=None)
            b = features.shape[0]
            features = features.view(b, 32, 32, self.hidden_size).permute(0, 3, 1, 2)
            features = self.image_token_compression(features).permute(0, 2, 3, 1)
        if not encoder_grad:
            features = features.detach()

        # Official order is sub-grid, one global separator, global-grid. For a square
        # single-crop robot frame the sub/global pixels are identical, so reusing this grid is
        # exactly equivalent to evaluating the same 448x448 crop twice.
        row_sep = self.sub_GN.expand(b, self.feature_grid, 1, self.hidden_size)
        block = torch.cat([features, row_sep], dim=2).reshape(b, -1, self.hidden_size)
        image_tokens = torch.cat([block, self.glb_GN.expand(b, -1, -1), block], dim=1)
        if image_tokens.shape[1] != self.num_tokens:
            raise RuntimeError(f"expected {self.num_tokens} image tokens, got {image_tokens.shape[1]}")
        return self.img_projection(image_tokens)


class MoTWorldModel(nn.Module):
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        self.mot = MoTModel(config.mot)
        self.vision = Phi4MMVisionEmbedding(config.mot.und_hidden_size)
        self.vision_image_size = self.vision.image_size
        self.vision_grid = self.vision.feature_grid
        self.latent_side = config.latent_grid // config.mot.latent_patch_size
        self.apply_trainable_scope()

    # ------------------------------------------------------------------ setup

    def apply_trainable_scope(self) -> None:
        """Set ``requires_grad`` across the four weight groups from the configured scope."""
        s = self.config.scope()
        self.vision.set_trainable(s.vision, s.projector)
        self.mot.set_und_trainable(s.und, kv_only=True)
        self.mot.set_gen_trainable(s.gen)

    def load_pretrained(self) -> None:
        self.mot.load_phi_weights()
        self.vision.load_pretrained(self.config.mot.phi_dir)
        self.apply_trainable_scope()

    @property
    def und_needs_grad(self) -> bool:
        """Whether the und stack must build a graph.

        True if und itself trains, or if anything feeding it does -- a trainable projector or
        vision tower sits upstream, so their gradients only exist if und is differentiated
        through. Getting this wrong is silent: no_grad here would leave those weights with
        ``grad = None`` while the loss still fell on the gen expert alone.
        """
        s = self.config.scope()
        return s.und or s.projector or s.vision

    # ------------------------------------------------------------------ und

    def _prepare_und_batch(
        self,
        pixel_values: torch.Tensor | None,
        text_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        device = text_ids.device
        text_embeds = self.mot.embed_tokens(text_ids)

        if pixel_values is None:
            inputs_embeds = text_embeds
            use_vision_lora = False
        else:
            vision_trains = self.config.scope().vision
            image_embeds = self.vision(pixel_values, encoder_grad=vision_trains)
            inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)
            use_vision_lora = True

        # Phi-4-MM was pretrained with ordinary 1-D LongRoPE over the flattened image/text
        # sequence. Repeating the same index on all three mRoPE axes is exactly that 1-D RoPE.
        seq = torch.arange(inputs_embeds.shape[1], device=device)
        pos = seq.view(1, 1, -1).expand(3, text_ids.shape[0], -1)
        return inputs_embeds, pos, use_vision_lora

    def encode_und(self, pixel_values: torch.Tensor | None, text_ids: torch.Tensor):
        """Build the reusable inference cache. Returns ``(per_layer_kv, rope_und)``.

        ``pixel_values`` is optional: the text-to-image and text-to-video tasks have no input
        frame, and feeding a blank one would spend a full vision-tower pass teaching the model
        that "no image" looks like a particular grey rectangle.
        """
        batch = text_ids.shape[0]
        micro = self.config.und_microbatch_size
        if pixel_values is not None and micro is not None and batch > micro:
            if pixel_values.shape[0] != batch:
                raise ValueError(
                    f"pixel/text batch mismatch: {pixel_values.shape[0]} vs {batch}"
                )
            pieces = [
                self._encode_und_batch(
                    pixel_values[start : start + micro],
                    text_ids[start : start + micro],
                )
                for start in range(0, batch, micro)
            ]
            kv = []
            for layer in range(self.config.mot.num_hidden_layers):
                kv.append(
                    (
                        torch.cat([piece_kv[layer][0] for piece_kv, _ in pieces], dim=0),
                        torch.cat([piece_kv[layer][1] for piece_kv, _ in pieces], dim=0),
                    )
                )
                # Release each slice as soon as its merged layer exists. Otherwise torch.cat
                # temporarily keeps two complete copies of all 32 layers' K/V.
                for piece_kv, _ in pieces:
                    piece_kv[layer] = None
            rope = tuple(
                torch.cat([piece_rope[i] for _, piece_rope in pieces], dim=0)
                for i in range(2)
            )
            return kv, rope
        return self._encode_und_batch(pixel_values, text_ids)

    def _encode_und_batch(self, pixel_values: torch.Tensor | None, text_ids: torch.Tensor):
        inputs_embeds, pos, use_vision_lora = self._prepare_und_batch(
            pixel_values,
            text_ids,
        )
        run = self._forward_und_checkpointed if self.und_needs_grad else self._forward_und_nograd
        _, kv, rope = run(inputs_embeds, pos, use_vision_lora)
        return kv, rope

    def _forward_und_nograd(self, inputs_embeds, pos, use_vision_lora):
        with torch.no_grad():
            kv, rope = self.mot.forward_und_kv(inputs_embeds, pos, use_vision_lora)
        return None, kv, rope

    def _forward_und_checkpointed(self, inputs_embeds, pos, use_vision_lora):
        if not (self.config.und_grad_checkpointing and self.training):
            kv, rope = self.mot.forward_und_kv(inputs_embeds, pos, use_vision_lora)
            return None, kv, rope
        rope = tuple(r.to(inputs_embeds.dtype) for r in self.mot.rotary_emb(pos))
        hidden = inputs_embeds
        kv = []
        for layer in self.mot.layers[:-1]:
            hidden, k, v = checkpoint(
                layer.und_forward,
                hidden,
                rope,
                use_vision_lora,
                use_reentrant=False,
            )
            kv.append((k, v))
        k, v = checkpoint(
            self.mot.layers[-1].und_kv,
            hidden,
            use_vision_lora,
            use_reentrant=False,
        )
        kv.append((k, v))
        return None, kv, rope

    def _forward_interleaved_batch(
        self,
        pixel_values: torch.Tensor | None,
        text_ids: torch.Tensor,
        gen_tokens: torch.Tensor,
        gen_position_ids: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        und_hidden, und_position_ids, use_vision_lora = self._prepare_und_batch(
            pixel_values,
            text_ids,
        )
        checkpoint_layers = self.training and (
            self.mot.gradient_checkpointing
            or (self.und_needs_grad and self.config.und_grad_checkpointing)
        )
        return self.mot.forward_interleaved(
            und_hidden=und_hidden,
            und_position_ids=und_position_ids,
            gen_hidden=gen_tokens,
            gen_position_ids=gen_position_ids,
            timestep=timestep,
            use_vision_lora=use_vision_lora,
            und_requires_grad=self.und_needs_grad,
            checkpoint_layers=checkpoint_layers,
            checkpoint_segment_size=self.config.mot_checkpoint_segment_size,
        )

    def forward_interleaved(
        self,
        pixel_values: torch.Tensor | None,
        text_ids: torch.Tensor,
        gen_tokens: torch.Tensor,
        gen_position_ids: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Run Cosmos3-style training without retaining an all-layer UND cache."""
        batch = text_ids.shape[0]
        if gen_tokens.shape[0] != batch or gen_position_ids.shape[1] != batch:
            raise ValueError("UND and GEN batch dimensions must match")
        if pixel_values is not None and pixel_values.shape[0] != batch:
            raise ValueError(
                f"pixel/text batch mismatch: {pixel_values.shape[0]} vs {batch}"
            )

        micro = self.config.und_microbatch_size
        if pixel_values is not None and micro is not None and batch > micro:
            outputs = []
            for start in range(0, batch, micro):
                end = min(start + micro, batch)
                outputs.append(
                    self._forward_interleaved_batch(
                        pixel_values[start:end],
                        text_ids[start:end],
                        gen_tokens[start:end],
                        gen_position_ids[:, start:end],
                        timestep[start:end],
                    )
                )
            return torch.cat(outputs, dim=0)

        return self._forward_interleaved_batch(
            pixel_values,
            text_ids,
            gen_tokens,
            gen_position_ids,
            timestep,
        )

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

        pos = self.gen_positions(t_lat, n_action, device).unsqueeze(1).expand(3, b, -1)
        timestep = token_sigma * 1000.0
        if self.training and self.config.training_execution == "interleaved":
            hidden = self.forward_interleaved(
                pixel_values,
                text_ids,
                gen_tokens,
                pos,
                timestep,
            )
        else:
            kv, rope_und = self.encode_und(pixel_values, text_ids)
            hidden = self.mot.forward_gen(gen_tokens, pos, kv, rope_und, timestep)

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
        vision = count(self.vision.img_processor)
        projector = count(self.vision) - vision
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "vision_frozen": vision,
            "vision_projector": projector,
            "und_frozen": mot["und"],
            "gen_trainable": mot["gen"],
            "total": vision + projector + mot["total"],
            "trainable": trainable,
        }
