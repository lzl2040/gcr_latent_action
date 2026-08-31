"""RoboContrast: contrastive alignment of the perception side and the physical side of a robot.

Perception side
    ``vision(t)``, ``vision(t + H)`` and ``language``. A set of latent *change queries*, seeded
    from the instruction embedding, cross-attends over an evidence bank built from the two
    frames and their difference. Language therefore decides *which* visual change matters,
    which is what makes the target comparable across scenes.

Physical side
    canonical ``state``, the ``action`` chunk and ``tactile`` sensing. Tactile arrives either as
    a low-dimensional signal (forces/torques) or as images, and both are folded into a single
    token stream so that datasets with, without, or with a different kind of tactile sensor all
    train together.

The two sides are projected into a shared space and aligned with a symmetric InfoNCE loss
computed over the union of all ranks.

Guarding against tactile domination
    Tactile images are 4 x 3 x 64 x 64 = 49k raw dimensions versus 40 for the state, so if they
    were encoded with the same capacity as the rest they would trivially dominate. Three
    mechanisms prevent that:
      * a deliberately shallow CNN pooled into a *single* token, so tactile never outnumbers
        the action tokens;
      * a learnable gate initialised at zero, so training starts from a tactile-free model and
        only opens the channel if it helps;
      * per-sample modality dropout, so no modality is ever guaranteed present.
"""

from __future__ import annotations

import itertools
import math
import os
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
from torch.utils.checkpoint import checkpoint

from lerobot.common.policies.ace.configuration_robo_contrast import RoboContrastConfig
from lerobot.common.policies.ace.ftp1_tactile import FTP1_SENSOR_NAMES, FTP1TactileTower
from lerobot.common.policies.pretrained import PreTrainedPolicy

# DINOv3 is normalised with ImageNet statistics (see its preprocessor_config.json), unlike
# SigLIP2 which uses 0.5/0.5.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
# SigLIP-family towers (including Cosmos3's) are trained on symmetric [-1, 1] inputs.
SIGLIP_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
SIGLIP_STD = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)


def _all_gather_detached(tensor: torch.Tensor) -> torch.Tensor:
    """Gather ``tensor`` from every rank, detaching the contributions of other ranks.

    Gradients only flow through the local slice, which keeps the backward pass local while
    still exposing ``world_size * batch_size`` negatives to the loss.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor
    with torch.no_grad():
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor.detach().contiguous())
    # Re-attach the local slice so that gradients flow through it (and only it).
    gathered[dist.get_rank()] = tensor
    return torch.cat(gathered, dim=0)


def _rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def pairwise_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Similarity between every row of ``a`` and every row of ``b``.

    ``(N, D) x (M, D) -> (N, M)`` is the ordinary single-vector case: a dot product of two
    already-L2-normalised embeddings.

    ``(N, K, D) x (M, J, D) -> (N, M)`` is ColBERT-style late interaction: each of ``a``'s K
    tokens finds its best match among ``b``'s J tokens, and those matches are averaged. This
    lets a chunk containing two distinct sub-motions be represented by two tokens that match
    independently, instead of by their average -- which is the whole reason for K > 1.

    The mean (rather than ColBERT's sum) is deliberate: it keeps the result in [-1, 1] exactly
    as the K=1 dot product is, so `logit_scale` and `temperature` stay calibrated and a K
    sweep does not silently rescale the loss.
    """
    if a.dim() == 2:
        return a @ b.t()
    return torch.einsum("nkd,mjd->nmkj", a, b).amax(dim=-1).mean(dim=-1)


def paired_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise similarity of matched pairs: ``(N, ...) -> (N,)``. Diagnostics only."""
    if a.dim() == 2:
        return (a * b).sum(-1)
    return torch.einsum("nkd,njd->nkj", a, b).amax(dim=-1).mean(dim=-1)


# ---------------------------------------------------------------------------
# generic blocks
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """Pre-norm multi-head attention supporting self- and cross-attention."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor | None = None,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        kv_source = query if context is None else context
        q = self.norm_q(query)
        kv = self.norm_q(kv_source) if context is None else self.norm_kv(kv_source)

        b, n, _ = q.shape
        m = kv.shape[1]

        q = self.q_proj(q).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(b, m, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(b, m, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if key_padding_mask is not None:
            # True = keep. Expand to (B, 1, 1, M) for scaled_dot_product_attention.
            attn_mask = key_padding_mask[:, None, None, :].to(torch.bool)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(b, n, self.num_heads * self.head_dim)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


class _CheckpointMixin:
    """Optional activation checkpointing for the two token-heavy trunks.

    The evidence bank is ~420 tokens wide at batch 256, so storing every intermediate
    activation costs more memory than the parameters themselves. Recomputing them in the
    backward pass trades ~30% extra compute for a large memory saving, which is the right
    trade here: training is bound by the disk, not by the GPU.
    """

    use_checkpointing: bool = False

    def _run_block(self, block, *args):
        if self.use_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, *args, use_reentrant=False)
        return block(*args)


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.ffn = FeedForward(dim, dropout=dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(x, None, key_padding_mask)
        return x + self.ffn(x)


class ChangeQueryBlock(nn.Module):
    """One round of the change-query decoder: read the fused evidence, then think.

    The instruction lives *inside* the evidence bank (see ``PerceptionEncoder``), so a single
    cross-attention is enough; an extra text-only cross-attention would re-read tokens the
    queries can already see.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.ffn = FeedForward(dim, dropout=dropout)

    def forward(self, queries, evidence, evidence_mask=None):
        queries = queries + self.self_attn(queries)
        queries = queries + self.cross_attn(queries, evidence, evidence_mask)
        return queries + self.ffn(queries)


# ---------------------------------------------------------------------------
# perception side
# ---------------------------------------------------------------------------
class ChangePredictor(nn.Module, _CheckpointMixin):
    """Predicts the frame-``t+H`` visual features from frame ``t``, the text and the queries.

    This is the perception-side counterpart of the tactile reconstruction head: without it the
    only thing shaping the change queries is the contrastive loss, which is happy with any
    representation that happens to separate the batch -- scene identity, camera pose, dataset
    style -- and need not encode motion at all.

    The construction is what makes the objective meaningful. The predictor sees the *raw*
    projected frame-``t`` patches, never the evidence trunk's output: the trunk has already
    attended over the ``v1 - v0`` difference stream, so its output determines ``v1`` exactly
    and a predictor reading it could ignore the queries entirely. Here the 16 change queries
    are the only channel through which anything about frame ``t+H`` can reach the prediction,
    so lowering this loss requires putting the change into them.
    """

    def __init__(self, dim: int, out_dim: int, num_layers: int, num_heads: int, dropout: float,
                 max_patches: int = 4096, use_checkpointing: bool = False):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.patch_pos = nn.Parameter(torch.randn(1, max_patches, dim) * 0.02)
        self.blocks = nn.ModuleList(
            [ChangeQueryBlock(dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_dim)

    def forward(self, v0, memory, memory_mask):
        x = v0 + self.patch_pos[:, : v0.shape[1]].to(v0.dtype)
        for block in self.blocks:
            x = self._run_block(block, x, memory, memory_mask)
        return self.head(self.norm(x))


class PerceptionEncoder(nn.Module, _CheckpointMixin):
    """Text-conditioned extractor of the visual change between ``t`` and ``t + H``."""

    def __init__(self, config: RoboContrastConfig):
        super().__init__()
        self.use_checkpointing = config.gradient_checkpointing
        from transformers import AutoModel, AutoTokenizer

        vision_name = config.vision_model_name
        if not os.path.exists(vision_name):
            vision_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
        text_name = config.text_model_name
        if not os.path.exists(text_name):
            text_name = "google/siglip2-base-patch16-224"

        self.backbone_kind = config.vision_backbone
        if self.backbone_kind == "cosmos3":
            from .cosmos3_encoders import build_cosmos3_vision

            self.vision_backbone, _, self.image_size = build_cosmos3_vision(config.cosmos3_dir)
            # This tower emits patch tokens only -- there is no CLS and no register token to
            # strip -- and it was trained with SigLIP's symmetric [-1, 1] normalisation, not
            # ImageNet's. Getting either wrong puts the input off-distribution for weights we
            # are not training, which is silent: the model still runs, just worse.
            self.num_prefix_tokens = 0
            self.pixel_mean, self.pixel_std = SIGLIP_MEAN, SIGLIP_STD
        elif self.backbone_kind == "qwen3vl":
            from .qwen3vl_encoder import build_qwen3vl_vision

            self.vision_backbone, _, self.image_size = build_qwen3vl_vision(config.qwen3vl_dir)
            # Patch tokens only, already reordered to row-major by the wrapper, and the same
            # symmetric [-1, 1] normalisation SigLIP uses.
            self.num_prefix_tokens = 0
            self.pixel_mean, self.pixel_std = SIGLIP_MEAN, SIGLIP_STD
        else:
            self.vision_backbone = AutoModel.from_pretrained(vision_name, dtype=torch.float32)
            # DINOv3 prepends one CLS token and `num_register_tokens` register tokens; only the
            # patch tokens are spatially meaningful, so the prefix is dropped.
            self.num_prefix_tokens = 1 + int(
                getattr(self.vision_backbone.config, "num_register_tokens", 0)
            )
            self.image_size = 224
            self.pixel_mean, self.pixel_std = IMAGENET_MEAN, IMAGENET_STD

        # Only SigLIP2's *text* tower is kept; dropping its vision tower saves 93M frozen
        # parameters that the vision backbone now replaces.
        text_full = AutoModel.from_pretrained(text_name, dtype=torch.float32)
        self.text_backbone = text_full.text_model
        del text_full
        self.tokenizer = AutoTokenizer.from_pretrained(text_name)
        self.text_max_length = config.text_max_length

        vision_dim = self.vision_backbone.config.hidden_size
        text_dim = self.text_backbone.config.hidden_size
        self.vision_dim = vision_dim
        dim = config.hidden_dim
        self.patch_stride = max(1, config.patch_token_stride)

        self.freeze_vision = config.freeze_vision_encoder
        self.freeze_text = config.freeze_text_encoder
        if self.freeze_vision:
            for p in self.vision_backbone.parameters():
                p.requires_grad = False
        if self.freeze_text:
            for p in self.text_backbone.parameters():
                p.requires_grad = False

        # Normalise the frozen features before projecting them. Backbones differ wildly in
        # output scale -- DINOv3 patch tokens have L2 ~12.6 where SigLIP2's have ~43.9 -- and
        # without this the rest of the encoder is implicitly tuned to one particular backbone.
        self.vision_norm = nn.LayerNorm(vision_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        self.visual_proj = nn.Linear(vision_dim, dim)
        self.text_proj = nn.Linear(text_dim, dim)
        # Two consecutive frames are nearly identical, so `v1 - v0` is an order of magnitude
        # smaller than either. Left alone it is swamped by the appearance streams and by the
        # type embeddings, and the pre-norm blocks then read a token that is mostly type
        # embedding. Rescaling the difference to unit scale is what keeps the change legible.
        self.diff_norm = nn.LayerNorm(dim)
        # 0 = frame t, 1 = frame t+H, 2 = their difference, 3 = language
        self.evidence_type_embed = nn.Embedding(4, dim)
        # nn.Embedding defaults to N(0, 1), which here is a per-token L2 of ~32 against a
        # visual signal of ~8: the tag would drown the content it is supposed to label.
        nn.init.normal_(self.evidence_type_embed.weight, std=0.02)

        self.change_queries = nn.Parameter(torch.randn(config.num_change_queries, dim) * 0.02)
        # What seeds the change queries when a sample has no instruction. Feeding the empty
        # string instead would be subtly wrong twice over: the tokenizer's BOS/EOS still pool
        # to a non-zero, *arbitrary* vector, and that vector is identical for every caption-free
        # sample, which makes "this dataset has no language" a constant the contrastive stage
        # can read off as a dataset fingerprint. A dedicated learned vector says "no
        # instruction" explicitly, and starting it at zero means the queries begin as the bare
        # learned prototypes -- the honest prior for an unconditioned sample.
        self.null_text = nn.Parameter(torch.zeros(dim))
        # Trunk: joint vision-vision-difference-language reasoning over the full evidence bank.
        self.evidence_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(dim, config.fusion_num_heads, config.dropout)
                for _ in range(config.num_evidence_layers)
            ]
        )
        self.evidence_norm = nn.LayerNorm(dim)
        # Decoder: a handful of latent queries distil the bank into "what changed".
        self.blocks = nn.ModuleList(
            [
                ChangeQueryBlock(dim, config.fusion_num_heads, config.dropout)
                for _ in range(config.num_fusion_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, config.projection_dim),
        )
        # Pool the change queries down to `num_cls_tokens` summary vectors. A learned mixing
        # matrix over the query axis, initialised to a uniform 1/Q, means K=1 reproduces the
        # previous `queries.mean(dim=1)` exactly at init -- so switching this on cannot by
        # itself change the K=1 result -- while K>1 lets each output token learn to draw on a
        # different subset of queries instead of being handed an arbitrary slice of them.
        self.num_cls_tokens = config.num_cls_tokens
        self.query_pool = nn.Parameter(
            torch.full(
                (config.num_change_queries, config.num_cls_tokens),
                1.0 / config.num_change_queries,
            )
        )

        self.recon_weight = config.perception_recon_weight
        # The reconstruction target is either the vision tower's own frame-(t+H) features or
        # the Cosmos3/Wan2.2 VAE latent of that frame. They are interchangeable because the
        # two grids coincide: both the ViT and the VAE divide the image into 16-pixel cells,
        # so each patch token has exactly one latent cell, and "predict t+H" stays a per-token
        # regression in either case -- only the channel count changes.
        self.recon_target = config.perception_recon_target
        self.vae = None
        target_dim = vision_dim
        if self.predictor_enabled(config) and self.recon_target == "vae":
            from .cosmos3_encoders import build_cosmos3_vae

            vae, z_dim, _, lat_mean, lat_std = build_cosmos3_vae(config.cosmos3_dir)
            for p in vae.parameters():
                p.requires_grad = False
            vae.eval()
            self.vae = vae
            self.vae_repeat_frames = config.vae_repeat_frames
            target_dim = z_dim
            # Wan2.2's published per-channel latent statistics. Without them the 48 channels
            # differ in scale by over an order of magnitude and the loss is dominated by
            # whichever few happen to be largest -- the same failure section 19 hit with
            # unnormalised tactile targets.
            self.register_buffer("vae_latent_mean", lat_mean.view(1, 1, -1), persistent=False)
            self.register_buffer("vae_latent_std", lat_std.view(1, 1, -1), persistent=False)

        self.predictor = (
            ChangePredictor(
                dim,
                target_dim,
                config.num_predictor_layers,
                config.fusion_num_heads,
                config.dropout,
                use_checkpointing=config.gradient_checkpointing,
            )
            if self.predictor_enabled(config)
            else None
        )
        # Targets are normalised per token before the loss (as in I-JEPA), so the objective is
        # about the *pattern* of the feature vector rather than its magnitude, which otherwise
        # dominates the L1 and is trivially predictable from frame t.
        self.target_norm = nn.LayerNorm(target_dim, elementwise_affine=False)

    @staticmethod
    def predictor_enabled(config: RoboContrastConfig) -> bool:
        return config.num_predictor_layers > 0 and config.perception_recon_weight > 0

    def _vae_target(self, image_t1: torch.Tensor) -> torch.Tensor:
        """Frame ``t+H`` -> ``(B, N, z_dim)`` VAE latent tokens on the ViT's patch grid.

        The VAE is causal and pads its own temporal history, so a single frame already yields
        one latent frame; ``vae_repeat_frames`` is kept only because the config exposes it and
        was measured to change nothing (T=1 and T=4 give bit-identical latents).
        """
        dtype = _module_dtype(self.vae, default=torch.float32)
        size = self.image_size
        x = image_t1.to(dtype=torch.float32)
        if x.shape[-1] != size or x.shape[-2] != size:
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        # The Wan VAE consumes [-1, 1], which is exactly the SigLIP normalisation.
        x = (x / 255.0 - 0.5) / 0.5
        video = x.to(dtype).unsqueeze(2)
        if self.vae_repeat_frames > 1:
            video = video.repeat(1, 1, self.vae_repeat_frames, 1, 1)
        with torch.no_grad():
            latent = self.vae.encode(video).latent_dist.mean  # (B, z, 1, h, w)
        latent = latent[:, :, 0].flatten(2).transpose(1, 2)  # (B, h*w, z)
        latent = (latent.float() - self.vae_latent_mean) / self.vae_latent_std
        if self.patch_stride > 1:
            # Both grids are row-major over the same 16-pixel cells, so the identical stride
            # keeps the predicted tokens and the target tokens pointing at the same cells.
            latent = latent[:, :: self.patch_stride, :]
        return latent    # -- raw input handling -------------------------------------------------
    def _to_pixel_values(self, images: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """uint8 ``(B, 3, H, W)`` in ``[0, 255]`` -> normalised float tensor.

        Resolution and normalisation follow the *backbone*, not the dataset: the loader emits
        224px but Cosmos3's tower was trained at 256px with symmetric normalisation, and
        feeding it 224px ImageNet-normalised input degrades frozen weights silently.

        Doing this on-device replaces the previous PIL round-trip, which was the single most
        expensive step of the old training loop.
        """
        x = images.to(dtype=torch.float32)
        size = self.image_size
        if x.shape[-1] != size or x.shape[-2] != size:
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        mean = self.pixel_mean.to(device=x.device, dtype=x.dtype)
        std = self.pixel_std.to(device=x.device, dtype=x.dtype)
        x = (x / 255.0 - mean) / std
        return x.to(dtype=dtype)

    def tokenize(self, texts: list[str], device: torch.device):
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.text_max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        return input_ids, attention_mask

    def _encode_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        ctx = torch.no_grad() if self.freeze_vision else torch.enable_grad()
        with ctx:
            out = self.vision_backbone(pixel_values=pixel_values)
        tokens = out.last_hidden_state[:, self.num_prefix_tokens :, :]
        if self.patch_stride > 1:
            tokens = tokens[:, :: self.patch_stride, :]
        return tokens.detach() if self.freeze_vision else tokens

    def _encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        ctx = torch.no_grad() if self.freeze_text else torch.enable_grad()
        with ctx:
            out = self.text_backbone(input_ids)
        tokens = out.last_hidden_state
        return tokens.detach() if self.freeze_text else tokens

    def forward(self, image_t0, image_t1, texts, has_text=None, probe: bool = False):
        """``has_text``: per-sample 0/1: 0 means the sample carries no real instruction.

        Stage-1 video pre-training runs largely on caption-free data, so this is the common
        case there rather than an edge case. ``None`` means "assume every sample has text",
        which keeps the contrastive stage's behaviour unchanged.

        ``probe`` adds the (cheap, no-grad, diagnostic-only) change-query ablation described
        in ``_recon_loss``. Returns ``(embedding, recon_loss, aux)``.
        """
        device = self.visual_proj.weight.device
        dtype = self.visual_proj.weight.dtype
        batch = image_t0.shape[0]

        # The pixels are consumed by the vision backbone, not by `visual_proj`; a frozen
        # backbone is exactly the kind of module a mixed-precision backend may leave in a
        # different dtype than the trainable trunk.
        pixel_dtype = _module_dtype(self.vision_backbone, default=dtype)
        pixels = torch.cat(
            [self._to_pixel_values(image_t0, pixel_dtype), self._to_pixel_values(image_t1, pixel_dtype)],
            dim=0,
        )
        patches = self._encode_vision(pixels).to(dtype)
        p0, p1 = patches[:batch], patches[batch:]

        input_ids, text_mask = self.tokenize(texts, device)
        text_tokens = self.text_proj(self.text_norm(self._encode_text(input_ids).to(dtype)))

        # Drop the caption-free samples' text tokens out of every attention that reads them.
        # The evidence bank always keeps its visual tokens unmasked, so no row can end up
        # fully masked -- which would make the softmax return NaN rather than "ignore this".
        keep_text = None
        if has_text is not None:
            keep_text = has_text.to(device=device).reshape(-1, 1) > 0.5
            if text_mask is None:
                text_mask = torch.ones(batch, text_tokens.shape[1], device=device, dtype=torch.long)
            text_mask = text_mask * keep_text.to(text_mask.dtype)

        v0 = self.visual_proj(self.vision_norm(p0))
        v1 = self.visual_proj(self.vision_norm(p1))
        diff = self.diff_norm(v1 - v0)

        type_ids = torch.arange(4, device=device)
        type_emb = self.evidence_type_embed(type_ids).to(dtype)
        # The bank carries the scene at `t`, what moved over the horizon, and the instruction.
        # `v1` is deliberately *not* included: it equals `v0 + diff`, so a third visual stream
        # would add 196 tokens of compute and memory for zero extra information. Keeping the
        # difference explicit (rather than letting attention derive it) preserves the useful
        # part: patches are spatially aligned across the two frames, so `diff[i]` is directly
        # "what changed at location i".
        evidence = torch.cat(
            [
                v0 + type_emb[0],
                diff + type_emb[2],
                text_tokens + type_emb[3],
            ],
            dim=1,
        )  # (B, 2N + L, D)

        num_visual = v0.shape[1] * 2
        if text_mask is not None:
            visual_mask = torch.ones(batch, num_visual, device=device, dtype=torch.bool)
            evidence_mask = torch.cat([visual_mask, text_mask.to(torch.bool)], dim=1)
        else:
            evidence_mask = None

        for block in self.evidence_blocks:
            evidence = self._run_block(block, evidence, evidence_mask)
        evidence = self.evidence_norm(evidence)

        # Seed the change queries with the sentence embedding so the *instruction* selects
        # which change to look for, instead of the queries being scene-agnostic.
        if text_mask is not None:
            denom = text_mask.sum(dim=1, keepdim=True).clamp(min=1).to(dtype)
            text_pooled = (text_tokens * text_mask.unsqueeze(-1).to(dtype)).sum(dim=1) / denom
        else:
            text_pooled = text_tokens.mean(dim=1)

        queries = self.change_queries.unsqueeze(0).expand(batch, -1, -1).to(dtype)
        if keep_text is not None:
            # A fully-masked caption pools to exactly zero above, so this substitutes the
            # learned null-instruction vector rather than adding it to garbage.
            text_pooled = torch.where(keep_text, text_pooled, self.null_text.to(dtype).expand(batch, -1))
        queries = queries + text_pooled.unsqueeze(1)

        for block in self.blocks:
            queries = block(queries, evidence, evidence_mask)

        queries = self.out_norm(queries)
        # (B, Q, D) -> (B, K, D): one summary vector per contrastive token.
        pooled = torch.einsum("bqd,qk->bkd", queries, self.query_pool.to(dtype))
        embedding = self.out_proj(pooled)
        if self.num_cls_tokens == 1:
            embedding = embedding.squeeze(1)

        recon_loss = None
        aux: dict[str, float] = {}
        if self.predictor is not None and self.training:
            if self.recon_target == "vae":
                target = self._vae_target(image_t1).to(dtype)
            else:
                # From the frozen backbone and detached: there is no trainable path into the
                # target, so the pair cannot collapse onto a constant the way a jointly
                # trained student/teacher would.
                target = p1.detach()
            recon_loss, aux = self._recon_loss(v0, queries, text_tokens, text_mask, target, probe=probe)
        return embedding, recon_loss, aux

    def _recon_loss(self, v0, queries, text_tokens, text_mask, target, probe: bool = False):
        """Predict the frame-``t+H`` patch features and score them against the real ones.

        Returns ``(loss, aux)``.

        The failure mode this objective has to be watched for is that most of ``p1`` is
        predictable from ``v0`` alone -- background, table, static objects -- so the predictor
        can drive the loss down while ignoring the change queries entirely, which are the only
        thing being pre-trained *for* stage 2. The loss curve looks healthy either way.

        ``probe`` measures it directly: rerun the prediction with each sample's queries
        replaced by another sample's. If the queries carry information about this pair, that
        substitution must hurt. ``percep_query_gain`` is the increase in loss; a value near
        zero means the change queries are decorative and the pre-training is not transferring
        anything the contrastive stage will care about.
        """
        memory = torch.cat([queries, text_tokens], dim=1)
        if text_mask is not None:
            query_mask = torch.ones(
                queries.shape[0], queries.shape[1], device=queries.device, dtype=torch.bool
            )
            memory_mask = torch.cat([query_mask, text_mask.to(torch.bool)], dim=1)
        else:
            memory_mask = None

        pred = self.predictor(v0, memory, memory_mask).float()
        if pred.shape[1] != target.shape[1]:
            raise RuntimeError(
                f"Reconstruction grid mismatch: predictor emits {pred.shape[1]} tokens but the "
                f"'{self.recon_target}' target has {target.shape[1]}. The two must index the "
                "same 16-pixel cells of the same image."
            )
        target = self.target_norm(target.float())
        loss = F.smooth_l1_loss(pred, target)

        aux: dict[str, float] = {}
        if probe and queries.shape[0] > 1:
            with torch.no_grad():
                # Roll rather than a random permutation: it is guaranteed to be derangement-like
                # (no sample keeps its own queries), which a random shuffle is not.
                shuffled = torch.cat([queries.roll(1, dims=0), text_tokens], dim=1)
                bad = self.predictor(v0, shuffled, memory_mask).float()
                aux["percep_query_gain"] = (F.smooth_l1_loss(bad, target) - loss).item()
        return loss, aux


# ---------------------------------------------------------------------------
# physical side
# ---------------------------------------------------------------------------
def _module_dtype(module: nn.Module, default: torch.dtype = torch.float32) -> torch.dtype:
    """The dtype a tensor must have to be fed to ``module``.

    Read it off the module that actually consumes the tensor, never off a neighbouring one.
    Mixed-precision backends do not cast a model uniformly: several keep normalisation layers
    in fp32 while casting Linear and Conv weights to bf16, so a LayerNorm's dtype says nothing
    about the convolution two lines below it. Inferring one from the other produced
    ``Input type (torch.cuda.FloatTensor) and weight type (CUDABFloat16Type) should be the
    same`` on the cluster while being silently correct on a box that casts everything.
    """
    for tensor in itertools.chain(module.parameters(), module.buffers()):
        if tensor.is_floating_point():
            return tensor.dtype
    return default


class _FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d with frozen statistics that runs in whatever dtype its input arrives in.

    ``torchvision.ops.misc.FrozenBatchNorm2d`` stores ``weight``/``bias``/``running_*`` as
    *buffers* and has no parameters at all. A mixed-precision backend that casts parameters
    but leaves buffers alone therefore turns it into an fp32 island in a bf16 network: it
    accepts the bf16 activation, promotes it against its fp32 statistics and hands fp32 to the
    next convolution, which fails with

        Input type (torch.cuda.FloatTensor) and weight type (CUDABFloat16Type) should be the
        same

    inside ``torchvision/models/resnet.py``, several layers away from anything we wrote.
    Casting the statistics to the activation's dtype at call time makes the module immune to
    however the surrounding framework decided to split parameters from buffers.

    The buffer names match torchvision's, so checkpoints stay interchangeable.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The statistics are folded in fp32 -- rsqrt of a small variance in bf16 loses real
        # precision -- and only the resulting affine is cast down to meet the activation.
        scale = self.weight.float() * (self.running_var.float() + self.eps).rsqrt()
        bias = self.bias.float() - self.running_mean.float() * scale
        shape = (1, -1) + (1,) * (x.dim() - 2)
        return x * scale.to(x.dtype).view(shape) + bias.to(x.dtype).view(shape)

    def extra_repr(self) -> str:
        return f"{self.weight.shape[0]}, eps={self.eps}"


def _freeze_batchnorm(module: nn.Module) -> nn.Module:
    """Replace every ``BatchNorm2d`` with a ``_FrozenBatchNorm2d`` carrying the same statistics."""
    if isinstance(module, nn.BatchNorm2d):
        frozen = _FrozenBatchNorm2d(module.num_features, eps=module.eps)
        frozen.weight.data.copy_(module.weight.data)
        frozen.bias.data.copy_(module.bias.data)
        frozen.running_mean.data.copy_(module.running_mean.data)
        frozen.running_var.data.copy_(module.running_var.data)
        return frozen
    for name, child in module.named_children():
        module.add_module(name, _freeze_batchnorm(child))
    return module


class TactileImageEncoder(nn.Module):
    """ResNet-18 tactile encoder, after UniVTAC (``UniVTAC/encoder/network.py::Tactile``).

    UniVTAC unifies heterogeneous *optical* tactile sensors (GelSight Mini, ViTAI GF225,
    XenseWS) by pushing all of them through one ImageNet-initialised ResNet-18 with
    ``num_classes=512``, i.e. the final FC is reused as an embedding head. It relies on the
    backbone alone to abstract over gel colour, marker pattern and resolution; there is no
    per-sensor adapter. We keep that choice — our sensors are equally heterogeneous and we
    have no calibration data — but add an explicit view embedding downstream, because unlike
    UniVTAC we must also cope with a *varying number* of sensors per dataset.

    Note UniVTAC feeds raw ``[0, 1]`` images (no ImageNet normalisation) because its backbone
    was re-trained from scratch on tactile data; we start from ImageNet weights, so we apply
    ImageNet statistics instead.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, out_dim: int = 512, pretrained: bool = True):
        super().__init__()
        import torchvision

        weights = None
        if pretrained:
            try:
                weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            except Exception:  # offline / older torchvision
                weights = None
        try:
            net = torchvision.models.resnet18(weights=weights)
        except Exception:
            net = torchvision.models.resnet18(weights=None)
        # Frozen BatchNorm, as UniVTAC does when it plugs the encoder into the policy
        # (`policy/ACT/detr/models/backbone.py`). Here it is not optional: the number of
        # tactile pads varies per sample, so the effective BN batch size is data dependent
        # and would make the features jitter with the mixture composition.
        net = _freeze_batchnorm(net)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layers = nn.Sequential(net.layer1, net.layer2, net.layer3, net.layer4)
        self.proj = nn.Linear(512, out_dim) if out_dim != 512 else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)

        self.register_buffer("mean", torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B, V, 3, H, W)`` uint8 -> per-view embedding ``(B, V, out_dim)`` and feature map."""
        b, v = images.shape[:2]
        conv_dtype = _module_dtype(self.stem)
        x = images.reshape(b * v, *images.shape[2:]).to(dtype=torch.float32) / 255.0
        x = ((x - self.mean) / self.std).to(conv_dtype)
        feat_map = self.layers(self.stem(x))
        pooled = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)
        # The head may sit in a different dtype than the convolutions (see `_module_dtype`).
        head_dtype = self.proj.weight.dtype if isinstance(self.proj, nn.Linear) else self.norm.weight.dtype
        emb = self.norm(self.proj(pooled.to(head_dtype)))
        return emb.view(b, v, -1), feat_map


class TactileReconHead(nn.Module):
    """Reconstructs the tactile image from its embedding.

    UniVTAC pretrains its encoder with MSE reconstruction of the gel image plus marker
    positions, depth and contact pose. Those three extra targets only exist in simulation, so
    for real sensors we keep the one head whose supervision is always available. The point is
    not the reconstruction itself but that the tactile features are shaped by an objective of
    their own instead of being dragged around by the contrastive gradient.

    The target is z-scored per dataset before the MSE; see ``_tactile_recon_loss``. That is
    what makes this head do anything at all, so the output is deliberately unbounded -- no
    sigmoid, unlike UniVTAC's ``RGBDecoder``, which predicts into [0, 1].
    """

    def __init__(self, in_dim: int, out_size: int = 28):
        super().__init__()
        self.out_size = out_size
        self.fc = nn.Linear(in_dim, 256 * 7 * 7)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 3, 3, 1, 1),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        x = self.fc(emb).view(-1, 256, 7, 7)
        x = self.net(x)
        if x.shape[-1] != self.out_size:
            x = F.interpolate(x, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        return x


class TactilePadTemporal(nn.Module):
    """Fuse one pad's ``F`` frame features into ``T`` tokens.

    The pad is the unit of fusion, not the batch: each pad is a separate physical contact
    surface, so mixing pads here would force the module to disentangle "which finger" before
    it can describe "what happened", and the physical transformer downstream already has a
    per-pad embedding for exactly that. Keeping fusion inside the pad also leaves the
    per-pad mask semantics untouched -- a dead pad is still replaced wholesale by the
    ``missing`` token, which is not expressible once pads are pooled together.

    Frames are marked with a learned temporal embedding and read by ``T`` learned queries in a
    single self-attention block over ``T + F`` tokens. The queries are what leaves; the frame
    tokens are scratch. ``F`` is 4, so the attention is negligible next to the ResNet passes
    that produced its input.

    The first query is initialised to read the window start and the second the change across
    it, matching what the previous ``[feat_t, feat_t1 - feat_t]`` concatenation encoded, so
    ``T = 2`` starts from the old behaviour and is free to depart from it.
    """

    def __init__(self, dim: int, num_frames: int, num_tokens: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_frames = num_frames
        self.num_tokens = num_tokens
        self.frame_embed = nn.Parameter(torch.randn(1, num_frames, dim) * 0.02)
        self.query = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)
        self.block = SelfAttentionBlock(dim, num_heads, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """``(N, F, D) -> (N, T, D)``."""
        x = feats + self.frame_embed.to(feats.dtype)
        q = self.query.to(feats.dtype).expand(feats.shape[0], -1, -1)
        out = self.block(torch.cat([q, x], dim=1))
        return self.norm(out[:, : self.num_tokens])


class PhysicalEncoder(nn.Module):
    """Encodes state + action chunk + tactile into one embedding, tolerating missing modalities.

    Every modality is read over the same window ``[t, t+H]`` that the perception side sees,
    at a resolution set by what it costs to read, and is then grouped into
    ``G = chunk_size / group_size`` tokens:

    ==================  ============  ==========================
    modality            frames read   tokens
    ==================  ============  ==========================
    CLS                 --            1
    state               chunk         G
    action              chunk         G
    tactile signal      chunk         G
    tactile image       t, t+H        ``max_tactile_views``
    ==================  ============  ==========================

    The token budget is the main defence against tactile dominance. A tactile camera carries
    far more raw capacity per token than a 40-dim state vector, so it is folded into one token
    per pad -- ``[feat_t, feat_t1 - feat_t]`` -- rather than the 64-per-pad that a naive
    patchwise encoding would produce. The zero-initialised gates, the tactile dropout and the
    reduced tactile learning rate handle the rest.
    """

    MOD_CLS, MOD_STATE, MOD_ACTION, MOD_TAC_SIG, MOD_TAC_IMG = range(5)

    def __init__(self, config: RoboContrastConfig):
        super().__init__()
        dim = config.hidden_dim
        self.config = config
        self.chunk_size = config.chunk_size
        self.group_size = config.group_size
        self.num_groups = config.chunk_size // config.group_size
        self.action_dim = config.max_action_dim
        self.state_dim = config.max_state_dim
        self.signal_dim = config.max_tactile_signal_dim

        # Every projection consumes ``[value * mask, mask]`` so an all-zero slot is
        # distinguishable from a genuine zero measurement.
        #
        # State is grouped exactly like the action chunk and shares its positional embedding:
        # group *i* of the state trajectory and group *i* of the action chunk cover the same
        # frames, and giving them the same positional code is what lets the trunk compare
        # "commanded" against "achieved" at matching times.
        self.state_proj = nn.Linear(self.group_size * self.state_dim * 2, dim)
        self.action_proj = nn.Linear(self.group_size * self.action_dim * 2, dim)
        # Tactile images are read at the two ends of the window, so their projection sees a
        # pair of frames and keeps its original token count: the useful thing about tactile is
        # the *change* in contact, but spending more tokens on it would let it crowd out the
        # action chunk. The tactile signal is chunked like state and action.
        self.signal_proj = nn.Linear(self.group_size * self.signal_dim * 2, dim)
        self.use_ftp1_tactile = config.tactile_backbone == "ftp1"
        if self.use_ftp1_tactile:
            self.tactile_cnn = FTP1TactileTower(
                config.ftp1_tactile_dir,
                list(config.ftp1_tactile_sensors) or FTP1_SENSOR_NAMES,
                out_dim=config.tactile_feat_dim,
                img_size=config.tactile_img_size,
            )
        else:
            self.tactile_cnn = TactileImageEncoder(config.tactile_feat_dim, config.tactile_pretrained)
        self.tactile_frames = config.tactile_frames
        self.tactile_tokens_per_pad = config.tactile_tokens_per_pad
        self.tactile_temporal = TactilePadTemporal(
            config.tactile_feat_dim,
            config.tactile_frames,
            config.tactile_tokens_per_pad,
            num_heads=8,
            dropout=config.dropout,
        )
        self.tactile_img_proj = nn.Linear(config.tactile_feat_dim, dim)
        self.tactile_recon = (
            TactileReconHead(config.tactile_feat_dim, config.tactile_recon_size)
            if config.tactile_recon_weight > 0
            else None
        )

        # `num_cls_tokens` read-out tokens. They are *not* tied together: each is a separate
        # learned vector, so with K>1 they can specialise on different parts of the chunk
        # (e.g. an approach phase and a grasp) rather than being forced to average them.
        self.num_cls_tokens = config.num_cls_tokens
        self.cls_token = nn.Parameter(torch.randn(1, config.num_cls_tokens, dim) * 0.02)
        self.modality_embed = nn.Embedding(5, dim)
        self.group_pos_embed = nn.Embedding(self.num_groups, dim)
        # Which finger/pad a tactile token came from. UniVTAC has no such embedding because it
        # always sees a fixed sensor set; our datasets ship 0, 1, 4 or 6 pads.
        self.tactile_view_embed = nn.Embedding(config.max_tactile_views, dim)
        # Which of a pad's tokens this is. Without it the two tokens of a pad are distinguished
        # only by their content, and both carry the same view embedding.
        self.tactile_token_embed = nn.Embedding(config.tactile_tokens_per_pad, dim)
        self.sample_rate_embed = nn.Embedding(64, dim)
        # Learned stand-ins used when a modality is absent or dropped.
        self.missing_embed = nn.Embedding(5, dim)

        # Tactile gates start closed so training begins from a state/action-only model and
        # only opens the tactile channel if it actually reduces the contrastive loss.
        self.tactile_signal_gate = nn.Parameter(torch.zeros(1))
        self.tactile_image_gate = nn.Parameter(torch.zeros(1))

        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(dim, config.num_attention_heads, config.dropout)
                for _ in range(config.num_physical_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, config.projection_dim),
        )

    def _with_mask(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Concatenate values (already masked) with their validity mask along the last dim."""
        return torch.cat([value * mask, mask], dim=-1)

    def _maybe_drop(self, keep: torch.Tensor, p: float) -> torch.Tensor:
        if not self.training or p <= 0.0:
            return keep
        draw = torch.rand_like(keep)
        return keep * (draw >= p).to(keep.dtype)

    def forward(self, batch: dict) -> torch.Tensor:
        dtype = self.state_proj.weight.dtype
        device = self.state_proj.weight.device

        state = batch["observation.state"].to(device=device, dtype=dtype)
        state_mask = batch["state_mask"].to(device=device, dtype=dtype)
        action = batch["action"].to(device=device, dtype=dtype)
        action_mask = batch["action_mask"].to(device=device, dtype=dtype)
        signal = batch["tactile_signal"].to(device=device, dtype=dtype)
        signal_present = batch["tactile_signal_mask"].to(device=device, dtype=dtype).reshape(-1, 1)
        tac_images = batch["tactile_image"].to(device=device)
        tac_img_mask = batch["tactile_image_mask"].to(device=device, dtype=dtype)
        sample_rate = batch["sample_rate"].to(device=device).long().clamp(0, 63).reshape(-1)

        b = state.shape[0]
        mod = self.modality_embed.weight.to(dtype)
        missing = self.missing_embed.weight.to(dtype)

        # -- state ---------------------------------------------------------
        # The state trajectory, not just the state at t. The action chunk says what was
        # commanded; the state says what the arm actually did, and the visual change we align
        # against is the consequence of the latter. The two also diverge in the ways that
        # matter most here -- contact, compliance, a slipping gripper.
        keep_state = self._maybe_drop(
            (state_mask.sum(dim=-1, keepdim=True) > 0).to(dtype), self.config.modality_dropout_state
        )
        st_mask_chunk = state_mask.unsqueeze(1).expand(-1, self.chunk_size, -1)
        st_feat = self._with_mask(state, st_mask_chunk)
        st_feat = st_feat.view(b, self.num_groups, self.group_size * st_feat.shape[-1])
        state_tokens = self.state_proj(st_feat)
        state_tokens = keep_state.unsqueeze(1) * state_tokens + (
            1 - keep_state
        ).unsqueeze(1) * missing[self.MOD_STATE]
        group_ids = torch.arange(self.num_groups, device=device)
        group_pos = self.group_pos_embed(group_ids).to(dtype).unsqueeze(0)
        rate_embed = self.sample_rate_embed(sample_rate).to(dtype).unsqueeze(1)
        state_tokens = state_tokens + group_pos + mod[self.MOD_STATE] + rate_embed

        # -- action chunk --------------------------------------------------
        keep_action = self._maybe_drop(
            (action_mask.sum(dim=-1, keepdim=True) > 0).to(dtype), self.config.modality_dropout_action
        )
        act_mask_chunk = action_mask.unsqueeze(1).expand(-1, self.chunk_size, -1)
        act_feat = self._with_mask(action, act_mask_chunk)
        act_feat = act_feat.view(b, self.num_groups, self.group_size * act_feat.shape[-1])
        action_tokens = self.action_proj(act_feat)
        action_tokens = keep_action.unsqueeze(1) * action_tokens + (
            1 - keep_action
        ).unsqueeze(1) * missing[self.MOD_ACTION]
        action_tokens = action_tokens + group_pos + mod[self.MOD_ACTION] + rate_embed

        # -- tactile signal ------------------------------------------------
        keep_signal = self._maybe_drop(signal_present, self.config.modality_dropout_tactile)
        sig_mask = signal_present.unsqueeze(1).expand(-1, self.chunk_size, self.signal_dim)
        sig_feat = self._with_mask(signal, sig_mask)
        sig_feat = sig_feat.view(b, self.num_groups, self.group_size * sig_feat.shape[-1])
        signal_tokens = self.signal_proj(sig_feat)
        signal_tokens = torch.tanh(self.tactile_signal_gate).to(dtype) * signal_tokens
        signal_tokens = keep_signal.unsqueeze(1) * signal_tokens + (
            1 - keep_signal
        ).unsqueeze(1) * missing[self.MOD_TAC_SIG]
        signal_tokens = signal_tokens + group_pos + mod[self.MOD_TAC_SIG] + rate_embed

        # -- tactile images ------------------------------------------------
        # One token per pad rather than a single pooled token: pads touch different parts of
        # the object and averaging them destroys exactly the contact pattern we want. The zero
        # initialised gate, the modality dropout and the reduced tactile learning rate are what
        # stop these extra tokens from taking over.
        #
        # Each pad arrives as ``tactile_frames`` samples spread across the window, fused inside
        # the pad by ``TactilePadTemporal`` into ``tactile_tokens_per_pad`` tokens. The window
        # interior is the point: a grasp that closes and settles mid-window is invisible at both
        # endpoints, and doc/results.md §21 measured that what the endpoints miss is structured
        # deformation rather than noise. Two tokens per pad give the transformer separate access
        # to the contact's state and its evolution, which a single projected concatenation mixes
        # before attention can see it -- at the price of raising the tactile image share of the
        # sequence from 19% to 32%, which is why ``tactile_tokens_per_pad`` is a knob.
        num_views = tac_images.shape[1]
        any_view = (tac_img_mask.sum(dim=-1, keepdim=True) > 0).to(dtype)
        keep_tac_img = self._maybe_drop(any_view, self.config.modality_dropout_tactile)
        recon_loss = None
        flat_valid = tac_img_mask.reshape(-1) > 0
        flat_images = tac_images.reshape(b * num_views, *tac_images.shape[2:])
        # Only the pads that actually exist are pushed through the CNN: most batches contain a
        # handful of tactile samples among 256, so encoding the zero-filled placeholders would
        # waste almost all of the ResNet's compute (and pollute the reconstruction target).
        #
        # The selection must never become *empty*, though. Under ZeRO-2 the set of parameters
        # that receive a gradient determines the gradient-reduction schedule, so a rank whose
        # batch happens to contain no tactile data would skip the tactile parameters and desync
        # from its peers -- which shows up as an NCCL collective timeout rather than an error.
        # Keeping one dummy row and zeroing its contribution makes the schedule data independent.
        selected = flat_valid.nonzero(as_tuple=True)[0]
        has_tactile = selected.numel() > 0
        if not has_tactile:
            selected = torch.zeros(1, dtype=torch.long, device=device)
        sel_images = flat_images[selected]
        # ``sel_images`` is (N, F, 3, H, W); the encoder treats the frame axis as its view axis.
        if self.use_ftp1_tactile:
            # The FTP-1 tower has one tokenizer per physical sensor, so it additionally needs
            # to know which sensor each selected pad came from and the per-dataset z-score
            # that sensor's weights were calibrated against.
            sensor_ids = batch["tactile_sensor_id"].to(device).reshape(-1)[selected]
            sel_mean = batch["tactile_img_mean"].to(device).reshape(-1, 3)[selected]
            sel_std = batch["tactile_img_std"].to(device).reshape(-1, 3)[selected]
            # The tower is frozen; running it without building a graph saves the activations
            # of 12 transformer layers over 197 tokens per pad-frame.
            with torch.no_grad():
                sel_pair = self.tactile_cnn(sel_images, sensor_ids, sel_mean, sel_std)
            sel_pair = sel_pair.to(dtype)
        else:
            # Same `.to(dtype)` as the FTP-1 branch above: the encoder's output dtype follows
            # its own head, which need not match the trunk that consumes it.
            sel_pair = self.tactile_cnn(sel_images)[0].to(dtype)
        sel_feats = sel_pair[:, 0]
        if not has_tactile:
            sel_pair = sel_pair * 0.0
            sel_feats = sel_feats * 0.0
        # ``(N, F, D) -> (N, T, D)``: the window's frames are fused inside the pad, so a pad
        # stays one maskable unit and the tokens that leave describe the contact rather than
        # individual frames.
        sel_tokens = self.tactile_temporal(sel_pair)

        tok_per_pad, feat_dim = self.tactile_tokens_per_pad, sel_tokens.shape[-1]
        view_feats = torch.zeros(
            b * num_views, tok_per_pad, feat_dim, device=device, dtype=sel_tokens.dtype
        )
        view_feats = view_feats.index_put((selected,), sel_tokens)
        img_tokens = self.tactile_img_proj(view_feats.view(b, num_views * tok_per_pad, feat_dim))
        if self.tactile_recon is not None and self.training:
            # Reconstruct the frame at ``t`` only -- the head exists to shape the features, and
            # running it over all ``tactile_frames`` would multiply its cost by that factor for
            # supervision of the same kind. ``sel_feats`` is frame 0's feature, taken before the
            # temporal fusion, so the pixel target ``sel_images[:, 0]`` is the matching frame.
            # The dataset ships FTP-1's statistics in its [-1, 1] convention
            # (``x/255*2-1``); the target here is in [0, 1], so shift them across. A dataset
            # with no registry entry gets the identity default (mean 0, std 1 in [-1, 1]),
            # which maps to 0.5/0.5 here -- not unit variance, but a graceful ~4x rather than
            # the 80x the correct statistics give.
            sel_mean = batch["tactile_img_mean"].to(device).reshape(-1, 3)[selected]
            sel_std = batch["tactile_img_std"].to(device).reshape(-1, 3)[selected]
            recon_loss = self._tactile_recon_loss(
                sel_feats,
                sel_images[:, 0],
                (sel_mean.float() + 1.0) * 0.5,
                (sel_std.float() * 0.5).clamp_min(1e-3),
            )
            if not has_tactile:
                recon_loss = recon_loss * 0.0

        img_tokens = torch.tanh(self.tactile_image_gate).to(dtype) * img_tokens
        # A pad that this dataset does not have is replaced by the learned "missing" token, so
        # a 4-pad dataset and a 0-pad dataset produce sequences of the same shape. The mask is
        # per pad, so it repeats across that pad's tokens.
        view_keep = (keep_tac_img * tac_img_mask).repeat_interleave(tok_per_pad, dim=1).unsqueeze(-1)
        img_tokens = view_keep * img_tokens + (1 - view_keep) * missing[self.MOD_TAC_IMG]
        view_ids = torch.arange(num_views, device=device).repeat_interleave(tok_per_pad)
        token_ids = torch.arange(tok_per_pad, device=device).repeat(num_views)
        img_tokens = (
            img_tokens
            + self.tactile_view_embed(view_ids).to(dtype).unsqueeze(0)
            + self.tactile_token_embed(token_ids).to(dtype).unsqueeze(0)
            + mod[self.MOD_TAC_IMG]
        )

        # -- transformer ---------------------------------------------------
        # K CLS + G state + G action + G tactile signal + V*T tactile image tokens, where
        # G = chunk_size / group_size, V = max_tactile_views, T = tactile_tokens_per_pad.
        cls = (self.cls_token.to(dtype).expand(b, -1, -1) + mod[self.MOD_CLS])
        tokens = torch.cat([cls, state_tokens, action_tokens, signal_tokens, img_tokens], dim=1)
        for block in self.blocks:
            tokens = block(tokens)

        summary = self.out_proj(self.out_norm(tokens[:, : self.num_cls_tokens]))
        if self.num_cls_tokens == 1:
            summary = summary.squeeze(1)
        return summary, recon_loss

    def _tactile_recon_loss(self, valid_feats, valid_images, mean, std) -> torch.Tensor:
        """MSE between the decoded and the true tactile image (UniVTAC's `rgb` head).

        ``mean``/``std`` are the per-dataset per-channel pixel statistics, in [0, 1] space,
        for each selected pad. Z-scoring the target is not cosmetic: gel images occupy a very
        narrow slice of [0, 1] (measured per-channel pixel std 0.079-0.125 across our tactile
        datasets), so a raw-[0, 1] MSE bottoms out around 0.003-0.015 -- the value a decoder
        reaches by emitting one fixed image. At ``tactile_recon_weight=0.1`` that contributes
        ~1e-3 against a contrastive loss of ~7.6, i.e. nothing. Dividing by the per-dataset
        std brings the target to unit variance and the objective back into a range where its
        gradient competes; see ``scripts/tactile_recon_floor.py``, which measures both.

        Statistics are per dataset rather than global on purpose. A single pooled tactile
        z-score is numerically almost identical to ImageNet's (measured pooled std
        0.240/0.228/0.222 vs 0.229/0.224/0.225) because most of the pooled variance is
        *between* datasets, so it would leave sharpa's target at 0.23 variance and
        neo_aloha's at 0.55 instead of 1.
        """
        pred = self.tactile_recon(valid_feats).float()
        size = self.config.tactile_recon_size
        target = valid_images.float() / 255.0
        target = F.interpolate(target, size=(size, size), mode="bilinear", align_corners=False)
        # Downsampling is linear and the z-score is a per-channel affine map, so the two
        # commute; normalising afterwards just touches fewer pixels.
        target = (target - mean.view(-1, 3, 1, 1)) / std.view(-1, 3, 1, 1)
        return F.mse_loss(pred, target)


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------
class RoboContrast(PreTrainedPolicy):
    config_class = RoboContrastConfig
    name = "robo_contrast"

    def __init__(self, config: RoboContrastConfig, dataset_stats=None):
        super().__init__(config)
        self.config = config
        config.validate_features()

        self.perception_encoder = PerceptionEncoder(config)
        # Stage 1 never touches the physical branch; not building it saves both its parameters
        # and the ZeRO bookkeeping for a module that would receive no gradient on any rank.
        self.physical_encoder = None if config.perception_only else PhysicalEncoder(config)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / config.temperature)))
        self._max_logit_scale = math.log(config.logit_scale_max)

    # -- PreTrainedPolicy API ---------------------------------------------
    def get_optim_params(self):
        """Two groups: the tactile CNN gets a reduced learning rate.

        UniVTAC gives its tactile backbone its own ``lr_tactile_backbone`` group. Here the
        motivation is sharper: the ResNet-18 arrives ImageNet-pretrained and is by far the
        highest-capacity-per-token module on the physical side, so at a shared learning rate
        it converges first and the contrastive loss learns to read tactile and ignore the
        action chunk.

        With ``tactile_backbone="ftp1"`` the tower is frozen, so the ``requires_grad`` filter
        below leaves only its output LayerNorm in this group -- the reduced learning rate then
        just makes the model cautious about rescaling pretrained features.
        """
        tactile_params, other_params = [], []
        tactile_prefix = ("physical_encoder.tactile_cnn.", "physical_encoder.tactile_recon.")
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            (tactile_params if name.startswith(tactile_prefix) else other_params).append(param)
        groups = [{"params": other_params}]
        if tactile_params:
            # LambdaLR captures each group's `lr` as its `initial_lr` and scales it by the same
            # schedule, so a per-group lr survives warmup and cosine decay.
            groups.append(
                {"params": tactile_params, "lr": self.config.optimizer_lr * self.config.tactile_lr_scale}
            )
        return groups

    def reset(self):
        self._queues = {"action": deque(maxlen=self.config.n_action_steps)}

    def select_action(self, batch):
        raise NotImplementedError("RoboContrast is a representation-learning model, not a controller.")

    # -- embeddings --------------------------------------------------------
    def encode_perception(self, batch, probe: bool = False):
        emb, recon_loss, aux = self.perception_encoder(
            batch["image_t0"], batch["image_t1"], batch["task"], batch.get("has_text"), probe=probe
        )
        return F.normalize(emb.float(), dim=-1), recon_loss, aux

    def encode_physical(self, batch) -> tuple[torch.Tensor, torch.Tensor | None]:
        emb, recon_loss = self.physical_encoder(batch)
        return F.normalize(emb.float(), dim=-1), recon_loss

    # -- loss --------------------------------------------------------------
    def _false_negative_mask(self, episode_uid, frame_index, all_episode_uid, all_frame_index):
        """``True`` where a candidate must be excluded from the denominator.

        Two frames of the same episode that are only a few steps apart describe almost the
        same motion, so treating them as negatives would push apart embeddings that *should*
        match. The batch sampler already keeps same-episode frames far apart; this is the
        safety net for the residual cases.
        """
        same_episode = episode_uid[:, None] == all_episode_uid[None, :]
        close = (frame_index[:, None] - all_frame_index[None, :]).abs() < self.config.false_negative_frame_gap
        return same_episode & close

    def forward(self, batch, task_type: str = "train_contrastive", step: int = 0):
        if task_type == "train_perception":
            return self._perception_forward(batch, step)
        perception, percep_recon, percep_aux = self.encode_perception(
            batch, probe=self._should_probe(step)
        )
        physical, tactile_recon = self.encode_physical(batch)

        rank, world_size = _rank_world()
        local_bs = perception.shape[0]

        all_perception = _all_gather_detached(perception)
        all_physical = _all_gather_detached(physical)

        device = perception.device
        episode_uid = batch["episode_uid"].to(device).long().reshape(-1)
        frame_index = batch["frame_index"].to(device).long().reshape(-1)
        all_episode_uid = _all_gather_detached(episode_uid)
        all_frame_index = _all_gather_detached(frame_index)

        logit_scale = self.logit_scale.clamp(max=self._max_logit_scale).exp().float()
        logits_p2r = logit_scale * pairwise_similarity(perception, all_physical)
        logits_r2p = logit_scale * pairwise_similarity(physical, all_perception)

        labels = torch.arange(local_bs, device=device) + rank * local_bs
        invalid = self._false_negative_mask(episode_uid, frame_index, all_episode_uid, all_frame_index)
        invalid[torch.arange(local_bs, device=device), labels] = False
        logits_p2r = logits_p2r.masked_fill(invalid, float("-inf"))
        logits_r2p = logits_r2p.masked_fill(invalid, float("-inf"))

        loss_p2r = F.cross_entropy(logits_p2r, labels)
        loss_r2p = F.cross_entropy(logits_r2p, labels)
        contrastive = 0.5 * (loss_p2r + loss_r2p)

        loss = contrastive
        recon_value = 0.0
        percep_recon_value = 0.0
        if tactile_recon is not None and self.config.tactile_recon_weight > 0:
            loss = loss + self.config.tactile_recon_weight * tactile_recon
            recon_value = tactile_recon.item()
        if percep_recon is not None and self.config.perception_recon_weight > 0:
            loss = loss + self.config.perception_recon_weight * percep_recon
            percep_recon_value = percep_recon.item()

        with torch.no_grad():
            correct = (logits_p2r.argmax(dim=-1) == labels).float()
            acc = correct.mean()
            pos_sim = paired_similarity(perception, physical).mean()
            # Retrieval accuracy restricted to the rows that actually carry tactile.
            # The tactile datasets are only ~2.7% of this mixture, so the aggregate accuracy
            # is nearly blind to anything the tactile path does: a change that helped tactile
            # enormously would still move `retrieval_acc` by a couple of points at most. This
            # is the metric to read when judging tactile work.
            has_tac = (
                (batch["tactile_image_mask"].to(device).sum(dim=-1) > 0)
                | (batch["tactile_signal_mask"].to(device).reshape(-1) > 0)
            ).float()
            n_tac = has_tac.sum()
        loss_dict = {
            "contrastive_loss": contrastive.item(),
            "recon_loss": recon_value,
            "percep_recon_loss": percep_recon_value,
            "retrieval_acc": acc.item(),
            # Reported as a hit count and a row count rather than a ratio: most batches contain
            # no tactile at all, and averaging a per-step ratio over those would fold in a
            # meaningless zero. Summing both over a window and dividing gives the true
            # conditional accuracy.
            "tactile_hits": (correct * has_tac).sum().item(),
            "tactile_rows": n_tac.item(),
            "pos_sim": pos_sim.item(),
            "logit_scale": logit_scale.item(),
            "tactile_sig_gate": torch.tanh(self.physical_encoder.tactile_signal_gate).item(),
            "tactile_img_gate": torch.tanh(self.physical_encoder.tactile_image_gate).item(),
        }
        loss_dict.update(percep_aux)
        return loss, loss_dict

    def _should_probe(self, step: int) -> bool:
        freq = self.config.query_probe_freq
        return freq > 0 and step % freq == 0

    def _perception_forward(self, batch, step: int):
        """Stage-1: train the perception branch on its reconstruction objective alone.

        No physical modality is touched, so this runs on plain video. The loss is exactly the
        term the contrastive stage already carries as ``percep_recon_loss``, at weight 1.0 --
        the point of the stage is to spend the whole budget on it while action-bearing data is
        scarce, then hand the weights over.

        ``percep_query_gain`` is the metric that decides whether the stage worked. It is
        probed on a schedule rather than every step because it costs a second predictor pass.
        """
        _, recon, aux = self.encode_perception(batch, probe=self._should_probe(step))
        if recon is None:
            raise RuntimeError(
                "Perception pre-training needs the reconstruction head. Set "
                "perception_recon_weight > 0 (it builds the predictor) and keep the policy in "
                "train mode."
            )
        loss_dict = {"percep_recon_loss": recon.item()}
        loss_dict.update(aux)
        return recon, loss_dict
