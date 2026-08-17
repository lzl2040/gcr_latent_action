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

import math
import os
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
from torch.utils.checkpoint import checkpoint

from lerobot.common.policies.ace.configuration_robo_contrast import RoboContrastConfig
from lerobot.common.policies.pretrained import PreTrainedPolicy

# DINOv3 is normalised with ImageNet statistics (see its preprocessor_config.json), unlike
# SigLIP2 which uses 0.5/0.5.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


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

        self.vision_backbone = AutoModel.from_pretrained(vision_name, dtype=torch.float32)
        # Only SigLIP2's *text* tower is kept; dropping its vision tower saves 93M frozen
        # parameters that DINOv3 now replaces.
        text_full = AutoModel.from_pretrained(text_name, dtype=torch.float32)
        self.text_backbone = text_full.text_model
        del text_full
        self.tokenizer = AutoTokenizer.from_pretrained(text_name)
        self.text_max_length = config.text_max_length

        vision_dim = self.vision_backbone.config.hidden_size
        text_dim = self.text_backbone.config.hidden_size
        self.vision_dim = vision_dim
        # DINOv3 prepends one CLS token and `num_register_tokens` register tokens; only the
        # patch tokens are spatially meaningful, so the prefix is dropped.
        self.num_prefix_tokens = 1 + int(getattr(self.vision_backbone.config, "num_register_tokens", 0))
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

        self.recon_weight = config.perception_recon_weight
        self.predictor = (
            ChangePredictor(
                dim,
                vision_dim,
                config.num_predictor_layers,
                config.fusion_num_heads,
                config.dropout,
                use_checkpointing=config.gradient_checkpointing,
            )
            if config.num_predictor_layers > 0 and config.perception_recon_weight > 0
            else None
        )
        # Targets are normalised per token before the loss (as in I-JEPA), so the objective is
        # about the *pattern* of the feature vector rather than its magnitude, which otherwise
        # dominates the L1 and is trivially predictable from frame t.
        self.target_norm = nn.LayerNorm(vision_dim, elementwise_affine=False)

    # -- raw input handling -------------------------------------------------
    @staticmethod
    def _to_pixel_values(images: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """uint8 ``(B, 3, H, W)`` in ``[0, 255]`` -> ImageNet-normalised float tensor.

        Doing this on-device replaces the previous PIL round-trip, which was the single most
        expensive step of the old training loop.
        """
        x = images.to(dtype=torch.float32)
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        mean = IMAGENET_MEAN.to(device=x.device, dtype=x.dtype)
        std = IMAGENET_STD.to(device=x.device, dtype=x.dtype)
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

    def forward(self, image_t0, image_t1, texts) -> torch.Tensor:
        device = self.visual_proj.weight.device
        dtype = self.visual_proj.weight.dtype
        batch = image_t0.shape[0]

        pixels = torch.cat(
            [self._to_pixel_values(image_t0, dtype), self._to_pixel_values(image_t1, dtype)], dim=0
        )
        patches = self._encode_vision(pixels).to(dtype)
        p0, p1 = patches[:batch], patches[batch:]

        input_ids, text_mask = self.tokenize(texts, device)
        text_tokens = self.text_proj(self.text_norm(self._encode_text(input_ids).to(dtype)))

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
        queries = queries + text_pooled.unsqueeze(1)

        for block in self.blocks:
            queries = block(queries, evidence, evidence_mask)

        queries = self.out_norm(queries)
        embedding = self.out_proj(queries.mean(dim=1))

        recon_loss = None
        if self.predictor is not None and self.training:
            recon_loss = self._recon_loss(v0, queries, text_tokens, text_mask, p1)
        return embedding, recon_loss

    def _recon_loss(self, v0, queries, text_tokens, text_mask, p1) -> torch.Tensor:
        """Predict the frame-``t+H`` patch features and score them against the real ones."""
        memory = torch.cat([queries, text_tokens], dim=1)
        if text_mask is not None:
            query_mask = torch.ones(
                queries.shape[0], queries.shape[1], device=queries.device, dtype=torch.bool
            )
            memory_mask = torch.cat([query_mask, text_mask.to(torch.bool)], dim=1)
        else:
            memory_mask = None

        pred = self.predictor(v0, memory, memory_mask).float()
        # The target comes from the frozen backbone and is detached: there is no trainable
        # path into it, so the pair cannot collapse onto a constant the way a jointly trained
        # student/teacher would.
        target = self.target_norm(p1.detach().float())
        return F.smooth_l1_loss(pred, target)


# ---------------------------------------------------------------------------
# physical side
# ---------------------------------------------------------------------------
def _freeze_batchnorm(module: nn.Module) -> nn.Module:
    """Replace every ``BatchNorm2d`` with a ``FrozenBatchNorm2d`` carrying the same statistics."""
    import torchvision

    if isinstance(module, nn.BatchNorm2d):
        frozen = torchvision.ops.misc.FrozenBatchNorm2d(module.num_features, eps=module.eps)
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
        dtype = self.proj.weight.dtype if isinstance(self.proj, nn.Linear) else self.norm.weight.dtype
        x = images.reshape(b * v, *images.shape[2:]).to(dtype=torch.float32) / 255.0
        x = ((x - self.mean) / self.std).to(dtype)
        feat_map = self.layers(self.stem(x))
        pooled = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)
        emb = self.norm(self.proj(pooled))
        return emb.view(b, v, -1), feat_map


class TactileReconHead(nn.Module):
    """Reconstructs the tactile image from its embedding.

    UniVTAC pretrains its encoder with MSE reconstruction of the gel image plus marker
    positions, depth and contact pose. Those three extra targets only exist in simulation, so
    for real sensors we keep the one head whose supervision is always available. The point is
    not the reconstruction itself but that the tactile features are shaped by an objective of
    their own instead of being dragged around by the contrastive gradient.
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


class PhysicalEncoder(nn.Module):
    """Encodes state + action chunk + tactile into one embedding, tolerating missing modalities."""

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
        self.state_proj = nn.Linear(self.state_dim * 2, dim)
        self.action_proj = nn.Linear(self.group_size * self.action_dim * 2, dim)
        self.signal_proj = nn.Linear(self.signal_dim * 2, dim)
        self.tactile_cnn = TactileImageEncoder(config.tactile_feat_dim, config.tactile_pretrained)
        self.tactile_img_proj = nn.Linear(config.tactile_feat_dim, dim)
        self.tactile_recon = (
            TactileReconHead(config.tactile_feat_dim, config.tactile_recon_size)
            if config.tactile_recon_weight > 0
            else None
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.modality_embed = nn.Embedding(5, dim)
        self.group_pos_embed = nn.Embedding(self.num_groups, dim)
        # Which finger/pad a tactile token came from. UniVTAC has no such embedding because it
        # always sees a fixed sensor set; our datasets ship 0, 1, 4 or 6 pads.
        self.tactile_view_embed = nn.Embedding(config.max_tactile_views, dim)
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
        keep_state = self._maybe_drop(
            (state_mask.sum(dim=-1, keepdim=True) > 0).to(dtype), self.config.modality_dropout_state
        )
        state_token = self.state_proj(self._with_mask(state, state_mask))
        state_token = keep_state * state_token + (1 - keep_state) * missing[self.MOD_STATE]
        state_token = (state_token + mod[self.MOD_STATE]).unsqueeze(1)

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
        group_ids = torch.arange(self.num_groups, device=device)
        action_tokens = (
            action_tokens
            + self.group_pos_embed(group_ids).to(dtype).unsqueeze(0)
            + mod[self.MOD_ACTION]
            + self.sample_rate_embed(sample_rate).to(dtype).unsqueeze(1)
        )

        # -- tactile signal ------------------------------------------------
        keep_signal = self._maybe_drop(signal_present, self.config.modality_dropout_tactile)
        signal_token = self.signal_proj(
            self._with_mask(signal, signal_present.expand(-1, self.signal_dim))
        )
        signal_token = torch.tanh(self.tactile_signal_gate).to(dtype) * signal_token
        signal_token = keep_signal * signal_token + (1 - keep_signal) * missing[self.MOD_TAC_SIG]
        signal_token = (signal_token + mod[self.MOD_TAC_SIG]).unsqueeze(1)

        # -- tactile images ------------------------------------------------
        # One token per pad rather than a single pooled token: pads touch different parts of
        # the object and averaging them destroys exactly the contact pattern we want. The zero
        # initialised gate, the modality dropout and the reduced tactile learning rate are what
        # stop these extra tokens from taking over.
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
        sel_feats = self.tactile_cnn(sel_images.unsqueeze(1))[0].squeeze(1)
        if not has_tactile:
            sel_feats = sel_feats * 0.0

        feat_dim = sel_feats.shape[-1]
        view_feats = torch.zeros(b * num_views, feat_dim, device=device, dtype=sel_feats.dtype)
        view_feats = view_feats.index_put((selected,), sel_feats)
        img_tokens = self.tactile_img_proj(view_feats.view(b, num_views, feat_dim))
        if self.tactile_recon is not None and self.training:
            recon_loss = self._tactile_recon_loss(sel_feats, sel_images)
            if not has_tactile:
                recon_loss = recon_loss * 0.0

        img_tokens = torch.tanh(self.tactile_image_gate).to(dtype) * img_tokens
        # A pad that this dataset does not have is replaced by the learned "missing" token, so
        # a 4-pad dataset and a 0-pad dataset produce sequences of the same shape.
        view_keep = (keep_tac_img * tac_img_mask).unsqueeze(-1)
        img_tokens = view_keep * img_tokens + (1 - view_keep) * missing[self.MOD_TAC_IMG]
        view_ids = torch.arange(num_views, device=device)
        img_tokens = (
            img_tokens + self.tactile_view_embed(view_ids).to(dtype).unsqueeze(0) + mod[self.MOD_TAC_IMG]
        )

        # -- transformer ---------------------------------------------------
        cls = (self.cls_token.to(dtype).expand(b, -1, -1) + mod[self.MOD_CLS])
        tokens = torch.cat([cls, state_token, action_tokens, signal_token, img_tokens], dim=1)
        for block in self.blocks:
            tokens = block(tokens)

        return self.out_proj(self.out_norm(tokens[:, 0])), recon_loss

    def _tactile_recon_loss(self, valid_feats, valid_images) -> torch.Tensor:
        """MSE between the decoded and the true tactile image (UniVTAC's `rgb` head)."""
        pred = self.tactile_recon(valid_feats).float()
        size = self.config.tactile_recon_size
        target = valid_images.float() / 255.0
        target = F.interpolate(target, size=(size, size), mode="bilinear", align_corners=False)
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
        self.physical_encoder = PhysicalEncoder(config)
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
    def encode_perception(self, batch) -> tuple[torch.Tensor, torch.Tensor | None]:
        emb, recon_loss = self.perception_encoder(
            batch["image_t0"], batch["image_t1"], batch["task"]
        )
        return F.normalize(emb.float(), dim=-1), recon_loss

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
        perception, percep_recon = self.encode_perception(batch)
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
        logits_p2r = logit_scale * perception @ all_physical.t()
        logits_r2p = logit_scale * physical @ all_perception.t()

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
            acc = (logits_p2r.argmax(dim=-1) == labels).float().mean()
            pos_sim = (perception * physical).sum(-1).mean()
        loss_dict = {
            "contrastive_loss": contrastive.item(),
            "recon_loss": recon_value,
            "percep_recon_loss": percep_recon_value,
            "retrieval_acc": acc.item(),
            "pos_sim": pos_sim.item(),
            "logit_scale": logit_scale.item(),
            "tactile_sig_gate": torch.tanh(self.physical_encoder.tactile_signal_gate).item(),
            "tactile_img_gate": torch.tanh(self.physical_encoder.tactile_image_gate).item(),
        }
        return loss, loss_dict
