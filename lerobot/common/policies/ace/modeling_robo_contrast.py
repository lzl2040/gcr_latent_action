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

from lerobot.common.policies.ace.configuration_robo_contrast import RoboContrastConfig
from lerobot.common.policies.pretrained import PreTrainedPolicy

SIGLIP_MEAN = 0.5
SIGLIP_STD = 0.5


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


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.ffn = FeedForward(dim, dropout=dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(x, None, key_padding_mask)
        return x + self.ffn(x)


class ChangeQueryBlock(nn.Module):
    """One round of: read the instruction, then read the visual evidence."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.text_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.visual_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.ffn = FeedForward(dim, dropout=dropout)

    def forward(self, queries, text_tokens, visual_tokens, text_mask=None):
        queries = queries + self.text_attn(queries, text_tokens, text_mask)
        queries = queries + self.visual_attn(queries, visual_tokens)
        return queries + self.ffn(queries)


# ---------------------------------------------------------------------------
# perception side
# ---------------------------------------------------------------------------
class PerceptionEncoder(nn.Module):
    """Text-conditioned extractor of the visual change between ``t`` and ``t + H``."""

    def __init__(self, config: RoboContrastConfig):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        model_name = config.vision_model_name
        if not os.path.exists(model_name):
            model_name = "google/siglip2-base-patch16-224"
        self.backbone = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_max_length = config.text_max_length

        vision_dim = self.backbone.config.vision_config.hidden_size
        text_dim = self.backbone.config.text_config.hidden_size
        dim = config.hidden_dim
        self.patch_stride = max(1, config.patch_token_stride)

        self.freeze_vision = config.freeze_vision_encoder
        self.freeze_text = config.freeze_text_encoder
        if self.freeze_vision:
            for p in self.backbone.vision_model.parameters():
                p.requires_grad = False
        if self.freeze_text:
            for p in self.backbone.text_model.parameters():
                p.requires_grad = False

        self.visual_proj = nn.Linear(vision_dim, dim)
        self.text_proj = nn.Linear(text_dim, dim)
        # 0 = frame t, 1 = frame t+H, 2 = their difference
        self.evidence_type_embed = nn.Embedding(3, dim)

        self.change_queries = nn.Parameter(torch.randn(config.num_change_queries, dim) * 0.02)
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

    # -- raw input handling -------------------------------------------------
    @staticmethod
    def _to_pixel_values(images: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """uint8 ``(B, 3, H, W)`` in ``[0, 255]`` -> SigLIP-normalised float tensor.

        Doing this on-device replaces the previous PIL round-trip, which was the single most
        expensive step of the old training loop.
        """
        x = images.to(dtype=torch.float32)
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x / 255.0 - SIGLIP_MEAN) / SIGLIP_STD
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
            out = self.backbone.vision_model(pixel_values=pixel_values)
        tokens = out.last_hidden_state
        if self.patch_stride > 1:
            tokens = tokens[:, :: self.patch_stride, :]
        return tokens.detach() if self.freeze_vision else tokens

    def _encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        ctx = torch.no_grad() if self.freeze_text else torch.enable_grad()
        with ctx:
            out = self.backbone.text_model(input_ids)
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
        text_tokens = self.text_proj(self._encode_text(input_ids).to(dtype))

        v0 = self.visual_proj(p0)
        v1 = self.visual_proj(p1)
        diff = v1 - v0

        type_ids = torch.arange(3, device=device)
        type_emb = self.evidence_type_embed(type_ids).to(dtype)
        evidence = torch.cat(
            [v0 + type_emb[0], v1 + type_emb[1], diff + type_emb[2]], dim=1
        )  # (B, 3N, D)

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
            queries = block(queries, text_tokens, evidence, text_mask)

        pooled = self.out_norm(queries).mean(dim=1)
        return self.out_proj(pooled)


# ---------------------------------------------------------------------------
# physical side
# ---------------------------------------------------------------------------
class TactileImageEncoder(nn.Module):
    """Deliberately small CNN: tactile images must not out-capacity state and action."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, out_dim, 3, 2, 1),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """``(B, V, 3, H, W)`` uint8 -> ``(B, V, out_dim)``."""
        b, v = images.shape[:2]
        x = images.reshape(b * v, *images.shape[2:]).to(dtype=self.net[0].weight.dtype)
        x = x / 127.5 - 1.0
        x = self.net(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.norm(x).view(b, v, -1)


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
        self.tactile_cnn = TactileImageEncoder(config.tactile_feat_dim)
        self.tactile_img_proj = nn.Linear(config.tactile_feat_dim, dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.modality_embed = nn.Embedding(5, dim)
        self.group_pos_embed = nn.Embedding(self.num_groups, dim)
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
        any_view = (tac_img_mask.sum(dim=-1, keepdim=True) > 0).to(dtype)
        keep_tac_img = self._maybe_drop(any_view, self.config.modality_dropout_tactile)
        if bool(any_view.any()):
            view_feats = self.tactile_cnn(tac_images)
            denom = tac_img_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            pooled = (view_feats * tac_img_mask.unsqueeze(-1)).sum(dim=1) / denom
            img_token = self.tactile_img_proj(pooled)
        else:
            img_token = torch.zeros(b, self.config.hidden_dim, device=device, dtype=dtype)
        img_token = torch.tanh(self.tactile_image_gate).to(dtype) * img_token
        img_token = keep_tac_img * img_token + (1 - keep_tac_img) * missing[self.MOD_TAC_IMG]
        img_token = (img_token + mod[self.MOD_TAC_IMG]).unsqueeze(1)

        # -- transformer ---------------------------------------------------
        cls = (self.cls_token.to(dtype).expand(b, -1, -1) + mod[self.MOD_CLS])
        tokens = torch.cat([cls, state_token, action_tokens, signal_token, img_token], dim=1)
        for block in self.blocks:
            tokens = block(tokens)

        return self.out_proj(self.out_norm(tokens[:, 0]))


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
        return [p for p in self.parameters() if p.requires_grad]

    def reset(self):
        self._queues = {"action": deque(maxlen=self.config.n_action_steps)}

    def select_action(self, batch):
        raise NotImplementedError("RoboContrast is a representation-learning model, not a controller.")

    # -- embeddings --------------------------------------------------------
    def encode_perception(self, batch) -> torch.Tensor:
        emb = self.perception_encoder(batch["image_t0"], batch["image_t1"], batch["task"])
        return F.normalize(emb.float(), dim=-1)

    def encode_physical(self, batch) -> torch.Tensor:
        emb = self.physical_encoder(batch)
        return F.normalize(emb.float(), dim=-1)

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
        perception = self.encode_perception(batch)
        physical = self.encode_physical(batch)

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
        loss = 0.5 * (loss_p2r + loss_r2p)

        with torch.no_grad():
            acc = (logits_p2r.argmax(dim=-1) == labels).float().mean()
            pos_sim = (perception * physical).sum(-1).mean()
        loss_dict = {
            "contrastive_loss": loss.item(),
            "recon_loss": 0.0,
            "retrieval_acc": acc.item(),
            "pos_sim": pos_sim.item(),
            "logit_scale": logit_scale.item(),
            "tactile_sig_gate": torch.tanh(self.physical_encoder.tactile_signal_gate).item(),
            "tactile_img_gate": torch.tanh(self.physical_encoder.tactile_image_gate).item(),
        }
        return loss, loss_dict
