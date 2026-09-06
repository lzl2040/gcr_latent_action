"""Load Qwen3-VL's vision tower as a plain patch-feature extractor.

Qwen3-VL's ViT is not interchangeable with a SigLIP-style tower, and three of the
differences are silent if you get them wrong:

1. **The forward interface is not `pixel_values`.** It takes already-flattened patches of
   shape ``(seq_len, C * temporal_patch_size * patch_size**2)`` plus a ``grid_thw`` table,
   and runs variable-length attention over ``cu_seqlens``. We patchify ourselves.

2. **Patches are ordered by 2x2 merge block, not row-major.** Qwen's processor emits
   ``(grid_t, gh/m, gw/m, m, m, C, T, p, p)``, so token *i* is not image location *i*.
   Feeding that order straight into this repo's evidence bank would break the one property
   the difference stream depends on -- that ``v1[i] - v0[i]`` is "what changed at location
   i" -- and would misalign the VAE reconstruction target. We reorder back to row-major.

3. **The tower ends in a merger that pools 2x2 patches**, turning 256 tokens into 64 and
   1024 channels into ``out_hidden_size``. That destroys both the spatial grid and the
   dimensionality this model wants, so only the merger's *norm* is kept -- which is the
   tower's own trained output LayerNorm, applied per token at ``hidden_size`` before the
   pooling reshape, so keeping it costs nothing and drops nothing.

DeepStack is disabled. Qwen3-VL taps intermediate layers and injects them into the first
three LLM layers; with no LLM there is nowhere to inject them, and the three merger MLPs
that produce them are ~82M frozen parameters that would never run. The trunk's final
hidden state remains a perfectly ordinary ViT feature map. Note this does mean the
features are not everything Qwen's pretraining optimised the tower to provide.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn

# Qwen3-VL's processor scales images to at least `shortest_edge` = 65536 *pixels*, i.e.
# 256x256. A 224x224 frame (50176 pixels) is below that, so it would be upscaled anyway;
# feeding 256 directly keeps the input in distribution and makes the 16x16 patch grid match
# the Wan VAE's 16x16 latent grid exactly.
QWEN3VL_IMAGE_SIZE = 256


class _NormOnlyMerger(nn.Module):
    """Keep the pretrained output norm while removing spatial pooling and width projection.

    The stock merger is an adapter for the Qwen language model: it groups each 2x2 patch
    block and projects the concatenated feature to the LLM width. This repository instead
    compares frame pairs token by token and reconstructs a 16x16 target grid, so collapsing
    that grid to 8x8 would discard the spatial correspondence the change queries supervise.

    Transformers still exposes this module's result as ``pooler_output`` because that is the
    fixed Qwen3-VL output field. In this wrapper the name is historical: the tensor remains
    full-resolution ``(B * num_patches, hidden_size)`` and is not pooled.
    """

    def __init__(self, norm: nn.Module):
        super().__init__()
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


def _uniform_batched_attention(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb=None,
    position_embeddings=None,
    **kwargs,
) -> torch.Tensor:
    """Batched replacement for Qwen3-VL's variable-length vision attention.

    The stock non-FlashAttention path splits the concatenated sequence into one chunk per
    image and calls SDPA in a Python loop. With a batch of 256 frames and 24 layers that is
    ~6k tiny kernel launches per forward, and it measured 0.94 s/step against Cosmos3-Edge's
    0.71 s despite Qwen's tower being the *smaller* of the two (306M vs 413M).

    Every frame here is the same fixed resolution, so all chunks have equal length and
    block-diagonal attention is exactly batched attention. We reshape to ``(B, H, n, d)``
    and issue a single call. This is an algebraic identity, not an approximation, and
    ``scripts/check_qwen3vl_vision.py`` asserts it against the stock path.

    Non-uniform batches fall back to the original implementation, so mixed-resolution input
    would still be correct, just slow.
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb_vision,
        eager_attention_forward,
    )

    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    if lengths.numel() == 0 or not bool(torch.all(lengths == lengths[0])):
        return self._stock_forward(
            hidden_states, cu_seqlens, rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings, **kwargs,
        )

    seq_length = hidden_states.shape[0]
    n = int(lengths[0])
    bsz = seq_length // n

    query_states, key_states, value_states = (
        self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    )
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

    # (S, H, d) -> (B, H, n, d); rows are contiguous per image, so the view is the split.
    def _split(t):
        return t.view(bsz, n, self.num_heads, -1).transpose(1, 2)

    query_states, key_states, value_states = _split(query_states), _split(key_states), _split(value_states)

    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, _ = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask=None,
        scaling=self.scaling,
        dropout=0.0 if not self.training else self.attention_dropout,
        is_causal=False,
        **kwargs,
    )
    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    return self.proj(attn_output)


def _install_grid_caches(vision_model: nn.Module) -> None:
    """Memoise the two per-call tensors that only depend on the (constant) patch grid.

    ``fast_pos_embed_interpolate`` and ``rot_pos_emb`` are recomputed on every forward, but
    they are pure functions of ``grid_thw`` -- and every frame here is the same fixed
    resolution, so they return the identical tensor every step. Measured on a 256-image
    batch they cost 144 ms and 65 ms of a 1001 ms tower forward, i.e. ~21% spent
    recomputing constants.

    ``rot_pos_emb`` has no parameters, so caching it is unconditionally safe.
    ``fast_pos_embed_interpolate`` reads ``pos_embed.weight``; caching that would detach it
    from autograd, so the cache is bypassed whenever that weight is trainable. The tower is
    frozen in this repo, but the guard means unfreezing it stays correct rather than
    silently freezing the position embedding.
    """
    if not hasattr(vision_model, "rot_pos_emb") or not hasattr(vision_model, "fast_pos_embed_interpolate"):
        # Newer Transformers computes these through shared vision utilities in forward.
        return

    stock_rot = vision_model.rot_pos_emb
    stock_pos = vision_model.fast_pos_embed_interpolate
    rot_cache: dict = {}
    pos_cache: dict = {}

    def _key(grid_thw, extra=()):
        return (tuple(map(tuple, grid_thw.tolist())), *extra)

    def rot_pos_emb(grid_thw):
        key = _key(grid_thw)
        if key not in rot_cache:
            rot_cache[key] = stock_rot(grid_thw)
        return rot_cache[key]

    def fast_pos_embed_interpolate(grid_thw):
        weight = vision_model.pos_embed.weight
        if weight.requires_grad and torch.is_grad_enabled():
            return stock_pos(grid_thw)
        # Autocast changes the output dtype, so it has to be part of the key.
        dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else weight.dtype
        key = _key(grid_thw, (weight.device, dtype))
        if key not in pos_cache:
            pos_cache[key] = stock_pos(grid_thw)
        return pos_cache[key]

    vision_model.rot_pos_emb = rot_pos_emb
    vision_model.fast_pos_embed_interpolate = fast_pos_embed_interpolate



class Qwen3VLPatchTrunk(nn.Module):
    """Wraps ``Qwen3VLVisionModel`` behind a ``pixel_values -> (B, N, D)`` interface.

    Mirrors what ``SiglipVisionModel``/DINOv3 expose, so the perception encoder does not
    need to know which backbone it is holding.
    """

    def __init__(self, vision_model: nn.Module, config):
        super().__init__()
        self.vision_model = vision_model
        self.config = config
        self.patch_size = int(config.patch_size)
        self.merge_size = int(config.spatial_merge_size)
        self.temporal_patch_size = int(config.temporal_patch_size)
        self.in_channels = int(getattr(config, "in_channels", 3))

    def forward(self, pixel_values: torch.Tensor):
        b, c, h, w = pixel_values.shape
        p, m = self.patch_size, self.merge_size
        if h % (p * m) or w % (p * m):
            raise ValueError(
                f"Qwen3-VL needs a resolution divisible by patch_size*spatial_merge_size "
                f"({p * m}); got {h}x{w}."
            )
        gh, gw = h // p, w // p
        bh, bw = gh // m, gw // m

        # (B, C, bh, m, p, bw, m, p) -> Qwen's (B, bh, bw, m, m, C, p, p) patch order.
        x = pixel_values.view(b, c, bh, m, p, bw, m, p)
        x = x.permute(0, 2, 5, 3, 6, 1, 4, 7)
        # A still frame is repeated to fill the temporal patch, which is what Qwen's own
        # image processor does for single images.
        x = x.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, self.temporal_patch_size, -1, -1)
        flat = x.reshape(b * gh * gw, c * self.temporal_patch_size * p * p)

        # Transformers 4.57 computes rotary positions with an int64 `torch.prod`; keeping
        # that legacy path on CPU avoids an NVRTC dependency. Transformers 5 instead builds
        # embedding indices directly on `grid_thw.device`, so the grid must follow the
        # CUDA-resident position embedding or `nn.Embedding` receives CPU indices.
        grid_device = flat.device if hasattr(self.vision_model, "interpolation_mode") else "cpu"
        grid_thw = torch.tensor([[1, gh, gw]] * b, device=grid_device, dtype=torch.long)
        out = self.vision_model(flat, grid_thw)
        if getattr(out, "pooler_output", None) is not None:
            # Transformers 5 keeps the historical `pooler_output` field even though our
            # replacement merger only applies LayerNorm. Stock Qwen3-VL would return
            # (B*64, out_hidden_size) here; we retain all patches as
            # (B*gh*gw, hidden_size), which is (B*256, 1024) for the 4B checkpoint.
            # `last_hidden_state` has the same shape but is before this pretrained norm.
            tokens = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            tokens = out.last_hidden_state
        elif isinstance(out, tuple):
            tokens = out[0]
        else:
            tokens = out

        # Undo the merge-block ordering: (B, bh, bw, m, m, D) -> (B, gh, gw, D).
        d = tokens.shape[-1]
        tokens = tokens.view(b, bh, bw, m, m, d).permute(0, 1, 3, 2, 4, 5).reshape(b, gh * gw, d)

        class _Out:
            pass

        result = _Out()
        result.last_hidden_state = tokens
        return result


def build_qwen3vl_vision(model_dir: str | Path, optimized: bool = True):
    """Load the vision tower from a local Qwen3-VL snapshot.

    Returns ``(module, hidden_size, image_size)``. Works for any Qwen3-VL size: the tower's
    width and depth are read from the checkpoint's own ``vision_config``, so switching
    between the 2B/4B tower (1024-wide, 24 layers) and the 8B/32B one (1152-wide, 27
    layers, the same shape as Cosmos3-Edge's) is a directory change.

    ``optimized`` enables two changes that are mathematically no-ops: batched attention in
    place of the per-image loop (``_uniform_batched_attention``) and memoised grid tensors
    (``_install_grid_caches``). It is only ever turned off to check the two paths against
    each other.
    """
    from safetensors.torch import load_file
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    model_dir = Path(model_dir)
    vision_cfg = Qwen3VLVisionConfig(**json.loads((model_dir / "config.json").read_text())["vision_config"])
    # Pin the attention kernel. Left to auto-detection it varies with what happens to be
    # installed on a given host, and this tower runs variable-length attention over
    # `cu_seqlens`, where the fallbacks are not all equivalent.
    vision_cfg._attn_implementation = "sdpa"
    model = Qwen3VLVisionModel._from_config(vision_cfg)

    prefix = "model.visual."
    state: dict[str, torch.Tensor] = {}
    for shard in sorted(model_dir.glob("*.safetensors")):
        for key, tensor in load_file(str(shard)).items():
            if key.startswith(prefix):
                state[key[len(prefix) :]] = tensor.to(torch.float32)
    if not state:
        raise FileNotFoundError(
            f"No '{prefix}*' tensors found under {model_dir}. Download the shard that holds "
            "the vision tower (for Qwen3-VL-4B that is model-00002-of-00002.safetensors)."
        )

    missing, unexpected = model.load_state_dict(state, strict=False)
    # A partially loaded frozen tower is invisible -- it trains and merely learns less --
    # so refuse rather than warn.
    if missing or unexpected:
        raise RuntimeError(
            f"Qwen3-VL vision weights did not load cleanly: {len(missing)} missing "
            f"{missing[:5]}, {len(unexpected)} unexpected {unexpected[:5]}."
        )

    full = sum(p.numel() for p in model.parameters()) / 1e6
    # The stock merger is an LLM adapter: it pools the 16x16 grid to 8x8 and maps four
    # concatenated 1024-d patches to the text model's 2560-d width. Our change representation
    # and reconstruction target both require one token per original 16x16 cell, so retain the
    # trained per-token output norm but remove the spatial pooling/projection.
    model.merger = _NormOnlyMerger(model.merger.norm)
    model.deepstack_merger_list = nn.ModuleList()
    model.deepstack_visual_indexes = []

    if optimized:
        import types

        for block in model.blocks:
            block.attn._stock_forward = block.attn.forward
            block.attn.forward = types.MethodType(_uniform_batched_attention, block.attn)
        _install_grid_caches(model)

    trunk = Qwen3VLPatchTrunk(model, vision_cfg)
    kept = sum(p.numel() for p in trunk.parameters()) / 1e6
    grid = QWEN3VL_IMAGE_SIZE // vision_cfg.patch_size
    logging.info(
        "Loaded Qwen3-VL vision trunk: %.1fM params (%.1fM in checkpoint; merger MLP and "
        "DeepStack heads dropped), hidden=%d, %d layers, %dx%d input -> %d patch tokens",
        kept,
        full,
        vision_cfg.hidden_size,
        vision_cfg.depth,
        QWEN3VL_IMAGE_SIZE,
        QWEN3VL_IMAGE_SIZE,
        grid * grid,
    )
    return trunk, int(vision_cfg.hidden_size), QWEN3VL_IMAGE_SIZE
