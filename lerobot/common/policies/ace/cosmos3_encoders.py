"""Loaders for the NVIDIA Cosmos 3 (Edge) sub-modules used by the perception branch.

Cosmos 3 ships *two* unrelated visual pipelines and it matters which one you take:

* ``vision_encoder/`` -- the Reasoner's semantic ViT. SigLIP-shaped (1152 wide, 27 layers,
  4304 MLP, patch 16, 256 learned positions), plus a 77M projector to the 2048-d LLM stream.
  This is the analogue of DINOv3 here and the module Cosmos itself documents swapping via
  ``export_model --vit-checkpoint-path``.
* ``vae/`` -- the Generator's tokenizer, which is the stock Alibaba **Wan2.2** causal 3D video
  VAE (``AutoencoderKLWan``, 48 latent channels, 16x spatial / 4x temporal). It is a video
  *compressor*, not a semantic encoder, and it is not NVIDIA-trained.

``transformers`` 4.57.4 has no ``cosmos3_edge_vision`` entry, so the tower cannot be loaded
through ``AutoModel``. It does not need to be: the checkpoint's tensor names are ordinary
SigLIP names under a ``model.visual.`` prefix, so the weights map onto ``SiglipVisionModel``
one-to-one. The single exception is ``patch_embedding.weight``, stored ``(1152, 768)`` as a
Linear over flattened 3x16x16 patches where SigLIP keeps a ``(1152, 3, 16, 16)`` Conv2d; the
two hold the same numbers in the same order, so a ``view`` converts it.

Everything here is loaded frozen and in eval mode. Nothing in this file trains.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn

# The Cosmos3 vision tower carries no CLS or register tokens: every output position is a
# patch. Kept as a named constant because the perception encoder has to strip a prefix for
# DINOv3 and must not strip one here.
COSMOS3_NUM_PREFIX_TOKENS = 0


def _vision_config(model_dir: Path) -> dict:
    cfg = json.loads((model_dir / "config.json").read_text())
    if "vision_config" not in cfg:
        raise ValueError(f"{model_dir}/config.json has no `vision_config`; is this Cosmos3-Edge?")
    return cfg["vision_config"]


def build_cosmos3_vision(model_dir: str | Path) -> tuple[nn.Module, int, int]:
    """Load ``vision_encoder/`` as a ``SiglipVisionModel``.

    Returns ``(module, hidden_size, native_image_size)``. The projector to the 2048-d LLM
    stream is deliberately *not* loaded: it exists to feed Cosmos's own language model, and
    this branch has its own ``visual_proj`` into ``hidden_dim``.
    """
    from safetensors.torch import load_file
    from transformers import SiglipVisionConfig, SiglipVisionModel

    model_dir = Path(model_dir)
    vc = _vision_config(model_dir)
    # ``num_patches`` is a square grid, so the image side is sqrt(num_patches) * patch_size.
    grid = int(round(vc["num_patches"] ** 0.5))
    image_size = grid * vc["patch_size"]
    config = SiglipVisionConfig(
        hidden_size=vc["hidden_size"],
        intermediate_size=vc["intermediate_size"],
        num_hidden_layers=vc["num_hidden_layers"],
        num_attention_heads=vc["num_attention_heads"],
        patch_size=vc["patch_size"],
        image_size=image_size,
        layer_norm_eps=vc.get("layer_norm_eps", 1e-6),
        hidden_act=vc.get("hidden_act", "gelu_pytorch_tanh"),
        # SigLIP's attention-pooling head has no counterpart in this checkpoint and the
        # perception branch pools with its own change queries anyway.
        vision_use_head=False,
    )
    model = SiglipVisionModel(config)

    raw = load_file(model_dir / "vision_encoder" / "model.safetensors")
    remapped: dict[str, torch.Tensor] = {}
    for key, tensor in raw.items():
        if not key.startswith("model.visual."):
            continue  # `model.projector.*` -- Cosmos's adapter into its LLM, not wanted here
        new_key = "vision_model." + key[len("model.visual.") :]
        if new_key.endswith("embeddings.patch_embedding.weight") and tensor.ndim == 2:
            out_dim, flat = tensor.shape
            tensor = tensor.view(out_dim, config.num_channels, config.patch_size, config.patch_size)
            if flat != config.num_channels * config.patch_size**2:
                raise ValueError(f"patch_embedding has width {flat}, incompatible with {config}")
        remapped[new_key] = tensor

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    # `strict=False` is needed only because SigLIP registers a `position_ids` buffer that the
    # checkpoint does not carry. Anything else missing means the remap is wrong, and silently
    # training on a partly random 412M tower is exactly the failure this check exists to stop.
    real_missing = [k for k in missing if not k.endswith("position_ids")]
    if real_missing or unexpected:
        raise RuntimeError(
            f"Cosmos3 vision remap incomplete: {len(real_missing)} missing "
            f"(e.g. {real_missing[:3]}), {len(unexpected)} unexpected (e.g. {unexpected[:3]})"
        )
    logging.info(
        "Loaded Cosmos3 vision tower: %.1fM params, %dx%d input, %d patch tokens",
        sum(p.numel() for p in model.parameters()) / 1e6,
        image_size,
        image_size,
        vc["num_patches"],
    )
    return model, config.hidden_size, image_size


def build_cosmos3_vae(model_dir: str | Path):
    """Load ``vae/`` (the Wan2.2 causal 3D video VAE) for use as a reconstruction target.

    Returns ``(vae, z_dim, temporal_compression, latents_mean, latents_std)``.

    The decoder is dropped. It is 555M of the checkpoint's 705M parameters and this branch
    only ever calls ``encode``; keeping it would cost more memory than the entire trainable
    perception trunk for a module that never runs.

    ``latents_mean``/``latents_std`` are Wan2.2's own per-channel latent statistics. They
    matter for the same reason section 19's tactile statistics did: an unnormalised target
    lets the reconstruction loss sit at a scale where its gradient cannot compete with the
    contrastive term.
    """
    from diffusers import AutoencoderKLWan

    model_dir = Path(model_dir)
    vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=torch.float32)
    cfg = json.loads((model_dir / "vae" / "config.json").read_text())
    total = sum(p.numel() for p in vae.parameters()) / 1e6
    vae.decoder = None
    logging.info(
        "Loaded Cosmos3 VAE (Wan2.2) encoder: %.1fM params (decoder dropped, was %.1fM total), "
        "z_dim=%d, spatial/%d, temporal/%d",
        sum(p.numel() for p in vae.parameters()) / 1e6,
        total,
        cfg["z_dim"],
        cfg["scale_factor_spatial"],
        cfg["scale_factor_temporal"],
    )
    mean = torch.tensor(cfg["latents_mean"], dtype=torch.float32)
    std = torch.tensor(cfg["latents_std"], dtype=torch.float32)
    return vae, int(cfg["z_dim"]), int(cfg["scale_factor_temporal"]), mean, std
