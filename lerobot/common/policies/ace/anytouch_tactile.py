"""Frozen AnyTouch tactile encoder.

This is an encoder-only reconstruction of the MIT-licensed AnyTouch release at
https://github.com/GeWu-Lab/AnyTouch/tree/9c43a1a6eb38d904fd767712eb9dcb2d98b8d56b.
Copyright (c) 2025 GeWu-Lab.
The official checkpoint contains a complete text/vision/touch model plus an MAE decoder.
Only the 24-layer CLIP ViT-L/14 touch encoder, shared 3-D patch embedding, sensor tokens and
1024->768 touch projection are loaded here.

AnyTouch's released dynamic path consumes three tactile frames. The source implementation
stores them as ``(B, T, C, H, W)`` and passes that tensor directly to ``Conv3d``. Since both
T and C equal three, the learned weight treats frames as input channels and RGB as depth.
That unusual layout is preserved deliberately; permuting to conventional ``(B, C, T, H, W)``
would silently change the operation represented by the pretrained weights.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import CLIPVisionConfig, CLIPVisionModel


logger = logging.getLogger(__name__)

ANYTOUCH_IMAGE_SIZE = 224
ANYTOUCH_HIDDEN_DIM = 1024
ANYTOUCH_OUT_DIM = 768
ANYTOUCH_PATCH_SIZE = 14
ANYTOUCH_SENSOR_TOKENS = 5
ANYTOUCH_UNIVERSAL_SENSOR_ID = -1


class AnyTouchTactileTower(nn.Module):
    """Encode four-frame tactile windows into one or two pretrained AnyTouch features.

    With two output tokens, four input frames become the overlapping dynamic windows
    ``[0, 1, 2]`` and ``[1, 2, 3]``. This uses every frame while requiring two ViT passes per
    pad rather than four independent static-image passes. With one token, three frames are
    sampled evenly across the input window.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    output_dim = ANYTOUCH_OUT_DIM

    def __init__(
        self,
        checkpoint: str,
        num_tokens: int = 2,
        forward_batch_size: int = 128,
    ):
        super().__init__()
        if num_tokens not in (1, 2):
            raise ValueError(f"AnyTouch supports one or two output tokens, got {num_tokens}.")
        if forward_batch_size < 1:
            raise ValueError(
                f"AnyTouch forward_batch_size must be positive, got {forward_batch_size}."
            )
        self.num_tokens = num_tokens
        self.forward_batch_size = forward_batch_size

        config = CLIPVisionConfig(
            hidden_size=ANYTOUCH_HIDDEN_DIM,
            intermediate_size=4096,
            num_hidden_layers=24,
            num_attention_heads=16,
            image_size=ANYTOUCH_IMAGE_SIZE,
            patch_size=ANYTOUCH_PATCH_SIZE,
            num_channels=3,
            hidden_act="gelu",
            layer_norm_eps=1e-5,
            attention_dropout=0.0,
        )
        clip_vision = CLIPVisionModel(config)
        # Transformers 4 wraps the encoder under ``vision_model``; Transformers 5 exposes
        # the same embeddings/encoder/layernorm modules directly on CLIPVisionModel.
        self.touch_model = getattr(clip_vision, "vision_model", clip_vision)
        self.touch_model.embeddings.patch_embedding = nn.Conv3d(
            in_channels=3,
            out_channels=ANYTOUCH_HIDDEN_DIM,
            kernel_size=(3, ANYTOUCH_PATCH_SIZE, ANYTOUCH_PATCH_SIZE),
            stride=(3, ANYTOUCH_PATCH_SIZE, ANYTOUCH_PATCH_SIZE),
            bias=False,
        )
        self.sensor_token = nn.Parameter(
            torch.empty(10, ANYTOUCH_SENSOR_TOKENS, ANYTOUCH_HIDDEN_DIM)
        )
        self.touch_projection = nn.Linear(
            ANYTOUCH_HIDDEN_DIM, ANYTOUCH_OUT_DIM, bias=False
        )
        self.register_buffer(
            "mean",
            torch.tensor(self.IMAGENET_MEAN).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(self.IMAGENET_STD).view(1, 1, 3, 1, 1),
            persistent=False,
        )

        self._load(checkpoint)
        self.requires_grad_(False)
        self.eval()

    def _load(self, checkpoint: str) -> None:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(
                f"AnyTouch checkpoint not found: {path}. Set "
                "--policy.anytouch_checkpoint to the official stage-2 checkpoint."
            )

        blob = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(blob, dict):
            raise TypeError(f"Expected an AnyTouch checkpoint dictionary, got {type(blob).__name__}.")
        state = blob.get("model", blob)
        if not isinstance(state, dict):
            raise TypeError("AnyTouch checkpoint key 'model' is not a state dictionary.")
        state = {key.removeprefix("module."): value for key, value in state.items()}

        full_prefix = "touch_mae_model."
        if any(key.startswith(full_prefix) for key in state):
            touch_prefix = full_prefix + "touch_model."
            model_state = {
                key.removeprefix(touch_prefix): value
                for key, value in state.items()
                if key.startswith(touch_prefix)
            }
            video_patch = state.get(full_prefix + "video_patch_embedding.weight")
            sensor_token = state.get(full_prefix + "sensor_token")
            projection = state.get(full_prefix + "touch_projection.weight")
        else:
            touch_prefix = "touch_model."
            model_state = {
                key.removeprefix(touch_prefix): value
                for key, value in state.items()
                if key.startswith(touch_prefix)
            }
            video_patch = state.get("video_patch_embedding.weight")
            sensor_token = state.get("sensor_token")
            projection = state.get("touch_projection.weight")

        if not model_state:
            raise KeyError(
                "Checkpoint has no AnyTouch tactile encoder keys under "
                "'touch_mae_model.touch_model.*' or 'touch_model.*'."
            )
        required = {
            "video patch embedding": video_patch,
            "sensor token": sensor_token,
            "touch projection": projection,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise KeyError(f"AnyTouch checkpoint is missing: {', '.join(missing)}.")

        # The checkpoint also carries the original 2-D CLIP patch embedding, but the released
        # AnyTouch model uses this shared 3-D embedding for both static and dynamic touch.
        model_state["embeddings.patch_embedding.weight"] = video_patch
        self.touch_model.load_state_dict(model_state, strict=True)
        with torch.no_grad():
            self.sensor_token.copy_(sensor_token)
            self.touch_projection.weight.copy_(projection)

        logger.info(
            "Loaded frozen AnyTouch tactile encoder from %s (%.1fM parameters, "
            "universal sensor token).",
            path,
            sum(parameter.numel() for parameter in self.parameters()) / 1e6,
        )

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def _windows(self, images: torch.Tensor) -> torch.Tensor:
        num_frames = images.shape[1]
        if num_frames < 3:
            raise ValueError(
                f"AnyTouch dynamic encoding needs at least 3 tactile frames, got {num_frames}."
            )
        if self.num_tokens == 1:
            # Match the official inference API, which consumes x[:, :3].
            return images[:, :3].unsqueeze(1)
        if num_frames < 4:
            raise ValueError(
                "Two AnyTouch tactile tokens need at least 4 frames so their windows differ."
            )
        return torch.stack([images[:, :3], images[:, -3:]], dim=1)

    def _encode_windows(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, 3 frames, 3 RGB, H, W)`` using AnyTouch's released layout."""
        x = x.float().div_(255.0)
        if x.shape[-2:] != (ANYTOUCH_IMAGE_SIZE, ANYTOUCH_IMAGE_SIZE):
            x = x.reshape(-1, 3, *x.shape[-2:])
            x = F.interpolate(
                x,
                size=(ANYTOUCH_IMAGE_SIZE, ANYTOUCH_IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            x = x.view(-1, 3, 3, ANYTOUCH_IMAGE_SIZE, ANYTOUCH_IMAGE_SIZE)
        x = ((x - self.mean) / self.std).to(
            self.touch_model.embeddings.patch_embedding.weight.dtype
        )

        patch_embeddings = self.touch_model.embeddings.patch_embedding(x)
        patch_embeddings = patch_embeddings.flatten(2).transpose(1, 2)
        if patch_embeddings.shape[1] != 256:
            raise RuntimeError(
                f"Expected 256 AnyTouch patches, got {patch_embeddings.shape[1]}."
            )

        position = self.touch_model.embeddings.position_embedding(
            self.touch_model.embeddings.position_ids
        ).to(patch_embeddings.dtype)
        patch_embeddings = patch_embeddings + position[:, 1:]
        cls = (
            self.touch_model.embeddings.class_embedding.to(patch_embeddings.dtype)
            + position[:, 0]
        )
        cls = cls.expand(patch_embeddings.shape[0], 1, -1)
        sensor = self.sensor_token[ANYTOUCH_UNIVERSAL_SENSOR_ID].to(
            patch_embeddings.dtype
        )
        sensor = sensor.unsqueeze(0).expand(patch_embeddings.shape[0], -1, -1)
        hidden = torch.cat([cls, sensor, patch_embeddings], dim=1)
        hidden = self.touch_model.pre_layrnorm(hidden)
        hidden = self.touch_model.encoder(inputs_embeds=hidden).last_hidden_state
        pooled = self.touch_model.post_layernorm(hidden[:, 0])
        return self.touch_projection(pooled)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """``(N, F, 3, H, W)`` uint8 -> ``(N, num_tokens, 768)``."""
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError(
                f"Expected tactile images shaped (N,F,3,H,W), got {tuple(images.shape)}."
            )
        n = images.shape[0]
        windows = self._windows(images).reshape(
            n * self.num_tokens, 3, 3, *images.shape[-2:]
        )
        # A tactile-heavy same-dataset batch can contain hundreds of live pads. The tower is
        # frozen, so chunking changes neither gradients nor BatchNorm statistics (there is no
        # BatchNorm), while bounding the transient ViT attention/MLP memory independently of
        # the training batch size.
        features = [
            self._encode_windows(chunk)
            for chunk in windows.split(self.forward_batch_size, dim=0)
        ]
        return torch.cat(features, dim=0).view(n, self.num_tokens, ANYTOUCH_OUT_DIM)
