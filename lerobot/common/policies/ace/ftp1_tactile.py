"""Reuse of the FTP-1 pretrained tactile tower (the UniVTAC line of work).

Why this exists
---------------
``TactileImageEncoder`` (see ``modeling_robo_contrast.py``) is an ImageNet ResNet-18 that has
to learn what a tactile image *means* from our contrastive signal alone. That signal is thin:
tactile datasets are ~2.7% of ``debug_research_data``, so the encoder sees a few rows per batch.
FTP-1 solved the same problem with ~3000 h of tactile manipulation data, and publishes the
resulting per-sensor tokenizers. Starting from those instead of ImageNet is strictly more
informed, and -- because we keep them frozen -- costs no trainable parameters.

What is actually reused
-----------------------
Note the repository contains *two* tactile encoders and only one of them is downloadable:

* ``UniVTAC/encoder/network.py::Tactile`` -- a ResNet-18 pretrained by reconstruction on
  *simulated GelSight Mini* data. This is what our ResNet path is modelled on, but its weights
  are **not published** anywhere in the repo (training writes to a local ``ablation/`` path),
  and it saw a single simulated sensor, so it would not transfer to SharpaWave/OpenLoongVTouch.
* ``src/openpi/models_pytorch/`` -- the T3-derived ViT tower that the shipped FTP-1 policy
  actually uses. Weights **are** published, under MIT, at ``MJJJJ1064/ftp1_v0426_50kstep``.

So this module reuses the second one. The architecture per tactile pad is

    (3, 224, 224) --patch16--> 196+1 tokens --ViT depth 3, width 768--> shared trunk depth 9
        --> LayerNorm --> take CLS --> Linear(768 -> 512)

with the ViT *per sensor* (a distinct 22.0M-parameter file per sensor type) and the trunk plus
projection *shared* across all sensors (64.2M parameters). The 512-wide output is not an
accident: FTP-1's tactile expert is ``gemma_small``, whose width is 512, which happens to be
exactly our ``tactile_feat_dim``, so the pretrained ``image_proj`` can be reused verbatim as
the head rather than discarded.

Why frozen
----------
Three independent reasons, any one of which would be sufficient:

1. Trainable-parameter budget. Four sensors' ViTs plus the trunk is 152M parameters. Unfreezing
   would blow the 500-800M total and wreck the perception/physical balance.
2. Sample size. Tactile is 2.7% of the mixture. Fine-tuning 152M parameters on that is
   hopelessly under-determined; we would destroy the pretrained features long before the
   contrastive loss taught us anything better.
3. ZeRO-2 safety. ``forward`` dispatches rows to a *different ViT per sensor*, so which
   parameters receive a gradient would depend on which datasets landed in the local batch.
   Under ZeRO-2 that desyncs the gradient-reduction schedule across ranks and surfaces as a
   600 s NCCL timeout rather than an exception. Frozen parameters never enter the schedule, so
   the hazard disappears entirely. **Do not unfreeze this without first making the sensor
   dispatch data-independent.**

Preprocessing
-------------
FTP-1 normalises tactile images in two steps, and both matter:

    x = uint8 / 255 * 2 - 1            # "div255_mul2_minus1"
    x = (x - mean) / std               # per-channel z-score

The z-score statistics are **per dataset**, not per sensor -- the same GelSight Mini has a
channel mean of -0.18 in ``Unit`` and -0.85 in ``RDP_Bimanual``, because gel colour, lighting
and camera gain differ per rig. ``FTP1_TACTILE_DATASETS`` therefore carries the statistics that
FTP-1 itself computed for each of its domains, taken from
``normalization/<domain>/independent_norm_stats_all_t0_zscore.json`` in the checkpoint.

Channel order was verified empirically rather than assumed: FTP-1's own eval script feeds BGR
(its zarr was built with ``cv2.imdecode``), so it was not obvious which order our decoded
LeRobot videos should be in. Measuring the per-channel mean of our
``VisuoTactile_D-WHEEL`` tactile video -- the only sensor whose three channel means are far
apart -- gives (-0.330, -0.180, -0.170) against a published (-0.410, -0.201, -0.219): a mean
absolute error of 0.050 as-is versus 0.124 reversed. Our frames are already in FTP-1's order,
so **no channel flip is applied**.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

logger = logging.getLogger(__name__)

# Native input geometry of the published weights. ``pos_embed`` is stored as (1, 197, 768),
# i.e. 14x14 patches plus a CLS token, so feeding anything other than 224 requires the
# positional embedding to be resampled (see ``_ViTEncoder.forward``).
FTP1_IMAGE_SIZE = 224
FTP1_PATCH_SIZE = 16
FTP1_EMBED_DIM = 768
FTP1_NUM_HEADS = 12
FTP1_ENCODER_DEPTH = 3
FTP1_TRUNK_DEPTH = 9
FTP1_OUT_DIM = 512

# Every image-type tactile sensor that appears in the datasets registered in
# ``canonical_space.py``. The index into this list is the ``tactile_sensor_id`` the dataset
# hands to the model; -1 means "no pad here" and is never dispatched.
FTP1_SENSOR_NAMES: list[str] = [
    "SharpaWave",
    "OpenLoongVTouch",
    "GelSightMini",
    "MCTac",
    "ViTaMIn",
    "FreeTacMan",
    "exUMI",
]
FTP1_SENSOR_IDS: dict[str, int] = {name: i for i, name in enumerate(FTP1_SENSOR_NAMES)}


def _stats(mean: list[float], std: list[float]) -> dict[str, list[float]]:
    return {"mean": mean, "std": std}


# Per-dataset tactile-image metadata, transcribed from the FTP-1 checkpoint:
#   * ``sensors`` -- which physical sensor produces each entry of the dataset's
#     ``tactile_image`` list in ``canonical_space.PHYSICAL_SPECS``, in the *same order*. Derived
#     from ``tactile_input_config_file.json`` by sorting FTP-1's function-area indices (24-47 is
#     the left hand, 0-23 the right), which reproduces our left-then-right key ordering.
#   * ``stats`` -- the per-channel z-score applied after ``div255_mul2_minus1``, from
#     ``normalization/<domain>/independent_norm_stats_all_t0_zscore.json``.
FTP1_TACTILE_DATASETS: dict[str, dict] = {
    "ftp_1_FreeTacMan": {
        "sensors": ["FreeTacMan", "FreeTacMan"],
        "stats": {"FreeTacMan": _stats([-0.003485, -0.288809, 0.003777], [0.368280, 0.326942, 0.320351])},
    },
    "ftp_1_RDP": {
        # FTP-1 records a GelSight and an MCTac here, but our conversion keeps a single pad,
        # `tactile_left_0`, and canonical_space.py documents that rig as the MCTac camera.
        # Listing only the pad we actually have keeps the positional assignment honest and
        # stops the truncation path from silently handing this dataset the GelSight tokenizer.
        # Unverified by measurement -- this dataset is not in the local mixture -- but the same
        # pad in RDP_Bimanual measures as an MCTac, and the two come from the same rig.
        "sensors": ["MCTac"],
        "stats": {
            "GelSightMini": _stats([-0.567995, -0.307983, -0.287494], [0.210233, 0.138887, 0.211952]),
            "MCTac": _stats([0.039227, 0.052511, 0.123231], [0.252232, 0.252368, 0.251900]),
        },
    },
    # Our copy is not the copy FTP-1 measured, so the published statistics for this domain do
    # not describe it and are kept only for reference. Measured over 600 frames spanning all
    # 50 episodes, `tactile_left_0` is (0.013, 0.008, 0.090) with std 0.243, which is 0.034
    # from RDP's *MCTac* and 0.81 from the GelSight statistics FTP-1 published for this domain
    # -- so the pad is an MCTac and the numbers below are ours, not FTP-1's.
    # `tactile_right_0` is dead: every frame is uniform black (ffprobe reports Y constant at
    # 16 across the whole file, 2.8 MB against 16 MB for the live pad), so it is not listed as
    # a tactile view in canonical_space.py at all.
    "ftp_1_RDP_Bimanual": {
        "sensors": ["MCTac"],
        "stats": {
            "MCTac": _stats([0.013404, 0.008264, 0.089951], [0.243333, 0.241419, 0.244142]),
        },
    },
    "ftp_1_sharpa": {
        "sensors": ["SharpaWave"] * 6,
        "stats": {"SharpaWave": _stats([-0.867698] * 3, [0.199941] * 3)},
    },
    "ftp_1_sharpa_split_0": {
        "sensors": ["SharpaWave"] * 6,
        "stats": {"SharpaWave": _stats([-0.867698] * 3, [0.199941] * 3)},
    },
    "ftp_1_Unit": {
        "sensors": ["GelSightMini"],
        "stats": {"GelSightMini": _stats([-0.175476, -0.262557, -0.539569], [0.247713, 0.178803, 0.211461])},
    },
    "ftp_1_VLA_touch": {
        "sensors": ["GelSightMini"],
        "stats": {"GelSightMini": _stats([-0.397731, -0.235207, -0.145355], [0.251950, 0.176359, 0.275244])},
    },
    "ftp_1_ViTaMIn": {
        "sensors": ["ViTaMIn", "ViTaMIn"],
        "stats": {"ViTaMIn": _stats([-0.003436, 0.015401, -0.075011], [0.443797, 0.451132, 0.473751])},
    },
    "ftp_1_VisuoTactile_D-WHEEL": {
        "sensors": ["OpenLoongVTouch"] * 4,
        "stats": {"OpenLoongVTouch": _stats([-0.410254, -0.200973, -0.218633], [0.246950, 0.210361, 0.232070])},
    },
    "ftp_1_VisuoTactile_D-WHEEL_split_0": {
        "sensors": ["OpenLoongVTouch"] * 4,
        "stats": {"OpenLoongVTouch": _stats([-0.410254, -0.200973, -0.218633], [0.246950, 0.210361, 0.232070])},
    },
    "ftp_1_VisuoTactile_QINGLOONG": {
        "sensors": ["OpenLoongVTouch"] * 4,
        "stats": {"OpenLoongVTouch": _stats([-0.479236, -0.252941, -0.326944], [0.283776, 0.221937, 0.250982])},
    },
    "ftp_1_exUMI": {
        "sensors": ["exUMI"],
        "stats": {"exUMI": _stats([-0.974438, -0.977952, 0.224915], [0.071599, 0.058624, 0.179451])},
    },
    # --- Datasets FTP-1 never trained on -------------------------------------------------
    # OpenNeo is not one of FTP-1's domains, so there is nothing to transcribe. These numbers
    # were measured from our own copies with scripts/measure_tactile_stats.py, using the same
    # definition FTP-1 uses (per-channel mean/std of uint8/255*2-1). Each rig's pads agree with
    # one another to within 0.013, so a single per-dataset statistic is used.
    #
    # The sensor is not documented anywhere we have, and it is not a GelSight: of every
    # published statistic, GelSight Mini is the *farthest* from these (0.34-0.37), while MCTac
    # is by far the nearest (0.046-0.063). MCTac is therefore the tokenizer these pads are
    # dispatched to. Treat the identity as inferred from appearance, not confirmed.
    "open_neo_aloha": {
        "sensors": ["MCTac"] * 4,
        "stats": {"MCTac": _stats([0.083175, 0.096035, 0.072995], [0.341184, 0.335290, 0.320280])},
    },
    "open_neo_arx5_single": {
        "sensors": ["MCTac"] * 2,
        "stats": {"MCTac": _stats([0.132388, 0.111184, 0.086841], [0.337444, 0.327955, 0.305640])},
    },
    # Not held locally, so this one is not measured. It is the bimanual build of the same ARX5
    # rig as open_neo_arx5_single, so that rig's statistics are reused.
    "open_neo_arx5": {
        "sensors": ["MCTac"] * 4,
        "stats": {"MCTac": _stats([0.132388, 0.111184, 0.086841], [0.337444, 0.327955, 0.305640])},
    },
    "open_neo_ur": {
        "sensors": ["MCTac"] * 2,
        "stats": {"MCTac": _stats([0.086441, 0.082853, 0.060759], [0.342134, 0.331121, 0.313760])},
    },
    # Also not held locally, and unlike arx5 it has no measured sibling, so it gets the average
    # of the three OpenNeo rigs we did measure. Those three agree to within 0.05 of each other,
    # so this is a much better prior than the identity fallback; replace it with a real
    # measurement once the data is available.
    "open_neo_flexiv": {
        "sensors": ["MCTac"] * 2,
        "stats": {"MCTac": _stats([0.100668, 0.096691, 0.073532], [0.340254, 0.331455, 0.313227])},
    },
}

# Fallback for a tactile dataset we have no FTP-1 entry for. ``div255_mul2_minus1`` already
# centres a mid-grey image near zero, so an identity z-score is the least-damaging default; the
# sensor falls back to GelSight Mini, the most common gel camera in the checkpoint.
_DEFAULT_SENSOR = "GelSightMini"
_DEFAULT_STATS = _stats([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])


def tactile_image_sensors(dataset_name: str, num_views: int) -> tuple[list[int], list[list[float]], list[list[float]]]:
    """Per-view sensor id and z-score statistics for ``dataset_name``.

    Returns three lists of length ``num_views``. ``num_views`` is the number of tactile image
    keys the dataset actually exposes, which may be smaller than FTP-1's view count when our
    conversion dropped a pad; a mismatch is logged rather than raised because the resulting
    per-view assignment is still positionally correct for the pads we kept.
    """
    entry = FTP1_TACTILE_DATASETS.get(dataset_name)
    if entry is None:
        if num_views:
            logger.warning(
                "%s has %d tactile image view(s) but no FTP-1 sensor entry; falling back to %s "
                "with an identity z-score. Add it to FTP1_TACTILE_DATASETS.",
                dataset_name,
                num_views,
                _DEFAULT_SENSOR,
            )
        sensors = [_DEFAULT_SENSOR] * num_views
        stats = {_DEFAULT_SENSOR: _DEFAULT_STATS}
    else:
        sensors = list(entry["sensors"])
        stats = entry["stats"]
        if len(sensors) != num_views:
            logger.warning(
                "%s exposes %d tactile view(s) but FTP-1 lists %d; using the first %d.",
                dataset_name,
                num_views,
                len(sensors),
                num_views,
            )
            sensors = (sensors + [_DEFAULT_SENSOR] * num_views)[:num_views]

    ids, means, stds = [], [], []
    for name in sensors:
        ids.append(FTP1_SENSOR_IDS.get(name, FTP1_SENSOR_IDS[_DEFAULT_SENSOR]))
        s = stats.get(name, _DEFAULT_STATS)
        means.append(list(s["mean"]))
        stds.append(list(s["std"]))
    return ids, means, stds


def required_sensors(dataset_names: list[str]) -> list[str]:
    """The distinct sensors a mixture needs, so we only build/load the ViTs we will use."""
    needed: list[str] = []
    for name in dataset_names:
        entry = FTP1_TACTILE_DATASETS.get(name)
        sensors = entry["sensors"] if entry else []
        for s in sensors:
            if s not in needed:
                needed.append(s)
    return needed


# --------------------------------------------------------------------------------------
# The network. Layer names deliberately mirror ``timm.models.vision_transformer`` because the
# published checkpoints were saved from a timm ViT; keeping the names identical means the
# state dict loads with ``strict=True`` and any future key drift fails loudly.
# --------------------------------------------------------------------------------------


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(b, n, c))


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _Mlp(dim, dim * mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class _PatchEmbed(nn.Module):
    def __init__(self, dim: int, patch: int):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class _ViTEncoder(nn.Module):
    """One sensor's tokenizer: ``vit_encoder.*`` in ``<Sensor>_image_224_224_3.safetensors``.

    The published module deletes ``head`` and ``norm`` (the final LayerNorm lives in the shared
    trunk instead), so neither exists here.
    """

    def __init__(self, dim: int = FTP1_EMBED_DIM, depth: int = FTP1_ENCODER_DEPTH,
                 num_heads: int = FTP1_NUM_HEADS, patch: int = FTP1_PATCH_SIZE,
                 img_size: int = FTP1_IMAGE_SIZE):
        super().__init__()
        self.patch = patch
        self.grid = img_size // patch
        self.patch_embed = _PatchEmbed(dim, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid * self.grid + 1, dim))
        self.blocks = nn.ModuleList([_Block(dim, num_heads) for _ in range(depth)])

    def _resample_pos_embed(self, n_patches: int) -> torch.Tensor:
        """Bicubic-resample the patch positions when the input is not 224x224.

        Only used if ``tactile_img_size`` is changed away from the native resolution; the
        pretrained features are calibrated for 224 and anything else is off-distribution.
        """
        if n_patches == self.pos_embed.shape[1] - 1:
            return self.pos_embed
        cls_pos, patch_pos = self.pos_embed[:, :1], self.pos_embed[:, 1:]
        src = int(round(patch_pos.shape[1] ** 0.5))
        dst = int(round(n_patches**0.5))
        patch_pos = patch_pos.reshape(1, src, src, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos.float(), size=(dst, dst), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, dst * dst, -1).to(cls_pos.dtype)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self._resample_pos_embed(x.shape[1])
        for block in self.blocks:
            x = block(x)
        return x


class _SharedTrunk(nn.Module):
    """``shared_chunk_encoder.*`` plus ``image_proj.*`` from the shared checkpoint.

    FTP-1 runs this over the tokens of a temporal chunk, but the shipped configuration sets
    ``disable_history: true`` (chunk length 1), so it is applied per frame here too.
    """

    def __init__(self, dim: int = FTP1_EMBED_DIM, depth: int = FTP1_TRUNK_DEPTH,
                 num_heads: int = FTP1_NUM_HEADS, out_dim: int = FTP1_OUT_DIM):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.image_proj = nn.Linear(dim, out_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(tokens)
        return self.image_proj(self.norm(tokens)[:, 0])


class FTP1TactileTower(nn.Module):
    """Per-sensor FTP-1 tokenizers plus the shared trunk, as a drop-in tactile image encoder.

    Signature matches ``TactileImageEncoder``: ``(B, V, 3, H, W)`` uint8 in, ``(B, V, 512)``
    out. Unlike the ResNet it additionally needs to know *which sensor* produced each row,
    because the tokenizer is per sensor, so ``forward`` takes the sensor ids and the per-view
    normalisation the dataset carries alongside the pixels.
    """

    def __init__(self, weights_dir: str, sensors: list[str], out_dim: int = FTP1_OUT_DIM,
                 img_size: int = FTP1_IMAGE_SIZE, freeze: bool = True):
        super().__init__()
        self.sensors = list(sensors)
        # ``ModuleDict`` keyed by sensor name; ids are resolved to names at dispatch time.
        self.encoders = nn.ModuleDict(
            {name: _ViTEncoder(img_size=img_size) for name in self.sensors}
        )
        self.trunk = _SharedTrunk(out_dim=FTP1_OUT_DIM)
        self.proj = nn.Linear(FTP1_OUT_DIM, out_dim) if out_dim != FTP1_OUT_DIM else nn.Identity()
        # The pretrained features have their own scale; the physical encoder mixes them with
        # nn.Embedding-initialised tokens, so normalise before handing them over. See the
        # backbone feature-scale audit in doc/results.md.
        self.norm = nn.LayerNorm(out_dim)

        self._load(weights_dir)

        self.frozen = freeze
        if freeze:
            for p in self.encoders.parameters():
                p.requires_grad_(False)
            for p in self.trunk.parameters():
                p.requires_grad_(False)
            self.encoders.eval()
            self.trunk.eval()
        else:
            logger.warning(
                "The FTP-1 tactile tower is unfrozen. Its per-sensor dispatch makes the set of "
                "parameters receiving gradients depend on which datasets are in the local "
                "batch, which desyncs ZeRO-2 across ranks (600 s NCCL timeout, not an error)."
            )

    def _load(self, weights_dir: str) -> None:
        from safetensors.torch import load_file

        shared_path = os.path.join(weights_dir, "hpt_tokenizer", "shared_image_chunk_encoder.safetensors")
        if not os.path.exists(shared_path):
            raise FileNotFoundError(
                f"{shared_path} not found. Fetch the FTP-1 tactile weights first, e.g.\n"
                "  huggingface-cli download MJJJJ1064/ftp1_v0426_50kstep "
                "--include 'hpt_tokenizer/*' --local-dir <dir>"
            )
        shared = load_file(shared_path)
        trunk_sd = {k[len("shared_chunk_encoder.") :]: v for k, v in shared.items()
                    if k.startswith("shared_chunk_encoder.")}
        trunk_sd["image_proj.weight"] = shared["image_proj.weight"]
        trunk_sd["image_proj.bias"] = shared["image_proj.bias"]
        self.trunk.load_state_dict(trunk_sd, strict=True)

        for name in self.sensors:
            if name not in FTP1_SENSOR_IDS:
                raise ValueError(f"Unknown FTP-1 tactile sensor {name!r}; valid: {FTP1_SENSOR_NAMES}")
            path = os.path.join(weights_dir, "hpt_tokenizer", f"{name}_image_224_224_3.safetensors")
            if not os.path.exists(path):
                raise FileNotFoundError(f"No FTP-1 tokenizer for sensor {name!r} at {path}")
            sd = load_file(path)
            sd = {k[len("vit_encoder.") :]: v for k, v in sd.items() if k.startswith("vit_encoder.")}
            # ``strict=True`` on purpose: a silently partial load would leave a randomly
            # initialised ViT that still produces plausible-looking features.
            self.encoders[name].load_state_dict(sd, strict=True)
        logger.info(
            "Loaded FTP-1 tactile tower: %d sensor tokenizer(s) %s + shared trunk.",
            len(self.sensors),
            self.sensors,
        )

    def train(self, mode: bool = True):  # noqa: D102
        super().train(mode)
        if self.frozen:
            # Keep the pretrained tower in eval mode regardless of the parent's mode. It has no
            # dropout or batch norm today, but this makes the guarantee explicit.
            self.encoders.eval()
            self.trunk.eval()
        return self

    def forward(self, images: torch.Tensor, sensor_ids: torch.Tensor,
                mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """``(N, F, 3, H, W)`` uint8 -> ``(N, F, out_dim)``.

        ``sensor_ids`` is ``(N,)``; ``mean``/``std`` are ``(N, 3)`` and already correspond to
        the dataset each row came from.
        """
        n, f = images.shape[:2]
        dtype = self.trunk.image_proj.weight.dtype
        x = images.reshape(n * f, *images.shape[2:]).float() / 255.0 * 2.0 - 1.0
        m = mean.repeat_interleave(f, dim=0).to(torch.float32).view(n * f, 3, 1, 1)
        s = std.repeat_interleave(f, dim=0).to(torch.float32).view(n * f, 3, 1, 1)
        x = ((x - m) / s.clamp_min(1e-6)).to(dtype)

        flat_ids = sensor_ids.repeat_interleave(f, dim=0)
        grid = next(iter(self.encoders.values())).grid
        tokens = x.new_zeros(n * f, grid * grid + 1, FTP1_EMBED_DIM)
        # Dispatch per sensor: each tokenizer only sees the rows it was trained on. Rows whose
        # id matches no loaded sensor keep their zeros and are masked out downstream anyway.
        for name in self.sensors:
            sel = (flat_ids == FTP1_SENSOR_IDS[name]).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            tokens = tokens.index_put((sel,), self.encoders[name](x[sel]).to(tokens.dtype))

        feats = self.trunk(tokens)
        return self.norm(self.proj(feats)).view(n, f, -1)
