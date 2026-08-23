"""Measure how much signal the tactile reconstruction objective actually carries.

``PhysicalEncoder._tactile_recon_loss`` trains the tactile encoder by reconstructing the gel
image. Whether that teaches anything depends on two quantities this script measures:

1. **The floor.** The MSE a decoder reaches by emitting one fixed image, i.e. the variance of
   the target around its mean. The reconstruction objective can never do better, and the
   gradient it produces is proportional to how far above this floor the loss sits. Gel images
   occupy a narrow slice of [0, 1], so in raw pixel space this floor is ~0.003-0.015 -- small
   enough that the term is invisible next to a contrastive loss of ~7. Z-scoring the target
   per dataset lifts it to ~1, which is why ``_tactile_recon_loss`` does that.

2. **How much of the floor is contact rather than appearance.** Subtracting each episode's own
   mean image instead of the global one removes everything explained by "which episode is
   this" (gel wear, lighting, sensor identity). What survives can only come from the contact
   itself. If that residual were small, the objective would mostly be teaching the encoder to
   recognise episodes, and raising its weight would make things worse rather than better.

Both are reported at the size the loss actually uses (``tactile_recon_size``, default 28),
after the same 112 -> 28 bilinear path as the model.

Usage
-----
Check one dataset::

    python scripts/tactile_recon_floor.py --root /Data/lerobot_data_ort6d/v30/FTP-1/sharpa

Several at once, at the resolution a different config would use::

    python scripts/tactile_recon_floor.py --root DIR1 --root DIR2 --recon-size 56

Pass ``--name`` alongside ``--root`` to also report the floor under the per-dataset z-score
that dataset's registry entry would apply, which is what the training loss sees::

    python scripts/tactile_recon_floor.py --root .../sharpa --name ftp_1_sharpa
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# The dataloader resizes tactile pads to `config.tactile_img_size` before the encoder sees
# them, and the loss downsamples from there. Reproduce both hops so the numbers match training.
TACTILE_IMG_SIZE = 112


def tactile_image_keys(info: dict) -> list[str]:
    return sorted(
        k
        for k, v in info["features"].items()
        if "tactile" in k and v.get("dtype") in ("video", "image")
    )


def video_files(root: Path, info: dict, key: str) -> list[Path]:
    """Every mp4 belonging to ``key``, for both the v2.1 and v3.0 layouts."""
    template = info["video_path"]
    if "{video_key}" in template and "file_index" in template:  # v3.0
        return sorted((root / "videos" / key).rglob("*.mp4"))
    return sorted((root / "videos").rglob(f"{key}/*.mp4"))


def episode_frames(path: Path, n: int, size: int) -> torch.Tensor | None:
    """``n`` frames spread over the whole episode, as float [0, 1] at ``size``.

    Spreading matters. The opening of an episode is pre-contact idle where the gel is
    literally constant -- sampling sharpa's first 96 frames gives a pixel std of 3e-4 -- so a
    prefix would measure the idle period rather than the task.
    """
    from torchcodec.decoders import VideoDecoder

    try:
        dec = VideoDecoder(str(path), device="cpu")
        total = dec.metadata.num_frames
        if not total:
            return None
        idx = sorted({int(i) for i in np.linspace(0, total - 1, min(n, total))})
        x = dec.get_frames_at(indices=idx).data.float() / 255.0
    except Exception as exc:  # noqa: BLE001
        logger.debug("  skipping %s: %s", path.name, exc)
        return None
    x = F.interpolate(x, size=(TACTILE_IMG_SIZE, TACTILE_IMG_SIZE), mode="bilinear", align_corners=False)
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


def registry_stats(name: str, num_views: int) -> tuple[np.ndarray, np.ndarray] | None:
    """The [0, 1]-space z-score ``_tactile_recon_loss`` would apply to view 0 of ``name``."""
    from lerobot.common.policies.ace.ftp1_tactile import tactile_image_sensors

    _, means, stds = tactile_image_sensors(name, num_views)
    # The registry is in FTP-1's [-1, 1] convention; the loss shifts it to [0, 1].
    return (np.asarray(means[0]) + 1.0) * 0.5, np.asarray(stds[0]) * 0.5


def report(root: Path, name: str | None, size: int, episodes: int, frames: int) -> None:
    info = json.loads((root / "meta" / "info.json").read_text())
    keys = tactile_image_keys(info)
    if not keys:
        logger.info("%s: no tactile image keys", root.name)
        return
    logger.info("\n=== %s === (recon target %dx%d)", name or root.name, size, size)

    for key in keys:
        per_ep = []
        for f in video_files(root, info, key)[:episodes]:
            x = episode_frames(f, frames, size)
            if x is not None:
                per_ep.append(x)
        if not per_ep:
            logger.info("  %-46s no readable video", key)
            continue

        allx = torch.cat(per_ep)
        floor_global = ((allx - allx.mean(0, keepdim=True)) ** 2).mean().item()
        within = torch.cat([x - x.mean(0, keepdim=True) for x in per_ep])
        floor_episode = (within**2).mean().item()
        contact = 100 * floor_episode / max(floor_global, 1e-12)

        logger.info(
            "  %-46s n=%d eps=%d\n"
            "      raw [0,1]      floor %.5f   (of which %.0f%% is contact, not appearance)",
            key,
            len(allx),
            len(per_ep),
            floor_global,
            contact,
        )
        if name is not None:
            m, s = registry_stats(name, len(keys))
            z = (allx - torch.tensor(m, dtype=torch.float32).view(1, 3, 1, 1)) / torch.tensor(
                s, dtype=torch.float32
            ).view(1, 3, 1, 1)
            zf = ((z - z.mean(0, keepdim=True)) ** 2).mean().item()
            logger.info(
                "      z-scored       floor %.5f   (%.0fx the raw gradient scale)",
                zf,
                zf / max(floor_global, 1e-12),
            )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", action="append", required=True, type=Path, help="dataset root (repeatable)")
    p.add_argument("--name", action="append", default=[], help="registry name for the matching --root")
    p.add_argument("--recon-size", type=int, default=28, help="config.tactile_recon_size")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--frames", type=int, default=24, help="frames per episode, spread over the whole episode")
    args = p.parse_args()

    names = list(args.name) + [None] * (len(args.root) - len(args.name))
    for root, name in zip(args.root, names):
        if not (root / "meta" / "info.json").exists():
            logger.warning("%s: no meta/info.json, skipping", root)
            continue
        report(root, name, args.recon_size, args.episodes, args.frames)


if __name__ == "__main__":
    main()
