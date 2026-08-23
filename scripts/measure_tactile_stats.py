"""Measure the per-channel z-score FTP-1 would have computed for a tactile dataset.

``lerobot/common/policies/ace/ftp1_tactile.py`` carries per-dataset normalisation statistics
transcribed from the FTP-1 checkpoint. A dataset that FTP-1 never trained on has no published
entry, so the loader falls back to an identity z-score and warns. This script produces the
missing numbers by measuring them the same way FTP-1 did, and additionally reports which of
the known FTP-1 sensors the measurement is closest to, which is how an unlabelled pad gets
identified.

The pipeline mirrors ``FTP1TactileTower.forward`` exactly:

    x = uint8 / 255 * 2 - 1

and the mean/std are taken per channel over that quantity, in the decoded frame's own colour
order (no BGR flip -- see the channel-order note in ``ftp1_tactile.py``).

Usage
-----
Measure a dataset and print a ready-to-paste ``FTP1_TACTILE_DATASETS`` entry::

    python scripts/measure_tactile_stats.py \
        --root "/media/v-wangxiaofa/新加卷/lerobot_data/OpenNeoData/aloha/aloha" \
        --name open_neo_aloha

Measure every tactile pad separately instead of pooling them (use this when the pads on a rig
might be different sensors)::

    python scripts/measure_tactile_stats.py --root ... --name ... --per-view

Point it at several datasets at once::

    python scripts/measure_tactile_stats.py \
        --root DIR1 --name NAME1 --root DIR2 --name NAME2

Statistics converge quickly; the defaults sample 400 episodes x 8 frames per key, which takes
well under a minute per dataset. Raise ``--episodes`` if the numbers look unstable (the script
reports the standard error so you can tell).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lerobot.common.policies.ace.ftp1_tactile import (  # noqa: E402
    FTP1_TACTILE_DATASETS,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


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
    return sorted((root / "videos").rglob(f"{key}/*.mp4")) or sorted(
        (root / "videos").rglob("*.mp4")
    )


def sample_frames(path: Path, n: int) -> np.ndarray | None:
    """``n`` frames spread across ``path``, as float32 in FTP-1's [-1, 1] range."""
    from torchcodec.decoders import VideoDecoder

    try:
        dec = VideoDecoder(str(path), device="cpu")
        total = dec.metadata.num_frames
        if not total:
            return None
        idx = sorted({int(i) for i in np.linspace(0, total - 1, min(n, total))})
        batch = dec.get_frames_at(indices=idx)
        arr = batch.data.numpy()  # (n, 3, H, W) uint8
    except Exception as exc:  # noqa: BLE001
        logger.debug("  skipping %s: %s", path.name, exc)
        return None
    return arr.astype(np.float32) / 255.0 * 2.0 - 1.0


class ChannelAccumulator:
    """Streaming per-channel sum / sum-of-squares, so nothing is held in memory."""

    def __init__(self) -> None:
        self.n = 0
        self.s = np.zeros(3, dtype=np.float64)
        self.ss = np.zeros(3, dtype=np.float64)
        self.frames = 0

    def add(self, x: np.ndarray) -> None:
        self.frames += x.shape[0]
        flat = x.transpose(1, 0, 2, 3).reshape(3, -1)
        self.n += flat.shape[1]
        # float64 accumulation is mandatory, not defensive. A float32 sum saturates once the
        # running total passes 2**24: summing 3e7 pixels that are all exactly -1.0 stops at
        # -16777216 and reports a mean of -0.557 for an all-black video.
        self.s += flat.sum(axis=1, dtype=np.float64)
        self.ss += np.square(flat, dtype=np.float64).sum(axis=1, dtype=np.float64)

    @property
    def mean(self) -> np.ndarray:
        return self.s / max(1, self.n)

    @property
    def std(self) -> np.ndarray:
        var = self.ss / max(1, self.n) - np.square(self.mean)
        return np.sqrt(np.clip(var, 0.0, None))

    @property
    def stderr(self) -> np.ndarray:
        """Standard error of the mean, treating each *frame* as the independent unit.

        Pixels inside one tactile frame are heavily correlated, so dividing by the pixel count
        would understate the error by orders of magnitude.
        """
        return self.std / math.sqrt(max(1, self.frames))


def nearest_known_sensor(mean: np.ndarray) -> list[tuple[float, str, str]]:
    """Rank published (dataset, sensor) statistics by distance to ``mean``."""
    out = []
    for ds, entry in FTP1_TACTILE_DATASETS.items():
        for sensor, st in entry["stats"].items():
            d = float(np.abs(np.asarray(st["mean"]) - mean).mean())
            out.append((d, ds, sensor))
    return sorted(out)


def measure(root: Path, episodes: int, frames: int, seed: int) -> dict[str, ChannelAccumulator]:
    info = json.loads((root / "meta" / "info.json").read_text())
    keys = tactile_image_keys(info)
    if not keys:
        logger.warning("  %s exposes no tactile image keys", root)
        return {}

    rng = random.Random(seed)
    accs: dict[str, ChannelAccumulator] = {}
    for key in keys:
        files = video_files(root, info, key)
        if not files:
            logger.warning("  no mp4 found for %s", key)
            continue
        picked = files if len(files) <= episodes else rng.sample(files, episodes)
        acc = ChannelAccumulator()
        for path in picked:
            x = sample_frames(path, frames)
            if x is not None:
                acc.add(x)
        if acc.frames:
            accs[key] = acc
            logger.info(
                "  %-52s %5d frames from %4d/%d file(s)",
                key.replace("observation.images.", ""),
                acc.frames,
                len(picked),
                len(files),
            )
    return accs


def fmt(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:.6f}" for x in v) + "]"


def report(name: str, accs: dict[str, ChannelAccumulator], per_view: bool) -> None:
    if not accs:
        return

    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")

    print("\n  per-pad measurement")
    for key, acc in accs.items():
        short = key.replace("observation.images.", "")
        print(
            f"    {short:<46} mean {fmt(acc.mean)}  ±{acc.stderr.max():.4f}\n"
            f"    {'':<46} std  {fmt(acc.std)}"
        )

    pooled = ChannelAccumulator()
    for acc in accs.values():
        pooled.n += acc.n
        pooled.s += acc.s
        pooled.ss += acc.ss
        pooled.frames += acc.frames

    spread = float(
        np.abs(np.stack([a.mean for a in accs.values()]) - pooled.mean).max()
    ) if len(accs) > 1 else 0.0
    print(f"\n  pooled  mean {fmt(pooled.mean)}   std {fmt(pooled.std)}")
    if len(accs) > 1:
        print(
            f"  max deviation of any single pad from the pooled mean: {spread:.4f}"
            f"  ({'pads differ -- consider --per-view' if spread > 0.05 else 'pads agree'})"
        )

    print("\n  closest published FTP-1 statistics (mean absolute channel difference)")
    for d, ds, sensor in nearest_known_sensor(pooled.mean)[:4]:
        print(f"    {d:.4f}   {ds}  /  {sensor}")

    print("\n  suggested FTP1_TACTILE_DATASETS entry")
    sensor = "<SENSOR>"
    if per_view:
        body = "\n".join(
            f'            # {k.replace("observation.images.", "")}'
            f'\n            "{sensor}_{i}": _stats({fmt(a.mean)}, {fmt(a.std)}),'
            for i, (k, a) in enumerate(accs.items())
        )
        print(f'    "{name}": {{\n        "sensors": [...],\n        "stats": {{\n{body}\n        }},\n    }},')
    else:
        print(
            f'    "{name}": {{\n'
            f'        "sensors": ["{sensor}"] * {len(accs)},\n'
            f'        "stats": {{"{sensor}": _stats({fmt(pooled.mean)}, {fmt(pooled.std)})}},\n'
            f"    }},"
        )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Measure FTP-1 style tactile z-score statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--root", type=Path, action="append", required=True,
                   help="dataset root (containing meta/ and videos/); repeatable")
    p.add_argument("--name", action="append", required=True,
                   help="registry name for the matching --root; repeatable")
    p.add_argument("--episodes", type=int, default=400, help="video files sampled per key")
    p.add_argument("--frames", type=int, default=8, help="frames sampled per video file")
    p.add_argument("--per-view", action="store_true",
                   help="report each pad separately instead of pooling them")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if len(args.root) != len(args.name):
        p.error("--root and --name must be given the same number of times")

    for root, name in zip(args.root, args.name, strict=True):
        if not (root / "meta" / "info.json").exists():
            logger.error("skipping %s: no meta/info.json", root)
            continue
        logger.info("\nreading %s", root)
        report(name, measure(root, args.episodes, args.frames, args.seed), args.per_view)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
