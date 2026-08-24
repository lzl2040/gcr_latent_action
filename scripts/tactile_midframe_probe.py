"""Does a tactile camera window need more than its two endpoints?

The contrastive loader reads each tactile pad at ``t`` and ``t + horizon`` only, and folds the
pair into one token as ``[feat_t, feat_t1 - feat_t]``. ``doc/results.md`` §9.1 records that
this was chosen for **cost, not principle**, and states the risk plainly: a contact transient
is a few frames wide, so two samples at the ends of the window can *straddle* it and see
almost nothing -- a failure that looks like a successful read.

Reading four frames instead of two doubles the decode load on the tactile video path, which
§8 identifies as the throughput constraint of the whole pipeline. That is only worth paying if
the interior of the window actually carries something the endpoints do not. This script
measures whether it does, on real windows, before anything is built.

Two statistics per pad, both on the raw 0-255 pixel scale:

``lerp_ratio``
    ``mean|mid_k - lerp(t0, t1, a_k)| / mean|t1 - t0|``. How much of each interior frame is
    *not* explained by linearly interpolating the endpoints, relative to the endpoint change
    the current design already sees. Near 0 means the interior is redundant: the endpoints
    plus a straight line reconstruct it, so four frames would buy nothing. Near or above 1
    means the interior moves off the line joining the endpoints -- real, unseen structure.

``straddle``
    ``mean_k min(|mid_k - t0|, |mid_k - t1|) / mean|t1 - t0|``. Distance from the interior to
    its *nearer* endpoint, in units of the endpoint separation. Above 1 is the failure §9.1
    warned about: the interior is further from both endpoints than they are from each other,
    i.e. an event happened and finished inside the window and neither endpoint saw it.

``straddle_frac``
    Fraction of windows whose ``straddle`` exceeds 1. A mean can be carried by a few windows;
    this says how often it happens.

``noise``
    ``mean|x(mid+1) - x(mid)| / mean|t1 - t0|``. The adjacent-frame difference, i.e. how much
    the pad changes over one frame. This is the control: a noisy pad produces a large
    ``lerp_ratio`` for free, because no smooth predictor can fit noise. ``lerp_ratio`` is only
    evidence of real unseen dynamics if it stands well above ``noise``.

``rcorr``
    Pearson correlation between the interpolation residual at ``mid`` and at ``mid+1``.
    Independent sensor noise decorrelates in one frame, so this sits near 0; a genuine
    deformation the endpoints missed persists across adjacent frames and drives it towards 1.
    This is the statistic that separates "the interior is unpredictable" from "the interior is
    unpredictable *and* structured", and only the latter is learnable.

Dead pads are skipped with the same rule the loader uses (``tactile_dead_std``, spatial std
per channel on a [0, 1] scale), because a constant pad is trivially "redundant" and would
dilute every average towards 0.

Windows are drawn from the interior of each episode. Episodes open with a pre-contact idle
stretch -- sharpa's first 96 frames have a pixel std of 3e-4 -- so sampling a prefix measures
the idle period and reports that nothing ever moves. Same trap as
``scripts/tactile_feature_probe.py``; see its docstring.

Usage
-----
    python scripts/tactile_midframe_probe.py \
        --dataset ftp_1_sharpa=/Data/lerobot_data_ort6d/v30/FTP-1/sharpa \
        --dataset ftp_1_VisuoTactile_D-WHEEL=/Data/.../VisuoTactile_D-WHEEL

Also reports wall-clock decode time for 2 vs 4 frames per window, which is the cost side of
the same decision.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def tactile_image_keys(info: dict) -> list[str]:
    return sorted(
        k
        for k, v in info["features"].items()
        if "tactile" in k and v.get("dtype") in ("video", "image")
    )


def video_files(root: Path, info: dict, key: str) -> list[Path]:
    template = info.get("video_path", "")
    if "{video_key}" in template and "file_index" in template:
        return sorted((root / "videos" / key).rglob("*.mp4"))
    return sorted((root / "videos").rglob(f"{key}/*.mp4"))


def resolve_horizon(fps: float, chunk_seconds: float, lo: int, hi: int) -> int:
    """The loader's duration-based window, mirrored from ``resolve_pair_horizon``."""
    return int(np.clip(round(chunk_seconds * fps), lo, hi))


def is_dead(frame: torch.Tensor, dead_std: float) -> bool:
    """``frame``: (3, H, W) on 0-255. Spatial std per channel, matching the loader's test."""
    return float((frame / 255.0).reshape(3, -1).std(dim=-1).max()) < dead_std


def probe_key(
    path: Path,
    horizon: int,
    n_windows: int,
    dead_std: float,
    rng: np.random.Generator,
) -> tuple[list[float], list[float], list[float], list[float], int, int]:
    from torchcodec.decoders import VideoDecoder

    lerps: list[float] = []
    straddles: list[float] = []
    noises: list[float] = []
    rcorrs: list[float] = []
    live = dead = 0
    try:
        dec = VideoDecoder(str(path), device="cpu")
        total = dec.metadata.num_frames
    except Exception as exc:  # noqa: BLE001
        logger.debug("  skipping %s: %s", path.name, exc)
        return lerps, straddles, noises, rcorrs, live, dead
    if total is None or total < horizon + 4:
        return lerps, straddles, noises, rcorrs, live, dead

    # Interior only: never start in the opening idle stretch, never run past the end.
    lo = max(1, int(0.1 * total))
    hi = total - horizon - 3
    if hi <= lo:
        return lerps, straddles, noises, rcorrs, live, dead
    starts = rng.integers(lo, hi, size=min(n_windows, hi - lo))

    for t0 in starts:
        m1, m2 = int(t0 + horizon / 3), int(t0 + 2 * horizon / 3)
        base = [int(t0), m1, m2, int(t0 + horizon)]
        if len(set(base)) < 4:
            continue
        # m1 + 1 and m2 + 1 give the one-frame noise floor and the residual correlation.
        idx = sorted(set(base + [m1 + 1, m2 + 1]))
        try:
            x = dec.get_frames_at(indices=idx).data.float()  # (N, 3, H, W) on 0-255
        except Exception:  # noqa: BLE001
            continue
        at = {j: i for i, j in enumerate(idx)}
        f0, f1 = x[at[base[0]]], x[at[base[3]]]
        # The loader drops a pad only when *both* endpoints are flat; match that exactly.
        if is_dead(f0, dead_std) and is_dead(f1, dead_std):
            dead += 1
            continue
        live += 1

        d_end = float((f1 - f0).abs().mean())
        if d_end < 1e-6:
            continue
        span = base[3] - base[0]

        def residual(j: int) -> torch.Tensor:
            a = (j - base[0]) / span
            return x[at[j]] - (1 - a) * f0 - a * f1

        res, near, noise, corr = [], [], [], []
        for m in (m1, m2):
            r_m, r_next = residual(m), residual(m + 1)
            res.append(float(r_m.abs().mean()))
            near.append(min(float((x[at[m]] - f0).abs().mean()), float((x[at[m]] - f1).abs().mean())))
            noise.append(float((x[at[m + 1]] - x[at[m]]).abs().mean()))
            a, b = r_m.flatten(), r_next.flatten()
            a, b = a - a.mean(), b - b.mean()
            denom = float(a.norm() * b.norm())
            if denom > 1e-9:
                corr.append(float((a @ b) / denom))
        lerps.append(float(np.mean(res)) / d_end)
        straddles.append(float(np.mean(near)) / d_end)
        noises.append(float(np.mean(noise)) / d_end)
        if corr:
            rcorrs.append(float(np.mean(corr)))
    return lerps, straddles, noises, rcorrs, live, dead


def decode_cost(path: Path, horizon: int, n: int, rng: np.random.Generator) -> dict[str, float]:
    """Seconds per window for 2- vs 4-frame reads, measured cold and warm.

    Both arms matter and they answer different questions. *Cold* opens a new decoder per
    window, which is what happens on a cache miss. *Warm* reuses one decoder, which is what
    happens on a hit -- the loader keeps a decoder cache (``LEROBOT_VIDEO_DECODER_CACHE_SIZE``),
    so the marginal cost of two extra frames is only visible once decoder construction is not
    dominating the measurement.
    """
    from torchcodec.decoders import VideoDecoder

    dec = VideoDecoder(str(path), device="cpu")
    total = dec.metadata.num_frames
    hi = total - horizon - 1
    if hi <= 1:
        return {}
    starts = rng.integers(1, hi, size=n)

    def indices(t0: int, k: int) -> list[int]:
        if k == 2:
            return [int(t0), int(t0 + horizon)]
        return sorted({int(t0 + horizon * a / 3) for a in range(4)})

    def run(k: int, warm: bool) -> float:
        t = time.perf_counter()
        for t0 in starts:
            try:
                d = dec if warm else VideoDecoder(str(path), device="cpu")
                d.get_frames_at(indices=indices(t0, k))
            except Exception:  # noqa: BLE001
                pass
        return (time.perf_counter() - t) / len(starts)

    run(2, True)  # discard: first pass warms page cache, and only the ratio is reported
    return {
        "cold2": run(2, False), "cold4": run(4, False),
        "warm2": run(2, True), "warm4": run(4, True),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", required=True, metavar="NAME=ROOT")
    ap.add_argument("--episodes", type=int, default=12, help="video files per tactile key")
    ap.add_argument("--windows", type=int, default=24, help="windows per video file")
    ap.add_argument("--chunk-seconds", type=float, default=1.6)
    ap.add_argument("--chunk-frames-min", type=int, default=8)
    ap.add_argument("--chunk-frames-max", type=int, default=48)
    ap.add_argument("--dead-std", type=float, default=0.002)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cost", action="store_true", help="also time 2-frame vs 4-frame decoding")
    args = ap.parse_args()

    rows = []
    for spec in args.dataset:
        name, _, root_s = spec.partition("=")
        root = Path(root_s)
        info = json.loads((root / "meta" / "info.json").read_text())
        fps = float(info.get("fps", 30))
        horizon = resolve_horizon(fps, args.chunk_seconds, args.chunk_frames_min, args.chunk_frames_max)
        keys = tactile_image_keys(info)
        logger.info("\n=== %s  fps=%.1f horizon=%d frames  pads=%d", name, fps, horizon, len(keys))

        for key in keys:
            rng = np.random.default_rng(args.seed)
            files = video_files(root, info, key)[: args.episodes]
            if not files:
                logger.info("  %-42s no video files", key.split(".")[-1])
                continue
            L, S, N, R, live, dead = [], [], [], [], 0, 0
            for f in files:
                l_, s_, n_, r_, li, de = probe_key(f, horizon, args.windows, args.dead_std, rng)
                L += l_
                S += s_
                N += n_
                R += r_
                live += li
                dead += de
            if not L:
                logger.info(
                    "  %-42s all dead (%d windows)", key.split(".")[-1], dead
                )
                rows.append((name, key.split(".")[-1], *[float("nan")] * 5, live, dead))
                continue
            frac = float(np.mean(np.asarray(S) > 1.0))
            lerp, strad = float(np.mean(L)), float(np.mean(S))
            noise = float(np.mean(N)) if N else float("nan")
            rcorr = float(np.mean(R)) if R else float("nan")
            logger.info(
                "  %-38s lerp %.3f  noise %.3f  rcorr %+.3f  straddle %.3f  frac %.2f  live/dead %d/%d",
                key.split(".")[-1], lerp, noise, rcorr, strad, frac, live, dead,
            )
            rows.append((name, key.split(".")[-1], lerp, strad, frac, noise, rcorr, live, dead))

        if args.cost and keys:
            files = video_files(root, info, keys[0])
            if files:
                rng = np.random.default_rng(args.seed)
                c = decode_cost(files[0], horizon, 40, rng)
                if c:
                    logger.info(
                        "  decode/window  cold: %.1f -> %.1f ms (%.2fx)   warm: %.1f -> %.1f ms (%.2fx)",
                        c["cold2"] * 1e3, c["cold4"] * 1e3, c["cold4"] / max(c["cold2"], 1e-9),
                        c["warm2"] * 1e3, c["warm4"] * 1e3, c["warm4"] / max(c["warm2"], 1e-9),
                    )

    live_rows = [r for r in rows if not np.isnan(r[2])]
    if live_rows:
        logger.info(
            "\nOVERALL over %d live pads: lerp %.3f  noise %.3f  rcorr %+.3f  straddle %.3f  frac %.2f",
            len(live_rows),
            float(np.mean([r[2] for r in live_rows])),
            float(np.nanmean([r[5] for r in live_rows])),
            float(np.nanmean([r[6] for r in live_rows])),
            float(np.mean([r[3] for r in live_rows])),
            float(np.mean([r[4] for r in live_rows])),
        )
        logger.info(
            "read: lerp >> noise and rcorr >> 0 => the interior holds structured motion the "
            "endpoints cannot reconstruct, so extra frames are worth their decode cost. "
            "lerp ~ noise or rcorr ~ 0 => the residual is sensor noise and extra frames buy nothing. "
            "straddle > 1 => an event starts and ends inside the window and neither endpoint sees it."
        )


if __name__ == "__main__":
    main()
