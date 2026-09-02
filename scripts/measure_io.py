"""Measure what the loader actually demands from storage, per clip.

The training benchmark reports ``data ~1 ms`` because 12 workers hide the decode behind the
GPU. That number is useless for sizing a *mounted* filesystem: it says the prefetch is deep
enough on a local NVMe, not how many bytes per second the job needs. On a network mount the
relevant quantities are bytes read per clip and the resulting aggregate bandwidth, which the
wall-clock in a prefetched pipeline never reveals.

Bytes are taken from /proc/self/io read_bytes, which counts actual block-device reads, so it
accounts for the real access pattern: decoding 9 frames of a 1.6 s window means seeking to a
keyframe and decoding forward, not reading 9 isolated frames.

num_workers=0 on purpose -- with workers the I/O happens in child processes and this process's
counters stay flat.

Run:  python -u scripts/measure_io.py --samples 64
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from lerobot.common.datasets.contrastive_dataset import (  # noqa: E402
    MultiModalContrastiveDataset,
    contrastive_collate_fn,
)
from lerobot.configs import parser  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402


def io_counters() -> tuple[int, int]:
    """(read_bytes, read_syscalls) for this process."""
    vals = {}
    for line in Path("/proc/self/io").read_text().splitlines():
        k, _, v = line.partition(": ")
        vals[k] = int(v)
    return vals["read_bytes"], vals["syscr"]


@parser.wrap()
def main(cfg: TrainPipelineConfig) -> None:
    n = int(os.environ.get("SAMPLES", "64"))
    batch = int(os.environ.get("BATCH", "8"))

    # Match the training config: 9 pixel frames per clip (-> 3 Wan latent frames), no tactile.
    cfg.policy.rgb_frames = int(os.environ.get("RGB_FRAMES", "9"))
    cfg.policy.use_tactile = os.environ.get("USE_TACTILE", "0") == "1"

    ds = MultiModalContrastiveDataset(cfg, data_mix=cfg.data_mix, seed=cfg.seed)
    loader = DataLoader(
        ds,
        batch_size=batch,
        num_workers=0,  # see module docstring
        collate_fn=contrastive_collate_fn,
        shuffle=True,
    )

    it = iter(loader)
    next(it)  # warm the file handles and codec so the first open is not charged to the mean

    b0, s0 = io_counters()
    t0 = time.perf_counter()
    seen = 0
    frames = 0
    while seen < n:
        item = next(it)
        seen += item["image_clip"].shape[0]
        frames += item["image_clip"].shape[0] * item["image_clip"].shape[1]
    dt = time.perf_counter() - t0
    b1, s1 = io_counters()

    mb = (b1 - b0) / 2**20
    print(f"samples             : {seen}")
    print(f"decoded frames      : {frames}  ({frames / seen:.0f} per clip)")
    print(f"wall clock          : {dt:.2f} s  -> {dt / seen * 1000:.1f} ms/clip (1 worker)")
    print(f"block-device reads  : {mb:.1f} MiB  -> {mb / seen * 1024:.0f} KiB/clip")
    print(f"read syscalls       : {s1 - s0}  ({(s1 - s0) / seen:.0f} per clip)")
    print(f"single-worker rate  : {mb / dt:.1f} MiB/s")
    print()
    print("Aggregate demand at a given per-GPU step rate (16 GPUs, batch 32/GPU):")
    kib = mb / seen * 1024
    for step_s, label in ((3.184, "A6000"), (1.038, "A100 eff .90"), (1.188, "A100 eff .75")):
        clips_s = 32 / step_s * 16
        print(f"  {label:<14s} step {step_s:.3f}s -> {clips_s:6.1f} clips/s "
              f"-> {clips_s * kib / 1024:7.1f} MiB/s aggregate")


if __name__ == "__main__":
    main()
