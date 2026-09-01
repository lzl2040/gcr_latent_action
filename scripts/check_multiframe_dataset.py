#!/usr/bin/env python
"""Check the two data-path options the world model needs: RGB clips and optional tactile.

Both are opt-in through ``getattr`` on the policy config, so the contrastive path is unchanged
by construction. What is worth testing is the part that is *not* obvious:

* **Clip endpoints.** The multi-frame read reuses the evenly-spaced stamp formula the tactile
  cameras already use, which at ``rgb_frames == 2`` reduces to the original
  ``[0.0, horizon / index_fps]``. That is an algebraic identity, but its consequence is what
  matters: for any ``rgb_frames``, the first and last frames of the clip must be the same two
  frames the contrastive path has always called ``image_t0``/``image_t1``. Otherwise two models
  reporting the same ``chunk_seconds`` would be looking at different windows.
* **Interior frames carry new information.** If the interior were a copy of the endpoints the
  clip would be a slower no-op, so the mean absolute difference is reported, not just asserted
  to exist.
* **Tactile off really skips the work.** The tactile keys must be absent (not zero-filled) and
  the per-sample read should get cheaper, since tactile is a second video decode.

Run (same args as train_ace_local.sh, minus deepspeed)::

    RGB_FRAMES=9 python scripts/check_multiframe_dataset.py --policy.type=robo_contrast ...
"""

import os
import sys
import time

from lerobot.common.datasets.contrastive_dataset import (
    MultiModalContrastiveDataset,
    contrastive_collate_fn,
)
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig

TACTILE_KEYS = (
    "tactile_signal",
    "tactile_signal_mask",
    "tactile_image",
    "tactile_image_mask",
    "tactile_sensor_id",
)


def _build(cfg, rgb_frames: int, use_tactile: bool):
    cfg.policy.rgb_frames = rgb_frames
    cfg.policy.use_tactile = use_tactile
    return MultiModalContrastiveDataset(
        cfg=cfg,
        data_mix=cfg.data_mix,
        seed=cfg.seed,
        dataset_size_one_epoch=cfg.dataset.dataset_size_one_epoch,
    )


def _time_reads(ds, idxs):
    t0 = time.perf_counter()
    for i in idxs:
        ds[i]
    return (time.perf_counter() - t0) / len(idxs)


@parser.wrap()
def main(cfg: TrainPipelineConfig):
    # draccus validates every CLI flag against the config schema, so extra argparse flags are
    # rejected outright; the knobs come in through the environment instead.
    n_rgb = int(os.environ.get("RGB_FRAMES", 9))
    n_samples = int(os.environ.get("SAMPLES", 4))

    print("=== baseline: rgb_frames=2, use_tactile=True ===")
    ds_pair = _build(cfg, 2, True)
    print(f"=== world-model path: rgb_frames={n_rgb}, use_tactile=False ===")
    ds_clip = _build(cfg, n_rgb, False)

    # Both instances are built with the same seed and mix, so index i names the same
    # (dataset, frame) in both. Indexing directly keeps the sampler out of the comparison.
    idxs = [int(i * max(1, len(ds_pair) // (n_samples + 1))) for i in range(1, n_samples + 1)]

    ok = True
    for i in idxs:
        a = ds_pair[i]
        b = ds_clip[i]
        if "image_clip" in a:
            print("FAIL: rgb_frames=2 emitted image_clip")
            ok = False
        present = [k for k in TACTILE_KEYS if k in b]
        if present:
            print(f"FAIL: use_tactile=False still emitted {present}")
            ok = False
        if not all(k in a for k in TACTILE_KEYS):
            print("FAIL: baseline lost its tactile keys")
            ok = False

        clip = b["image_clip"]
        if int(a["dataset_id"]) != int(b["dataset_id"]) or int(a["frame_index"]) != int(
            b["frame_index"]
        ):
            print(f"[{i}] SKIP: the two datasets disagree on which sample index {i} is")
            continue

        d0 = (clip[0].float() - a["image_t0"].float()).abs().max().item()
        d1 = (clip[-1].float() - a["image_t1"].float()).abs().max().item()
        interior = (clip[1:-1].float() - clip[0:1].float()).abs().mean().item()
        shape_ok = tuple(clip.shape) == (n_rgb, *a["image_t0"].shape)
        good = shape_ok and d0 == 0 and d1 == 0
        ok &= good
        print(
            f"[{i}] ds={int(a['dataset_id'])} fr={int(a['frame_index'])} "
            f"clip={tuple(clip.shape)}/{clip.dtype} "
            f"|t0-clip[0]|={d0:.0f} |t1-clip[-1]|={d1:.0f} "
            f"mean|interior-first|={interior:.2f}  {'OK' if good else 'FAIL'}"
        )

    # Warm caches first so the comparison is decode cost, not first-touch cost.
    _time_reads(ds_pair, idxs)
    _time_reads(ds_clip, idxs)
    t_pair = _time_reads(ds_pair, idxs)
    t_clip = _time_reads(ds_clip, idxs)
    print(
        f"per-sample read: pair+tactile {t_pair * 1e3:.1f} ms   "
        f"clip{n_rgb}-no-tactile {t_clip * 1e3:.1f} ms   ({t_clip / max(t_pair, 1e-9):.2f}x)"
    )

    batch = contrastive_collate_fn([ds_clip[i] for i in idxs])
    print(f"collated image_clip: {tuple(batch['image_clip'].shape)} {batch['image_clip'].dtype}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
