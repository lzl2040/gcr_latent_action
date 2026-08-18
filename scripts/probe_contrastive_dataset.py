#!/usr/bin/env python
"""Load one dataset through the contrastive pipeline and report where it breaks.

The training job builds the whole mixture before it touches a single sample, so a dataset
that fails costs several minutes and buries the cause under nine healthy datasets. This
runs the identical code path -- same root resolution, same canonical spec, same
`MultiModalContrastiveDataset` -- against one dataset, and prints the stage each failure
belongs to instead of a bare traceback.

    python scripts/probe_contrastive_dataset.py --dataset agibot_alpha \
        --parent_dir_extra "/media/v-wangxiaofa/新加卷/lerobot_data"

Stages, reported in order:
  1. root      -- which directory the name resolves to, via the same `_resolve_root`
  2. meta      -- codebase version, fps, feature shapes from meta/info.json
  3. parquet   -- the stored column types against the ones info.json declares
  4. spec      -- the canonical-space mapping, and which of the 40 slots it fills
  5. build     -- constructing the dataset (delta_timestamps, video keys, norm stats)
  6. samples   -- pulling frames and checking shapes, dtypes and masks
  7. collate   -- the batch the model would actually receive
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import traceback
from dataclasses import dataclass, field

import numpy as np
import torch

# Run from anywhere: vla2root.json and the package are both resolved from the repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from lerobot.common.datasets.canonical_space import get_spec  # noqa: E402
from lerobot.common.datasets.contrastive_dataset import (  # noqa: E402
    MultiModalContrastiveDataset,
    contrastive_collate_fn,
)
from lerobot.common.datasets.mixtures import OXE_NAMED_MIXTURES  # noqa: E402
from lerobot.common.policies.ace.configuration_robo_contrast import RoboContrastConfig  # noqa: E402
from lerobot.configs.default import DatasetConfig  # noqa: E402

logger = logging.getLogger("probe")


@dataclass
class _Cfg:
    """The subset of TrainPipelineConfig the dataset actually reads."""

    policy: RoboContrastConfig = field(default_factory=RoboContrastConfig)
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(repo_id="probe"))


def _stage(name: str) -> None:
    print(f"\n{'=' * 78}\n[{name}]\n{'=' * 78}")


def _fail(stage: str, exc: BaseException) -> int:
    print(f"\n>>> FAILED at stage: {stage}")
    print(f">>> {type(exc).__name__}: {exc}\n")
    traceback.print_exc()
    return 1


def _describe(value, indent: str = "    ") -> str:
    if isinstance(value, torch.Tensor):
        text = f"{tuple(value.shape)} {value.dtype}"
        if value.numel() and value.is_floating_point():
            finite = torch.isfinite(value)
            text += f" range=[{value[finite].min():.4g}, {value[finite].max():.4g}]"
            if not bool(finite.all()):
                text += f" !! {int((~finite).sum())} non-finite"
        elif value.numel():
            text += f" range=[{value.min()}, {value.max()}]"
        return text
    if isinstance(value, np.ndarray):
        return f"ndarray{value.shape} {value.dtype}"
    if isinstance(value, str):
        return f"str {value!r}"
    return f"{type(value).__name__} {value}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="dataset name as it appears in vla2root.json")
    parser.add_argument("--parent_dir_v21", default="/Data/lerobot_data_ort6d")
    parser.add_argument("--parent_dir_v30", default="/Data/lerobot_data_ort6d/v30")
    parser.add_argument("--parent_dir_extra", default="", help="comma-separated extra roots")
    parser.add_argument("--video_backend", default="torchcodec")
    parser.add_argument("--samples", type=int, default=4, help="how many frames to pull")
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--chunk_seconds", type=float, default=None)
    parser.add_argument("--window_mode", default=None, choices=["duration", "frames"])
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--verbose", action="store_true", help="show library INFO logs")
    parser.add_argument(
        "--debug", action="store_true", help="show library DEBUG logs, including the traceback "
        "behind a swallowed 'Failed to open ...' warning"
    )
    args = parser.parse_args()

    level = logging.DEBUG if args.debug else logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="[%(levelname)s] %(name)s - %(message)s")
    # The dataset code logs the resolved window and every skip decision at INFO, and the cause
    # of a swallowed load failure at DEBUG; both are exactly what is worth seeing here.
    logging.getLogger("lerobot").setLevel(level)

    os.chdir(REPO_ROOT)  # vla2root.json is read by relative path

    cfg = _Cfg()
    cfg.dataset.parent_dir_v21 = args.parent_dir_v21
    cfg.dataset.parent_dir_v30 = args.parent_dir_v30
    cfg.dataset.parent_dir_extra = args.parent_dir_extra
    cfg.dataset.video_backend = args.video_backend
    for name in ("chunk_size", "group_size", "chunk_seconds", "window_mode"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg.policy, name, value)
    cfg.policy.__post_init__()

    print(f"dataset      : {args.dataset}")
    print(f"chunk_size   : {cfg.policy.chunk_size}  group_size={cfg.policy.group_size}")
    print(f"window_mode  : {cfg.policy.window_mode}  chunk_seconds={cfg.policy.chunk_seconds}")

    # -- 1. root -----------------------------------------------------------------
    _stage("1. root")
    try:
        with open("vla2root.json") as f:
            vla2root = json.load(f)
        if args.dataset not in vla2root:
            print(f"'{args.dataset}' is not in vla2root.json.")
            print("Datasets that are: " + ", ".join(sorted(vla2root)[:20]) + " ...")
            return 1
        relative = vla2root[args.dataset]
        print(f"vla2root.json entry : {relative}")
        root = MultiModalContrastiveDataset._resolve_root(cfg, relative)
        if root is None:
            print("\n>>> FAILED at stage: root -- no candidate contained meta/info.json.")
            print("    Searched:")
            extra = [p for p in (cfg.dataset.parent_dir_extra or "").split(",") if p.strip()]
            for parent in [cfg.dataset.parent_dir_v21, cfg.dataset.parent_dir_v30, *extra]:
                candidate = os.path.join(parent.strip(), relative)
                print(f"      {candidate}  exists={os.path.exists(candidate)}")
            return 1
        print(f"resolved root       : {root}")
        for sub in ("meta", "data", "videos"):
            path = os.path.join(root, sub)
            print(f"  {sub:7s} exists={os.path.isdir(path)}")
    except Exception as exc:
        return _fail("root", exc)

    # -- 2. meta -----------------------------------------------------------------
    _stage("2. meta")
    try:
        with open(os.path.join(root, "meta", "info.json")) as f:
            info = json.load(f)
        features = info.get("features", {})
        print(f"codebase_version : {info.get('codebase_version', 'v2.1 (absent)')}")
        print(f"declared fps     : {info.get('fps')}")
        print(f"episodes / frames: {info.get('total_episodes')} / {info.get('total_frames')}")
        print(f"chunks_size      : {info.get('chunks_size')}")
        print("\nfeatures:")
        for key in sorted(features):
            shape = features[key].get("shape")
            print(f"  {key:45s} {features[key].get('dtype', '?'):10s} shape={shape}")
        for name in ("meta/episodes.jsonl", "meta/tasks.jsonl", "meta/stats.json", "meta/episodes_stats.jsonl"):
            print(f"  {name:45s} exists={os.path.exists(os.path.join(root, name))}")
    except Exception as exc:
        return _fail("meta", exc)

    # -- 3. parquet schema -------------------------------------------------------
    _stage("3. parquet schema vs info.json")
    try:
        import glob

        import pyarrow.parquet as pq

        paths = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
        print(f"parquet files: {len(paths)}")
        if not paths:
            print("!! no parquet under data/ -- the dataset cannot be read")
            return 1
        schema = pq.ParquetFile(paths[0]).schema_arrow
        # `datasets` casts every column to the type declared in info.json and refuses
        # element-type mismatches outright, reporting only "An error occurred while generating
        # the dataset". Comparing the two here names the offending column instead.
        expected_element = {
            "float32": "float",
            "float64": "double",
            "int64": "int64",
            "int32": "int32",
            "bool": "bool",
            "string": "string",
        }
        mismatches = []
        for key in sorted(features):
            declared = features[key].get("dtype", "?")
            if declared in ("video", "image"):
                continue
            if key not in schema.names:
                mismatches.append((key, declared, "ABSENT from parquet"))
                continue
            actual = str(schema.field(key).type)
            want = expected_element.get(declared)
            # Only the *element* type matters. A column may be stored as a bare scalar, a
            # `list<element: float>` or a `fixed_size_list<element: float>[16]`; all three are
            # float32 and all three are accepted. It is `double` under a float32 declaration
            # that `datasets` refuses.
            match = re.fullmatch(r"(?:fixed_size_)?list<element: (.+?)>(?:\[\d+\])?", actual)
            element = match.group(1) if match else actual
            ok = want is not None and element == want
            print(f"  {key:35s} info={declared:8s} parquet={actual:30s} {'ok' if ok else '<-- MISMATCH'}")
            if not ok:
                mismatches.append((key, declared, actual))
        if mismatches:
            print("\n>>> FAILED at stage: parquet schema")
            print("    `datasets` will refuse to cast these columns:")
            for key, declared, actual in mismatches:
                print(f"      {key}: info.json says {declared}, parquet holds {actual}")
            print("\n    Either the parquet was written with the wrong precision, or info.json")
            print("    mislabels it. Both produce the opaque error above at the build stage.")
            return 1
    except ImportError:
        print("pyarrow unavailable, skipping this check")
    except Exception as exc:
        return _fail("parquet schema", exc)

    # -- 4. spec -----------------------------------------------------------------
    _stage("4. canonical spec")
    try:

        def _dim(kind: str) -> int:
            shape = features.get(kind, {}).get("shape") or [0]
            return int(np.prod(shape))

        spec = get_spec(args.dataset, action_dim=_dim("action"), state_dim=_dim("observation.state"))
        if spec is None:
            print(">>> get_spec returned None: the dataset has no canonical mapping.")
            return 1
        for kind in ("action", "state"):
            entries = spec.get(kind, [])
            slots: set[int] = set()
            print(f"\n{kind} ({len(entries)} rule(s)):")
            for entry in entries:
                print(f"  {entry}")
                target = entry.get("canonical") if isinstance(entry, dict) else None
                if isinstance(target, (list, tuple)):
                    slots.update(int(s) for s in target)
                elif isinstance(target, slice):
                    slots.update(range(*target.indices(40)))
            if slots:
                print(f"  -> fills {len(slots)} canonical slot(s): {sorted(slots)}")
        tactile = {k: v for k, v in spec.items() if "tactile" in k}
        print(f"\ntactile keys in spec: {tactile if tactile else 'none'}")
    except Exception as exc:
        return _fail("spec", exc)

    # -- 5. build ----------------------------------------------------------------
    _stage("5. build dataset")
    try:
        # A private one-entry mixture, so the probe exercises the real constructor without
        # depending on the dataset being a member of any published mixture.
        mix_name = f"__probe_{args.dataset}"
        OXE_NAMED_MIXTURES[mix_name] = [(args.dataset, 1.0)]
        dataset = MultiModalContrastiveDataset(cfg, data_mix=mix_name, seed=args.seed)
        if not dataset.datasets:
            print(">>> The dataset was skipped during construction (see the warning above).")
            if not args.debug:
                print(">>> Re-run with --debug to print the traceback behind that warning.")
            return 1
        print(f"length            : {len(dataset)}")
        print(f"sub-dataset size  : {dataset.dataset_sizes[0]}")
        print(f"frame horizon     : {dataset.frame_horizons[0]} frames")
        print(f"true fps          : {dataset.true_fps[0]}")
        print(f"image keys        : {dataset.image_key_maps[0]}")
        print(f"episodes          : {len(dataset.episode_ranges[0])}")
    except Exception as exc:
        return _fail("build", exc)

    # -- 6. samples --------------------------------------------------------------
    _stage("6. samples")
    samples = []
    try:
        rng = np.random.default_rng(args.seed)
        indices = rng.integers(0, len(dataset), size=args.samples)
        for n, index in enumerate(indices):
            item = dataset[int(index)]
            samples.append(item)
            print(f"\n--- sample {n} (index {index}) ---")
            for key in sorted(item):
                print(f"  {key:26s} {_describe(item[key])}")
    except Exception as exc:
        return _fail("samples", exc)

    # -- 7. collate --------------------------------------------------------------
    _stage("7. collate")
    try:
        batch = contrastive_collate_fn(samples)
        for key in sorted(batch):
            print(f"  {key:26s} {_describe(batch[key])}")
    except Exception as exc:
        return _fail("collate", exc)

    print(f"\n{'=' * 78}\nOK: {args.dataset} loaded and produced a batch.\n{'=' * 78}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
