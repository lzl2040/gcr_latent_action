"""Multi-modal, multi-dataset loader for perception <-> physical contrastive learning.

This loader produces, for every sample:

Perception side
    * ``image_t0`` / ``image_t1``  : the primary camera at time ``t`` and ``t + horizon``
    * ``task``                     : the language instruction
    * ``pair_is_valid``            : 0 when ``t + horizon`` had to be clamped inside the episode

Physical side
    * ``action`` / ``action_mask`` : action chunk in the canonical 40-dim slotted space
    * ``observation.state`` / ``state_mask`` : state trajectory over the same chunk
    * ``tactile_signal`` / ``tactile_signal_mask`` : low-dimensional tactile readings at t, t+H
    * ``tactile_image``  / ``tactile_image_mask``  : tactile camera views at t, t+H

Heterogeneity is handled by :mod:`lerobot.common.datasets.canonical_space`: every dataset is
mapped into the same slotted vector plus a validity mask, so datasets that only expose joint
positions, only end-effector poses, or both, all coexist without silently overlapping.

Indexing supports two forms:
    * ``dataset[i]``                    -> the ``i``-th entry of the per-epoch sampling plan
    * ``dataset[(ds_idx, frame_idx)]``  -> an explicit sample, used by the contrastive sampler
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from tabulate import tabulate

from lerobot.common.datasets.dataset_fps import resolve_true_fps
from lerobot.common.datasets.canonical_space import (
    CANON_DIM,
    MAX_TACTILE_SIGNAL_DIM,
    MAX_TACTILE_VIEWS,
    get_spec,
    tactile_image_keys,
    tactile_signal_keys,
)
from lerobot.common.datasets.lerobot_dataset_for_ace import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    resolve_delta_timestamps,
)
from lerobot.common.datasets.mixtures import OXE_NAMED_MIXTURES
from lerobot.common.datasets.oxe_configs import OXE_DATASET_CONFIGS
from lerobot.common.datasets_v30.dataset_metadata import (
    LeRobotDatasetMetadata as LeRobotDatasetMetadataV30,
)
from lerobot.common.datasets_v30.lerobot_dataset import LeRobotDataset as LeRobotDatasetV30
from lerobot.common.policies.ace.ftp1_tactile import tactile_image_sensors

logger = logging.getLogger(__name__)


def _to_tensor(value) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value
    try:
        return torch.as_tensor(value)
    except Exception:  # noqa: BLE001 - defensive: heterogeneous raw dataset payloads
        return None


class MultiModalContrastiveDataset(torch.utils.data.Dataset):
    """Mixture of LeRobot datasets emitting aligned perception/physical modalities."""

    def __init__(
        self,
        cfg,
        image_transforms=None,
        seed: int = 1000,
        data_mix: str = "debug_research_data",
        vla2root_json: str = "vla2root.json",
        dataset_size_one_epoch: int = 100_000,
    ):
        super().__init__()
        self.cfg = cfg
        self.seed = seed
        self.epoch = 0
        self.image_transforms = image_transforms

        policy_cfg = cfg.policy
        self.chunk_size = policy_cfg.chunk_size
        # "duration" resolves the window from ``chunk_seconds`` per dataset; "frames" uses the
        # consecutive frames 0..chunk_size-1, which is what downstream VLA training wants.
        self.window_mode = getattr(policy_cfg, "window_mode", "duration")
        # Temporal window, expressed in seconds and resolved to a raw frame count per dataset
        # in ``_window_offsets``. ``frame_horizon`` overrides it with a fixed frame count.
        self.chunk_seconds = getattr(policy_cfg, "chunk_seconds", 1.6)
        self.chunk_frames_min = getattr(policy_cfg, "chunk_frames_min", 8)
        self.chunk_frames_max = getattr(policy_cfg, "chunk_frames_max", 48)
        self.frame_horizon_override = getattr(policy_cfg, "frame_horizon", None)
        # Filled in per dataset as they are built, so the resolved windows can be logged.
        self.frame_horizons: list[int] = []
        # Wall-clock capture rate per dataset, which is not always the declared one.
        self.true_fps: list[float] = []
        self.tactile_img_size = getattr(policy_cfg, "tactile_img_size", 64)
        self.tactile_dead_std = getattr(policy_cfg, "tactile_dead_std", 0.002)
        self.max_tactile_views = min(getattr(policy_cfg, "max_tactile_views", MAX_TACTILE_VIEWS), MAX_TACTILE_VIEWS)
        self.use_wrist_image = getattr(policy_cfg, "use_wrist_image", False)

        mixture_spec = OXE_NAMED_MIXTURES[data_mix]
        included_datasets, sample_weights = [], []
        for d_name, d_weight in mixture_spec:
            if d_name in included_datasets:
                continue
            included_datasets.append(d_name)
            sample_weights.append(d_weight)

        with open(vla2root_json) as f:
            vla2data_root = json.load(f)

        self.datasets: list = []
        self.dataset_names: list[str] = []
        self.dataset_sizes: list[int] = []
        self.specs: list[dict] = []
        self.image_key_maps: list[dict] = []
        self.norm_stats: list[dict] = []
        self.episode_ranges: list[np.ndarray] = []
        kept_weights: list[float] = []
        meta_features = None

        for dataset_name, weight in zip(included_datasets, sample_weights, strict=True):
            if dataset_name not in vla2data_root:
                logger.warning("%s missing from %s, skipping.", dataset_name, vla2root_json)
                continue
            data_root = self._resolve_root(cfg, vla2data_root[dataset_name])
            if data_root is None:
                logger.warning("%s not found on disk, skipping.", dataset_name)
                continue

            with open(os.path.join(data_root, "meta", "info.json")) as f:
                info = json.load(f)
            version = info.get("codebase_version", "v2.1")

            spec = get_spec(
                dataset_name,
                action_dim=self._feature_dim(info, "action"),
                state_dim=self._feature_dim(info, "observation.state"),
            )

            dataset, ds_meta, img_keys, horizon_ds, true_fps_ds = self._build_dataset(
                cfg, dataset_name, data_root, version, spec
            )
            if dataset is None:
                continue
            if meta_features is None:
                meta_features = dict(ds_meta.features)

            self.datasets.append(dataset)
            self.dataset_names.append(dataset_name)
            self.dataset_sizes.append(len(dataset))
            self.specs.append(spec)
            self.image_key_maps.append(img_keys)
            self.norm_stats.append(self._build_norm_stats(dataset, spec))
            self.episode_ranges.append(self._build_episode_ranges(dataset, version))
            # The horizon recorded here is the one ``_build_dataset`` actually read with, not a
            # recomputation: the sampler trims episodes by it, so a second derivation from a
            # second fps source could silently disagree with the window the loader used. The
            # same applies to the true fps, which `sample_rate` is built from.
            self.frame_horizons.append(horizon_ds)
            self.true_fps.append(true_fps_ds)
            declared = float(ds_meta.fps)
            logger.info(
                "Dataset: %s: fps=%g%s -> %s window %d frames (%.2fs), %d steps",
                dataset_name, true_fps_ds,
                f" (declared {declared:g})" if true_fps_ds != declared else "",
                self.window_mode, horizon_ds, horizon_ds / true_fps_ds, self.chunk_size,
            )
            kept_weights.append(weight)

        if not self.datasets:
            raise RuntimeError(f"No dataset of mixture '{data_mix}' could be loaded.")

        # Which datasets carry tactile at all. Tactile is a small slice of most mixtures, so
        # any metric computed over the whole mixture barely sees it; the evaluator uses this
        # to build a tactile-only split that can.
        self.has_tactile = np.array(
            [bool(tactile_signal_keys(s)) or bool(tactile_image_keys(s)) for s in self.specs],
            dtype=bool,
        )

        # Which physical sensor produced each tactile pad, and the z-score FTP-1 measured for
        # it. Only the "ftp1" tactile backbone consumes these, but they are cheap (a few
        # floats per dataset) and building them unconditionally keeps the batch schema fixed,
        # so the two backbones stay swappable without touching the collate function.
        self.tactile_sensor_meta = [
            self._build_tactile_sensor_meta(name, spec)
            for name, spec in zip(self.dataset_names, self.specs, strict=True)
        ]

        # Balance by dataset size, as in the original pipeline.
        weights = np.array(kept_weights, dtype=np.float64) * np.array(self.dataset_sizes, dtype=np.float64)
        self.sample_weights = weights / weights.sum()
        self.dataset_size_one_epoch = dataset_size_one_epoch
        self.dataset_sample_counts = (self.sample_weights * dataset_size_one_epoch).astype(int)

        print(
            tabulate(
                [
                    [self.dataset_names[i], self.dataset_sizes[i], f"{self.sample_weights[i]:.4f}", len(self.episode_ranges[i])]
                    for i in range(len(self.datasets))
                ],
                headers=["Dataset", "Frames", "Ratio", "Episodes"],
                tablefmt="grid",
            )
        )

        self.id2dataset, self.num_episodes = self._build_sampling_plan(seed)
        self.dataset_len = len(self.id2dataset)

        self.meta = self._build_unified_meta(cfg, meta_features)

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_root(cfg, relative_root: str) -> str | None:
        """First root under which ``relative_root`` is an actual LeRobot dataset.

        The test is ``meta/info.json``, not mere directory existence. Datasets are often nested
        one level deeper than the path in ``vla2root.json`` suggests (``OpenNeoData/aloha`` in
        fact contains ``aloha/``), and an existence check happily returns that outer directory,
        so the failure surfaces much later as a confusing FileNotFoundError on info.json rather
        than as "this dataset was not found".
        """
        extra = getattr(cfg.dataset, "parent_dir_extra", "") or ""
        parents = [cfg.dataset.parent_dir_v21, cfg.dataset.parent_dir_v30]
        parents += [p.strip() for p in extra.split(",") if p.strip()]
        for parent in parents:
            if not parent:
                continue
            candidate = os.path.join(parent, relative_root)
            if os.path.exists(os.path.join(candidate, "meta", "info.json")):
                return candidate
            # Tolerate the common one-level nesting (``<root>/<name>/<name>``) rather than
            # making every such dataset need a hand-edited path.
            nested = os.path.join(candidate, os.path.basename(relative_root))
            if os.path.exists(os.path.join(nested, "meta", "info.json")):
                return nested
        return None

    @staticmethod
    def _feature_dim(info: dict, key: str) -> int | None:
        feature = info.get("features", {}).get(key)
        if feature is None:
            return None
        shape = feature.get("shape")
        if not shape:
            return None
        return int(np.prod(shape))

    def _window_offsets(self, fps: float) -> tuple[list[int], int]:
        """``(frame_offsets, horizon)`` for one dataset, given its fps.

        Two modes, selected by ``policy.window_mode``, because contrastive pre-training and
        downstream VLA training want genuinely different things from the same field.

        ``"frames"`` is the ordinary action chunk: the consecutive frames ``0..chunk_size-1`` at
        the dataset's own rate. A policy that *emits* an action sequence has to produce commands
        the robot can execute back-to-back, so resampling would either skip commands or emit
        duplicates. There is nothing to equalise -- the chunk length is the action horizon.

        ``"duration"`` is for the contrastive stage, where the physical branch is never executed,
        only pooled into an embedding that has to explain a visual change. There a fixed frame
        count is the wrong unit: 16 frames is 5.3 s at fractal's 3 fps and 0.53 s at 30 fps, so
        the perception side was being asked to explain wildly different amounts of change for no
        reason but the recording rate. ``horizon`` raw frames are covered instead, then resampled
        onto the fixed ``chunk_size`` grid. Two properties are deliberate:

        * **Nearest-frame, not interpolation in time.** Every returned offset is a real frame
          index, so ``delta_timestamps`` never asks the loader for a timestamp that falls
          between two frames, which would trip its tolerance check. A window shorter than
          ``chunk_size`` repeats offsets -- honest, there is no more information to be had --
          and a longer one strides them.
        * **The grid spans ``[0, horizon]`` inclusive**, so its last step lands exactly on the
          frame the perception side uses as ``image_t1``. Dividing by ``chunk_size`` instead of
          ``chunk_size - 1`` would stop the physical window short of the visual one (46 of 48
          frames at 30 fps) and, worse, do so by a dataset-dependent amount, quietly breaking
          the correspondence this whole design rests on.

        The returned length is always ``chunk_size`` in both modes, so the physical token count
        never varies with the dataset or the mode.
        """
        # The caller turns every offset into a timestamp by dividing by fps, so a bad fps is
        # fatal in both modes, not just the duration one.
        fps = float(fps or 0.0)
        if not fps > 0:
            raise ValueError(f"fps must be positive to build a temporal window, got {fps!r}.")

        if self.window_mode == "frames":
            # `frame_horizon` still controls how far ahead the *pair* looks; it defaults to the
            # end of the chunk so the image pair and the action chunk describe the same span.
            horizon = (
                int(self.frame_horizon_override)
                if self.frame_horizon_override is not None
                else self.chunk_size - 1
            )
            return list(range(self.chunk_size)), max(1, horizon)

        if self.frame_horizon_override is not None:
            horizon = int(self.frame_horizon_override)
        else:
            exact = self.chunk_seconds * fps
            horizon = int(round(exact))
            horizon = max(self.chunk_frames_min, min(self.chunk_frames_max, horizon))
            # Outside [chunk_frames_min, chunk_frames_max] the clamp wins and the window is no
            # longer `chunk_seconds` long, so datasets stop sharing a temporal receptive field.
            # That is a deliberate cost cap, not a bug, but it should never be silent.
            if abs(horizon - exact) > 0.5:
                logger.warning(
                    "fps=%g wants a %.1f-frame window for %.2fs but the clamp gives %d frames "
                    "(%.2fs); this dataset does not share the mixture's temporal window. "
                    "Widen chunk_frames_min/max if that matters.",
                    fps, exact, self.chunk_seconds, horizon, horizon / fps,
                )
        horizon = max(1, horizon)
        denom = max(1, self.chunk_size - 1)
        offsets = [round(i * horizon / denom) for i in range(self.chunk_size)]
        return offsets, horizon

    def _build_dataset(self, cfg, dataset_name, data_root, version, spec):
        repo_id = f"bulldog-{dataset_name}"
        meta_cls = LeRobotDatasetMetadata if version == "v2.1" else LeRobotDatasetMetadataV30
        ds_meta = meta_cls(repo_id, root=data_root)
        # Two different fps, and conflating them is a silent data bug.
        #
        # ``index_fps`` is what info.json declares, and it is the time base the ``timestamp``
        # column was written on: frame ``i`` sits at ``i / index_fps``. Every delta_timestamp we
        # hand the loader must be built with it, or we match the wrong frame.
        #
        # ``true_fps`` is the rate the data was really captured at. It decides how much
        # wall-clock motion a window of ``H`` frames covers, and it is what ``sample_rate``
        # should report. The two agree for most datasets but not for FTP-1, which declares 30
        # while capturing at 10-15 Hz -- see ``dataset_fps.py``.
        index_fps = float(ds_meta.fps)
        true_fps = resolve_true_fps(dataset_name, index_fps)

        img_keys = self._resolve_image_keys(dataset_name, None, ds_meta=ds_meta)
        rgb_keys = [img_keys["primary"]] if img_keys["primary"] else []
        if self.use_wrist_image and img_keys.get("wrist"):
            rgb_keys.append(img_keys["wrist"])
        tac_img_keys = [k for k in tactile_image_keys(spec) if k in ds_meta.video_keys][
            : self.max_tactile_views
        ]

        # Only the action sources this dataset's canonical spec actually reads are chunked;
        # chunking every ``action.*`` column (e.g. 44-dim hand joints) wastes a lot of IO.
        wanted_action_keys = {src for src, *_ in spec.get("action", [])}
        resolved = resolve_delta_timestamps(cfg.policy, ds_meta) or {}
        delta_timestamps = {k: v for k, v in resolved.items() if k in wanted_action_keys}

        # How far each modality is read over the window is decided by what it costs to read.
        #
        # Everything that lives in parquet -- state, action, tactile signal -- is in the same
        # row group, so the whole chunk comes back for almost nothing. The state trajectory is
        # worth having next to the action chunk: the action is what was *commanded*, the state
        # is what actually *happened*, and the visual change we align against is the result of
        # the latter. The tactile signal is worth having at full rate for a different reason --
        # a contact transient is a few frames wide, so two samples can straddle it and see
        # almost nothing of it.
        #
        # All three read the *same* grid, which is the duration-based one from
        # ``_window_offsets``. The action keys are overridden here rather than left to
        # ``resolve_delta_timestamps``: that helper builds a consecutive-frame chunk from
        # ``action_delta_indices``, which would put the action on a different time base to the
        # state and the image pair it is supposed to explain.
        offsets, horizon = self._window_offsets(true_fps)
        chunk_stamps = [o / index_fps for o in offsets]
        for key in wanted_action_keys:
            if key in ds_meta.features:
                delta_timestamps[key] = chunk_stamps
        for key in {src for src, *_ in spec.get("state", [])}:
            if key in ds_meta.features:
                delta_timestamps[key] = chunk_stamps
        for key in tactile_signal_keys(spec):
            if key in ds_meta.features:
                delta_timestamps[key] = chunk_stamps

        # The video streams are read at the two ends of the window only. For RGB that is the
        # whole design; for tactile cameras it is a deliberate compromise. Tactile carries
        # contact events -- grasp closure, slip, impact -- which are things that *happen during*
        # the window, so a single frame at ``t`` can miss the entire signal (if the grasp closes
        # at t+8, the frame at t shows no contact at all). But decoding 16 frames x 4 pads on a
        # spinning disk is exactly where this pipeline is already bottlenecked, so two frames
        # buy "before contact -> after contact" at 2x the decode cost instead of 16x.
        pair_stamps = [0.0, horizon / index_fps]
        for key in rgb_keys:
            if key in ds_meta.video_keys:
                delta_timestamps[key] = pair_stamps
        for key in tac_img_keys:
            delta_timestamps[key] = pair_stamps

        try:
            common = dict(
                root=data_root,
                delta_timestamps=delta_timestamps or None,
                image_transforms=None,  # applied per-modality in this class
                video_backend=cfg.dataset.video_backend,
                dataset_name=dataset_name,
            )
            if version == "v2.1":
                dataset = LeRobotDataset(repo_id, **common)
            else:
                dataset = LeRobotDatasetV30(repo_id, video_return_type="uint8", **common)
        except Exception as exc:  # noqa: BLE001 - a broken dataset must not kill the whole mixture
            logger.warning("Failed to open %s (%s): %s", dataset_name, data_root, exc)
            return None, None, None, None, None

        dataset.video_keys_to_decode = rgb_keys + tac_img_keys
        return dataset, ds_meta, img_keys, horizon, true_fps

    def _resolve_image_keys(self, dataset_name, dataset, ds_meta=None) -> dict:
        """Map the OXE ``primary``/``secondary``/``wrist`` roles onto real dataset keys."""
        meta = ds_meta if ds_meta is not None else dataset.meta
        available = set(meta.video_keys) | set(getattr(meta, "image_keys", []) or [])
        config = OXE_DATASET_CONFIGS.get(dataset_name, {})
        role_map = config.get("image_obs_keys", {})

        resolved = {}
        for role in ("primary", "secondary", "wrist"):
            raw = role_map.get(role)
            resolved[role] = None
            if raw is None:
                continue
            for prefix in ("observation.images.", "observations.images.", "images.rgb."):
                candidate = f"{prefix}{raw}"
                if candidate in available:
                    resolved[role] = candidate
                    break
        if resolved["primary"] is None:
            # Fall back to any non-tactile camera so the sample is still usable.
            for key in sorted(available):
                if "tactile" not in key:
                    resolved["primary"] = key
                    break
        return resolved

    def _build_norm_stats(self, dataset, spec) -> dict:
        """Project each dataset's own statistics into the canonical slots.

        Per-dataset (rather than mixture-wide) normalisation is used on purpose: it removes the
        dataset-specific scale that a contrastive model would otherwise exploit as a shortcut.
        """
        stats = getattr(dataset.meta, "stats", None) or {}

        def project(instructions):
            mean = np.zeros(CANON_DIM, dtype=np.float32)
            std = np.ones(CANON_DIM, dtype=np.float32)
            mask = np.zeros(CANON_DIM, dtype=np.float32)
            for src_key, s0, s1, d0 in instructions:
                width = s1 - s0
                mask[d0 : d0 + width] = 1.0
                src = stats.get(src_key)
                if src is None or "mean" not in src:
                    continue
                src_mean = np.asarray(src["mean"], dtype=np.float32).reshape(-1)
                src_std = np.asarray(src["std"], dtype=np.float32).reshape(-1)
                if src_mean.shape[0] < s1:
                    continue
                mean[d0 : d0 + width] = src_mean[s0:s1]
                std[d0 : d0 + width] = np.maximum(src_std[s0:s1], 1e-3)
            return {
                "mean": torch.from_numpy(mean),
                "std": torch.from_numpy(std),
                "mask": torch.from_numpy(mask),
            }

        out = {"action": project(spec.get("action", [])), "state": project(spec.get("state", []))}

        sig_mean = np.zeros(MAX_TACTILE_SIGNAL_DIM, dtype=np.float32)
        sig_std = np.ones(MAX_TACTILE_SIGNAL_DIM, dtype=np.float32)
        sig_mask = np.zeros(MAX_TACTILE_SIGNAL_DIM, dtype=np.float32)
        # The slot layout is decided *here* and handed to `_build_tactile_signal`, rather than
        # each of them walking its own running offset over the same key list. The two walks skip
        # on different conditions -- this one skips keys with no statistics, the other skips keys
        # the item did not return -- so a key present in one and absent in the other would shift
        # every later slot and silently normalise the wrong dimensions.
        sig_slots: list[tuple[str, int, int]] = []
        offset = 0
        for key in tactile_signal_keys(spec):
            src = stats.get(key)
            if src is None or "mean" not in src:
                continue
            src_mean = np.asarray(src["mean"], dtype=np.float32).reshape(-1)
            src_std = np.asarray(src["std"], dtype=np.float32).reshape(-1)
            width = min(src_mean.shape[0], MAX_TACTILE_SIGNAL_DIM - offset)
            if width <= 0:
                break
            sig_mean[offset : offset + width] = src_mean[:width]
            sig_std[offset : offset + width] = np.maximum(src_std[:width], 1e-3)
            sig_mask[offset : offset + width] = 1.0
            sig_slots.append((key, offset, width))
            offset += width
        out["tactile_signal"] = {
            "mean": torch.from_numpy(sig_mean),
            "std": torch.from_numpy(sig_std),
            "mask": torch.from_numpy(sig_mask),
            "slots": sig_slots,
        }
        return out

    @staticmethod
    def _build_episode_ranges(dataset, version) -> np.ndarray:
        """``(num_episodes, 2)`` array of ``[from_index, to_index)`` absolute frame ranges."""
        if version == "v2.1":
            idx = dataset.episode_data_index
            starts = np.asarray(idx["from"], dtype=np.int64)
            ends = np.asarray(idx["to"], dtype=np.int64)
        else:
            episodes = dataset.meta.episodes
            starts = np.asarray(episodes["dataset_from_index"], dtype=np.int64)
            ends = np.asarray(episodes["dataset_to_index"], dtype=np.int64)
        return np.stack([starts, ends], axis=1)

    def _build_unified_meta(self, cfg, meta_features):
        img_feature = None
        features = {}
        for key, value in (meta_features or {}).items():
            if value.get("dtype") in ("image", "video"):
                img_feature = value
            else:
                features[key] = value
        if img_feature is None:
            img_feature = {"dtype": "video", "shape": (224, 224, 3), "names": None, "info": {}}
        img_size = cfg.dataset.image_transforms.img_size
        img_feature = dict(img_feature)
        img_feature["shape"] = (img_size, img_size, 3)
        for key in ("observation.images.primary", "observation.images.secondary", "observation.images.wrist"):
            features[key] = img_feature

        canon = {
            "dtype": "float32",
            "shape": (CANON_DIM,),
            "names": None,
        }
        features["action"] = canon
        features["observation.state"] = canon

        stats = {
            "action": {
                "mean": np.zeros(CANON_DIM, dtype=np.float32),
                "std": np.ones(CANON_DIM, dtype=np.float32),
                "max": np.ones(CANON_DIM, dtype=np.float32),
                "min": -np.ones(CANON_DIM, dtype=np.float32),
                "count": np.array([1]),
            },
        }
        stats["observation.state"] = stats["action"]
        meta = LeRobotDatasetMetadata.create_with_stats_feats(stats=stats, features=features)
        meta.repo_id = "Prometheus"
        return meta

    # ------------------------------------------------------------------
    # sampling plan
    # ------------------------------------------------------------------
    def _build_sampling_plan(self, seed: int):
        rng = random.Random(seed)
        plan: list[tuple[int, int]] = []
        episode_count = 0
        for ds_idx, (dataset, count) in enumerate(zip(self.datasets, self.dataset_sample_counts, strict=True)):
            size = len(dataset)
            if count <= 0 or size == 0:
                continue
            if count <= size:
                indices = rng.sample(range(size), count)
            else:
                indices = rng.choices(range(size), k=count)
            plan.extend((ds_idx, i) for i in indices)
            episode_count += int(len(self.episode_ranges[ds_idx]) * min(1.0, count / size))
        return plan, max(episode_count, 1)

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.id2dataset, self.num_episodes = self._build_sampling_plan(self.seed + epoch)
        self.dataset_len = len(self.id2dataset)

    def __len__(self):
        return self.dataset_len

    # ------------------------------------------------------------------
    # item construction
    # ------------------------------------------------------------------
    def __getitem__(self, index):
        if isinstance(index, (tuple, list)) and len(index) == 2:
            ds_idx, frame_idx = int(index[0]), int(index[1])
        else:
            ds_idx, frame_idx = self.id2dataset[int(index)]

        dataset = self.datasets[ds_idx]
        frame_idx = int(np.clip(frame_idx, 0, len(dataset) - 1))
        try:
            item = dataset[frame_idx]
        except Exception as exc:  # noqa: BLE001 - never let one broken frame kill training
            logger.warning("Failed to read %s[%d]: %s", self.dataset_names[ds_idx], frame_idx, exc)
            item = dataset[0]
            frame_idx = 0

        return self._to_canonical(item, ds_idx, frame_idx)

    def _resize_rgb(self, image: torch.Tensor) -> torch.Tensor:
        size = self.cfg.dataset.image_transforms.img_size
        if image.shape[-1] == size and image.shape[-2] == size:
            return image
        return F.interpolate(
            image.unsqueeze(0).float(), size=(size, size), mode="bilinear", align_corners=False
        ).squeeze(0).to(torch.uint8)

    def _extract_frames(self, item, primary_key):
        """Return ``(image_t0, image_t1, pair_is_valid)`` as uint8 CHW tensors."""
        size = self.cfg.dataset.image_transforms.img_size
        frames = item.get(primary_key) if primary_key else None
        if frames is None:
            zeros = torch.zeros(3, size, size, dtype=torch.uint8)
            return zeros, zeros.clone(), 0.0

        if frames.ndim == 4:  # (T, C, H, W)
            first = frames[0]
            last = frames[-1] if frames.shape[0] > 1 else frames[0]
            is_pad = item.get(f"{primary_key}_is_pad")
            valid = 0.0 if (is_pad is not None and bool(is_pad[-1])) else 1.0
        else:
            first = frames
            last = frames
            valid = 0.0

        first = self._resize_rgb(first)
        last = self._resize_rgb(last)
        return first, last, valid

    def _build_canonical_vector(self, item, instructions, norm, is_chunk: bool):
        width = CANON_DIM
        chunk = self.chunk_size if is_chunk else 1
        out = torch.zeros(chunk, width, dtype=torch.float32)
        mask = torch.zeros(width, dtype=torch.float32)

        for src_key, s0, s1, d0 in instructions:
            value = _to_tensor(item.get(src_key))
            if value is None:
                continue
            value = value.to(torch.float32)
            if is_chunk:
                if value.ndim == 1:
                    value = value.unsqueeze(0).expand(chunk, -1)
                elif value.ndim > 2:
                    value = value.reshape(value.shape[0], -1)
                if value.shape[0] < chunk:
                    pad = value[-1:].expand(chunk - value.shape[0], -1)
                    value = torch.cat([value, pad], dim=0)
                value = value[:chunk]
            else:
                value = value.reshape(-1) if value.ndim == 1 else value.reshape(value.shape[0], -1)[0]
                value = value.unsqueeze(0)
            if value.shape[-1] < s1:
                continue
            span = s1 - s0
            out[:, d0 : d0 + span] = value[:, s0:s1]
            mask[d0 : d0 + span] = 1.0

        mask = mask * norm["mask"]
        out = (out - norm["mean"]) / norm["std"]
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0) * mask
        return (out if is_chunk else out.squeeze(0)), mask

    def _build_tactile_signal(self, item, spec, norm):
        """``(chunk, 32)``: the tactile signal over the whole window.

        Read at full rate rather than at the two ends, because a contact transient is only a
        few frames wide: two samples can straddle it and see almost nothing of it. Unlike the
        tactile cameras this costs nothing, the signal being a parquet column.

        Slots come from the normalisation stats so that dimension ``i`` of the output always
        means the same physical channel as dimension ``i`` of ``mean``/``std``/``mask``.
        """
        chunk = self.chunk_size
        signal = torch.zeros(chunk, MAX_TACTILE_SIGNAL_DIM, dtype=torch.float32)
        found = False
        for key, offset, width in norm.get("slots", ()):
            value = _to_tensor(item.get(key))
            if value is None:
                continue
            value = value.to(torch.float32)
            # A key may come back as (D,) if this dataset had no window to read.
            value = value.reshape(1, -1) if value.ndim == 1 else value.reshape(value.shape[0], -1)
            if value.shape[0] < chunk:
                value = torch.cat([value, value[-1:].expand(chunk - value.shape[0], -1)], dim=0)
            value = value[:chunk]
            take = min(width, value.shape[1])
            signal[:, offset : offset + take] = value[:, :take]
            found = True
        if not found:
            return signal, torch.zeros((), dtype=torch.float32)
        signal = (signal - norm["mean"]) / norm["std"]
        signal = torch.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0) * norm["mask"]
        return signal, torch.ones((), dtype=torch.float32)

    def _build_tactile_sensor_meta(self, dataset_name, spec):
        """Per-pad sensor id and z-score, padded out to ``max_tactile_views``.

        Slot order follows ``tactile_image_keys(spec)`` exactly, which is also the slot order
        ``_build_tactile_images`` writes, so a pad's pixels and its normalisation can never
        drift apart. Unused slots get id -1; the model never dispatches them because their
        mask is zero.
        """
        keys = tactile_image_keys(spec)[: self.max_tactile_views]
        ids, means, stds = tactile_image_sensors(dataset_name, len(keys))

        sensor_id = torch.full((self.max_tactile_views,), -1, dtype=torch.long)
        mean = torch.zeros(self.max_tactile_views, 3, dtype=torch.float32)
        std = torch.ones(self.max_tactile_views, 3, dtype=torch.float32)
        for slot in range(len(keys)):
            sensor_id[slot] = ids[slot]
            mean[slot] = torch.tensor(means[slot], dtype=torch.float32)
            std[slot] = torch.tensor(stds[slot], dtype=torch.float32)
        return sensor_id, mean, std

    def _build_tactile_images(self, item, spec):
        """``(V, 2, 3, S, S)``: each pad at ``t`` and at ``t + horizon``.

        Pads whose frames carry no spatial structure are masked out here. A large share of our
        tactile pixels are dead: ``sharpa``'s six pads are blank 38-100% of the time and
        ``RDP_Bimanual``'s right pad is constant for its whole file (measured distributions in
        ``doc/results.md`` §10.5). A constant image is not a cheap input -- it costs a CNN pass
        and a physical-sequence token, it is a trivially satisfiable reconstruction target, and
        it teaches the encoder that "tactile present" often means "tactile says nothing".

        The test is the *spatial* std within each channel, not the std over the whole frame: a
        pad that is uniformly some non-black colour is spatially dead but has a non-zero global
        std, so a global test would let it through. A pad is dropped only when **both** frames
        are flat -- one flat and one live is a contact onset or release, which is exactly the
        kind of event this branch exists to capture.
        """
        size = self.tactile_img_size
        dead_std = self.tactile_dead_std
        views = torch.zeros(self.max_tactile_views, 2, 3, size, size, dtype=torch.uint8)
        mask = torch.zeros(self.max_tactile_views, dtype=torch.float32)
        for slot, key in enumerate(tactile_image_keys(spec)[: self.max_tactile_views]):
            frame = item.get(key)
            if frame is None:
                continue
            if frame.ndim == 3:  # (C, H, W) -- no window was read
                frame = frame.unsqueeze(0)
            if frame.shape[0] < 2:
                frame = frame[:1].expand(2, -1, -1, -1)
            frame = frame[:2]
            frame = F.interpolate(
                frame.float(), size=(size, size), mode="bilinear", align_corners=False
            )
            views[slot] = frame.clamp(0, 255).to(torch.uint8)
            if dead_std > 0:
                # Scale to [0, 1] so the threshold is expressed in the same units as the
                # measured distributions; one 8-bit grey level is 1/255 = 0.0039.
                spatial_std = (frame / 255.0).reshape(2, 3, -1).std(dim=-1).max()
                if spatial_std.item() < dead_std:
                    continue
            mask[slot] = 1.0
        return views, mask

    def _to_canonical(self, item, ds_idx, frame_idx):
        spec = self.specs[ds_idx]
        norm = self.norm_stats[ds_idx]
        primary_key = self.image_key_maps[ds_idx]["primary"]

        image_t0, image_t1, pair_valid = self._extract_frames(item, primary_key)
        action, action_mask = self._build_canonical_vector(item, spec.get("action", []), norm["action"], True)
        state, state_mask = self._build_canonical_vector(item, spec.get("state", []), norm["state"], True)
        tactile_signal, tactile_signal_mask = self._build_tactile_signal(item, spec, norm["tactile_signal"])
        tactile_image, tactile_image_mask = self._build_tactile_images(item, spec)
        sensor_id, sensor_mean, sensor_std = self.tactile_sensor_meta[ds_idx]

        episode_index = int(_to_tensor(item.get("episode_index", 0)).reshape(-1)[0].item())
        task = item.get("task", "")
        if isinstance(task, (list, tuple)):
            task = task[0] if task else ""

        return {
            "image_t0": image_t0,
            "image_t1": image_t1,
            "pair_is_valid": torch.tensor(pair_valid, dtype=torch.float32),
            "task": str(task),
            "action": action,
            "action_mask": action_mask,
            "observation.state": state,
            "state_mask": state_mask,
            "tactile_signal": tactile_signal,
            "tactile_signal_mask": tactile_signal_mask,
            "tactile_image": tactile_image,
            "tactile_image_mask": tactile_image_mask,
            "tactile_sensor_id": sensor_id,
            "tactile_img_mean": sensor_mean,
            "tactile_img_std": sensor_std,
            # Not `item["fps"]`: the v2.1 loader never sets it (so every v2.1 dataset silently
            # reported 10) and the v3.0 loader sets it from the *declared* fps, which is wrong
            # for FTP-1. `self.true_fps` is the rate the data was actually captured at.
            "sample_rate": torch.tensor(int(round(self.true_fps[ds_idx])), dtype=torch.long),
            "dataset_id": torch.tensor(ds_idx, dtype=torch.long),
            "episode_uid": torch.tensor(ds_idx * 1_000_000 + episode_index, dtype=torch.long),
            "frame_index": torch.tensor(frame_idx, dtype=torch.long),
        }

    # ------------------------------------------------------------------
    @property
    def num_frames(self) -> int:
        return self.dataset_len

    @property
    def features(self):
        return self.meta.features


def contrastive_collate_fn(batch: list[dict]) -> dict:
    """Stack tensors, keep language instructions as a python list."""
    out = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        if isinstance(values[0], torch.Tensor):
            out[key] = torch.stack(values, dim=0)
        else:
            out[key] = values
    return out
