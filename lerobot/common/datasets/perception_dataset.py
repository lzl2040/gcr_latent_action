"""Vision-only loader for stage-1 pre-training of the perception branch.

Why this exists
---------------
The contrastive stage needs a dataset that carries *both* a camera and a robot: actions,
states, ideally tactile. That is a small and expensive slice of the video that exists. The
perception branch, however, has an objective it can train on alone -- predict the patch
features of frame ``t+H`` from frame ``t``, the instruction, and the change queries -- and
that objective needs nothing but two frames and (optionally) a caption. So it can be
pre-trained on the much larger pool of plain robot video before the physical branch is
introduced.

What this loader deliberately does *not* do
-------------------------------------------
No ``get_spec``, no canonical 40-d projection, no normalisation statistics, no tactile. A
dataset with no ``action`` column, no ``observation.state`` and no ``OXE_DATASET_CONFIGS``
entry is a first-class citizen here. The only hard requirement is one decodable camera.

The IO consequence is the point: :class:`MultiModalContrastiveDataset` reads a
``chunk_size``-long grid from parquet plus up to ``1 + max_tactile_views`` video streams,
while this reads exactly two frames of one stream.

Missing language
----------------
Plenty of video has no instruction, and the two ways of handling that are not equivalent.
Feeding the empty string makes "no instruction" indistinguishable from an instruction that
happens to tokenize to nothing, and the resulting constant embedding is a dataset
fingerprint the model can lean on. Instead every sample carries an explicit ``has_text``
flag and the encoder substitutes a learned null-instruction embedding, exactly as the
physical branch already does for its missing modalities. Empty, whitespace and a short list
of placeholder strings ("none", "n/a", the dataset's own name, ...) all count as absent --
these appear often enough in converted datasets to be worth catching, and a silent
"instruction: none" would otherwise be trained on as if it were language.

Emitted per sample
------------------
``image_t0`` / ``image_t1`` uint8 CHW, ``pair_is_valid``, ``task``, ``has_text``,
``dataset_id``, ``episode_uid``, ``frame_index``, ``sample_rate``.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import torch
from tabulate import tabulate

from lerobot.common.datasets.contrastive_dataset import (
    MultiModalContrastiveDataset,
    resolve_pair_horizon,
)
from lerobot.common.datasets.dataset_fps import resolve_true_fps
from lerobot.common.datasets.instruction_text import dataset_task_strings, is_real_instruction
from lerobot.common.datasets.lerobot_dataset_for_ace import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from lerobot.common.datasets.mixtures import OXE_NAMED_MIXTURES
from lerobot.common.datasets_v30.dataset_metadata import (
    LeRobotDatasetMetadata as LeRobotDatasetMetadataV30,
)
from lerobot.common.datasets_v30.lerobot_dataset import LeRobotDataset as LeRobotDatasetV30

logger = logging.getLogger(__name__)

class PerceptionVideoDataset(torch.utils.data.Dataset):
    """Mixture of LeRobot datasets emitting ``(frame_t, frame_t+H, instruction)`` triples."""

    # Helpers shared verbatim with the contrastive loader. Each encodes a correctness fix that
    # cost real debugging -- the uint8-vs-float decoder convention, the one-level dataset root
    # nesting, the torchcodec availability check -- so they are bound here rather than copied,
    # which would let stage 1 and stage 2 drift apart in exactly the places that already bit us.
    _resolve_root = staticmethod(MultiModalContrastiveDataset._resolve_root)
    _resolve_video_backend = staticmethod(MultiModalContrastiveDataset._resolve_video_backend)
    _build_episode_ranges = staticmethod(MultiModalContrastiveDataset._build_episode_ranges)
    _as_uint8 = staticmethod(MultiModalContrastiveDataset._as_uint8)
    _resolve_image_keys = MultiModalContrastiveDataset._resolve_image_keys
    _resize_rgb = MultiModalContrastiveDataset._resize_rgb
    _extract_frames = MultiModalContrastiveDataset._extract_frames
    # ``make_policy`` dimensions the policy from dataset metadata, and this builds the same
    # synthetic canonical-space metadata the contrastive loader uses. Stage 1 never reads an
    # action, but the policy config still validates its feature spec, and reusing this keeps
    # the two stages' feature definitions identical rather than nearly identical.
    _build_unified_meta = MultiModalContrastiveDataset._build_unified_meta

    def __init__(
        self,
        cfg,
        image_transforms=None,
        seed: int = 1000,
        data_mix: str = "debug_research_data",
        vla2root_json: str = "vla2root.json",
        dataset_size_one_epoch: int = 100_000,
        camera_mode: str = "primary",
    ):
        super().__init__()
        self.cfg = cfg
        self.seed = seed
        self.epoch = 0
        self.image_transforms = image_transforms

        policy_cfg = cfg.policy
        # Read through the same knobs as stage 2 so the two stages cannot disagree about how
        # much motion a "pair" spans. See ``resolve_pair_horizon``.
        self.chunk_size = policy_cfg.chunk_size
        self.window_mode = getattr(policy_cfg, "window_mode", "duration")
        self.chunk_seconds = getattr(policy_cfg, "chunk_seconds", 1.6)
        self.chunk_frames_min = getattr(policy_cfg, "chunk_frames_min", 8)
        self.chunk_frames_max = getattr(policy_cfg, "chunk_frames_max", 48)
        self.frame_horizon_override = getattr(policy_cfg, "frame_horizon", None)
        self._video_backend = self._resolve_video_backend(cfg.dataset.video_backend)

        if camera_mode not in ("primary", "all"):
            raise ValueError(f"camera_mode must be 'primary' or 'all', got {camera_mode!r}.")
        self.camera_mode = camera_mode

        mixture_spec = OXE_NAMED_MIXTURES[data_mix]
        included_datasets, mixture_weights = [], []
        for d_name, d_weight in mixture_spec:
            if d_name in included_datasets:
                continue
            included_datasets.append(d_name)
            mixture_weights.append(d_weight)

        with open(vla2root_json) as f:
            vla2data_root = json.load(f)

        # One entry per (dataset, camera). In ``camera_mode="all"`` a second camera becomes a
        # separate entry with its own LeRobotDataset object rather than a second video stream
        # on the existing one. That looks wasteful and is the opposite: each object then
        # declares exactly one video key in its ``delta_timestamps``, so reading a sample
        # decodes one stream. Multiplexing views inside a single object would decode every
        # camera on every access and throw all but one away.
        self.datasets: list = []
        self.dataset_names: list[str] = []
        self.image_keys: list[str] = []
        self.dataset_sizes: list[int] = []
        self.episode_ranges: list[np.ndarray] = []
        self.frame_horizons: list[int] = []
        self.true_fps: list[float] = []
        self.text_coverage: list[float] = []
        self.source_names: list[str] = []
        kept_weights: list[float] = []
        meta_features = None

        for dataset_name, weight in zip(included_datasets, mixture_weights, strict=True):
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

            entries = self._build_dataset(cfg, dataset_name, data_root, version)
            if not entries:
                continue
            if meta_features is None:
                meta_features = dict(entries[0][1].features)
            # Cameras of one dataset share its mixture weight instead of multiplying it; a
            # three-camera dataset should not become three times as likely as a one-camera
            # dataset purely because of how it was recorded.
            per_view_weight = weight / len(entries)
            for entry in entries:
                dataset, ds_meta, img_key, horizon, true_fps, coverage, view_name = entry
                self.datasets.append(dataset)
                self.dataset_names.append(view_name)
                self.source_names.append(dataset_name)
                self.image_keys.append(img_key)
                self.dataset_sizes.append(len(dataset))
                self.episode_ranges.append(self._build_episode_ranges(dataset, version))
                self.frame_horizons.append(horizon)
                self.true_fps.append(true_fps)
                self.text_coverage.append(coverage)
                kept_weights.append(per_view_weight)

            _, ds_meta, _, horizon, true_fps, coverage, _ = entries[0]
            declared = float(ds_meta.fps)
            logger.info(
                "Dataset: %s: fps=%g%s -> %s window %d frames (%.2fs), %d view(s), text %.0f%%",
                dataset_name, true_fps,
                f" (declared {declared:g})" if true_fps != declared else "",
                self.window_mode, horizon, horizon / true_fps, len(entries), 100 * coverage,
            )

        if not self.datasets:
            raise RuntimeError(f"No dataset of mixture '{data_mix}' could be loaded.")

        weights = np.array(kept_weights, dtype=np.float64) * np.array(
            self.dataset_sizes, dtype=np.float64
        )
        self.sample_weights = weights / weights.sum()
        self.dataset_size_one_epoch = dataset_size_one_epoch
        self.dataset_sample_counts = (self.sample_weights * dataset_size_one_epoch).astype(int)

        # Frames from which ``t + horizon`` still lands inside the same episode. Sampling is
        # restricted to these rather than clamping afterwards: a clamped pair has
        # ``image_t1 == image_t0``, i.e. a target the predictor can hit by copying its input,
        # and at short episode lengths those degenerate pairs are a large fraction of the data.
        self._valid_starts = [
            self._trim_episodes(rng_, h) for rng_, h in zip(self.episode_ranges, self.frame_horizons, strict=True)
        ]

        print(
            tabulate(
                [
                    [
                        self.dataset_names[i],
                        self.dataset_sizes[i],
                        f"{self.sample_weights[i]:.4f}",
                        len(self.episode_ranges[i]),
                        f"{100 * self.text_coverage[i]:.0f}%",
                    ]
                    for i in range(len(self.datasets))
                ],
                headers=["Dataset/view", "Frames", "Ratio", "Episodes", "Text"],
                tablefmt="grid",
            )
        )
        no_text = [n for n, c in zip(self.dataset_names, self.text_coverage, strict=True) if c < 0.5]
        if no_text:
            logger.info(
                "%d/%d views have no usable instruction (%s); they train with the learned "
                "null-instruction embedding.",
                len(no_text), len(self.datasets), ", ".join(no_text[:5]),
            )

        self.id2dataset = self._build_sampling_plan(seed)
        self.dataset_len = len(self.id2dataset)
        self.meta = self._build_unified_meta(cfg, meta_features)
        # Episode count is reported for logging only; stage 1 draws frames, not episodes.
        self.num_episodes = max(sum(len(r) for r in self.episode_ranges), 1)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _build_dataset(self, cfg, dataset_name, data_root, version):
        """One entry per usable camera: ``(dataset, meta, key, horizon, fps, coverage, name)``."""
        repo_id = f"bulldog-{dataset_name}"
        meta_cls = LeRobotDatasetMetadata if version == "v2.1" else LeRobotDatasetMetadataV30
        ds_meta = meta_cls(repo_id, root=data_root)

        # ``index_fps`` is the time base of the ``timestamp`` column, so every delta_timestamp
        # is built with it. ``true_fps`` is the real capture rate and decides how much motion
        # the window covers. FTP-1 declares 30 while capturing at 10-15 Hz.
        index_fps = float(ds_meta.fps)
        true_fps = resolve_true_fps(dataset_name, index_fps)
        horizon = resolve_pair_horizon(
            true_fps,
            window_mode=self.window_mode,
            chunk_size=self.chunk_size,
            chunk_seconds=self.chunk_seconds,
            chunk_frames_min=self.chunk_frames_min,
            chunk_frames_max=self.chunk_frames_max,
            frame_horizon=self.frame_horizon_override,
        )
        pair_stamps = [0.0, horizon / index_fps]

        keys = self._camera_keys(dataset_name, ds_meta)
        if not keys:
            logger.warning("%s exposes no non-tactile camera, skipping.", dataset_name)
            return []

        coverage = self._text_coverage(ds_meta, dataset_name)

        entries = []
        for key in keys:
            try:
                common = dict(
                    root=data_root,
                    delta_timestamps={key: pair_stamps},
                    image_transforms=None,
                    video_backend=self._video_backend,
                    dataset_name=dataset_name,
                )
                if version == "v2.1":
                    dataset = LeRobotDataset(repo_id, **common)
                else:
                    dataset = LeRobotDatasetV30(repo_id, video_return_type="uint8", **common)
            except Exception as exc:  # noqa: BLE001 - one broken dataset must not kill the mixture
                logger.warning(
                    "Failed to open %s (%s): %s",
                    dataset_name, data_root, exc,
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )
                return []
            # The whole IO saving of this loader lives on this line: decode this camera only.
            dataset.video_keys_to_decode = [key]
            view = dataset_name if len(keys) == 1 else f"{dataset_name}::{key.split('.')[-1]}"
            entries.append((dataset, ds_meta, key, horizon, true_fps, coverage, view))
        return entries

    def _camera_keys(self, dataset_name, ds_meta) -> list[str]:
        """Which camera(s) to read, tactile pads excluded.

        Reuses the contrastive loader's OXE role resolution so a dataset is read through the
        same camera in both stages; its fallback already covers datasets with no
        ``OXE_DATASET_CONFIGS`` entry, which is the common case for plain video.
        """
        roles = self._resolve_image_keys(dataset_name, None, ds_meta=ds_meta)
        if self.camera_mode == "primary":
            return [roles["primary"]] if roles["primary"] else []

        available = set(ds_meta.video_keys) | set(getattr(ds_meta, "image_keys", []) or [])
        keys = [k for k in sorted(available) if "tactile" not in k]
        # Keep the primary first so ``camera_mode="all"`` is a superset of "primary" in a
        # readable order, not an alphabetical reshuffle.
        if roles["primary"] in keys:
            keys.remove(roles["primary"])
            keys.insert(0, roles["primary"])
        return keys

    @staticmethod
    def _text_coverage(ds_meta, dataset_name) -> float:
        """Fraction of a dataset's declared tasks that are real instructions."""
        tasks = dataset_task_strings(ds_meta)
        if not tasks:
            return 0.0
        return sum(is_real_instruction(t, dataset_name) for t in tasks) / len(tasks)

    @staticmethod
    def _trim_episodes(episode_ranges: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        """``(starts, lengths)`` of the frames whose partner frame is in the same episode.

        Episodes shorter than ``horizon + 1`` frames have no such frame and are dropped. If
        that would empty the dataset the trim is abandoned and every episode contributes its
        first frame, so a short-episode dataset degrades to clamped pairs instead of raising.
        """
        starts = episode_ranges[:, 0]
        lengths = episode_ranges[:, 1] - starts - horizon
        keep = lengths > 0
        if not keep.any():
            logger.warning(
                "Every episode is shorter than the %d-frame window; pairs will be clamped "
                "and mostly static. Lower chunk_seconds for this dataset.", horizon,
            )
            return starts, np.ones_like(starts)
        return starts[keep], lengths[keep]

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------
    def _build_sampling_plan(self, seed: int) -> list[tuple[int, int]]:
        """``(ds_idx, frame_idx)`` pairs, uniform over the *valid* frames of each dataset.

        Sampling episode-then-offset with the episode drawn proportionally to its trimmed
        length is uniform over valid frames while never materialising the frame list, which
        for a multi-million-frame mixture matters.
        """
        rng = np.random.default_rng(seed)
        plan: list[tuple[int, int]] = []
        for ds_idx, count in enumerate(self.dataset_sample_counts):
            count = int(count)
            if count <= 0:
                continue
            starts, lengths = self._valid_starts[ds_idx]
            probs = lengths / lengths.sum()
            ep = rng.choice(len(starts), size=count, p=probs)
            offsets = (rng.random(count) * lengths[ep]).astype(np.int64)
            frames = starts[ep] + offsets
            plan.extend((ds_idx, int(f)) for f in frames)
        rng.shuffle(plan)
        return plan

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.id2dataset = self._build_sampling_plan(self.seed + epoch)
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

        image_t0, image_t1, pair_valid = self._extract_frames(item, self.image_keys[ds_idx])

        task = item.get("task", "")
        if isinstance(task, (list, tuple)):
            task = task[0] if task else ""
        task = str(task)
        has_text = is_real_instruction(task, self.source_names[ds_idx])

        episode_index = int(np.asarray(item.get("episode_index", 0)).reshape(-1)[0])

        return {
            "image_t0": image_t0,
            "image_t1": image_t1,
            "pair_is_valid": torch.tensor(pair_valid, dtype=torch.float32),
            # The string is passed through even when it is filler, so an audit can see what was
            # rejected; `has_text` is what the model is allowed to act on.
            "task": task if has_text else "",
            "has_text": torch.tensor(float(has_text), dtype=torch.float32),
            "sample_rate": torch.tensor(int(round(self.true_fps[ds_idx])), dtype=torch.long),
            "dataset_id": torch.tensor(ds_idx, dtype=torch.long),
            "episode_uid": torch.tensor(ds_idx * 1_000_000 + episode_index, dtype=torch.long),
            "frame_index": torch.tensor(frame_idx, dtype=torch.long),
        }

    @property
    def num_frames(self) -> int:
        return self.dataset_len
