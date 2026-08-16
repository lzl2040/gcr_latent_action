"""Batch sampler shaping the negatives of the perception <-> physical contrastive loss.

Uniform random batches make the task too easy: telling a kitchen scene apart from a
factory scene needs no understanding of motion at all. This sampler builds batches that
are hard *by construction*:

1. ``same_dataset_frac`` of a batch comes from a single dataset, so negatives share the
   embodiment, camera and scene statistics.
2. ``episode_group_frac`` of that majority is drawn as small groups of frames from the
   *same episode*. Those are the hardest negatives available: identical scene, identical
   robot, only the motion differs.
3. Frames inside one episode group are forced at least ``min_frame_gap`` frames apart, so
   a "negative" is never a near-duplicate of the anchor (which would be a false negative).
4. The remaining slots are filled from the whole mixture to keep some easy negatives and
   avoid collapsing onto a single dataset's statistics.

Every frame is additionally kept at least ``horizon`` frames before the end of its episode
so that both the future perception frame and the full action chunk exist without clamping.
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler


class ContrastiveBatchSampler(Sampler):
    """Distributed batch sampler yielding lists of ``(dataset_idx, frame_idx)`` tuples."""

    def __init__(
        self,
        episode_ranges: list[np.ndarray],
        sample_weights: np.ndarray,
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        samples_per_epoch: int = 100_000,
        horizon: int = 16,
        same_dataset_frac: float = 0.75,
        episode_group_frac: float = 0.75,
        episode_group_size: int = 8,
        min_frame_gap: int = 32,
    ):
        self.episode_ranges = episode_ranges
        self.sample_weights = np.asarray(sample_weights, dtype=np.float64)
        self.sample_weights = self.sample_weights / self.sample_weights.sum()
        self.batch_size = batch_size
        self.num_replicas = max(1, num_replicas)
        self.rank = rank
        self.seed = seed
        self.horizon = horizon
        self.same_dataset_frac = float(np.clip(same_dataset_frac, 0.0, 1.0))
        self.episode_group_frac = float(np.clip(episode_group_frac, 0.0, 1.0))
        self.episode_group_size = max(2, episode_group_size)
        self.min_frame_gap = max(1, min_frame_gap)
        self.epoch = 0

        # Usable frame span per episode: [start, end - horizon). Episodes too short to host a
        # full chunk fall back to their single first frame (clamping is handled downstream).
        self.usable = []
        for ranges in episode_ranges:
            starts = ranges[:, 0]
            ends = np.maximum(ranges[:, 1] - self.horizon, starts + 1)
            self.usable.append(np.stack([starts, ends], axis=1))

        total_batches = max(1, samples_per_epoch // (batch_size * self.num_replicas))
        self.num_batches = total_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def _pick_dataset(self, rng: np.random.Generator) -> int:
        return int(rng.choice(len(self.sample_weights), p=self.sample_weights))

    def _random_frame(self, rng: np.random.Generator, ds_idx: int) -> int:
        usable = self.usable[ds_idx]
        ep = int(rng.integers(0, len(usable)))
        start, end = usable[ep]
        return int(rng.integers(start, end))

    def _episode_group(self, rng: np.random.Generator, ds_idx: int, count: int) -> list[int]:
        """Sample up to ``count`` frames of one episode, pairwise ``min_frame_gap`` apart."""
        usable = self.usable[ds_idx]
        ep = int(rng.integers(0, len(usable)))
        start, end = int(usable[ep][0]), int(usable[ep][1])
        span = end - start
        if span <= 0:
            return [start]

        # A stride-based draw guarantees the minimum gap without rejection sampling.
        max_slots = max(1, span // self.min_frame_gap)
        k = int(min(count, max_slots))
        slots = rng.choice(max_slots, size=k, replace=False)
        frames = []
        for slot in slots:
            lo = start + int(slot) * self.min_frame_gap
            hi = min(lo + self.min_frame_gap, end)
            frames.append(int(rng.integers(lo, max(lo + 1, hi))))
        return frames

    def _build_batch(self, rng: np.random.Generator) -> list[tuple[int, int]]:
        batch: list[tuple[int, int]] = []
        main_ds = self._pick_dataset(rng)

        n_main = int(round(self.batch_size * self.same_dataset_frac))
        n_group = int(round(n_main * self.episode_group_frac))

        while len(batch) < n_group:
            remaining = n_group - len(batch)
            frames = self._episode_group(rng, main_ds, min(self.episode_group_size, remaining))
            batch.extend((main_ds, f) for f in frames)

        while len(batch) < n_main:
            batch.append((main_ds, self._random_frame(rng, main_ds)))

        while len(batch) < self.batch_size:
            ds_idx = self._pick_dataset(rng)
            batch.append((ds_idx, self._random_frame(rng, ds_idx)))

        batch = batch[: self.batch_size]
        perm = rng.permutation(len(batch))
        return [batch[i] for i in perm]

    def __iter__(self):
        for local_batch_id in range(self.num_batches):
            global_batch_id = local_batch_id * self.num_replicas + self.rank
            rng = np.random.default_rng(
                [self.seed, self.epoch, global_batch_id]
            )
            yield self._build_batch(rng)
