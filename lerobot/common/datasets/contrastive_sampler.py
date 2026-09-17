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
    """Distributed batch sampler yielding lists of ``(dataset_idx, frame_idx)`` tuples.

    With ``balance_across_ranks=True``, the old rank-local hard-negative batches are first
    constructed as one global batch and then re-sharded across ranks. This preserves the exact
    global sample set while preventing one rank from receiving nearly all expensive
    tactile/video samples and becoming a straggler or OOM victim before the contrastive
    all-gather. The option is explicit because fixed evaluation intentionally preserves its
    historical rank-local batch structure.
    """

    def __init__(
        self,
        episode_ranges: list[np.ndarray],
        sample_weights: np.ndarray,
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        samples_per_epoch: int = 100_000,
        horizon: int | list[int] = 16,
        same_dataset_frac: float = 0.75,
        episode_group_frac: float = 0.75,
        episode_group_size: int = 8,
        min_frame_gap: int = 32,
        sample_costs: np.ndarray | list[float] | None = None,
        balance_across_ranks: bool = False,
    ):
        self.episode_ranges = episode_ranges
        self.sample_weights = np.asarray(sample_weights, dtype=np.float64)
        self.sample_weights = self.sample_weights / self.sample_weights.sum()
        if sample_costs is None:
            self.sample_costs = np.ones(len(self.sample_weights), dtype=np.float64)
        else:
            self.sample_costs = np.asarray(sample_costs, dtype=np.float64)
            if self.sample_costs.shape != self.sample_weights.shape:
                raise ValueError(
                    f"sample_costs has shape {self.sample_costs.shape}, expected "
                    f"{self.sample_weights.shape}."
                )
            if not np.isfinite(self.sample_costs).all() or (self.sample_costs <= 0).any():
                raise ValueError("sample_costs must contain only finite positive values.")
        self.batch_size = batch_size
        self.num_replicas = max(1, num_replicas)
        self.rank = rank
        self.seed = seed
        # One horizon per dataset: the window is a fixed duration, so its length in *frames*
        # differs per dataset (8 at fractal's 3 fps, 48 at 30 fps). Trimming every episode by a
        # single global horizon would cut low-fps datasets by far more than their own window
        # needs -- trimming fractal's ~43-frame episodes by 48 would leave almost nothing.
        if isinstance(horizon, (list, tuple, np.ndarray)):
            if len(horizon) != len(episode_ranges):
                raise ValueError(
                    f"horizon has {len(horizon)} entries but there are {len(episode_ranges)} "
                    "datasets; it must supply one horizon per dataset."
                )
            self.horizons = [int(h) for h in horizon]
        else:
            self.horizons = [int(horizon)] * len(episode_ranges)
        self.same_dataset_frac = float(np.clip(same_dataset_frac, 0.0, 1.0))
        self.episode_group_frac = float(np.clip(episode_group_frac, 0.0, 1.0))
        self.episode_group_size = max(2, episode_group_size)
        self.min_frame_gap = max(1, min_frame_gap)
        self.balance_across_ranks = bool(balance_across_ranks)
        self.epoch = 0

        # Usable frame span per episode: [start, end - horizon). Episodes too short to host a
        # full chunk fall back to their single first frame (clamping is handled downstream).
        self.usable = []
        for ds_idx, ranges in enumerate(episode_ranges):
            starts = ranges[:, 0]
            ends = np.maximum(ranges[:, 1] - self.horizons[ds_idx], starts + 1)
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

    def _global_batch(self, local_batch_id: int) -> list[tuple[int, int]]:
        """Reproduce the old per-rank batches, concatenated in virtual-rank order."""
        batch = []
        for virtual_rank in range(self.num_replicas):
            global_batch_id = local_batch_id * self.num_replicas + virtual_rank
            rng = np.random.default_rng([self.seed, self.epoch, global_batch_id])
            batch.extend(self._build_batch(rng))
        return batch

    def _balanced_rank_batches(
        self,
        global_batch: list[tuple[int, int]],
    ) -> list[list[tuple[int, int]]]:
        """Partition a fixed global batch by dataset frequency and estimated sample cost."""
        if len(global_batch) != self.batch_size * self.num_replicas:
            raise ValueError(
                f"Expected {self.batch_size * self.num_replicas} global samples, "
                f"got {len(global_batch)}."
            )

        assignments: list[list[tuple[int, tuple[int, int]]]] = [
            [] for _ in range(self.num_replicas)
        ]
        loads = np.zeros(self.num_replicas, dtype=np.float64)
        dataset_counts = np.zeros(
            (self.num_replicas, len(self.sample_weights)),
            dtype=np.int64,
        )

        # Assign expensive samples first. For each dataset, prefer the rank that has seen the
        # fewest of its samples, then the lowest total estimated load. This balances unknown
        # dataset-specific I/O as well as the known tactile-view cost.
        indexed = list(enumerate(global_batch))
        indexed.sort(
            key=lambda item: (
                -self.sample_costs[item[1][0]],
                item[0],
            )
        )
        for original_position, sample in indexed:
            ds_idx = sample[0]
            candidates = [
                rank
                for rank in range(self.num_replicas)
                if len(assignments[rank]) < self.batch_size
            ]
            rank = min(
                candidates,
                key=lambda candidate: (
                    dataset_counts[candidate, ds_idx],
                    loads[candidate],
                    len(assignments[candidate]),
                    candidate,
                ),
            )
            assignments[rank].append((original_position, sample))
            dataset_counts[rank, ds_idx] += 1
            loads[rank] += self.sample_costs[ds_idx]

        # Group each local batch by dataset to retain decoder/cache locality. Relative order
        # within a dataset follows the original global plan.
        rank_batches = []
        for assignment in assignments:
            assignment.sort(key=lambda item: (item[1][0], item[0]))
            rank_batches.append([sample for _, sample in assignment])
        return rank_batches

    def __iter__(self):
        for local_batch_id in range(self.num_batches):
            if self.num_replicas == 1 or not self.balance_across_ranks:
                global_batch_id = local_batch_id * self.num_replicas + self.rank
                rng = np.random.default_rng(
                    [self.seed, self.epoch, global_batch_id]
                )
                yield self._build_batch(rng)
                continue
            global_batch = self._global_batch(local_batch_id)
            yield self._balanced_rank_batches(global_batch)[self.rank]
