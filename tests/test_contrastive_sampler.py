from collections import Counter

import numpy as np

from lerobot.common.datasets.contrastive_sampler import ContrastiveBatchSampler


def _episode_ranges(num_datasets: int) -> list[np.ndarray]:
    return [
        np.asarray([[dataset_idx * 10_000, dataset_idx * 10_000 + 8_000]])
        for dataset_idx in range(num_datasets)
    ]


def _sampler(rank: int, num_replicas: int = 8) -> ContrastiveBatchSampler:
    return ContrastiveBatchSampler(
        episode_ranges=_episode_ranges(4),
        sample_weights=np.asarray([0.25, 0.25, 0.25, 0.25]),
        sample_costs=np.asarray([1.0, 2.0, 5.0, 7.0]),
        batch_size=64,
        num_replicas=num_replicas,
        rank=rank,
        seed=1000,
        samples_per_epoch=64 * num_replicas * 2,
        horizon=32,
        same_dataset_frac=0.75,
        episode_group_frac=0.75,
        episode_group_size=8,
        min_frame_gap=32,
        balance_across_ranks=True,
    )


def _cost(batch: list[tuple[int, int]]) -> float:
    costs = (1.0, 2.0, 5.0, 7.0)
    return sum(costs[dataset_idx] for dataset_idx, _ in batch)


def test_distributed_rebalancing_preserves_global_batch() -> None:
    reference = _sampler(rank=0)
    old_rank_batches = []
    for virtual_rank in range(reference.num_replicas):
        rng = np.random.default_rng([reference.seed, reference.epoch, virtual_rank])
        old_rank_batches.append(reference._build_batch(rng))

    balanced = [next(iter(_sampler(rank))) for rank in range(reference.num_replicas)]

    assert all(len(batch) == reference.batch_size for batch in balanced)
    assert Counter(sample for batch in balanced for sample in batch) == Counter(
        sample for batch in old_rank_batches for sample in batch
    )

    for dataset_idx in range(4):
        counts = [
            sum(sample_dataset == dataset_idx for sample_dataset, _ in batch)
            for batch in balanced
        ]
        assert max(counts) - min(counts) <= 1

    old_costs = [_cost(batch) for batch in old_rank_batches]
    balanced_costs = [_cost(batch) for batch in balanced]
    assert max(balanced_costs) - min(balanced_costs) < max(old_costs) - min(old_costs)


def test_distributed_rebalancing_is_deterministic() -> None:
    first = [next(iter(_sampler(rank))) for rank in range(8)]
    second = [next(iter(_sampler(rank))) for rank in range(8)]
    assert first == second


def test_single_rank_keeps_original_sampling_order() -> None:
    sampler = _sampler(rank=0, num_replicas=1)
    expected = sampler._build_batch(
        np.random.default_rng([sampler.seed, sampler.epoch, 0])
    )
    assert next(iter(sampler)) == expected


def test_distributed_sampler_keeps_legacy_partition_when_balancing_is_disabled() -> None:
    sampler = _sampler(rank=3)
    sampler.balance_across_ranks = False
    expected = sampler._build_batch(
        np.random.default_rng([sampler.seed, sampler.epoch, 3])
    )
    assert next(iter(sampler)) == expected


def test_sampler_can_resume_from_a_batch_offset() -> None:
    sampler = _sampler(rank=0)
    full_epoch = list(sampler)

    sampler.set_epoch(0)
    sampler.set_start_batch(1)

    assert len(sampler) == len(full_epoch) - 1
    assert list(sampler) == full_epoch[1:]


def test_setting_epoch_resets_resume_offset() -> None:
    sampler = _sampler(rank=0)
    sampler.set_start_batch(1)

    sampler.set_epoch(2)

    assert len(sampler) == sampler.num_batches
    assert sampler.start_batch == 0


def test_sampler_keeps_physical_pairs_in_mixed_video_batches() -> None:
    sampler = ContrastiveBatchSampler(
        episode_ranges=_episode_ranges(3),
        sample_weights=np.asarray([0.499, 0.499, 0.002]),
        dataset_has_physical=np.asarray([False, False, True]),
        batch_size=64,
        num_replicas=1,
        seed=1000,
        samples_per_epoch=64 * 20,
        horizon=32,
    )

    for batch in sampler:
        assert sum(dataset_idx == 2 for dataset_idx, _ in batch) >= 2


def test_rank_balancing_preserves_physical_floor_on_every_rank() -> None:
    samplers = [
        ContrastiveBatchSampler(
            episode_ranges=_episode_ranges(12),
            sample_weights=np.asarray([0.24] * 4 + [0.005] * 8),
            dataset_has_physical=np.asarray([False] * 4 + [True] * 8),
            batch_size=16,
            num_replicas=2,
            rank=rank,
            seed=2,
            samples_per_epoch=16 * 2 * 20,
            horizon=32,
            balance_across_ranks=True,
        )
        for rank in range(2)
    ]

    for rank_batches in zip(*samplers):
        for sampler, batch in zip(samplers, rank_batches, strict=True):
            assert (
                sum(
                    sampler.dataset_has_physical[dataset_idx]
                    for dataset_idx, _ in batch
                )
                >= 2
            )


def test_sampler_allows_all_video_batches_when_no_physical_dataset_exists() -> None:
    sampler = ContrastiveBatchSampler(
        episode_ranges=_episode_ranges(2),
        sample_weights=np.asarray([0.5, 0.5]),
        dataset_has_physical=np.asarray([False, False]),
        batch_size=16,
        num_replicas=1,
        seed=1000,
        samples_per_epoch=16,
        horizon=32,
    )

    batch = next(iter(sampler))
    assert len(batch) == 16
    assert not any(sampler.dataset_has_physical[dataset_idx] for dataset_idx, _ in batch)


def test_stage2_sampler_keeps_video_only_datasets_with_micro_batch_one() -> None:
    sampler = ContrastiveBatchSampler(
        episode_ranges=_episode_ranges(2),
        sample_weights=np.asarray([0.5, 0.5]),
        dataset_has_physical=np.asarray([True, False]),
        batch_size=1,
        num_replicas=4,
        rank=0,
        seed=1000,
        samples_per_epoch=400,
        horizon=32,
        min_physical_per_batch=0,
        balance_across_ranks=True,
    )

    sampled_datasets = [batch[0][0] for batch in sampler]
    assert 0 in sampled_datasets
    assert 1 in sampled_datasets
