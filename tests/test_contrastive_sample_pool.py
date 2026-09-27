from pathlib import Path

import numpy as np
import pytest

from lerobot.common.datasets.contrastive_dataset import MultiModalContrastiveDataset
from lerobot.common.datasets.contrastive_sample_pool import (
    build_or_load_sample_pool,
    sample_anchor_stride,
    sample_pool_checkpoint_state,
    validate_sample_pool_resume,
)
from lerobot.common.datasets.contrastive_sampler import ContrastiveBatchSampler
from lerobot.common.datasets.perception_dataset import PerceptionVideoDataset


def _build_pool(
    *,
    root: Path | None = None,
    cache_root: Path | None = None,
    horizon: int = 31,
):
    return build_or_load_sample_pool(
        data_mix="test_mix",
        dataset_names=["video_30fps"],
        dataset_sizes=[90],
        episode_ranges=[np.asarray([[0, 40], [50, 90]], dtype=np.int64)],
        true_fps=[30.0],
        horizons=[horizon],
        keep_all_below_fps=10.0,
        target_hz=5.0,
        pool_root=root,
        local_cache_root=cache_root,
    )


def test_anchor_stride_keeps_low_fps_and_targets_five_hz() -> None:
    assert sample_anchor_stride(8) == 1
    assert sample_anchor_stride(10) == 1
    assert sample_anchor_stride(12) == 2
    assert sample_anchor_stride(15) == 3
    assert sample_anchor_stride(20) == 4
    assert sample_anchor_stride(24) == 5
    assert sample_anchor_stride(30) == 6


def test_compact_pool_stores_one_row_per_usable_episode() -> None:
    pool = _build_pool()

    np.testing.assert_array_equal(
        pool.episode_table,
        np.asarray(
            [
                [0, 2, 2],
                [50, 2, 4],
            ],
            dtype=np.int64,
        ),
    )
    np.testing.assert_array_equal(pool.dataset_episode_offsets, [0, 2])
    np.testing.assert_array_equal(pool.dataset_anchor_counts, [4])
    np.testing.assert_array_equal(pool.anchor_strides, [6])
    assert pool.total_anchors == 4


def test_pool_persists_and_reloads_through_local_mmap(tmp_path: Path) -> None:
    pool_root = tmp_path / "shared"
    cache_root = tmp_path / "local"

    first = _build_pool(root=pool_root, cache_root=cache_root)
    second = _build_pool(root=pool_root, cache_root=cache_root)

    assert first.fingerprint == second.fingerprint
    assert first.storage_dir == second.storage_dir
    assert (first.storage_dir / "manifest.json").is_file()
    assert (first.storage_dir / "episodes.npy").is_file()
    assert isinstance(second.episode_table, np.memmap)
    assert str(second.episode_table.filename).startswith(str(cache_root))
    np.testing.assert_array_equal(first.episode_table, second.episode_table)


def test_pool_fingerprint_changes_with_temporal_window(tmp_path: Path) -> None:
    first = _build_pool(root=tmp_path, horizon=31)
    second = _build_pool(root=tmp_path, horizon=15)

    assert first.fingerprint != second.fingerprint


def test_checkpoint_state_requires_the_same_pool() -> None:
    pool = _build_pool()
    state = sample_pool_checkpoint_state(pool)

    validate_sample_pool_resume(pool, state)
    validate_sample_pool_resume(None, {})

    with pytest.raises(ValueError, match="predates persistent sample pools"):
        validate_sample_pool_resume(pool, {})
    with pytest.raises(ValueError, match="changed across resume"):
        validate_sample_pool_resume(
            pool,
            {"sample_pool_fingerprint": "different"},
        )
    with pytest.raises(ValueError, match="current run disabled"):
        validate_sample_pool_resume(None, state)


def test_sampler_draws_only_sparse_valid_starts() -> None:
    pool = _build_pool()
    sampler = ContrastiveBatchSampler(
        episode_ranges=[np.asarray([[0, 40], [50, 90]], dtype=np.int64)],
        sample_weights=np.asarray([1.0]),
        batch_size=16,
        num_replicas=1,
        seed=7,
        samples_per_epoch=16,
        horizon=31,
        same_dataset_frac=1.0,
        episode_group_frac=0.0,
        sample_pool=pool,
    )

    batch = next(iter(sampler))
    starts = [frame for _, frame in batch]
    assert set(starts) <= {0, 6, 50, 56}
    assert all(start + 31 < (40 if start < 50 else 90) for start in starts)


def test_perception_sampling_plan_uses_the_same_sparse_pool() -> None:
    pool = _build_pool()
    dataset = PerceptionVideoDataset.__new__(PerceptionVideoDataset)
    dataset.sample_pool = pool
    dataset.dataset_sample_counts = np.asarray([128], dtype=np.int64)
    dataset._valid_starts = []
    dataset.total_pool_anchors = pool.total_anchors

    plan = dataset._build_sampling_plan(seed=11)

    assert len(plan) == 128
    assert {frame for _, frame in plan} <= {0, 6, 50, 56}
    assert dataset.num_frames == 4


def test_pool_episode_groups_respect_frame_gap() -> None:
    pool = build_or_load_sample_pool(
        data_mix="test_mix",
        dataset_names=["video_30fps"],
        dataset_sizes=[200],
        episode_ranges=[np.asarray([[0, 200]], dtype=np.int64)],
        true_fps=[30.0],
        horizons=[31],
        keep_all_below_fps=10.0,
        target_hz=5.0,
        pool_root=None,
        local_cache_root=None,
    )

    frames = pool.sample_episode_group(
        np.random.default_rng(3),
        dataset_idx=0,
        count=8,
        min_frame_gap=32,
    )

    assert len(frames) == 5
    assert all(frame % 6 == frames[0] % 6 for frame in frames)
    assert min(abs(left - right) for index, left in enumerate(frames) for right in frames[index + 1 :]) >= 32


def test_heldout_episode_split_is_deterministic_and_excluded_from_training() -> None:
    dataset = MultiModalContrastiveDataset.__new__(
        MultiModalContrastiveDataset
    )
    dataset.dataset_names = ["train_source", "validation_only"]
    dataset.all_episode_ranges = [
        np.asarray([[0, 50], [50, 100], [100, 150], [150, 200]], dtype=np.int64),
        np.asarray([[0, 60], [60, 120], [120, 180]], dtype=np.int64),
    ]
    dataset.frame_horizons = [31, 31]
    dataset.training_sources = [True, False]
    dataset.true_fps = [30.0, 15.0]

    first = dataset._split_heldout_eval_episodes(
        ("train_source", "validation_only"),
        episodes_per_dataset=2,
        seed=17,
    )
    second = dataset._split_heldout_eval_episodes(
        ("train_source", "validation_only"),
        episodes_per_dataset=2,
        seed=17,
    )
    train_ranges, heldout_ranges, manifest = first

    for first_ranges, second_ranges in zip(first[0], second[0], strict=True):
        np.testing.assert_array_equal(first_ranges, second_ranges)
    for first_ranges, second_ranges in zip(first[1], second[1], strict=True):
        np.testing.assert_array_equal(first_ranges, second_ranges)
    assert manifest == second[2]
    assert len(train_ranges[0]) == 2
    assert len(heldout_ranges[0]) == 2
    assert len(train_ranges[1]) == 0
    assert len(heldout_ranges[1]) == 2
    assert {
        tuple(row) for row in train_ranges[0]
    }.isdisjoint({tuple(row) for row in heldout_ranges[0]})

    dataset.episode_ranges = train_ranges
    dataset.heldout_episode_ranges = heldout_ranges
    dataset.heldout_eval_manifest = manifest
    dataset.heldout_sample_pool = build_or_load_sample_pool(
        data_mix="heldout_test",
        dataset_names=dataset.dataset_names,
        dataset_sizes=[200, 180],
        episode_ranges=heldout_ranges,
        true_fps=dataset.true_fps,
        horizons=dataset.frame_horizons,
        keep_all_below_fps=10.0,
        target_hz=5.0,
        pool_root=None,
        local_cache_root=None,
    )
    dataset.heldout_eval_fingerprint = dataset.heldout_sample_pool.fingerprint

    samples = dataset.heldout_validation_samples()

    assert len(samples) == 4
    for sample in samples:
        assert sample["episode_start"] <= sample["frame_index"]
        assert (
            sample["frame_index"] + sample["horizon"]
            < sample["episode_end"]
        )
