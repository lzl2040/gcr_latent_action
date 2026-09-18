import numpy as np
import pytest

from lerobot.common.datasets.contrastive_dataset import MultiModalContrastiveDataset


def _dataset_with_sizes(sizes: list[int]) -> MultiModalContrastiveDataset:
    dataset = MultiModalContrastiveDataset.__new__(MultiModalContrastiveDataset)
    dataset.dataset_frame_ends = np.cumsum(np.asarray(sizes, dtype=np.int64))
    dataset.dataset_len = int(dataset.dataset_frame_ends[-1])
    dataset.epoch = 0
    return dataset


def test_flat_index_maps_across_dataset_boundaries() -> None:
    dataset = _dataset_with_sizes([3, 5, 10_000_000_000])

    assert len(dataset) == 10_000_000_008
    assert dataset._resolve_index(0) == (0, 0)
    assert dataset._resolve_index(2) == (0, 2)
    assert dataset._resolve_index(3) == (1, 0)
    assert dataset._resolve_index(7) == (1, 4)
    assert dataset._resolve_index(8) == (2, 0)
    assert dataset._resolve_index(10_000_000_007) == (2, 9_999_999_999)
    assert dataset._resolve_index(-1) == (2, 9_999_999_999)


def test_explicit_tuple_index_bypasses_flat_mapping() -> None:
    dataset = _dataset_with_sizes([3, 5])
    assert dataset._resolve_index((1, 123)) == (1, 123)
    assert dataset._resolve_index([0, 456]) == (0, 456)


def test_flat_index_rejects_out_of_range_values() -> None:
    dataset = _dataset_with_sizes([3, 5])
    with pytest.raises(IndexError):
        dataset._resolve_index(8)
    with pytest.raises(IndexError):
        dataset._resolve_index(-9)


def test_set_epoch_does_not_rebuild_or_change_flat_indexing() -> None:
    dataset = _dataset_with_sizes([3, 5])
    original_ends = dataset.dataset_frame_ends

    dataset.set_epoch(7)

    assert dataset.epoch == 7
    assert dataset.dataset_frame_ends is original_ends
    assert len(dataset) == 8


def test_getitem_uses_flat_mapping_for_direct_integer_indices() -> None:
    dataset = _dataset_with_sizes([2, 3])
    dataset.datasets = [["a0", "a1"], ["b0", "b1", "b2"]]
    dataset.dataset_names = ["a", "b"]
    dataset._to_canonical = lambda item, ds_idx, frame_idx: (item, ds_idx, frame_idx)

    assert dataset[0] == ("a0", 0, 0)
    assert dataset[2] == ("b0", 1, 0)
    assert dataset[-1] == ("b2", 1, 2)
    assert dataset[(1, 1)] == ("b1", 1, 1)
