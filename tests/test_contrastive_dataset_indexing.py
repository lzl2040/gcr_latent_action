import numpy as np
import pytest
import torch

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
    dataset._to_canonical = lambda item, ds_idx, frame_idx: {
        "item": item,
        "dataset_id": torch.tensor(ds_idx),
        "frame_index": torch.tensor(frame_idx),
    }

    assert dataset[0]["item"] == "a0"
    assert dataset[2]["item"] == "b0"
    assert dataset[-1]["item"] == "b2"
    assert dataset[(1, 1)]["item"] == "b1"
    assert dataset[0]["data_read_fallback"].item() == 0
    assert dataset[(1, 1)]["requested_frame_index"].item() == 1


def test_getitem_marks_worker_read_fallbacks() -> None:
    class BrokenDataset:
        def __len__(self):
            return 3

        def __getitem__(self, index):
            if index == 1:
                raise RuntimeError("broken frame")
            return f"fallback-{index}"

    dataset = _dataset_with_sizes([3])
    dataset.datasets = [BrokenDataset()]
    dataset.dataset_names = ["broken"]
    dataset._random_fallback_index = lambda ds_idx, requested_frame_idx, dataset_length: 2
    dataset._to_canonical = lambda item, ds_idx, frame_idx: {
        "item": item,
        "dataset_id": torch.tensor(ds_idx),
        "frame_index": torch.tensor(frame_idx),
    }

    item = dataset[1]

    assert item["item"] == "fallback-2"
    assert item["frame_index"].item() == 2
    assert item["requested_frame_index"].item() == 1
    assert item["data_read_fallback"].item() == 1


def test_random_fallback_index_is_reproducible_and_excludes_failed_frame() -> None:
    dataset = _dataset_with_sizes([8])
    dataset.seed = 123
    dataset.epoch = 4

    first = [dataset._random_fallback_index(0, index, 8) for index in range(8)]
    second = [dataset._random_fallback_index(0, index, 8) for index in range(8)]

    assert first == second
    assert all(fallback != requested for requested, fallback in enumerate(first))
