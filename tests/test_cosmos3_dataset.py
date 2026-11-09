from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from lerobot.common.datasets.contrastive_dataset import MultiModalContrastiveDataset
from lerobot.common.datasets.cosmos3_dataset import Cosmos3LiberoDataset, libero_actions_to_cosmos3


def legacy_actions():
    controls = np.array([[0.2, -0.4, 0.5], [-0.3, 0.1, 0.4]])
    encoded = Rotation.from_euler("xyz", controls).as_matrix()[:, :, :2].reshape(2, 6)
    return np.concatenate((np.zeros((2, 3)), encoded, [[0.25], [0.75]]), -1), controls


def test_source_euler_actions_become_native_column_axis_angle_actions():
    source, controls = legacy_actions()
    result = libero_actions_to_cosmos3(source)
    expected = Rotation.from_rotvec(controls).as_matrix()
    np.testing.assert_allclose(result[:, 3:6], expected[:, :, 0], atol=1e-6)
    np.testing.assert_allclose(result[:, 6:9], expected[:, :, 1], atol=1e-6)
    np.testing.assert_array_equal(result[:, 9], source[:, 9])
    with pytest.raises(ValueError, match="finite 10D"):
        libero_actions_to_cosmos3(np.full((1, 10), np.nan))


def test_native_windows_trim_episode_tails_instead_of_training_on_padding():
    dataset = object.__new__(Cosmos3LiberoDataset)
    dataset.chunk_size = 16
    dataset.all_episode_ranges = [np.array([[0, 20], [20, 35], [35, 52]])]
    ranges, _, _ = dataset._split_heldout_eval_episodes([], 0, 0)
    assert ranges[0].tolist() == [[0, 4], [20, 20], [35, 36]]
    for (_, end), (start, last_anchor) in zip(dataset.all_episode_ranges[0], ranges[0], strict=True):
        if last_anchor > start:
            assert last_anchor - 1 + 16 == end - 1


def test_native_adapter_keeps_raw_actions_and_rejects_padded_windows(monkeypatch):
    source, _ = legacy_actions()
    source = torch.tensor(np.tile(source, (8, 1)), dtype=torch.float32)
    dataset = object.__new__(Cosmos3LiberoDataset)
    dataset.chunk_size = 16
    dataset.rgb_views = ("primary", "wrist")
    dataset.image_key_maps = [{"primary": "cam0", "wrist": "cam1"}]
    dataset.cfg = SimpleNamespace(policy=SimpleNamespace(training_action_layout="libero_euler_interleaved"))
    canonical = {
        "video_view_mask": torch.ones(2, dtype=torch.bool),
        "video_views": torch.zeros((2, 17, 3, 2, 2), dtype=torch.uint8),
        "task": "move",
        "has_physical": 1,
        "dataset_id": 0,
        "episode_uid": 0,
        "frame_index": 0,
        "sample_rate": 20,
        "tactile_image_mask": torch.zeros(1),
        "action": torch.full((16, 40), 999),
    }
    monkeypatch.setattr(MultiModalContrastiveDataset, "_to_canonical", lambda *args: canonical)
    item = {"action": source, "action_is_pad": torch.zeros(16, dtype=torch.bool)}
    result = dataset._to_canonical(item, 0, 0)
    assert result["observation.images.image"].shape == (17, 3, 2, 2)
    np.testing.assert_allclose(result["action"].numpy(), libero_actions_to_cosmos3(source.numpy()))
    item["action_is_pad"][-1] = True
    with pytest.raises(ValueError, match="padded"):
        dataset._to_canonical(item, 0, 0)
