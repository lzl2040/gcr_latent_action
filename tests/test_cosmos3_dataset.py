from types import SimpleNamespace

import numpy as np
import pytest
import torch
from datasets import Dataset
from scipy.spatial.transform import Rotation

from lerobot.common.datasets.canonical_space import CANON_DIM, get_spec
from lerobot.common.datasets.contrastive_dataset import MultiModalContrastiveDataset
from lerobot.common.policies.cosmos3.dataset_adapter import Cosmos3DataAdapter, libero_actions_to_cosmos3


class PhysicalSource(SimpleNamespace):
    def __len__(self):
        return len(self.hf_dataset)


def legacy_actions():
    controls = np.array([[0.2, -0.4, 0.5], [-0.3, 0.1, 0.4]])
    encoded = Rotation.from_euler("xyz", controls).as_matrix()[:, :, :2].reshape(2, 6)
    return np.concatenate((np.full((2, 3), 5.0), encoded, [[0.25], [0.75]]), -1), controls


def shared_dataset(names=("real_house_stack",), widths=None, use_state=False):
    widths = widths or [10] * len(names)
    dataset = object.__new__(MultiModalContrastiveDataset)
    dataset.cfg = SimpleNamespace(
        stage="finetune",
        dataset=SimpleNamespace(
            image_transforms=SimpleNamespace(img_size=2),
        ),
    )
    dataset.normalize_physical = False
    dataset.require_full_window = True
    dataset.sample_transform = None
    dataset.chunk_size = 16
    dataset.world_video_frames = 17
    dataset.rgb_views = ("primary", "wrist")
    dataset.datasets, dataset.specs, dataset.norm_stats = [], [], []
    dataset.dataset_names = list(names)
    dataset.true_fps = [20.0 if name.startswith("libero_") else 10.0 for name in names]
    dataset.image_key_maps = [{"primary": "front", "wrist": "wrist"} for _ in names]
    dataset.max_tactile_views = 1
    dataset.tactile_frames = 2
    dataset.tactile_img_size = 2
    dataset.tactile_dead_std = 0
    dataset.tactile_sensor_meta = [
        (torch.zeros(1, dtype=torch.long), torch.zeros(1, 3), torch.ones(1, 3)) for _ in names
    ]
    for name, width in zip(names, widths, strict=True):
        source = np.tile(legacy_actions()[0], (10, 1)) if width == 10 else np.full((20, width), 7.0)
        feature = {"dtype": "float32", "shape": (width,)}
        raw = PhysicalSource(
            hf_dataset=Dataset.from_dict({"action": source, "observation.state": source + 1}),
            meta=SimpleNamespace(
                features={"action": feature, "observation.state": feature},
                stats={
                    key: {"mean": np.full(width, 100), "std": np.full(width, 10)}
                    for key in ("action", "observation.state")
                },
            ),
        )
        # State rotations describe the actual pose, not Euler-encoded controls.
        if width == 10:
            state = source.copy()
            state[:, :3] += 1
            raw.hf_dataset = Dataset.from_dict({"action": source, "observation.state": state})
        dataset.datasets.append(raw)
        spec = get_spec(name, action_dim=width, state_dim=width)
        dataset.specs.append(spec)
        dataset.norm_stats.append(dataset._build_norm_stats(raw, spec))
    dataset.meta = SimpleNamespace(
        info={
            "features": {
                "observation.images.primary": {
                    "dtype": "video",
                    "shape": (2, 2, 3),
                    "names": ["height", "width", "channel"],
                },
                "action": {"dtype": "float32", "shape": (CANON_DIM,)},
            }
        },
        stats={},
    )
    config = SimpleNamespace(
        chunk_size=16,
        rgb_views=dataset.rgb_views,
        training_action_layout="auto",
        training_canonical_slots=None,
        max_action_dim=64,
        use_state=use_state,
    )
    return dataset, config


def raw_item(dataset, index=0):
    item = dataset.datasets[index].hf_dataset.with_format("torch")[:16]
    item.update(
        {
            "front": torch.zeros(17, 3, 2, 2, dtype=torch.uint8),
            "wrist": torch.zeros(17, 3, 2, 2, dtype=torch.uint8),
            "task": "move the object",
            "episode_index": torch.zeros(16, dtype=torch.long),
        }
    )
    return item


def test_source_euler_actions_become_native_column_axis_angle_actions():
    source, controls = legacy_actions()
    result = libero_actions_to_cosmos3(source)
    expected = Rotation.from_rotvec(controls).as_matrix()
    np.testing.assert_allclose(result[:, 3:6], expected[:, :, 0], atol=1e-6)
    np.testing.assert_allclose(result[:, 6:9], expected[:, :, 1], atol=1e-6)
    np.testing.assert_array_equal(result[:, 9], source[:, 9])
    with pytest.raises(ValueError, match="finite 10D"):
        libero_actions_to_cosmos3(np.full((1, 10), np.nan))


def test_shared_full_windows_trim_episode_tails_instead_of_training_on_padding():
    dataset = object.__new__(MultiModalContrastiveDataset)
    dataset.require_full_window = True
    dataset.frame_horizons = [16]
    dataset.all_episode_ranges = [np.array([[0, 20], [20, 35], [35, 52]])]
    ranges, _, _ = dataset._split_heldout_eval_episodes([], 0, 0)
    assert ranges[0].tolist() == [[0, 4], [20, 20], [35, 36]]
    assert dataset.all_episode_ranges[0].tolist() == [[0, 20], [20, 35], [35, 52]]


def test_full_window_statistics_distinguish_source_frames_from_valid_anchors():
    dataset, _ = shared_dataset()
    dataset.all_episode_ranges = [np.array([[0, 20]])]
    dataset.episode_ranges = [np.array([[0, 4]])]
    dataset.dataset_sizes = [20]
    dataset.sample_weights = [1.0]
    dataset.pool_anchor_counts = [4]
    dataset.anchor_strides = [1]
    dataset.heldout_anchor_counts = [0]
    dataset.training_sources = [True]
    dataset.has_physical = [True]
    result = dataset._build_dataset_statistics()[0]
    assert result["frames"] == 20
    assert result["anchor_count"] == 4
    assert result["anchor_ratio"] == 0.2


def test_raw_physical_interface_skips_normalization_and_stats_need_no_video_reads():
    dataset, _ = shared_dataset()
    raw, mask = dataset.get_raw_physical_values(0, "action")
    torch.testing.assert_close(raw[:, :3], torch.full((20, 3), 5.0))
    assert mask.sum() == 10 and not raw[:, 10:].any()
    normalized, _ = dataset._build_canonical_vector(
        raw_item(dataset),
        dataset.specs[0]["action"],
        dataset.norm_stats[0]["action"],
        True,
        normalize=True,
    )
    torch.testing.assert_close(normalized[:, :3], torch.full((16, 3), -9.5))
    sample = dataset._to_canonical(raw_item(dataset), 0, 0)
    torch.testing.assert_close(sample["action"], raw[:16])


def test_house_and_libero_share_readers_but_only_libero_gets_euler_reinterpretation():
    dataset, config = shared_dataset(("libero_spatial_no_noops", "real_house_stack"), use_state=True)
    adapter = Cosmos3DataAdapter(dataset, config)
    source, _ = legacy_actions()
    house = adapter(dataset._to_canonical(raw_item(dataset, 1), 1, 0))
    libero = adapter(dataset._to_canonical(raw_item(dataset, 0), 0, 0))
    expected_house = source[:, 3:9].reshape(2, 3, 2).transpose(0, 2, 1).reshape(2, 6)
    np.testing.assert_allclose(house["action"][:2, 3:9], expected_house, atol=1e-6)
    np.testing.assert_allclose(libero["action"][:2], libero_actions_to_cosmos3(source), atol=1e-6)
    np.testing.assert_allclose(libero["observation.state"][3:9], expected_house[0], atol=1e-6)
    assert house["observation.state"].shape == (10,)
    assert house["observation.state"][0] == 6
    assert house["sample_fps"] == 10 and libero["sample_fps"] == 20
    assert house["observation.images.image"].shape == (17, 3, 2, 2)
    assert config.training_canonical_slots == list(range(10))


def test_mixed_eef_and_joint_sources_keep_slots_masks_and_statistics_distinct():
    dataset, config = shared_dataset(("eef_source", "joint_source"), widths=[10, 8], use_state=True)
    adapter = Cosmos3DataAdapter(dataset, config)
    eef = adapter(dataset._to_canonical(raw_item(dataset), 0, 0))
    joints = adapter(dataset._to_canonical(raw_item(dataset, 1), 1, 0))
    assert config.training_canonical_slots == list(range(10)) + list(range(20, 28))
    assert eef["action_mask"].tolist() == [True] * 10 + [False] * 8
    assert joints["action_mask"].tolist() == [False] * 10 + [True] * 8
    assert not joints["action"][:, :10].any()
    assert dataset.meta.stats["action"]["mean"][0] == 5
    assert dataset.meta.stats["action"]["mean"][-1] == 7
    assert dataset.meta.info["features"]["observation.state"]["shape"] == (18,)
    config.training_canonical_slots = [0]
    with pytest.raises(ValueError, match="canonical slots changed"):
        Cosmos3DataAdapter(dataset, config)


@pytest.mark.parametrize("modality", ["action", "front", "wrist"])
def test_shared_native_windows_reject_any_padded_or_incomplete_modality(modality):
    dataset, _ = shared_dataset()
    item = raw_item(dataset)
    item[f"{modality}_is_pad"] = torch.zeros(item[modality].shape[0], dtype=torch.bool)
    item[f"{modality}_is_pad"][-1] = True
    with pytest.raises(ValueError, match="padded"):
        dataset._to_canonical(item, 0, 0)
    del item[f"{modality}_is_pad"]
    item[modality] = item[modality][:-1]
    with pytest.raises(ValueError, match="incomplete"):
        dataset._to_canonical(item, 0, 0)


@pytest.mark.parametrize("role,missing", [("primary", "wrist"), ("wrist", "primary")])
def test_camera_selection_is_explicit_instead_of_inventing_missing_views(role, missing):
    dataset, config = shared_dataset()
    dataset.image_key_maps[0][missing] = None
    with pytest.raises(ValueError, match="missing distinct cameras"):
        Cosmos3DataAdapter(dataset, config)
    config.rgb_views = dataset.rgb_views = (role,)
    adapter = Cosmos3DataAdapter(dataset, config)
    item = raw_item(dataset)
    item["wrist"].fill_(1)
    result = adapter(dataset._to_canonical(item, 0, 0))
    assert "observation.images.image" in result
    assert "observation.images.image2" not in result
    assert result["observation.images.image"].eq(int(role == "wrist")).all()


def test_shared_sample_hook_runs_after_rgb_augmentation_and_keeps_read_receipts():
    dataset, config = shared_dataset()
    dataset.sample_transform = Cosmos3DataAdapter(dataset, config)
    item = raw_item(dataset)
    dataset.datasets[0] = [item] * 20
    dataset.rgb_augmentation = lambda frames: frames + 1
    result = dataset._get_item(0, 0, fallback_pool=None, fallback_ranges=None, augment_rgb=True)
    assert result["observation.images.image"].eq(1).all()
    assert result["observation.images.image2"].eq(1).all()
    assert result["data_read_fallback"] == 0
    assert result["requested_frame_index"] == 0
