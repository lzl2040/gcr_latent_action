from types import SimpleNamespace

import numpy as np
import pytest
import torch
from datasets import Dataset

from lerobot.common.datasets.canonical_space import CANON_DIM, MAX_TACTILE_SIGNAL_DIM
from lerobot.common.datasets.contrastive_dataset import (
    MultiModalContrastiveDataset,
    contrastive_collate_fn,
)
from lerobot.common.datasets.lerobot_dataset_for_ace import LeRobotDataset
from lerobot.common.datasets_v30.lerobot_dataset import LeRobotDataset as LeRobotDatasetV30
from lerobot.common.policies.qwen3vl_mot.modeling_qwen3vl_mot import Qwen3VLMoTPolicy
from lerobot.common.policies.qwen3vl_mot.tasks import TASK_SPECS


def _norm(width, mean, std):
    mask = torch.zeros(width)
    mask[:len(mean)] = 1
    return {
        "mean": torch.tensor([*mean, *([0.0] * (width - len(mean)))]),
        "std": torch.tensor([*std, *([1.0] * (width - len(std)))]),
        "mask": mask,
    }


def _canonical_dataset():
    dataset = MultiModalContrastiveDataset.__new__(MultiModalContrastiveDataset)
    dataset.cfg = SimpleNamespace(
        dataset=SimpleNamespace(image_transforms=SimpleNamespace(img_size=2))
    )
    dataset.chunk_size = 32
    dataset.world_video_frames = 9
    dataset.rgb_views = ("primary", "wrist")
    dataset.tactile_frames = 3
    dataset.tactile_img_size = 2
    dataset.tactile_dead_std = 0.0
    dataset.max_tactile_views = 1
    dataset.specs = [{
        "action": [("action", 0, 2, 0)],
        "state": [("observation.state", 0, 2, 0)],
        "tactile_signal": ["touch_signal"],
        "tactile_image": ["touch_image"],
    }]
    tactile_norm = _norm(MAX_TACTILE_SIGNAL_DIM, [3.0], [2.0])
    tactile_norm["slots"] = [("touch_signal", 0, 1)]
    dataset.norm_stats = [{
        "action": _norm(CANON_DIM, [2.0, -3.0], [2.0, 4.0]),
        "state": _norm(CANON_DIM, [1.0, 4.0], [2.0, 3.0]),
        "tactile_signal": tactile_norm,
    }]
    dataset.image_key_maps = [{"primary": "front", "wrist": "wrist"}]
    dataset.tactile_sensor_meta = [
        (torch.zeros(1, dtype=torch.long), torch.zeros(1, 3), torch.ones(1, 3))
    ]
    dataset.dataset_names = ["tail_fixture"]
    dataset.true_fps = [20.0]
    return dataset


@pytest.mark.parametrize("version", ["v2.1", "v3.0"])
@pytest.mark.parametrize("frame_idx", [0, 8, 9, 38, 39, 40, 42])
def test_episode_windows_repeat_observations_and_zero_raw_actions(version, frame_idx):
    dataset = _canonical_dataset()
    raw_type = LeRobotDataset if version == "v2.1" else LeRobotDatasetV30
    raw = raw_type.__new__(raw_type)
    raw.dataset_name = "tail_fixture"
    raw.meta = SimpleNamespace(
        video_keys=["front", "wrist", "touch_image"],
        episodes=[
            {"dataset_from_index": 0, "dataset_to_index": 40},
            {"dataset_from_index": 40, "dataset_to_index": 43},
        ],
    )
    raw.episode_data_index = {"from": torch.tensor([0, 40]), "to": torch.tensor([40, 43])}
    raw._absolute_to_relative_idx = None
    rgb_offsets = [round(index * 31 / 8) for index in range(9)]
    raw.delta_indices = {
        "action": list(range(32)),
        "observation.state": list(range(32)),
        "touch_signal": list(range(32)),
        "front": rgb_offsets,
        "wrist": rgb_offsets,
        "touch_image": [0, 16, 31],
    }
    raw.video_keys_to_decode = raw.meta.video_keys
    source_action = torch.arange(43, dtype=torch.float32)[:, None] + torch.tensor([3.0, 4.0])
    source_state = torch.arange(43, dtype=torch.float32)[:, None] + torch.tensor([1.0, 10.0])
    source_touch = torch.arange(43, dtype=torch.float32)[:, None] * 2 + 1
    raw.hf_dataset = Dataset.from_dict({
        "episode_index": [0] * 40 + [1] * 3,
        "timestamp": [index / 20 for index in range(40)] + [index / 20 for index in range(3)],
        "action": source_action.tolist(),
        "observation.state": source_state.tolist(),
        "touch_signal": source_touch.tolist(),
    }).with_format("torch")
    episode = int(frame_idx >= 40)
    episode_end = 40 if episode == 0 else 43
    query_indices, padding = raw._get_query_indices(frame_idx, episode)
    item = {
        **raw.hf_dataset[frame_idx],
        **raw._query_hf_dataset(query_indices),
        **padding,
        "task": "move the object",
    }
    for key in raw.meta.video_keys:
        indices = torch.tensor(query_indices[key], dtype=torch.uint8)
        item[key] = indices[:, None, None, None].expand(-1, 3, 2, 2)
        assert max(query_indices[key]) < episode_end
    timestamps = raw._get_query_timestamps(float(item["timestamp"]), query_indices)
    for key in raw.meta.video_keys:
        expected = raw.hf_dataset[query_indices[key]]["timestamp"].tolist()
        np.testing.assert_allclose(timestamps[key], expected)

    result = dataset._to_canonical(item, 0, frame_idx)

    expected_action = source_action[query_indices["action"]].clone()
    expected_action[padding["action_is_pad"]] = 0
    norm = dataset.norm_stats[0]
    torch.testing.assert_close(
        result["action"][:, :2],
        (expected_action - norm["action"]["mean"][:2]) / norm["action"]["std"][:2],
    )
    torch.testing.assert_close(
        result["observation.state"][:, :2],
        (source_state[query_indices["observation.state"]] - norm["state"]["mean"][:2])
        / norm["state"]["std"][:2],
    )
    torch.testing.assert_close(
        result["tactile_signal"][:, :1],
        (source_touch[query_indices["touch_signal"]] - 3.0) / 2.0,
    )
    assert not result["action"][:, 2:].any()
    assert result["action_mask"][:2].tolist() == [1, 1]
    assert result["pair_is_valid"].item() == 1
    assert result["video_view_mask"].tolist() == [True, True]
    assert result["image_view_mask"].tolist() == [True, True]
    assert result["tactile_image_mask"].tolist() == [1]
    assert result["image_t0"][0, 0, 0].item() == frame_idx
    assert result["image_t1"][0, 0, 0].item() == min(frame_idx + 31, episode_end - 1)
    assert result["video"][:, 0, 0, 0].tolist() == query_indices["front"]
    assert result["tactile_image"][0, :, 0, 0, 0].tolist() == query_indices["touch_image"]
    if padding["action_is_pad"].any():
        assert not torch.equal(result["action"][padding["action_is_pad"]], torch.zeros_like(
            result["action"][padding["action_is_pad"]]
        ))

    batch = contrastive_collate_fn([result])
    for views in [("primary",), ("primary", "wrist")]:
        policy = SimpleNamespace(
            rgb_views=views,
            config=SimpleNamespace(tactile_generation_target="spatial_patches"),
        )
        for task in ("joint_prediction", "forward_dynamics", "inverse_dynamics"):
            valid = Qwen3VLMoTPolicy._task_valid_rows(policy, TASK_SPECS[task], batch, torch.device("cpu"))
            assert valid.tolist() == [True]


@pytest.mark.parametrize("action", [torch.tensor([7.0, 8.0]), torch.tensor([[5.0, 6.0], [7.0, 8.0]])])
def test_short_action_arrays_pad_with_raw_zero_while_states_repeat_last(action):
    dataset = _canonical_dataset()
    norm = dataset.norm_stats[0]
    actions, _ = dataset._build_canonical_vector(
        {"action": action}, dataset.specs[0]["action"], norm["action"], True, zero_pad=True
    )
    states, _ = dataset._build_canonical_vector(
        {"observation.state": action}, dataset.specs[0]["state"], norm["state"], True
    )
    raw_action = actions * norm["action"]["std"] + norm["action"]["mean"]
    raw_state = states * norm["state"]["std"] + norm["state"]["mean"]
    count = 1 if action.ndim == 1 else action.shape[0]
    torch.testing.assert_close(raw_action[:count, :2], action.reshape(-1, 2))
    assert not raw_action[count:, :2].any()
    torch.testing.assert_close(raw_state[count:, :2], action.reshape(-1, 2)[-1:].expand(32 - count, -1))


def test_action_padding_uses_each_source_keys_own_mask():
    dataset = _canonical_dataset()
    norm = _norm(CANON_DIM, [0.0] * 4, [1.0] * 4)
    actions, _ = dataset._build_canonical_vector(
        {
            "arm": torch.full((32, 2), 5.0),
            "arm_is_pad": torch.arange(32) >= 2,
            "gripper": torch.full((32, 2), 7.0),
            "gripper_is_pad": torch.arange(32) >= 1,
        },
        [("arm", 0, 2, 0), ("gripper", 0, 2, 2)],
        norm,
        True,
        zero_pad=True,
    )
    assert actions[:2, :2].eq(5).all()
    assert not actions[2:, :2].any()
    assert actions[0, 2:4].eq(7).all()
    assert not actions[1:, 2:4].any()


def test_action_padding_rejects_misaligned_time_mask():
    dataset = _canonical_dataset()
    with pytest.raises(ValueError, match="time dimension"):
        dataset._build_canonical_vector(
            {"action": torch.ones(32, 2), "action_is_pad": torch.ones(2, dtype=torch.bool)},
            dataset.specs[0]["action"],
            dataset.norm_stats[0]["action"],
            True,
            zero_pad=True,
        )


def test_short_tactile_video_preserves_frames_and_repeats_the_last():
    dataset = _canonical_dataset()
    frames = torch.tensor([30, 70], dtype=torch.uint8)[:, None, None, None].expand(-1, 3, 2, 2)
    views, mask = dataset._build_tactile_images({"touch_image": frames}, dataset.specs[0])
    assert views[0, :, 0, 0, 0].tolist() == [30, 70, 70]
    assert mask.tolist() == [1]
