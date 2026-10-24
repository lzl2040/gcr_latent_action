from types import SimpleNamespace

import numpy as np
import pytest
import torch
from datasets import Dataset

from lerobot.common.datasets.canonical_space import CANON_DIM, MAX_TACTILE_SIGNAL_DIM, get_spec
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
@pytest.mark.parametrize("eef_action", [False, True])
@pytest.mark.parametrize("action_padding", [None, "repeat_last"])
def test_episode_windows_apply_dataset_action_padding(version, frame_idx, eef_action, action_padding):
    dataset = _canonical_dataset()
    if action_padding is not None:
        dataset.specs[0]["action_padding"] = get_spec("robodojo")["action_padding"]
        assert dataset.specs[0]["action_padding"] == action_padding
    if eef_action:
        dataset.specs[0]["action"] = [("action", 0, 10, 0)]
        dataset.norm_stats[0]["action"] = _norm(
            CANON_DIM,
            [2.0, -3.0, 1.0, 0.999, 0.0, 0.0, 0.999, 0.0, 0.0, 0.5],
            [2.0, 4.0, 1.0, 0.002, 0.1, 0.1, 0.002, 0.1, 0.1, 0.5],
        )
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
    if eef_action:
        source_action = torch.cat([
            source_action,
            torch.zeros(43, 1),
            torch.eye(3)[:, :2].reshape(1, 6).expand(43, -1),
            torch.arange(43).remainder(2).float()[:, None],
        ], dim=-1)
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
    if action_padding is None:
        expected_action[padding["action_is_pad"]] = 0
        if eef_action:
            expected_action[padding["action_is_pad"], 3:9] = torch.eye(3)[:, :2].reshape(6)
            expected_action[padding["action_is_pad"], 9] = source_action[episode_end - 1, 9]
    norm = dataset.norm_stats[0]
    action_width = 10 if eef_action else 2
    torch.testing.assert_close(
        result["action"][:, :action_width],
        (expected_action - norm["action"]["mean"][:action_width]) / norm["action"]["std"][:action_width],
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
    assert not result["action"][:, action_width:].any()
    assert result["action_mask"][:action_width].eq(1).all()
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


def test_unknown_action_padding_fails_explicitly():
    dataset = _canonical_dataset()
    dataset.specs[0]["action_padding"] = "invalid"
    with pytest.raises(ValueError, match="Unsupported action padding"):
        dataset._to_canonical({}, 0, 0)


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
    norm = _norm(CANON_DIM, [0.0] * 20, [1.0] * 20)
    actions, _ = dataset._build_canonical_vector(
        {
            "arm": torch.full((32, 2), 5.0),
            "arm_is_pad": torch.arange(32) >= 2,
            "gripper": torch.full((32, 2), 7.0),
            "gripper_is_pad": torch.arange(32) >= 1,
        },
        [("arm", 0, 2, 0), ("gripper", 0, 1, 9), ("gripper", 1, 2, 19)],
        norm,
        True,
        zero_pad=True,
    )
    assert actions[:2, :2].eq(5).all()
    assert not actions[2:, :2].any()
    assert actions[0, [9, 19]].eq(7).all()
    assert actions[1:, [9, 19]].eq(7).all()


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


@pytest.mark.parametrize("source_frames", [1, 17, 32, 35])
@pytest.mark.parametrize("explicit_padding", [False, True])
@pytest.mark.parametrize("instructions", [
    [("action", 0, 10, 0)],
    [("action", 0, 20, 0)],
    [("action", 4, 24, 0)],
    [("xyz", 0, 3, 0), ("rotation", 2, 8, 3), ("gripper", 0, 1, 9)],
    [("left", 2, 12, 0), ("right", 4, 14, 10)],
    [("joints", 0, 8, 20), ("joints", 8, 16, 28)],
    [("eef", 0, 20, 0), ("joints", 0, 8, 20), ("joints", 8, 16, 28)],
])
def test_zero_control_padding_respects_canonical_slots_and_source_masks(
    source_frames, explicit_padding, instructions
):
    dataset = _canonical_dataset()
    source_widths = {}
    for key, _, end, _ in instructions:
        source_widths[key] = max(source_widths.get(key, 0), end)
    item = {}
    for index, (key, width) in enumerate(source_widths.items()):
        item[key] = torch.arange(source_frames * width, dtype=torch.float32).reshape(source_frames, width) + 1
        if explicit_padding:
            item[f"{key}_is_pad"] = torch.arange(source_frames) >= index + 2
    originals = {key: value.clone() for key, value in item.items()}
    norm = _norm(CANON_DIM, [0.2] * CANON_DIM, [0.3] * CANON_DIM)
    actions, mask = dataset._build_canonical_vector(item, instructions, norm, True, zero_pad=True)

    zero_control = torch.zeros(CANON_DIM)
    zero_control[3:9] = zero_control[13:19] = torch.eye(3)[:, :2].reshape(6)
    expected = torch.zeros(32, CANON_DIM)
    expected_mask = torch.zeros(CANON_DIM)
    for key, start, end, destination in instructions:
        width = end - start
        count = min(source_frames, 32)
        expected[:count, destination:destination + width] = item[key][:count, start:end]
        padded = torch.arange(32) >= source_frames
        if explicit_padding:
            padded[:count] |= item[f"{key}_is_pad"][:count]
        expected[padded, destination:destination + width] = zero_control[destination:destination + width]
        for gripper in (9, 19, 27, 35):
            if destination <= gripper < destination + width and padded.any():
                real_indices = torch.where(~padded)[0]
                expected[padded, gripper] = item[key][real_indices[-1], start + gripper - destination]
        expected_mask[destination:destination + width] = 1
    expected = (expected - norm["mean"]) / norm["std"] * expected_mask
    torch.testing.assert_close(actions, expected)
    torch.testing.assert_close(mask, expected_mask)
    for key, value in originals.items():
        torch.testing.assert_close(item[key], value)


def test_missing_actions_do_not_enable_identity_rotation_slots():
    dataset = _canonical_dataset()
    norm = _norm(CANON_DIM, [0.2] * CANON_DIM, [0.3] * CANON_DIM)
    actions, mask = dataset._build_canonical_vector(
        {}, [("action", 0, 20, 0)], norm, True, zero_pad=True
    )
    assert not actions.any()
    assert not mask.any()


def test_padded_libero_action_has_zero_translation_and_rotation_controls():
    from lerobot.evaluation.eval_qwen3vl_mot_libero import ort6d_to_controls

    dataset = _canonical_dataset()
    norm = _norm(
        CANON_DIM,
        [0.1, -0.1, 0.0, 0.998706, 0.001, 0.0, 0.997762, 0.0, 0.0, 0.5],
        [0.1, 0.1, 0.1, 0.002791, 0.01, 0.01, 0.003706, 0.01, 0.01, 0.5],
    )
    item = {"action": torch.ones(32, 10), "action_is_pad": torch.arange(32) > 0}
    actions, _ = dataset._build_canonical_vector(
        item, [("action", 0, 10, 0)], norm, True, zero_pad=True
    )
    raw = actions * norm["std"] + norm["mean"]
    assert actions[1:, :10].abs().max() < 2
    assert not raw[1:, :3].any()
    assert raw[1:, 9].eq(1).all()
    controls = ort6d_to_controls(raw[1:, 3:9].numpy())
    np.testing.assert_allclose(controls, np.zeros((31, 3)), atol=1e-7)


@pytest.mark.parametrize("gripper", [0.0, 1.0])
def test_single_action_holds_gripper_while_padding_pose_controls(gripper):
    dataset = _canonical_dataset()
    action = torch.tensor([0.3, 0.2, 0.1, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, gripper])
    norm = _norm(CANON_DIM, [0.2] * 10, [0.3] * 10)
    actions, _ = dataset._build_canonical_vector(
        {"action": action}, [("action", 0, 10, 0)], norm, True, zero_pad=True
    )
    raw = actions * norm["std"] + norm["mean"]
    torch.testing.assert_close(raw[0, :10], action)
    torch.testing.assert_close(raw[1:, :3], torch.zeros(31, 3), atol=1e-7, rtol=0)
    torch.testing.assert_close(raw[1:, 3:9], torch.eye(3)[:, :2].reshape(1, 6).expand(31, -1))
    torch.testing.assert_close(raw[1:, 9], torch.full((31,), gripper), atol=1e-7, rtol=0)


def test_gripper_padding_requires_a_real_action_value():
    dataset = _canonical_dataset()
    norm = _norm(CANON_DIM, [0.0] * 10, [1.0] * 10)
    item = {"action": torch.ones(32, 10), "action_is_pad": torch.ones(32, dtype=torch.bool)}
    with pytest.raises(ValueError, match="no real timestep"):
        dataset._build_canonical_vector(item, [("action", 0, 10, 0)], norm, True, zero_pad=True)
    norm["mask"][9] = 0
    actions, mask = dataset._build_canonical_vector(item, [("action", 0, 10, 0)], norm, True, zero_pad=True)
    assert not actions[:, 9].any()
    assert not mask[9]


def test_short_tactile_video_preserves_frames_and_repeats_the_last():
    dataset = _canonical_dataset()
    frames = torch.tensor([30, 70], dtype=torch.uint8)[:, None, None, None].expand(-1, 3, 2, 2)
    views, mask = dataset._build_tactile_images({"touch_image": frames}, dataset.specs[0])
    assert views[0, :, 0, 0, 0].tolist() == [30, 70, 70]
    assert mask.tolist() == [1]
