import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from lerobot.evaluation.eval_qwen3vl_mot_robodojo import (
    POLICY_NAME,
    build_parser,
    create_workspace,
    validate_result,
)
from lerobot.evaluation.robodojo_deploy import eval_one_episode_batch
from lerobot.evaluation.robodojo_qwen3vl import (
    RoboDojoAdapter,
    RoboDojoModel,
    camera_tensor,
    state_to_canonical,
)


def observation(index=0):
    rotation = Rotation.from_euler("xyz", [0.3, -0.2, 0.9]).as_quat()
    return {
        "env_idx": index,
        "instruction": "Stack the three bowls together.",
        "state": {
            "left_ee_pose": np.r_[[0.1, 0.2, 0.3], rotation[[3, 0, 1, 2]]],
            "right_ee_pose": np.r_[[-0.1, -0.2, 0.4], [1, 0, 0, 0]],
            "left_ee_joint_state": np.array([0.25]),
            "right_ee_joint_state": np.array([0.75]),
        },
        "vision": {
            name: {"color": np.full((4, 6, 3), value, dtype=np.uint8)}
            for name, value in (("cam_head", 17), ("cam_left_wrist", 53), ("cam_right_wrist", 91))
        },
    }


@pytest.fixture
def adapter(tmp_path):
    meta = tmp_path / "dataset/meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps({"fps": 25, "features": {"observation.state": {"shape": [16]}}})
    )
    (meta / "stats.json").write_text(
        json.dumps(
            {
                key: {"mean": [0.1] * 20, "std": [2] * 19 + [0]}
                for key in ("observation.ee_ort6d_pos", "action.ee_ort6d_pos")
            }
        )
    )
    return RoboDojoAdapter.from_dataset(meta.parent)


def config():
    return SimpleNamespace(
        max_state_dim=40,
        max_action_dim=40,
        max_tactile_views=6,
        rgb_views=("primary", "secondary", "wrist"),
        video_image_size=4,
        use_tactile_conditioning=False,
        n_action_steps=32,
    )


def test_state_uses_world_wxyz_and_interleaved_columns():
    canonical = state_to_canonical(observation())
    expected = Rotation.from_euler("xyz", [0.3, -0.2, 0.9]).as_matrix()[:, :2].reshape(6)
    np.testing.assert_allclose(canonical[3:9], expected, atol=1e-7)
    np.testing.assert_array_equal(canonical[13:19], [1, 0, 0, 1, 0, 0])
    assert canonical.shape == (20,)
    np.testing.assert_array_equal(canonical[[9, 19]], [0.25, 0.75])


@pytest.mark.parametrize("value", [[0, 0, 0, 0], [1, 1, 0, 0], [np.nan, 0, 0, 1]])
def test_invalid_native_quaternion_rejected(value):
    obs = observation()
    obs["state"]["right_ee_pose"][3:] = value
    with pytest.raises(ValueError):
        state_to_canonical(obs)


def test_normalization_masks_and_three_native_camera_roles(adapter):
    batch = adapter.observation_batch(observation(), config(), torch.device("cpu"))
    assert adapter.fps == 25
    assert adapter.state_std[-1].item() == pytest.approx(1e-3)
    torch.testing.assert_close(
        batch["observation.state"][0, :20],
        (torch.from_numpy(state_to_canonical(observation())) - 0.1) / adapter.state_std,
    )
    torch.testing.assert_close(batch["observation.state"][0, 20:], torch.zeros(20))
    torch.testing.assert_close(batch["state_mask"], batch["action_mask"])
    torch.testing.assert_close(batch["state_mask"][0], torch.tensor([1.0] * 20 + [0.0] * 20))
    assert batch["images_t0"].shape == (1, 3, 3, 4, 4)
    assert batch["images_t0"][0, :, 0, 0, 0].tolist() == [17, 53, 91]
    assert batch["image_view_mask"].all()
    assert batch["task"] == [observation()["instruction"]]


def test_no_rgb_flip_or_channel_swap():
    obs = observation()
    image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    obs["vision"]["cam_head"]["color"] = image
    tensor = camera_tensor(obs, "cam_head", 4)
    expected = torch.nn.functional.interpolate(
        torch.from_numpy(image).permute(2, 0, 1)[None].float(),
        size=(4, 4),
        mode="bilinear",
        align_corners=False,
    )[0].to(torch.uint8)
    torch.testing.assert_close(tensor, expected)


def test_missing_camera_and_instruction_fail_explicitly(adapter):
    obs = observation()
    del obs["vision"]["cam_left_wrist"]
    with pytest.raises(KeyError):
        adapter.observation_batch(obs, config(), torch.device("cpu"))
    obs = observation()
    obs["instruction"] = ""
    with pytest.raises(ValueError, match="instruction"):
        adapter.observation_batch(obs, config(), torch.device("cpu"))


def test_action_roundtrip_is_absolute_wxyz_not_joint_or_delta(adapter):
    raw = torch.from_numpy(state_to_canonical(observation()))
    normalized = torch.zeros(1, 3, 40)
    normalized[0, :, :20] = (raw - adapter.action_mean) / adapter.action_std
    actions = adapter.native_actions(normalized)
    for action in actions:
        assert set(action) == set(observation()["state"])
        for arm in ("left", "right"):
            np.testing.assert_allclose(
                action[f"{arm}_ee_pose"][:3], observation()["state"][f"{arm}_ee_pose"][:3], atol=1e-7
            )
            expected = observation()["state"][f"{arm}_ee_pose"][[4, 5, 6, 3]]
            actual = action[f"{arm}_ee_pose"][[4, 5, 6, 3]]
            np.testing.assert_allclose(
                Rotation.from_quat(actual).as_matrix(), Rotation.from_quat(expected).as_matrix(), atol=1e-7
            )
            np.testing.assert_allclose(
                action[f"{arm}_ee_joint_state"], observation()["state"][f"{arm}_ee_joint_state"]
            )


@pytest.mark.parametrize("kind", ["nan", "empty", "wrong_width", "degenerate"])
def test_invalid_predictions_are_rejected(adapter, kind):
    values = torch.zeros(1, 2, 40)
    if kind == "nan":
        values[0, 0, 0] = torch.nan
    elif kind == "empty":
        values = values[:, :0]
    elif kind == "wrong_width":
        values = values[:, :, :20]
    else:
        # Inverse normalization yields collinear columns, not a valid rotation.
        values[0, :, :20] = (torch.ones(20) - adapter.action_mean) / adapter.action_std
    with pytest.raises(ValueError):
        adapter.native_actions(values)


def test_grippers_clip_with_warning_but_translation_does_not(adapter, caplog):
    raw = torch.from_numpy(state_to_canonical(observation()))
    raw[0], raw[9], raw[19] = 123, -3, 4
    values = torch.zeros(1, 1, 40)
    values[0, 0, :20] = (raw - adapter.action_mean) / adapter.action_std
    action = adapter.native_actions(values)[0]
    assert action["left_ee_pose"][0] == pytest.approx(123)
    assert action["left_ee_joint_state"][0] == 0
    assert action["right_ee_joint_state"][0] == 1
    assert "Clipping" in caplog.text


def test_missing_converted_statistics_not_silently_replaced(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text('{"fps":25}')
    (tmp_path / "meta/stats.json").write_text('{"action": {"mean": [0]}}')
    with pytest.raises(KeyError, match="observation.ee_ort6d_pos"):
        RoboDojoAdapter.from_dataset(tmp_path)


def test_model_preserves_requested_env_order_and_resets(adapter, tmp_path):
    class Policy:
        config = config()

        def reset(self):
            pass

        def sample_canonical_action(self, batch):
            output = torch.zeros(1, 32, 40)
            raw = torch.from_numpy(state_to_canonical(observation()))
            raw[0] = batch["observation.state"][0, 0] * adapter.state_std[0] + adapter.state_mean[0]
            output[:, :, :20] = (raw - adapter.action_mean) / adapter.action_std
            return output

    model = RoboDojoModel(Policy(), adapter, device="cpu", replan_steps=10, output_dir=tmp_path / "chunks")
    model.reset()
    first, second = observation(3), observation(7)
    second["state"]["left_ee_pose"][0] = 2.5
    model.update_obs_batch([first, second])
    actions = model.get_action_batch([7, 3])
    assert [len(chunk) for chunk in actions] == [10, 10]
    assert actions[0][0]["left_ee_pose"][0] == pytest.approx(2.5)
    assert actions[1][0]["left_ee_pose"][0] == pytest.approx(0.1)
    assert len(list((tmp_path / "chunks").glob("*.npz"))) == 2
    model.reset()
    with pytest.raises(RuntimeError, match="update_obs"):
        model.get_action()
    with pytest.raises(KeyError):
        model.get_action_batch([3])


def test_batched_deploy_removes_finished_environments_and_records_each_step():
    class Env:
        steps = 0
        observations = []
        executed = []

        def is_episode_end(self):
            return self.steps == 3

        def get_running_env_idx_list(self):
            return [2, 8] if self.steps == 0 else [8]

        def get_obs_batch(self, indices):
            self.observations.append(list(indices))
            return [{"env_idx": index} for index in indices]

        def take_action_batch(self, actions, indices):
            self.executed.append((list(indices), actions))
            self.steps += 1

    class Client:
        def call(self, func_name, obs=None):
            if func_name == "get_action_batch":
                return [[f"{index}-{step}" for step in range(3)] for index in obs]
            return None

    env = Env()
    eval_one_episode_batch(env, Client())
    assert [indices for indices, _ in env.executed] == [[2, 8], [8], [8]]
    assert env.executed[-1][1] == ["8-2"]
    assert env.observations == [[2, 8], [8], [8]]


def test_workspace_isolated_from_shared_source(tmp_path):
    source = tmp_path / "source"
    for name in ("env", "env_cfg", "task", "src", "utils", "third_party", "Assets", "XPolicyLab/policy"):
        (source / name).mkdir(parents=True)
    (source / "XPolicyLab/__init__.py").write_text("")
    (source / "XPolicyLab/model_template.py").write_text("")
    workspace = tmp_path / "workspace"
    create_workspace(source, source / "Assets", workspace)
    assert (workspace / "Assets").resolve() == source / "Assets"
    assert not (source / "XPolicyLab/policy" / POLICY_NAME).exists()
    assert (workspace / "XPolicyLab/policy" / POLICY_NAME / "deploy.py").is_file()
    assert "eval_batch: true" in (workspace / "XPolicyLab/policy" / POLICY_NAME / "deploy.yml").read_text()
    with pytest.raises(FileExistsError):
        create_workspace(source, source / "Assets", workspace)


def result_fixture(tmp_path, *, layouts=(0, 1)):
    root = tmp_path / "eval_result/RoboDojo/stack_bowls" / POLICY_NAME / "arx_x5/seed/run"
    root.mkdir(parents=True)
    data = {
        "eval_time": 2,
        "success_rate": 0.5,
        "details": {str(i): {"layout_id": layout, "success": i == 0} for i, layout in enumerate(layouts)},
    }
    path = root / "_result.json"
    path.write_text(json.dumps(data))
    return path, data


def test_native_result_validation_and_layout_replacement_rejection(tmp_path):
    path, data = result_fixture(tmp_path)
    assert validate_result(tmp_path, "stack_bowls", 2) == (path, data)
    data["details"]["1"]["layout_id"] = 3
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="replaced"):
        validate_result(tmp_path, "stack_bowls", 2)


def test_incomplete_and_inconsistent_result_rejected(tmp_path):
    path, data = result_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="completed episodes"):
        validate_result(tmp_path, "stack_bowls", 3)
    data["success_rate"] = 1
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="success rate"):
        validate_result(tmp_path, "stack_bowls", 2)


def test_default_checkpoint_and_cli():
    args = build_parser().parse_args(["offline"])
    assert args.checkpoint.name == "step4k"
    assert args.replan_steps == 10
    assert args.eval_num == "1"
    assert args.sim_python.name == "python"
    assert isinstance(args.dataset_root, Path)
