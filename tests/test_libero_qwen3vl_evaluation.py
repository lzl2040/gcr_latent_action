import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from lerobot.evaluation.eval_qwen3vl_mot_libero import (
    DUMMY_ACTION,
    LiberoAdapter,
    checkpoint_paths,
    configure_libero,
    libero_image,
    load_dcp_model,
    ort6d_to_controls,
    rollout_episode,
    state_to_ort6d,
    write_results,
)


def observation():
    return {
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        "robot0_eef_quat": Rotation.from_euler("xyz", [2.8, 0.2, -0.3]).as_quat(),
        "robot0_gripper_qpos": np.array([0.032, -0.031]),
        "agentview_image": np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3),
        "robot0_eye_in_hand_image": np.full((4, 4, 3), 123, dtype=np.uint8),
    }


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "fps": 20,
                "features": {"action": {"shape": [10]}, "observation.state": {"shape": [10]}},
            }
        )
    )
    (root / "meta/stats.json").write_text(
        json.dumps({key: {"mean": [0] * 10, "std": [0] + [2] * 9} for key in ("observation.state", "action")})
    )
    return root


def test_state_reproduces_converter_not_true_quaternion_rotation():
    obs = observation()
    quat = obs["robot0_eef_quat"]
    axes = quat[:3] * (2 * np.arccos(quat[3]) / np.sqrt(1 - quat[3] ** 2))
    expected = np.r_[
        obs["robot0_eef_pos"], Rotation.from_euler("xyz", axes).as_matrix()[:, :2].reshape(6), -0.031
    ]
    actual = state_to_ort6d(obs)
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    assert not np.allclose(actual[3:9], Rotation.from_quat(quat).as_matrix()[:, :2].reshape(6))
    np.testing.assert_array_equal(obs["robot0_eef_quat"], quat)


def test_identity_quaternion_and_negative_w_match_reference():
    obs = observation()
    obs["robot0_eef_quat"] = np.array([0, 0, 0, 1.0])
    np.testing.assert_array_equal(state_to_ort6d(obs)[3:9], [1, 0, 0, 1, 0, 0])
    obs["robot0_eef_quat"] = np.array([np.sin(2), 0, 0, np.cos(2)])
    expected = Rotation.from_euler("xyz", [4, 0, 0]).as_matrix()[:, :2].reshape(6)
    np.testing.assert_allclose(state_to_ort6d(obs)[3:9], expected, atol=1e-7)


@pytest.mark.parametrize(
    "key,value",
    [
        ("robot0_eef_pos", [0, 0, np.nan]),
        ("robot0_eef_quat", [0, 0, 0, 2]),
        ("robot0_gripper_qpos", [0.03]),
    ],
)
def test_invalid_native_state_is_rejected(key, value):
    obs = observation()
    obs[key] = value
    with pytest.raises(ValueError):
        state_to_ort6d(obs)


def test_interleaved_rotation_roundtrip_and_projection():
    axes = np.array([[0.2, -0.1, 0.3], [-0.05, 0.25, -0.3], [0, 0, 0]])
    matrix = Rotation.from_euler("xyz", axes).as_matrix()
    columns = matrix[:, :, :2].copy()
    columns[:, :, 1] = 3 * columns[:, :, 1] + 0.3 * columns[:, :, 0]
    columns[:, :, 0] *= 2
    np.testing.assert_allclose(ort6d_to_controls(columns.reshape(-1, 6)), axes, atol=1e-8)


@pytest.mark.parametrize(
    "rotation",
    [
        [[0] * 6],
        [[1, 2, 0, 0, 0, 0]],
        [[1, 0, 0, 1, 0, np.nan]],
        [[1] * 7],
    ],
)
def test_degenerate_rotation_is_rejected(rotation):
    with pytest.raises(ValueError):
        ort6d_to_controls(np.array(rotation))


def test_suite_normalization_and_multiview_masks(dataset):
    adapter = LiberoAdapter.from_dataset(dataset)
    assert adapter.fps == 20
    assert adapter.state_std[0] == pytest.approx(1e-3)
    config = SimpleNamespace(
        max_state_dim=40, max_action_dim=40, rgb_views=("primary", "secondary", "wrist"), video_image_size=4
    )
    obs = observation()
    batch = adapter.observation_batch(obs, "pick up the cup", config, torch.device("cpu"))
    torch.testing.assert_close(
        batch["observation.state"][0, :10], torch.from_numpy(state_to_ort6d(obs)) / adapter.state_std
    )
    assert torch.count_nonzero(batch["observation.state"][0, 10:]) == 0
    assert batch["action_mask"].sum() == batch["state_mask"].sum() == 10
    assert batch["image_view_mask"].tolist() == [[True, False, True]]
    assert batch["images_t0"].shape == (1, 3, 3, 4, 4)
    assert batch["images_t0"].dtype == torch.uint8
    assert not batch["images_t0"][0, 1].any()
    assert batch["sample_rate"].item() == 20
    assert batch["task"] == ["pick up the cup"]
    torch.testing.assert_close(batch["image_t0"], batch["images_t0"][:, 0])


def test_image_rotation_and_resize_match_training():
    obs = observation()
    image = libero_image(obs, "agentview_image", 4)
    np.testing.assert_array_equal(image.permute(1, 2, 0).numpy(), obs["agentview_image"][::-1, ::-1])
    assert libero_image(obs, "agentview_image", 8).shape == (3, 8, 8)


def test_native_actions_inverse_normalization_euler_and_gripper(dataset, caplog):
    adapter = LiberoAdapter.from_dataset(dataset)
    axes = np.array([[0.2, -0.1, 0.3], [0, 0, 0], [-0.2, 0.1, -0.1]])
    raw = np.c_[
        np.array([[0.1, -0.2, 0.3], [2, 0, 0], [-0.1, 0.2, -0.3]]),
        Rotation.from_euler("xyz", axes).as_matrix()[:, :, :2].reshape(-1, 6),
        [0, 1, 0.5],
    ]
    normalized = torch.zeros(1, 3, 40)
    normalized[0, :, :10] = (
        torch.tensor(raw, dtype=torch.float32) - adapter.action_mean
    ) / adapter.action_std
    actions = adapter.native_actions(normalized)
    np.testing.assert_allclose(actions[:, 3:6], axes, atol=1e-7)
    np.testing.assert_array_equal(actions[:, 6], [1, -1, 0])
    np.testing.assert_allclose(actions[:, :3], np.clip(raw[:, :3], -1, 1), atol=1e-7)
    assert "Clipping 1 predicted control values" in caplog.text
    normalized[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="nonfinite"):
        adapter.native_actions(normalized)
    normalized[0, 0, 0] = 0
    normalized[0, 0, 1] = torch.finfo(torch.float32).max
    with pytest.raises(ValueError, match="Inverse-normalized"):
        adapter.native_actions(normalized)


def test_stats_wrong_width_and_missing_stats_fail(dataset):
    path = dataset / "meta/info.json"
    info = json.loads(path.read_text())
    info["features"]["action"]["shape"] = [7]
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="10D"):
        LiberoAdapter.from_dataset(dataset)
    (dataset / "meta/stats.json").unlink()
    with pytest.raises(FileNotFoundError):
        LiberoAdapter.from_dataset(dataset)


def test_dcp_single_process_strict_load(tmp_path):
    import torch.distributed.checkpoint as dcp

    model = torch.nn.Linear(3, 2)
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    dcp.save({"model": expected}, checkpoint_id=tmp_path)
    restored = torch.nn.Linear(3, 2)
    load_dcp_model(restored, tmp_path)
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected[key])
    with pytest.raises(ValueError, match="shape/type mismatch"):
        load_dcp_model(torch.nn.Linear(4, 2), tmp_path)
    with pytest.raises(ValueError, match="model keys differ"):
        load_dcp_model(torch.nn.Sequential(torch.nn.Linear(3, 2)), tmp_path)


def test_checkpoint_config_discovery_and_missing_config(tmp_path):
    model = tmp_path / "step12k/model"
    model.mkdir(parents=True)
    (model / ".metadata").touch()
    config = tmp_path / "config.json"
    config.write_text("{}")
    assert checkpoint_paths(model.parent, None) == (model, config)
    assert checkpoint_paths(model, None) == (model, config)
    config.unlink()
    with pytest.raises(FileNotFoundError, match="saved policy config"):
        checkpoint_paths(model.parent, None)


class FakeEnv:
    def __init__(self, success_at=None, done_at=None):
        self.success_at = success_at
        self.done_at = done_at
        self.history = []

    def seed(self, seed):
        self.saved_seed = seed

    def reset(self):
        self.history = []

    def set_init_state(self, state):
        self.initial_state = state
        return observation()

    def step(self, action):
        self.history.append(action)
        return observation(), 0, self.done_at == len(self.history), {}

    def check_success(self):
        return self.success_at is not None and len(self.history) >= self.success_at


def rollout(env, plan, **kwargs):
    return rollout_episode(
        env,
        "fixed-state",
        plan,
        seed=123,
        max_steps=5,
        wait_steps=2,
        replan_steps=3,
        save_video=True,
        **kwargs,
    )


def test_episode_settling_replan_and_timeout():
    env = FakeEnv()
    calls = []

    def plan(obs):
        calls.append(obs)
        return np.arange(28).reshape(4, 7) / 100

    result, frames = rollout(env, plan)
    assert env.initial_state == "fixed-state" and env.saved_seed == 123
    assert env.history[:2] == [DUMMY_ACTION, DUMMY_ACTION]
    assert result["policy_calls"] == len(calls) == 2
    assert result["steps"] == 5 and result["timeout"] and not result["success"]
    assert env.history[2] == env.history[5]
    assert len(frames) == 6
    np.testing.assert_array_equal(frames[0], observation()["agentview_image"][::-1, ::-1])
    rollout(env, plan)
    assert env.history[2] == result["actions"][0]


def test_success_is_not_the_environment_done_flag():
    def plan(obs):
        return np.zeros((3, 7))

    result, _ = rollout(FakeEnv(done_at=4), plan)
    assert result["environment_done"] and not result["success"] and not result["timeout"]
    result, _ = rollout(FakeEnv(success_at=4), plan)
    assert result["success"] and result["steps"] == 2 and not result["timeout"]


def test_invalid_chunk_and_inference_error_are_not_success_fallbacks():
    with pytest.raises(ValueError, match="finite 7D"):
        rollout(FakeEnv(), lambda obs: np.zeros((1, 7)))

    def broken_plan(obs):
        raise RuntimeError("inference failed")

    with pytest.raises(RuntimeError, match="inference failed"):
        rollout(FakeEnv(), broken_plan)


def test_results_count_episodes_per_task_and_are_persistent(tmp_path):
    path = tmp_path / "results.json"
    assert write_results(path, {}, [])["success_rate"] is None
    episodes = [
        {"task_id": 0, "task_name": "a", "success": True},
        {"task_id": 0, "task_name": "a", "success": False},
        {"task_id": 1, "task_name": "b", "success": False},
    ]
    result = write_results(path, {"seed": 0, "expected_episodes": 3}, episodes)
    assert result["success_rate"] == 1 / 3
    assert result["tasks"]["0"]["success_rate"] == 0.5
    assert result == json.loads(path.read_text())
    assert result["evaluation_complete"]
    assert not write_results(path, {"expected_episodes": 4}, episodes)["evaluation_complete"]
    assert not path.with_suffix(".json.tmp").exists()


def test_libero_configuration_does_not_modify_shared_home(tmp_path, monkeypatch):
    root = tmp_path / "libero_repo"
    benchmark = root / "libero/libero"
    for name in ("bddl_files", "init_files", "assets"):
        (benchmark / name).mkdir(parents=True)
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr("sys.path", list(__import__("sys").path))
    for key in ("LIBERO_CONFIG_PATH", "MUJOCO_GL", "PYOPENGL_PLATFORM", "MUJOCO_EGL_DEVICE_ID"):
        monkeypatch.setenv(key, "old")
    configure_libero(root, output, 3)
    import os

    assert Path(os.environ["LIBERO_CONFIG_PATH"]) == output / "libero_config"
    assert os.environ["MUJOCO_GL"] == "egl"
    assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "3"
    assert (output / "libero_config/config.yaml").is_file()
