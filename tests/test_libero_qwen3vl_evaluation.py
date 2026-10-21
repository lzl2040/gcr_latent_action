import json
import random
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from lerobot.evaluation.eval_qwen3vl_mot_libero import (
    DUMMY_ACTION,
    LiberoAdapter,
    PredictionVideoWriter,
    build_parser,
    checkpoint_paths,
    configure_libero,
    libero_image,
    load_dcp_model,
    ort6d_to_controls,
    prepare_output_directory,
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
        max_state_dim=40,
        max_action_dim=40,
        rgb_views=("primary", "secondary", "wrist"),
        video_image_size=4,
        max_tactile_views=6,
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
    assert batch["tactile_image_mask"].shape == (1, 6)
    assert not batch["tactile_image_mask"].any()
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
        raise AssertionError("Episode resets must not reseed the environment.")

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
        max_steps=5,
        wait_steps=2,
        replan_steps=3,
        save_video=True,
        **kwargs,
    )


def test_episode_settling_replan_and_timeout():
    env = FakeEnv()
    calls = []

    def plan(obs, **context):
        calls.append(context)
        return np.arange(28).reshape(4, 7) / 100

    result, frames = rollout(env, plan)
    assert env.initial_state == "fixed-state"
    assert env.history[:2] == [DUMMY_ACTION, DUMMY_ACTION]
    assert result["policy_calls"] == len(calls) == 2
    assert calls == [
        {"control_step": 0, "replan_index": 0},
        {"control_step": 3, "replan_index": 1},
    ]
    assert result["steps"] == 5 and result["timeout"] and not result["success"]
    assert env.history[2] == env.history[5]
    assert len(frames) == 6
    np.testing.assert_array_equal(frames[0], observation()["agentview_image"][::-1, ::-1])
    rollout(env, plan)
    assert env.history[2] == result["actions"][0]


@pytest.mark.parametrize("seed", [0, 123])
def test_episode_rng_continues_across_replans_and_episodes(seed):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    runs = []
    try:
        with torch.random.fork_rng(devices=[]):
            for _ in range(2):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                draws = []

                def plan(obs, *, draws=draws, **context):
                    draws.append((random.random(), np.random.random(), torch.rand(()).item()))
                    return np.zeros((3, 7))

                env = FakeEnv()
                for _ in range(2):
                    rollout(env, plan)
                runs.append(draws)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    python_rng = random.Random(seed)
    numpy_rng = np.random.RandomState(seed)
    torch_rng = torch.Generator().manual_seed(seed)
    expected = [
        (python_rng.random(), numpy_rng.random(), torch.rand((), generator=torch_rng).item())
        for _ in range(4)
    ]
    assert runs == [expected, expected]
    assert expected[:2] != expected[2:]


def test_default_replanning_executes_ten_of_thirty_two_actions():
    args = build_parser().parse_args(["--checkpoint", "/weights/step12k"])
    assert args.replan_steps == 10
    calls, chunks = [], []

    def plan(obs, **context):
        calls.append(context)
        chunk = np.arange(32 * 7).reshape(32, 7) / 1000 + len(chunks)
        chunks.append(chunk)
        return chunk

    result, _ = rollout_episode(
        FakeEnv(),
        "fixed-state",
        plan,
        max_steps=25,
        wait_steps=0,
        replan_steps=args.replan_steps,
        save_video=False,
    )
    assert result["steps"] == 25 and result["policy_calls"] == 3
    assert calls == [
        {"control_step": 0, "replan_index": 0},
        {"control_step": 10, "replan_index": 1},
        {"control_step": 20, "replan_index": 2},
    ]
    np.testing.assert_array_equal(
        result["actions"], np.concatenate([chunks[0][:10], chunks[1][:10], chunks[2][:5]])
    )


def test_success_is_not_the_environment_done_flag():
    def plan(obs, **context):
        return np.zeros((3, 7))

    result, _ = rollout(FakeEnv(done_at=4), plan)
    assert result["environment_done"] and not result["success"] and not result["timeout"]
    result, _ = rollout(FakeEnv(success_at=4), plan)
    assert result["success"] and result["steps"] == 2 and not result["timeout"]


def test_invalid_chunk_and_inference_error_are_not_success_fallbacks():
    with pytest.raises(ValueError, match="finite 7D"):
        rollout(FakeEnv(), lambda obs, **context: np.zeros((1, 7)))

    def broken_plan(obs, **context):
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


class FakeVideoDecoder:
    def __init__(self, frames=9):
        self.calls = 0
        self.frames = frames

    def decode(self, latents):
        self.calls += 1
        video = torch.linspace(-1, 1, self.frames).view(1, 1, -1, 1, 1).expand(1, 3, -1, 4, 4)
        return SimpleNamespace(sample=video)


def test_prediction_videos_per_replan_multiview_timing_and_output_directory(tmp_path):
    import imageio.v2 as imageio

    config = SimpleNamespace(
        rgb_views=("primary", "secondary", "wrist"),
        chunk_size=32,
        world_video_frames=9,
        video_image_size=4,
    )
    writer = PredictionVideoWriter(config, torch.device("cpu"), source_fps=20)
    decoder = FakeVideoDecoder()
    writer.decoder = decoder
    prediction = {
        "video_latents_views": torch.zeros(1, 3, 48, 3, 1, 1),
        "video_view_mask": torch.tensor([[True, False, True]]),
    }
    actions = np.zeros((32, 7))
    actions[:, 0] = np.arange(32) / 100
    for index, step in enumerate((0, 8)):
        directory = (
            tmp_path / "custom_results/predictions/task_00_trial_000" / f"replan_{index:04d}_step_{step:04d}"
        )
        metadata = writer.save(
            prediction, actions, directory, control_step=step, replan_index=index, seed=index
        )
        assert metadata["replan_index"] == index and metadata["control_step"] == step
        assert metadata["run_seed"] == index
        assert metadata["sampling_rng"] == "run_seeded_continuous"
        assert metadata["video_frame_times_s"] == [0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.15, 1.35, 1.55]
        assert metadata["video_fps"] == pytest.approx(160 / 31)
        assert metadata["includes_current_frame"] and metadata["video_frames"] == 9
        assert set(metadata["videos"]) == {"primary", "wrist"}
        assert not (directory / "secondary.mp4").exists()
        assert metadata == json.loads((directory / "metadata.json").read_text())
        np.testing.assert_allclose(np.load(metadata["action_chunk_path"]), actions)
        for path in metadata["videos"].values():
            assert Path(path).parent == directory
            with imageio.get_reader(path) as video:
                assert video.count_frames() == 9
                assert video.get_meta_data()["fps"] == pytest.approx(160 / 31, abs=0.01)
                assert video.get_data(0).max() == 0
                assert video.get_data(8).min() == 255
    assert decoder.calls == 4


def test_single_view_predicted_video_and_invalid_outputs(tmp_path):
    config = SimpleNamespace(
        rgb_views=("primary",),
        chunk_size=32,
        world_video_frames=9,
        video_image_size=4,
    )
    writer = PredictionVideoWriter(config, torch.device("cpu"), source_fps=20)
    writer.decoder = FakeVideoDecoder()
    prediction = {"video_latents": torch.zeros(1, 48, 3, 1, 1)}
    metadata = writer.save(
        prediction, np.zeros((32, 7)), tmp_path / "first", control_step=0, replan_index=0, seed=0
    )
    assert set(metadata["videos"]) == {"primary"}
    writer.decoder.frames = 8
    with pytest.raises(ValueError, match="Decoded primary video"):
        writer.save(
            prediction, np.zeros((32, 7)), tmp_path / "wrong_frames", control_step=0, replan_index=0, seed=0
        )
    prediction["video_latents"][0, 0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="latents contain nonfinite"):
        writer.save(
            prediction, np.zeros((32, 7)), tmp_path / "bad_latents", control_step=0, replan_index=0, seed=0
        )
    assert not (tmp_path / "bad_latents").exists()


def test_prediction_png_frames_are_exact_and_filmstrip_preserves_order(tmp_path):
    import imageio.v2 as imageio

    config = SimpleNamespace(
        rgb_views=("primary",),
        chunk_size=32,
        world_video_frames=9,
        video_image_size=4,
    )
    writer = PredictionVideoWriter(config, torch.device("cpu"), source_fps=25)
    writer.decoder = FakeVideoDecoder()
    metadata = writer.save(
        {"video_latents": torch.zeros(1, 48, 3, 1, 1)},
        np.zeros((32, 16)),
        tmp_path / "prediction",
        control_step=10,
        replan_index=1,
        seed=0,
        save_frames=True,
    )
    assert metadata["video_fps"] == pytest.approx(200 / 31)
    assert metadata["video_frame_times_s"][-1] == pytest.approx(1.24)
    frames = [imageio.imread(path) for path in metadata["frames"]["primary"]]
    assert len(frames) == 9
    for index, frame in enumerate(frames):
        np.testing.assert_array_equal(frame, np.full((4, 4, 3), round(index * 255 / 8), dtype=np.uint8))
    np.testing.assert_array_equal(
        imageio.imread(tmp_path / "prediction/primary_filmstrip.png"), np.concatenate(frames, axis=1)
    )
    assert metadata == json.loads((tmp_path / "prediction/metadata.json").read_text())


@pytest.mark.parametrize("existing", [False, True])
def test_output_directory_creates_missing_or_empty_directory(tmp_path, existing):
    output = tmp_path / "results/run"
    if existing:
        output.mkdir(parents=True)
    assert prepare_output_directory(output) == output.resolve()
    assert output.is_dir() and not list(output.iterdir())


def test_output_directory_overwrite_clears_all_old_artifacts_only_in_target(tmp_path, caplog):
    output = tmp_path / "results/run"
    output.mkdir(parents=True)
    sibling = tmp_path / "results/other_run"
    sibling.mkdir()
    (sibling / "results.json").write_text("keep")
    for name in (
        "results.json",
        "results.json.tmp",
        "evaluation.log",
        "task_09_trial_019_actions.npy",
        "task_09_trial_019_failure.mp4",
        "old-file.txt",
        "predictions/task_09_trial_019/replan_0000_step_0000/primary.mp4",
        "first_plans/task_09_trial_019.npz",
        "libero_config/config.yaml",
    ):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("old")
    assert prepare_output_directory(output) == output.resolve()
    assert not list(output.iterdir())
    assert (sibling / "results.json").read_text() == "keep"
    assert "Overwriting evaluation output directory" in caplog.text


@pytest.mark.parametrize("dangling", [False, True])
def test_output_directory_refuses_symlinks(tmp_path, dangling):
    target = tmp_path / "original"
    if not dangling:
        target.mkdir()
        (target / "keep.txt").write_text("keep")
    output = tmp_path / "symlink"
    output.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        prepare_output_directory(output)
    assert output.is_symlink()
    if dangling:
        assert not target.exists()
    else:
        assert (target / "keep.txt").read_text() == "keep"


def test_output_directory_protects_input_symlinks_inside_output(tmp_path):
    checkpoint = tmp_path / "weights/model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "keep.txt").write_text("keep")
    output = tmp_path / "evaluation"
    output.mkdir()
    reference = output / "checkpoint"
    reference.symlink_to(checkpoint, target_is_directory=True)
    with pytest.raises(ValueError, match="evaluation inputs"):
        prepare_output_directory(output, protected_paths=(reference,))
    assert reference.is_symlink()
    assert (checkpoint / "keep.txt").read_text() == "keep"


def test_output_directory_refuses_mount_points(tmp_path, monkeypatch):
    output = tmp_path / "mount"
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    monkeypatch.setattr(Path, "is_mount", lambda self: self == output)
    with pytest.raises(ValueError, match="protected"):
        prepare_output_directory(output)
    assert (output / "keep.txt").read_text() == "keep"


@pytest.mark.parametrize("kind", ["root", "top-level", "home", "cwd", "repository", "input", "input-parent"])
def test_output_directory_refuses_protected_paths_before_deleting(tmp_path, monkeypatch, kind):
    import lerobot.evaluation.eval_qwen3vl_mot_libero as evaluation

    checkpoint = tmp_path / "weights/checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "keep.txt").write_text("keep")
    paths = {
        "root": Path("/"),
        "top-level": Path("/tmp"),
        "home": Path.home(),
        "cwd": Path.cwd(),
        "repository": Path(evaluation.__file__).resolve().parents[2],
        "input": checkpoint,
        "input-parent": checkpoint.parent,
    }
    remove = Mock(side_effect=AssertionError("Protected paths must never be deleted."))
    monkeypatch.setattr(evaluation.shutil, "rmtree", remove)
    with pytest.raises(ValueError, match="protected"):
        prepare_output_directory(paths[kind], protected_paths=(checkpoint,))
    remove.assert_not_called()
    assert (checkpoint / "keep.txt").read_text() == "keep"


def test_output_directory_refuses_existing_file_and_preserves_cleanup_errors(tmp_path, monkeypatch):
    import lerobot.evaluation.eval_qwen3vl_mot_libero as evaluation

    output = tmp_path / "result"
    output.write_text("keep")
    with pytest.raises(FileExistsError, match="not a directory"):
        prepare_output_directory(output)
    assert output.read_text() == "keep"
    output.unlink()
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    monkeypatch.setattr(evaluation.shutil, "rmtree", Mock(side_effect=PermissionError("cleanup denied")))
    with pytest.raises(PermissionError, match="cleanup denied"):
        prepare_output_directory(output)
    assert (output / "keep.txt").read_text() == "keep"


@pytest.mark.parametrize(
    "input_flag",
    [
        "--checkpoint",
        "--policy-config",
        "--dataset-root",
        "--libero-root",
        "--qwen3vl-dir",
        "--cosmos3-dir",
        "--stage1-text-dir",
        "--fp8-training-config",
    ],
)
def test_main_protects_all_input_paths_from_output_overwrite(tmp_path, monkeypatch, input_flag):
    import lerobot.evaluation.eval_qwen3vl_mot_libero as evaluation

    output = tmp_path / "evaluation"
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    monkeypatch.setattr(evaluation.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(evaluation.torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(evaluation.torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(
        evaluation.torch.backends.cuda.matmul,
        "allow_tf32",
        evaluation.torch.backends.cuda.matmul.allow_tf32,
    )
    command = ["--output-dir", str(output), input_flag, str(output / "input")]
    if input_flag != "--checkpoint":
        command.extend(["--checkpoint", "/safe/checkpoint"])
    with pytest.raises(ValueError, match="evaluation inputs"):
        evaluation.main(command)
    assert (output / "keep.txt").read_text() == "keep"


def test_output_directory_and_independent_video_switches():
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "/weights/step12k",
            "--output-dir",
            "/tmp/custom_libero_results",
            "--no-save-video",
        ]
    )
    assert args.output_dir == Path("/tmp/custom_libero_results")
    assert args.no_save_video and not args.no_save_predicted_video
    args = build_parser().parse_args(["--checkpoint", "/weights/step12k", "--no-save-predicted-video"])
    assert not args.no_save_video and args.no_save_predicted_video


def test_stop_on_success_is_opt_in():
    args = build_parser().parse_args(["--checkpoint", "/weights/step18k"])
    assert not args.stop_on_success
    args = build_parser().parse_args(["--checkpoint", "/weights/step18k", "--stop-on-success"])
    assert args.stop_on_success


def test_actual_video_modes_are_mutually_exclusive():
    parser = build_parser()
    args = parser.parse_args(["--checkpoint", "/weights/step18k", "--save-failed-videos-only"])
    assert args.save_failed_videos_only and not args.no_save_video
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--checkpoint", "/weights/step18k", "--save-failed-videos-only", "--no-save-video"]
        )


@pytest.mark.parametrize(
    ("video_flags", "saved_names"),
    [
        (
            [],
            [
                "task_00_trial_000_success.mp4",
                "task_00_trial_001_failure.mp4",
                "task_01_trial_000_success.mp4",
                "task_01_trial_001_failure.mp4",
            ],
        ),
        (
            ["--save-failed-videos-only"],
            ["task_00_trial_001_failure.mp4", "task_01_trial_001_failure.mp4"],
        ),
        (["--no-save-video"], []),
    ],
)
def test_main_actual_video_saving_modes(tmp_path, monkeypatch, video_flags, saved_names):
    import imageio.v2 as imageio

    import lerobot.evaluation.eval_qwen3vl_mot_libero as evaluation

    closed, saved, draws = [], [], []
    suite = SimpleNamespace(
        n_tasks=2,
        get_task_init_states=lambda _: [0, 1],
        get_task=lambda _: SimpleNamespace(name="toy", language="toy task"),
        get_task_bddl_file_path=lambda _: "/unused/task.bddl",
    )
    package = ModuleType("libero")
    libero = ModuleType("libero.libero")
    envs = ModuleType("libero.libero.envs")
    package.libero = libero
    libero.benchmark = SimpleNamespace(get_benchmark_dict=lambda: {"libero_goal": lambda **kwargs: suite})
    envs.OffScreenRenderEnv = lambda **kwargs: SimpleNamespace(close=lambda: closed.append(True))
    for name, module in (("libero", package), ("libero.libero", libero), ("libero.libero.envs", envs)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(evaluation, "configure_libero", lambda *args: None)
    monkeypatch.setattr(evaluation, "load_libero_initial_states", lambda suite, task_id: [0, 1])
    monkeypatch.setattr(evaluation.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(evaluation.torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(evaluation.torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(
        evaluation.torch.backends.cuda.matmul,
        "allow_tf32",
        evaluation.torch.backends.cuda.matmul.allow_tf32,
    )
    config = SimpleNamespace(
        inference_steps=20, rgb_views=("primary",), task_names=("joint_prediction",), video_image_size=4
    )
    monkeypatch.setattr(
        evaluation,
        "load_policy",
        lambda args: SimpleNamespace(
            config=config, reset=lambda: None, libero_precision_details={"precision": "bf16"}
        ),
    )
    monkeypatch.setattr(evaluation.LiberoAdapter, "from_dataset", lambda path: SimpleNamespace(fps=20))

    def fake_rollout(env, initial_state, plan, **kwargs):
        assert "seed" not in kwargs
        draws.append((random.random(), np.random.random(), torch.rand(()).item()))
        return {
            "success": initial_state == 0,
            "steps": 1,
            "policy_calls": 0,
            "inference_seconds": 0.0,
            "actions": [[0.0] * 7],
        }, [np.zeros((4, 4, 3), dtype=np.uint8)] if kwargs["save_video"] else []

    monkeypatch.setattr(evaluation, "rollout_episode", fake_rollout)
    monkeypatch.setattr(imageio, "mimsave", lambda path, frames, **kwargs: saved.append(path))
    seeders = [
        (module, Mock(wraps=module.seed if module is not torch else module.manual_seed))
        for module in (random, np.random, torch)
    ]
    for module, seeder in seeders:
        monkeypatch.setattr(module, "manual_seed" if module is torch else "seed", seeder)
    output = tmp_path / "evaluation"
    output.mkdir()
    (output / "results.json").write_text('{"old": true}')
    (output / "evaluation.log").write_text("old evaluation log")
    stale = output / "predictions/task_09_trial_019/old.mp4"
    stale.parent.mkdir(parents=True)
    stale.write_text("old")
    (output / "task_09_trial_019_actions.npy").write_text("old")
    command = [
        "--checkpoint",
        "/weights/step18k",
        "--trials-per-task",
        "2",
        "--seed",
        "123",
        "--no-save-predicted-video",
        "--output-dir",
        str(output),
        *video_flags,
    ]
    evaluation.main(command)
    assert closed == [True, True]
    assert not (output / "predictions").exists()
    assert not (output / "task_09_trial_019_actions.npy").exists()
    assert "old evaluation log" not in (output / "evaluation.log").read_text()
    assert [path.name for path in saved] == saved_names
    result = json.loads((output / "results.json").read_text())
    assert result["evaluation_complete"] and result["episodes_completed"] == 4
    assert result["successes"] == 2
    assert result["settings"]["sampling_rng"] == "run_seeded_continuous"
    assert [episode["seed"] for episode in result["episodes"]] == [123] * 4
    assert [
        (episode["task_id"], episode["trial"], episode["init_state_index"]) for episode in result["episodes"]
    ] == [(0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 1, 1)]
    for _, seeder in seeders:
        seeder.assert_called_once_with(123)
    python_rng = random.Random(123)
    numpy_rng = np.random.RandomState(123)
    torch_rng = torch.Generator().manual_seed(123)
    expected = [
        (python_rng.random(), numpy_rng.random(), torch.rand((), generator=torch_rng).item())
        for _ in range(4)
    ]
    assert draws == expected
    assert [
        Path(episode["video_path"]).name for episode in result["episodes"] if "video_path" in episode
    ] == saved_names
    rerun = command.copy()
    rerun[rerun.index("--trials-per-task") + 1] = "1"
    evaluation.main(rerun)
    replacement = json.loads((output / "results.json").read_text())
    assert replacement["evaluation_complete"] and replacement["episodes_completed"] == 2
    assert replacement["successes"] == 2
    assert all(episode["trial"] == 0 for episode in replacement["episodes"])
    assert not (output / "task_00_trial_001_actions.npy").exists()
    assert not (output / "task_01_trial_001_actions.npy").exists()
    assert draws[4:] == expected[:2]
    assert closed == [True] * 4
    for _, seeder in seeders:
        assert seeder.call_count == 2
        seeder.assert_called_with(123)
