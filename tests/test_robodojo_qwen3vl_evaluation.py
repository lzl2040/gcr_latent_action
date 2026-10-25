import json
import os
import subprocess
import sys
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
    evaluate,
    validate_result,
)
from lerobot.evaluation.robodojo_deploy import eval_one_episode, eval_one_episode_batch
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
        task_names=("forward_dynamics", "inverse_dynamics", "joint_prediction"),
        task_weights=(0.15, 0.15, 0.7),
    )


def test_shared_policy_loader_precision_defaults():
    args = build_parser().parse_args(["eval"])
    assert args.precision == "bf16"
    assert args.fp8_training_config is None


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


def test_matching_futures_share_one_sample_and_preserve_next_replan_rng(adapter, tmp_path):
    import imageio.v2 as imageio

    class Policy:
        config = config()
        joint_calls = 0
        action_calls = 0

        def reset(self):
            pass

        def _sample(self):
            output = torch.zeros(1, 32, 40)
            raw = torch.from_numpy(state_to_canonical(observation()))
            output[:, :, :20] = (raw - adapter.action_mean) / adapter.action_std
            output[:, :, 0] += torch.rand(32)
            return output

        def sample_canonical_action(self, batch):
            self.action_calls += 1
            return self._sample()

        def generate_validation(self, batch, *, seed):
            assert seed is None
            self.joint_calls += 1
            self.prediction = {"action_normalized": self._sample(), "state_normalized": torch.ones(1, 9, 40)}
            return self.prediction

    class Writer:
        def __init__(self, policy):
            self.policy = policy
            self.chunks = []

        def save(self, prediction, actions, directory, **kwargs):
            assert prediction is self.policy.prediction
            assert kwargs["save_frames"] and kwargs["seed"] == 42
            # Simulate random draws during lazy VAE construction.
            torch.rand(1000)
            directory.mkdir(parents=True)
            self.chunks.append((actions, directory, kwargs))
            return {"directory": str(directory)}

    plain_policy, export_policy = Policy(), Policy()
    writer = Writer(export_policy)
    outputs = []
    for name, policy, exporter in (("plain", plain_policy, None), ("export", export_policy, writer)):
        torch.manual_seed(42)
        model = RoboDojoModel(
            policy,
            adapter,
            device="cpu",
            replan_steps=10,
            output_dir=tmp_path / name / "chunks",
            prediction_writer=exporter,
            seed=42,
        )
        model.reset()
        model.update_obs(observation())
        outputs.append([model.get_action(), model.get_action()])
    assert plain_policy.action_calls == 2 and plain_policy.joint_calls == 0
    assert export_policy.joint_calls == 2 and export_policy.action_calls == 0
    for chunk_index in range(2):
        exported, directory, metadata = writer.chunks[chunk_index]
        assert exported.shape == (32, 16)
        assert metadata["control_step"] == chunk_index * 10
        assert "reset_0000/env_000" in str(directory)
        for action_index, (plain, saved) in enumerate(
            zip(outputs[0][chunk_index], outputs[1][chunk_index], strict=True)
        ):
            for key in plain:
                np.testing.assert_array_equal(plain[key], saved[key])
            np.testing.assert_array_equal(
                exported[action_index],
                np.concatenate(
                    [
                        saved[f"{arm}_{key}"]
                        for arm in ("left", "right")
                        for key in ("ee_pose", "ee_joint_state")
                    ]
                ),
            )
        for view, value in (("primary", 17), ("secondary", 53), ("wrist", 91)):
            np.testing.assert_array_equal(
                imageio.imread(directory / f"conditioning_{view}.png"),
                np.full((4, 4, 3), value, dtype=np.uint8),
            )
    with np.load(tmp_path / "export/chunks/chunk_000000_env_000.npz") as data:
        np.testing.assert_array_equal(data["current_state"], state_to_canonical(observation()))
        np.testing.assert_array_equal(data["normalized_predicted_states"], np.ones((1, 9, 40)))
    records = [
        json.loads(line) for line in (tmp_path / "export/chunks/inference.jsonl").read_text().splitlines()
    ]
    assert [r["control_step"] for r in records] == [0, 10]
    assert all("prediction" in record for record in records)
    model.reset()
    model.update_obs(observation())
    model.get_action()
    assert writer.chunks[-1][2]["control_step"] == 0
    assert "reset_0001" in str(writer.chunks[-1][1])


@pytest.mark.parametrize("action_weight,joint_weight", [(1, 1), (0, 0)])
def test_future_export_rejects_sampler_contract_changes(adapter, tmp_path, action_weight, joint_weight):
    cfg = config()
    cfg.task_names = ("action_prediction", "joint_prediction")
    cfg.task_weights = (action_weight, joint_weight)
    with pytest.raises(ValueError, match="joint-only"):
        RoboDojoModel(
            SimpleNamespace(config=cfg),
            adapter,
            device="cpu",
            replan_steps=10,
            output_dir=tmp_path,
            prediction_writer=object(),
        )


def test_future_export_errors_are_not_silently_swallowed(adapter, tmp_path):
    raw = torch.from_numpy(state_to_canonical(observation()))
    normalized = torch.zeros(1, 32, 40)
    normalized[:, :, :20] = (raw - adapter.action_mean) / adapter.action_std
    policy = SimpleNamespace(
        config=config(),
        reset=lambda: None,
        generate_validation=lambda batch, seed: {"action_normalized": normalized},
    )

    class Writer:
        def save(self, *args, **kwargs):
            raise RuntimeError("decoder failure")

    model = RoboDojoModel(
        policy,
        adapter,
        device="cpu",
        replan_steps=10,
        output_dir=tmp_path / "chunks",
        prediction_writer=Writer(),
    )
    model.reset()
    model.update_obs(observation())
    with pytest.raises(RuntimeError, match="decoder failure"):
        model.get_action()
    assert not (tmp_path / "chunks/inference.jsonl").exists()
    assert model.control_steps == {}


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
    eval_one_episode_batch(TASK_ENV=env, model_client=Client())
    assert [indices for indices, _ in env.executed] == [[2, 8], [8], [8]]
    assert env.executed[-1][1] == ["8-2"]
    assert env.observations == [[2, 8], [8], [8]]


def test_workspace_isolated_from_shared_source(tmp_path):
    source = tmp_path / "source"
    for name in ("env", "env_cfg", "task", "src", "utils", "third_party", "Assets", "XPolicyLab/policy"):
        (source / name).mkdir(parents=True)
    (source / "XPolicyLab/__init__.py").write_text("")
    (source / "XPolicyLab/model_template.py").write_text("")
    (source / "env/global_configs.py").write_text(
        "import os\nROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')\n"
    )
    workspace = tmp_path / "workspace"
    create_workspace(source, source / "Assets", workspace)
    assert (workspace / "Assets").resolve() == source / "Assets"
    assert not (source / "XPolicyLab/policy" / POLICY_NAME).exists()
    assert (workspace / "XPolicyLab/policy" / POLICY_NAME / "deploy.py").is_file()
    assert "eval_batch: true" in (workspace / "XPolicyLab/policy" / POLICY_NAME / "deploy.yml").read_text()
    resolved_root = subprocess.check_output(
        [sys.executable, "-c", "from env.global_configs import ROOT_DIR; print(ROOT_DIR)"],
        cwd=workspace,
        env=dict(os.environ, PYTHONPATH=str(workspace)),
        text=True,
    ).strip()
    assert (Path(resolved_root) / "XPolicyLab/policy" / POLICY_NAME / "deploy.yml").is_file()
    assert Path(resolved_root).resolve() == workspace
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
    data["success_rate"] = float("nan")
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="success rate"):
        validate_result(tmp_path, "stack_bowls", 2)


def test_deploy_does_not_duplicate_observations_at_chunk_boundaries():
    class Env:
        steps = 0
        captured_steps = []

        def is_episode_end(self):
            return self.steps == 4

        def get_running_env_idx_list(self):
            return [0]

        def get_obs_batch(self, indices):
            self.captured_steps.append(self.steps)
            return [{"env_idx": 0}]

        def take_action_batch(self, actions, indices):
            self.steps += 1

    class Client:
        def call(self, func_name, obs=None):
            return [[{}, {}]] if func_name == "get_action_batch" else None

    env = Env()
    eval_one_episode_batch(TASK_ENV=env, model_client=Client())
    assert env.captured_steps == [0, 1, 2, 3]


def test_single_deploy_accepts_upstream_keyword_contract():
    class Env:
        steps = 0

        def is_episode_end(self):
            return self.steps == 1

        def get_obs(self):
            return {}

        def take_action(self, action):
            self.steps += 1

    class Client:
        def call(self, func_name, obs=None):
            return [{}] if func_name == "get_action" else None

    env = Env()
    eval_one_episode(TASK_ENV=env, model_client=Client())
    assert env.steps == 1


def test_default_checkpoint_and_cli():
    args = build_parser().parse_args(["offline"])
    assert args.checkpoint.name == "step4k"
    assert args.replan_steps == 10
    assert args.eval_num == "1"
    assert args.sim_python.name == "python"
    assert isinstance(args.dataset_root, Path)
    assert not args.save_predicted_video
    assert build_parser().parse_args(["eval", "--save-predicted-video"]).save_predicted_video
    diagnostic = build_parser().parse_args(["diagnose"])
    assert diagnostic.diagnostic_episodes == [3000, 3001, 3002]
    assert diagnostic.diagnostic_inference_steps == [20, 50]
    assert diagnostic.training_config is None
    assert not diagnostic.diagnostic_fp8
    assert build_parser().parse_args(["diagnose", "--diagnostic-fp8"]).diagnostic_fp8


def test_diagnostics_select_matching_training_config_and_reject_another_experiment(tmp_path, monkeypatch):
    from lerobot.evaluation.robodojo_diagnostics import diagnose, matching_training_config

    model = tmp_path / "step8k/model"
    model.mkdir(parents=True)
    (model / ".metadata").touch()
    payload = {"type": "qwen3vl_mot", "task_weights": [0.1, 0.1, 0.8]}
    (tmp_path / "config.json").write_text(json.dumps(payload))
    training = tmp_path / "train_config.json"
    training.write_text(json.dumps({"policy": payload}))
    args = build_parser().parse_args(["diagnose", "--checkpoint", str(model.parent)])
    assert matching_training_config(args) == training.resolve()
    other = tmp_path / "other_train_config.json"
    other.write_text(json.dumps({"policy": {**payload, "task_weights": [0.15, 0.15, 0.7]}}))
    args.training_config = other

    def unexpected_load(_args):
        pytest.fail("Mismatched diagnostics must fail before allocating the model.")

    monkeypatch.setattr("lerobot.evaluation.eval_qwen3vl_mot_robodojo.load_model", unexpected_load)
    with pytest.raises(ValueError, match="does not match"):
        diagnose(args)


def test_training_sample_roundtrip_and_input_parity(adapter):
    from lerobot.evaluation.robodojo_diagnostics import input_parity, validation_observation

    batch = adapter.observation_batch(observation(), config(), torch.device("cpu"))
    validation = {**batch, "observation.state": batch["observation.state"][:, None].expand(-1, 32, -1)}
    item = {key: value[0] for key, value in validation.items()}
    reconstructed = validation_observation(item, adapter, config().rgb_views)
    deployed = adapter.observation_batch(reconstructed, config(), torch.device("cpu"))
    parity = input_parity(validation, deployed)
    assert parity["normalized_current_state_max_error"] < 1e-5
    assert parity["images_t0"] == 0
    deployed["images_t0"][0, 1, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="images_t0"):
        input_parity(validation, deployed)


def test_diagnostic_action_metrics_use_physical_units_and_executed_prefix(adapter):
    from lerobot.evaluation.robodojo_diagnostics import action_metrics

    target = torch.zeros(1, 32, 40)
    target[..., :20] = (
        torch.from_numpy(state_to_canonical(observation())) - adapter.action_mean
    ) / adapter.action_std
    predicted = target.clone()
    predicted[:, 10:, 0] += 0.5
    metrics = action_metrics(predicted, target, adapter, 10)
    assert metrics["prefix_normalized_mse"] == 0
    assert metrics["normalized_mse"] > 0
    assert metrics["left"]["xyz_mean_l2_m"] == pytest.approx(22 / 32)
    assert metrics["left"]["prefix_xyz_mean_l2_m"] == 0
    assert metrics["left"]["rotation_mean_deg"] == pytest.approx(0, abs=1e-6)
    assert metrics["right"]["xyz_mean_l2_m"] == 0
    assert metrics["left"]["first_xyz_l2_m"] == 0
    predicted[0, 0, 0] += 0.25
    metrics = action_metrics(predicted, target, adapter, 10)
    assert metrics["left"]["first_xyz_l2_m"] == pytest.approx(0.5)
    assert metrics["left"]["first_predicted_xyz_m"] == pytest.approx([0.6, 0.2, 0.3])
    assert metrics["left"]["first_target_xyz_m"] == pytest.approx([0.1, 0.2, 0.3])
    assert metrics["left"]["first_rotation_deg"] == pytest.approx(0, abs=1e-6)
    predicted[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        action_metrics(predicted, target, adapter, 10)


def test_clean_action_autoencoding_uses_real_targets_and_canonical_masks():
    from lerobot.evaluation.robodojo_diagnostics import clean_action_autoencoding

    target = torch.randn(1, 32, 40)
    mask = torch.tensor([[1.0] * 20 + [0.0] * 20])
    batch = {"action": target, "action_mask": mask, "state_mask": mask.clone()}

    def encode(received, name, valid):
        assert received is batch
        assert name == "action"
        torch.testing.assert_close(valid, torch.tensor([True]))
        return target.reshape(1, 8, -1), None, None, None

    policy = SimpleNamespace(
        _clean_physical_embedding=encode,
        _decode_physical_embedding=lambda name, clean: clean,
    )
    decoded = clean_action_autoencoding(policy, batch)
    torch.testing.assert_close(decoded[..., :20], target[..., :20])
    assert not decoded[..., 20:].any()


@pytest.fixture
def delta_adapter(tmp_path):
    from lerobot.evaluation.scripts.add_robodojo_delta_actions import DELTA_CONTRACT

    meta = tmp_path / "delta/meta"
    meta.mkdir(parents=True)
    mean = np.zeros(20)
    mean[[3, 6, 13, 16]] = 1
    (meta / "info.json").write_text(
        json.dumps(
            {
                "fps": 25,
                "features": {"delta_action.ee_ort6d_pos": {"shape": [20]}},
                "delta_action_contract": DELTA_CONTRACT,
            }
        )
    )
    std = np.full(20, 0.1)
    std[[3, 6, 13, 16]] = 0.0001
    (meta / "stats.json").write_text(
        json.dumps(
            {
                "observation.ee_ort6d_pos": {"mean": [0.1] * 20, "std": [2] * 20},
                "action.ee_ort6d_pos": {"mean": [123] * 20, "std": [42] * 20},
                "delta_action.ee_ort6d_pos": {"mean": mean.tolist(), "std": std.tolist()},
            }
        )
    )
    return RoboDojoAdapter.from_dataset(meta.parent, action_mode="delta")


def normalized_delta(adapter, raw):
    values = torch.zeros(1, len(raw), 40)
    values[0, :, :20] = (torch.as_tensor(raw, dtype=torch.float32) - adapter.action_mean) / adapter.action_std
    return values


def test_delta_statistics_and_zero_motion_hold_current_pose(delta_adapter):
    assert delta_adapter.action_feature == "delta_action.ee_ort6d_pos"
    assert delta_adapter.action_std[3].item() == pytest.approx(1e-3)
    values = torch.zeros(1, 32, 40)
    obs = observation()
    for action in delta_adapter.native_actions(values, obs):
        for arm in ("left", "right"):
            np.testing.assert_allclose(action[f"{arm}_ee_pose"], obs["state"][f"{arm}_ee_pose"], atol=1e-7)
            np.testing.assert_allclose(
                action[f"{arm}_ee_joint_state"], obs["state"][f"{arm}_ee_joint_state"], atol=1e-7
            )
    with pytest.raises(ValueError, match="current observation"):
        delta_adapter.native_actions(values)
    obs["state"]["left_ee_pose"][3:] = 0
    with pytest.raises(ValueError, match="unit norm"):
        delta_adapter.native_actions(values, obs)


def test_delta_uses_world_translation_left_rotations_and_stepwise_gripper_clipping(delta_adapter, caplog):
    raw = delta_adapter.action_mean.repeat(2, 1).numpy()
    increments = Rotation.from_euler("xyz", [[0.1, -0.2, 0.3], [-0.15, 0.2, 0.1]])
    for offset in (0, 10):
        raw[:, offset : offset + 3] = [[0.01, -0.02, 0.03], [0.04, 0.05, -0.01]]
        raw[:, offset + 3 : offset + 9] = increments.as_matrix()[:, :, :2].reshape(2, 6)
        raw[:, offset + 9] = [1, -0.3]
    actions = delta_adapter.native_actions(normalized_delta(delta_adapter, raw), observation())
    for arm in ("left", "right"):
        initial = observation()["state"][f"{arm}_ee_pose"]
        rotation = Rotation.from_quat(initial[[4, 5, 6, 3]])
        for index, action in enumerate(actions):
            rotation = increments[index] * rotation
            np.testing.assert_allclose(
                action[f"{arm}_ee_pose"][:3],
                initial[:3] + raw[: index + 1, :3].sum(0),
                atol=1e-7,
            )
            np.testing.assert_allclose(
                Rotation.from_quat(action[f"{arm}_ee_pose"][[4, 5, 6, 3]]).as_matrix(),
                rotation.as_matrix(),
                atol=2e-7,
            )
        assert actions[0][f"{arm}_ee_joint_state"][0] == 1
        assert actions[1][f"{arm}_ee_joint_state"][0] == pytest.approx(0.7)
    assert "Clipping left" in caplog.text and "Clipping right" in caplog.text


def test_delta_model_reanchors_each_replan_and_keeps_environments_independent(delta_adapter, tmp_path):
    raw = delta_adapter.action_mean.repeat(32, 1)
    raw[:, 0] = 0.01
    normalized = normalized_delta(delta_adapter, raw)
    policy = SimpleNamespace(
        config=config(), reset=lambda: None, sample_canonical_action=lambda batch: normalized
    )
    model = RoboDojoModel(policy, delta_adapter, device="cpu", replan_steps=2, output_dir=tmp_path / "chunks")
    model.reset()
    model.update_obs(observation(3))
    first = model.get_action()
    np.testing.assert_allclose([row["left_ee_pose"][0] for row in first], [0.11, 0.12], atol=1e-7)
    changed = observation(3)
    changed["state"]["left_ee_pose"][0] = 2
    other = observation(7)
    other["state"]["left_ee_pose"][0] = -2
    model.update_obs_batch([changed, other])
    second, third = model.get_action_batch([7, 3])
    assert second[0]["left_ee_pose"][0] == pytest.approx(-1.99)
    assert third[0]["left_ee_pose"][0] == pytest.approx(2.01)
    record = np.load(tmp_path / "chunks/chunk_000002_env_003.npz")
    assert record["action_mode"].item() == "delta"
    np.testing.assert_allclose(record["unnormalized_actions"][:, 0], 0.01, atol=1e-8)
    assert (
        json.loads((tmp_path / "chunks/inference.jsonl").read_text().splitlines()[0])["action_mode"]
        == "delta"
    )
    model.reset()
    with pytest.raises(RuntimeError, match="update_obs"):
        model.get_action()


def test_delta_entrypoint_defaults_and_training_action_mode_guard(tmp_path):
    from lerobot.evaluation.eval_qwen3vl_mot_robodojo import validate_training_action_mode
    from lerobot.evaluation.eval_qwen3vl_mot_robodojo_delta import build_parser as delta_parser

    args = delta_parser().parse_args(["eval"])
    assert args.action_mode == "delta" and args.checkpoint.name == "step8k"
    assert args.timeout == 7200
    assert args.dataset_root.name == "stack_bowls_delta" and args.diagnostic_episodes == [0, 1, 2]
    assert build_parser().parse_args(["eval"]).action_mode == "absolute"
    checkpoint = tmp_path / "step8k/model"
    checkpoint.mkdir(parents=True)
    (checkpoint / ".metadata").touch()
    (tmp_path / "config.json").write_text('{"type":"qwen3vl_mot"}')
    (tmp_path / "train_config.json").write_text(
        json.dumps(
            {"policy": {"type": "qwen3vl_mot"}, "data_mix": "delta_robodojo_stack_bowl", "stage": "finetune"}
        )
    )
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/stats.json").write_text("{}")
    args.checkpoint, args.dataset_root = checkpoint.parent, tmp_path
    assert validate_training_action_mode(args)["training_configuration_verified"]
    args.action_mode = "absolute"
    with pytest.raises(ValueError, match="Action mode"):
        validate_training_action_mode(args)


def test_delta_contract_is_required_and_not_replaced_by_absolute_statistics(delta_adapter, tmp_path):
    root = tmp_path / "delta"
    path = root / "meta/info.json"
    payload = json.loads(path.read_text())
    payload["delta_action_contract"]["reference"] = "consecutive_action_target"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="contract"):
        RoboDojoAdapter.from_dataset(root, action_mode="delta")
    with pytest.raises(ValueError, match="Unsupported"):
        RoboDojoAdapter.from_dataset(root, action_mode="other")


def test_delta_diagnostics_measure_integrated_targets_not_isolated_increments(delta_adapter):
    from lerobot.evaluation.robodojo_diagnostics import action_metrics

    target = torch.zeros(1, 32, 40)
    predicted = target.clone()
    predicted[:, :, 0] = 0.1  # std=0.1, so each step adds one centimeter.
    metrics = action_metrics(predicted, target, delta_adapter, 10, observation())
    assert metrics["left"]["first_xyz_l2_m"] == pytest.approx(0.01, abs=1e-7)
    assert metrics["left"]["prefix_xyz_mean_l2_m"] == pytest.approx(0.055, abs=1e-7)
    assert metrics["left"]["xyz_mean_l2_m"] == pytest.approx(0.165, abs=1e-7)


def test_diagnostic_samples_include_configured_validation_and_check_episode_bounds():
    from lerobot.evaluation.robodojo_diagnostics import diagnostic_samples

    dataset = SimpleNamespace(
        heldout_validation_samples=lambda: [{"dataset_index": 0, "episode_index": 1, "frame_index": 101}],
        all_episode_ranges=[np.array([[0, 100], [100, 200]])],
    )
    samples = diagnostic_samples(dataset, [0], [0, 64])
    assert [s["frame_index"] for s in samples] == [101, 0, 64]
    assert samples[0]["source"] == "configured_generation_validation"
    with pytest.raises(ValueError, match="episode"):
        diagnostic_samples(dataset, [2], [0])
    with pytest.raises(ValueError, match="outside"):
        diagnostic_samples(dataset, [0], [100])


def test_conditional_losses_fix_noise_levels_and_restore_rng_and_sampler():
    from lerobot.evaluation.robodojo_diagnostics import conditional_losses

    class Policy:
        config = SimpleNamespace(task_names=("joint_prediction", "inverse_dynamics"))

        def _sample_sigma(self, size, device):
            return torch.rand(size, device=device)

        def __call__(self, batch, *, task_type):
            sigma = self._sample_sigma(1, torch.device("cpu")).item()
            torch.rand(100)
            return torch.tensor(0.0), {"action_prediction_reconstruction_loss": sigma, "state_flow_loss": 1}

    policy = Policy()
    torch.manual_seed(19)
    before = torch.random.get_rng_state()
    results = conditional_losses(policy, {}, 42, torch.device("cpu"))
    torch.testing.assert_close(torch.random.get_rng_state(), before)
    assert len(results) == 6
    assert [row["action_prediction_reconstruction_loss"] for row in results] == pytest.approx(
        [0.1, 0.5, 0.9] * 2
    )
    assert all("state_flow_loss" not in row for row in results)
    assert "_sample_sigma" not in policy.__dict__


def test_simulator_subprocess_uses_its_own_environment_bin(tmp_path, monkeypatch):
    args = build_parser().parse_args(["eval"])
    args.robodojo_root = tmp_path / "source"
    configs = args.robodojo_root / "task/RoboDojo/config"
    configs.mkdir(parents=True)
    (configs / "stack_bowls.yml").write_text("{}")
    (configs / "_task.yml").write_text("common:\n  eval_nums: 50\ntasks: {}\n")
    args.sim_python = tmp_path / "simulator_env/bin/python"
    args.sim_python.parent.mkdir(parents=True)
    args.sim_python.touch()
    args.output_dir = tmp_path / "output"
    args.port = 0
    monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "YES")
    monkeypatch.setenv("PATH", "/policy_env/bin:/usr/bin")
    monkeypatch.setattr("lerobot.evaluation.eval_qwen3vl_mot_robodojo.create_workspace", lambda *args: None)

    class PreflightReachedError(Exception):
        pass

    def check_output(command, *, text, env):
        assert command[0] == str(args.sim_python)
        assert env["PATH"].split(os.pathsep) == [str(args.sim_python.parent), "/policy_env/bin", "/usr/bin"]
        raise PreflightReachedError

    monkeypatch.setattr(subprocess, "check_output", check_output)
    with pytest.raises(PreflightReachedError):
        evaluate(args)
