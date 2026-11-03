import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from lerobot.evaluation import eval_qwen3vl_mot_libero_plus as plus


@pytest.fixture
def catalog():
    return [
        {
            "suite": suite,
            "task_id": index,
            "classification_id": index + 10,
            "task_name": name,
            "category": category,
            "difficulty_level": level,
            "instruction": f"official instruction for {name}",
            "bddl_file": f"/official/{name}.bddl",
        }
        for suite, index, name, category, level in (
            ("libero_spatial", 0, "table", "Background Textures", 1),
            ("libero_spatial", 1, "layout", "Objects Layout", None),
            ("libero_goal", 0, "camera", "Camera Viewpoints", 5),
        )
    ]


def selection(catalog):
    selected = {}
    for row in catalog:
        selected.setdefault(row["suite"], []).append(row["task_id"])
    return selected


def episode(row, success=True):
    return {**row, "success": success, "trial": 0, "init_state_index": 0}


def test_filters_use_actual_suite_indices_and_retain_unrated_by_default(catalog):
    args = plus.build_parser().parse_args(["--dry-run", "--task-suite", "libero_spatial", "--task-ids", "1"])
    assert plus.select_tasks(catalog, args) == [catalog[1]]
    args.difficulty_levels = [1, 2, 3, 4, 5]
    with pytest.raises(ValueError, match="no tasks"):
        plus.select_tasks(catalog, args)
    args.difficulty_levels = None
    args.task_ids = None
    args.categories = ["background"]
    assert plus.select_tasks(catalog, args) == [catalog[0]]


@pytest.mark.parametrize("ids", [[10], [0, 0], [-1]])
def test_reject_classification_ids_duplicates_and_invalid_indices(catalog, ids):
    args = plus.build_parser().parse_args(["--dry-run", "--task-suite", "libero_spatial"])
    args.task_ids = ids
    with pytest.raises(ValueError, match="zero-based"):
        plus.select_tasks(catalog, args)


def test_shards_are_disjoint_complete_and_keep_canonical_order(catalog):
    shards = []
    for index in range(2):
        args = plus.build_parser().parse_args(["--dry-run", "--num-shards", "2", "--shard-index", str(index)])
        shards.append(plus.select_tasks(catalog, args))
    assert shards == [catalog[::2], catalog[1::2]]
    assert {plus.task_key(row) for shard in shards for row in shard} == {
        plus.task_key(row) for row in catalog
    }


def test_task_ids_cannot_be_applied_to_multiple_suites(catalog):
    args = plus.build_parser().parse_args(["--dry-run", "--task-ids", "0"])
    with pytest.raises(ValueError, match="single"):
        plus.select_tasks(catalog, args)


def test_load_initial_state_uses_official_api_first_state_and_restores_flag(monkeypatch):
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "0")
    states = np.arange(6.0).reshape(2, 3)
    suite = SimpleNamespace(get_task_init_states=Mock())

    def load(index):
        assert index == 12
        assert os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"
        return states

    suite.get_task_init_states.side_effect = load
    actual = plus.load_initial_state(suite, 12)
    np.testing.assert_array_equal(actual, states[0])
    assert not np.shares_memory(actual, states)
    assert os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "0"
    monkeypatch.delenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    suite.get_task_init_states.side_effect = FileNotFoundError("missing official init")
    with pytest.raises(FileNotFoundError):
        plus.load_initial_state(suite, 12)
    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" not in os.environ


@pytest.mark.parametrize("states", [np.array([[np.nan]]), np.empty((1, 0)), np.ones((1, 2, 2))])
def test_invalid_initial_states_fail_instead_of_skipping_tasks(states):
    with pytest.raises(ValueError, match="Invalid official"):
        plus.load_initial_state(SimpleNamespace(get_task_init_states=lambda _: states), 0)


def test_partial_results_never_claim_complete_benchmark_and_null_difficulty_is_counted(catalog):
    settings = {"selected_tasks": selection(catalog[:2])}
    episodes = [episode(catalog[0]), episode(catalog[1], False)]
    result = plus.summarize(settings, episodes, catalog)
    assert result["evaluation_complete"]
    assert not result["benchmark_complete"]
    assert result["official_total_success_rate"] is None
    assert result["success_rate"] == 0.5
    assert result["per_difficulty"]["unrated"]["episodes_completed"] == 1
    assert result["per_category"]["Camera Viewpoints"]["success_rate"] is None
    partial = plus.summarize(settings, episodes[:1], catalog)
    assert not partial["evaluation_complete"]


def test_total_is_task_weighted_not_mean_of_suites_or_categories(catalog):
    result = plus.summarize(
        {"selected_tasks": selection(catalog)},
        [episode(row, row["suite"] == "libero_spatial") for row in catalog],
        catalog,
    )
    assert result["benchmark_complete"]
    assert result["official_total_success_rate"] == pytest.approx(2 / 3)
    assert result["per_suite"]["libero_spatial"]["success_rate"] == 1
    assert result["per_suite"]["libero_goal"]["success_rate"] == 0


@pytest.mark.parametrize(
    "change", [{"trial": 1}, {"init_state_index": 1}, {"instruction": "unperturbed"}, {"success": "false"}]
)
def test_wrong_trial_state_or_instruction_cannot_be_scored_as_official(catalog, change):
    with pytest.raises(ValueError, match="official"):
        plus.summarize({"selected_tasks": selection(catalog)}, [{**episode(catalog[0]), **change}], catalog)


def test_duplicate_episodes_are_not_counted_twice(catalog):
    with pytest.raises(ValueError, match="Duplicate"):
        plus.summarize(
            {"selected_tasks": selection(catalog)}, [episode(catalog[0]), episode(catalog[0])], catalog
        )


def raw_settings(catalog):
    return {
        "protocol": plus.PROTOCOL,
        "mode": "policy",
        "source_revision": plus.SOURCE_REVISION,
        "classification_sha256": "classification",
        "evaluator_sha256": "evaluator",
        "runtime_versions": {"torch": "test"},
        "selected_tasks": selection(catalog),
        "seed": 0,
        "environment_seed": 0,
        "trials_per_task": 1,
        "init_state_index": 0,
        "wait_steps": 10,
        "max_steps": plus.original.MAX_STEPS,
        "env_resolution": 256,
        "control_frequency": 20,
        "replan_steps": 10,
        "inference_steps": 20,
        "precision": "bf16",
        "rgb_views": ["primary", "wrist"],
        "policy_identity": {"checkpoint": "/model", "policy_config_sha256": "config"},
        "normalization": {suite: {"stats_sha256": suite} for suite in selection(catalog)},
    }


def save_raw(path, catalog):
    settings = raw_settings(catalog)
    plus.write_json(path, {"settings": settings, "episodes": [episode(row) for row in catalog]})
    return settings


def test_merge_disjoint_shards_recomputes_complete_score(tmp_path, catalog):
    paths = [tmp_path / "shard0.json", tmp_path / "shard1.json"]
    settings = save_raw(paths[0], catalog[::2])
    save_raw(paths[1], catalog[1::2])
    result = plus.merge_results(paths, catalog, settings)
    assert result["benchmark_complete"] and result["evaluation_complete"]
    assert result["official_total_success_rate"] == 1
    assert len(result["settings"]["inputs"]) == 2
    assert result["settings"]["inputs"][0]["sha256"] == plus.sha256(paths[0])


@pytest.mark.parametrize("change", ["overlap", "checkpoint", "normalization", "seed", "source", "runtime"])
def test_merge_rejects_repeated_or_incompatible_runs(tmp_path, catalog, change):
    paths = [tmp_path / "first.json", tmp_path / "second.json"]
    settings = save_raw(paths[0], catalog[:1])
    save_raw(paths[1], catalog[1:2] if change != "overlap" else catalog[:1])
    payload = json.loads(paths[1].read_text())
    current = payload["settings"]
    if change == "checkpoint":
        current["policy_identity"]["checkpoint"] = "/other_model"
    elif change == "normalization":
        current["normalization"]["libero_spatial"]["stats_sha256"] = "other"
    elif change == "seed":
        current["seed"] = 1
    elif change == "source":
        current["classification_sha256"] = "other"
    elif change == "runtime":
        current["runtime_versions"]["torch"] = "other"
    plus.write_json(paths[1], payload)
    with pytest.raises(ValueError):
        plus.merge_results(paths, catalog, settings)


def test_suite_specific_checkpoints_can_merge_without_being_misrepresented(tmp_path, catalog):
    paths = [tmp_path / "spatial.json", tmp_path / "goal.json"]
    settings = save_raw(paths[0], catalog[:2])
    save_raw(paths[1], catalog[2:])
    payload = json.loads(paths[1].read_text())
    payload["settings"]["policy_identity"]["checkpoint"] = "/goal_model"
    plus.write_json(paths[1], payload)
    result = plus.merge_results(paths, catalog, settings)
    assert result["benchmark_complete"]
    assert result["settings"]["suite_inputs"]["libero_goal"]["policy"]["checkpoint"] == "/goal_model"


def test_output_directory_is_never_overwritten(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    keep = output / "results.json"
    keep.write_text("keep")
    with pytest.raises(FileExistsError):
        plus.main(["--dry-run", "--output-dir", str(output)])
    assert keep.read_text() == "keep"


@pytest.mark.parametrize("flags", [["--trials-per-task", "20"], ["--max-steps", "2"], ["--stop-on-success"]])
def test_nonofficial_shortcuts_are_not_exposed(flags):
    with pytest.raises(SystemExit):
        plus.build_parser().parse_args(flags)


def test_load_catalog_matches_by_name_not_json_id_or_position(tmp_path, monkeypatch):
    import sys
    from types import ModuleType

    root = tmp_path / "plus"
    directory = root / "libero/libero/benchmark"
    directory.mkdir(parents=True)
    entries = [
        {"id": 100, "name": "second", "category": "Objects Layout", "difficulty_level": None},
        {"id": 10, "name": "first", "category": "Camera Viewpoints", "difficulty_level": 5},
    ]
    (directory / "task_classification.json").write_text(json.dumps({"libero_goal": entries}))
    tasks = [
        SimpleNamespace(name="first", language="rewritten language from BDDL"),
        SimpleNamespace(name="second", language="official second"),
    ]
    suite = SimpleNamespace(
        n_tasks=2,
        get_task=lambda index: tasks[index],
        get_task_bddl_file_path=lambda index: str(index),
    )
    benchmark = SimpleNamespace(get_benchmark_dict=lambda: {"libero_goal": lambda **_: suite})
    package = ModuleType("libero.libero")
    package.benchmark = benchmark
    monkeypatch.setitem(sys.modules, "libero.libero", package)
    monkeypatch.setattr(plus, "SUITE_COUNTS", {"libero_goal": 2})
    _, records = plus.load_catalog(root)
    assert [row["classification_id"] for row in records] == [10, 100]
    assert records[0]["instruction"] == tasks[0].language
    assert records[1]["difficulty_level"] is None
    entries[1]["name"] = "not-in-suite"
    (directory / "task_classification.json").write_text(json.dumps({"libero_goal": entries}))
    with pytest.raises(ValueError, match="do not match"):
        plus.load_catalog(root)


def test_policy_rollout_preserves_official_environment_and_instruction(tmp_path, monkeypatch, catalog):
    import sys
    from types import ModuleType

    observation = {"agentview_image": np.zeros((4, 4, 3), dtype=np.uint8)}

    class Environment:
        def __init__(self):
            self.steps = 0
            self.seed = Mock()
            self.close = Mock()
            self.set_init_state = Mock(return_value=observation)

        def reset(self):
            self.steps = 0

        def check_success(self):
            return self.steps >= 12

        def step(self, action):
            self.steps += 1
            if self.steps <= 10:
                assert action == plus.original.DUMMY_ACTION
            return observation, 0, False, {}

    env = Environment()
    module = ModuleType("libero.libero.envs")
    module.OffScreenRenderEnv = Mock(return_value=env)
    monkeypatch.setitem(sys.modules, "libero.libero.envs", module)
    policy = SimpleNamespace(
        config=object(), reset=Mock(), sample_canonical_action=Mock(return_value=torch.zeros(1, 2, 40))
    )
    adapter = SimpleNamespace(
        observation_batch=Mock(return_value={"task": [catalog[0]["instruction"]]}),
        native_actions=Mock(return_value=np.zeros((2, 7))),
    )
    args = SimpleNamespace(
        render_gpu_device_id=-1,
        precision="bf16",
        replan_steps=2,
        no_save_video=True,
        save_all_videos=False,
        seed=7,
    )
    suite = SimpleNamespace(get_task_init_states=lambda _: np.array([[1.0, 2.0], [3.0, 4.0]]))
    result = plus.evaluate_task(catalog[0], suite, policy, adapter, args, tmp_path, torch.device("cpu"))
    module.OffScreenRenderEnv.assert_called_once_with(
        bddl_file_name=catalog[0]["bddl_file"],
        camera_heights=256,
        camera_widths=256,
        control_freq=20,
        render_gpu_device_id=-1,
    )
    env.seed.assert_called_once_with(0)
    env.close.assert_called_once()
    np.testing.assert_array_equal(env.set_init_state.call_args.args[0], [1, 2])
    adapter.observation_batch.assert_called_once_with(
        observation, catalog[0]["instruction"], policy.config, torch.device("cpu")
    )
    policy.reset.assert_called_once()
    assert result["success"] and result["steps"] == 2 and result["policy_calls"] == 1
    assert result["init_state_index"] == 0 and result["trial"] == 0
    assert np.load(result["actions_path"]).shape == (2, 7)
    assert result["actions_sha256"] == plus.sha256(Path(result["actions_path"]))


def test_settling_success_does_not_skip_the_official_ten_wait_steps_or_policy_trial():
    steps = []
    env = SimpleNamespace(
        reset=Mock(),
        set_init_state=lambda _: {},
        step=lambda action: (steps.append(action) or {}, 0, len(steps) <= 10, {}),
        check_success=lambda: len(steps) >= 11,
    )
    plan = Mock(return_value=np.zeros((2, 7)))
    result, _ = plus.rollout_episode(env, np.zeros(2), plan, max_steps=2, replan_steps=2, save_video=False)
    assert len(steps) == 11 and steps[:10] == [plus.original.DUMMY_ACTION] * 10
    assert result["success"] and result["steps"] == 1
    plan.assert_called_once_with({}, control_step=0, replan_index=0)


@pytest.mark.parametrize("chunk", [np.zeros((1, 7)), np.zeros((2, 6)), np.full((2, 7), np.nan)])
def test_rollout_rejects_bad_native_action_chunks(chunk):
    env = SimpleNamespace(
        reset=lambda: None,
        set_init_state=lambda _: {},
        step=lambda _: ({}, 0, False, {}),
        check_success=lambda: False,
    )
    with pytest.raises(ValueError, match="native 7D"):
        plus.rollout_episode(
            env, np.zeros(2), lambda *_, **__: chunk, max_steps=2, replan_steps=2, save_video=False
        )


def test_main_keeps_suite_normalization_separate_and_writes_complete_results(tmp_path, monkeypatch, catalog):
    checkpoint = tmp_path / "step12k"
    (checkpoint / "model").mkdir(parents=True)
    (checkpoint / "model/.metadata").write_bytes(b"mock DCP metadata")
    (checkpoint / "config.json").write_text("{}")
    root = tmp_path / "datasets"
    for suite, mean in (("libero_spatial", 1), ("libero_goal", 5)):
        meta = root / plus.original.SUITE_DATASETS[suite] / "meta"
        meta.mkdir(parents=True)
        plus.write_json(
            meta / "info.json",
            {"fps": 20, "features": {key: {"shape": [10]} for key in ("observation.state", "action")}},
        )
        plus.write_json(
            meta / "stats.json",
            {key: {"mean": [mean] * 10, "std": [1] * 10} for key in ("observation.state", "action")},
        )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", Mock())
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    identity = {
        key: value
        for key, value in raw_settings(catalog).items()
        if key in ("source_revision", "classification_sha256", "evaluator_sha256", "runtime_versions")
    }
    monkeypatch.setattr(plus, "configure_plus", lambda *args: identity)
    monkeypatch.setattr(plus, "load_catalog", lambda root: (dict.fromkeys(selection(catalog)), catalog))
    policy = SimpleNamespace(
        config=SimpleNamespace(
            inference_steps=20, rgb_views=["primary", "wrist"], task_names=["joint_prediction"]
        ),
        libero_precision_details={"precision": "bf16"},
    )
    load = Mock(return_value=policy)
    monkeypatch.setattr(plus.original, "load_policy", load)

    def evaluate(record, suite, policy, adapter, args, output, device):
        mean = 1 if record["suite"] == "libero_spatial" else 5
        torch.testing.assert_close(adapter.action_mean, torch.full((10,), float(mean)))
        return episode(record)

    monkeypatch.setattr(plus, "evaluate_task", evaluate)
    output = tmp_path / "output"
    plus.main(["--checkpoint", str(checkpoint), "--dataset-root", str(root), "--output-dir", str(output)])
    result = json.loads((output / "results.json").read_text())
    assert result["evaluation_complete"] and result["benchmark_complete"]
    assert result["episodes_completed"] == 3
    assert set(result["settings"]["normalization"]) == {"libero_spatial", "libero_goal"}
    assert result["settings"]["policy_identity"]["dcp_metadata_sha256"] == plus.sha256(
        checkpoint / "model/.metadata"
    )
    assert len(json.loads((output / "task_manifest.json").read_text())["tasks"]) == 3
    load.assert_called_once()
