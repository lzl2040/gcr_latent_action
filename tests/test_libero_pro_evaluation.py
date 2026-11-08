import json
from pathlib import Path

import numpy as np
import pytest

from lerobot.evaluation import eval_qwen3vl_mot_libero_pro as pro


def record(tmp_path):
    path = tmp_path / "states.npy"
    np.save(path, np.array([[1.0, 2.0], [3.0, 4.0]]))
    return {
        "suite": "libero_goal_lan",
        "base_suite": "libero_goal",
        "category": "semantic",
        "task_id": 0,
        "instruction": "Actual paraphrased BDDL instruction",
        "initial_state_file": str(path),
        "initial_states_sha256": pro.sha256(path),
    }


def test_requested_fixed_indices_and_mutation_guard(tmp_path):
    row = record(tmp_path)
    np.testing.assert_array_equal(pro.load_states(row, 2), [[1, 2], [3, 4]])
    np.save(row["initial_state_file"], np.zeros((2, 2)))
    with pytest.raises(ValueError, match="changed"):
        pro.load_states(row, 2)


def test_insufficient_or_nonfinite_states_are_rejected(tmp_path):
    row = record(tmp_path)
    with pytest.raises(ValueError, match="at least"):
        pro.load_states(row, 3)
    np.save(row["initial_state_file"], [[np.nan, 0]])
    row["initial_states_sha256"] = pro.sha256(Path(row["initial_state_file"]))
    with pytest.raises(ValueError, match="finite"):
        pro.load_states(row, 1)


def test_summary_requires_each_trial_and_preserves_real_instruction(tmp_path):
    row = record(tmp_path)
    settings = {"selected_tasks": [pro.task_key(row)], "trials_per_task": 2}
    episodes = [{**row, "trial": i, "init_state_index": i, "success": i == 0} for i in range(2)]
    partial = pro.summarize(settings, episodes[:1], [row])
    assert not partial["evaluation_complete"]
    full = pro.summarize(settings, episodes, [row])
    assert full["evaluation_complete"] and full["success_rate"] == 0.5
    assert not full["official_50_trials_per_task"]
    assert json.loads(json.dumps(full))["episodes"][0]["instruction"] == row["instruction"]
    with pytest.raises(ValueError, match="Duplicate"):
        pro.summarize(settings, episodes + episodes[:1], [row])
    episodes[0]["instruction"] = "Original filename instruction"
    with pytest.raises(ValueError, match="sources changed"):
        pro.summarize(settings, episodes, [row])


def test_initial_state_index_cannot_be_reused(tmp_path):
    row = record(tmp_path)
    settings = {"selected_tasks": [pro.task_key(row)], "trials_per_task": 2}
    episode = {**row, "trial": 1, "init_state_index": 0, "success": False}
    with pytest.raises(ValueError, match="fixed initial-state"):
        pro.summarize(settings, [episode], [row])
