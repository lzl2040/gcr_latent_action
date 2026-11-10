import json

import numpy as np
import pytest

from lerobot.evaluation import eval_cosmos3_libero as native
from lerobot.evaluation import eval_cosmos3_libero_pro as evaluator


class InitiallySolvedEnvironment:
    def __init__(self, events):
        self.events = events
        self.waited = 0
        self.controlled = 0
        self.observation = {"agentview_image": np.zeros((2, 2, 3), dtype=np.uint8)}

    def seed(self, value):
        self.events.append(("seed", value))

    def reset(self):
        self.events.append("reset")

    def set_init_state(self, value):
        self.events.append(("initial", value))
        return self.observation

    def step(self, action):
        if action[0] == 0:
            self.waited += 1
        else:
            self.controlled += 1
        return self.observation, 0, True, {}

    def check_success(self):
        return True


def test_pro_resets_each_environment_with_seed_and_always_settles_before_policy():
    events = []
    envs = [InitiallySolvedEnvironment(events), InitiallySolvedEnvironment(events)]

    def plan(observations):
        assert all(env.waited == 10 for env in envs)
        return np.full((len(observations), 16, 7), 0.25), np.zeros((len(observations), 16, 10))

    results = native.rollout_batch(
        envs,
        [0, 1],
        plan,
        max_steps=220,
        environment_seed=0,
        reference_settling=True,
    )
    assert events == [("seed", 0), "reset", ("initial", 0), ("seed", 0), "reset", ("initial", 1)]
    assert all(
        record["success"] and record["steps"] == 1 and record["wait_steps"] == 10 for record, _ in results
    )
    assert [env.controlled for env in envs] == [1, 1]

    original = InitiallySolvedEnvironment([])
    result = native.rollout_batch(
        [original],
        [0],
        lambda _: pytest.fail("Original runner should retain its pre-policy success check"),
        max_steps=220,
    )
    assert result[0][0]["success"] and result[0][0]["steps"] == 0
    assert original.waited == 1


def test_pro_merge_preserves_fixed_sources_and_reports_changed_environments(tmp_path):
    catalog = [
        {
            "suite": "libero_spatial_env",
            "base_suite": "libero_spatial",
            "category": "environment",
            "task_id": index,
            "environment_changed": index == 0,
            "instruction": f"actual BDDL instruction {index}",
        }
        for index in range(2)
    ]
    paths = []
    for index, row in enumerate(catalog):
        settings = {
            "protocol": evaluator.PROTOCOL,
            "trials_per_task": 1,
            "batch_size": 10,
            "policy_identity": {"config_sha256": "same-policy"},
            "selected_tasks": [[row["suite"], row["task_id"]]],
            "num_shards": 2,
            "shard_index": index,
        }
        episode = {**row, "trial": 0, "init_state_index": 0, "success": index == 1}
        path = tmp_path / f"shard{index}.json"
        path.write_text(json.dumps(evaluator.summarize(settings, [episode], catalog)))
        paths.append(path)
    result = evaluator.merge_results(paths, catalog)
    assert result["evaluation_complete"] and result["benchmark_task_coverage_complete"]
    assert not result["official_50_trials_per_task"]
    assert result["successes"] == 1 and result["episodes_completed"] == 2
    assert result["environment_change"]["changed"]["success_rate"] == 0
    assert result["environment_change"]["unchanged"]["success_rate"] == 1
    with pytest.raises(ValueError, match="Overlapping"):
        evaluator.merge_results([paths[0], paths[0]], catalog)
    other = json.loads(paths[1].read_text())
    other["settings"]["batch_size"] = 5
    paths[1].write_text(json.dumps(other))
    with pytest.raises(ValueError, match="different native"):
        evaluator.merge_results(paths, catalog)
    other["episodes"] = []
    paths[1].write_text(json.dumps(other))
    with pytest.raises(ValueError, match="Incomplete"):
        evaluator.merge_results(paths, catalog)
