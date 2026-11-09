import numpy as np
import pytest
import torch

from lerobot.evaluation.eval_cosmos3_libero import observation_batch, rollout_batch


class Environment:
    def __init__(self, success_after=None):
        self.success_after = success_after
        self.observation = {
            "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
        }

    def reset(self):
        self.actions = []
        self.settle_steps = 0

    def set_init_state(self, initial):
        self.initial = initial
        return self.observation

    def check_success(self):
        return self.success_after is not None and len(self.actions) >= self.success_after

    def step(self, action):
        assert not self.check_success(), "A finished environment was stepped again"
        if action[0] == 0:
            self.settle_steps += 1
        else:
            self.actions.append(action)
        return self.observation, 0, self.check_success(), {}


def test_batched_rollout_fixed_states_settling_completion_and_partial_chunks():
    envs = [Environment(3), Environment()]
    widths = []

    def plan(observations):
        widths.append(len(observations))
        return np.full((len(observations), 2, 7), 0.25), np.zeros((len(observations), 2, 10))

    results = rollout_batch(envs, ["state_0", "state_1"], plan, max_steps=5, replan_steps=2)
    assert [env.initial for env in envs] == ["state_0", "state_1"]
    assert [env.settle_steps for env in envs] == [10, 10]
    assert widths == [2, 2, 1]
    first, second = [record for record, _ in results]
    assert first["success"] and first["steps"] == 3 and not first["timeout"]
    assert not second["success"] and second["steps"] == 5 and second["timeout"]
    assert len(first["actions"]) == 3 and len(second["actions"]) == 5
    assert [len(frames) for _, frames in results] == [4, 6]


def test_rollout_rejects_nonfinite_actions_before_actuation():
    env = Environment()
    with pytest.raises(ValueError, match="finite"):
        rollout_batch(
            [env],
            [0],
            lambda _: (np.full((1, 16, 7), np.nan), np.zeros((1, 16, 10))),
            max_steps=100,
        )
    assert not env.actions


def test_native_observation_batch_keeps_cameras_uint8_and_rotates_both():
    env = Environment()
    env.observation["agentview_image"][0, 0] = [1, 2, 3]
    env.observation["robot0_eye_in_hand_image"][0, 0] = [4, 5, 6]
    batch = observation_batch([env.observation, env.observation], "move the bowl", torch.device("cpu"))
    assert batch["task"] == ["move the bowl", "move the bowl"]
    for key, expected in (
        ("observation.images.image", [1, 2, 3]),
        ("observation.images.image2", [4, 5, 6]),
    ):
        assert batch[key].dtype == torch.uint8
        assert batch[key].shape == (2, 3, 256, 256)
        assert batch[key][1, :, -1, -1].tolist() == expected
