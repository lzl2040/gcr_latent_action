from types import SimpleNamespace

import pytest
import torch

from lerobot.common.policies.qwen3vl_mot.sampling_qwen3vl_mot import Qwen3VLMoTSampler


class RecordingPolicy(torch.nn.Module):
    def __init__(self, mode="state_action", visual=False):
        super().__init__()
        self.config = SimpleNamespace(
            max_state_dim=4,
            max_action_dim=4,
            n_action_steps=4,
            physical_history_steps=3 if mode else 0,
            physical_history_modalities=tuple(mode.split("_")) if mode else (),
            rgb_views=("primary", "wrist"),
            visual_history_enabled=visual,
            visual_history_offsets=(-2,),
            visual_history_include_first=True,
        )
        self.output = torch.tensor([[[0.1, 0, 0.2, 0], [0.3, 0, 0.4, 0], [99, 0, 99, 0], [100, 0, 100, 0]]])
        self.seen = None

    @staticmethod
    def _current_image(batch):
        return batch["image_t0"]

    def sample_canonical_action(self, batch, *, generator=None):
        self.seen = batch
        return self.output.expand(batch["image_t0"].shape[0], -1, -1).clone()


def observation(step):
    return {
        "observation.state": torch.tensor([[float(step), float(step + 10), float("nan"), 0]]),
        "state_mask": torch.tensor([[1, 1, 0, 0]]),
        "action_mask": torch.tensor([[1, 0, 1, 0]]),
        "image_t0": torch.full((1, 3, 4, 4), step, dtype=torch.uint8),
        "images_t0": torch.stack(
            (torch.full((3, 4, 4), step, dtype=torch.uint8), torch.zeros(3, 4, 4, dtype=torch.uint8))
        )[None],
        "image_view_mask": torch.tensor([[True, False]]),
        "task": ["pick up"],
    }


@pytest.mark.parametrize("mode", ["", "state", "action", "state_action"])
@pytest.mark.parametrize("visual", [False, True])
def test_only_executed_steps_enter_history_and_repeated_sampling_does_not_advance_it(mode, visual):
    policy = RecordingPolicy(mode, visual).eval()
    sampler = Qwen3VLMoTSampler(policy)
    for step in range(7):
        current = observation(step)
        current["action"] = torch.full((1, 4, 4), 999.0)
        current["video"] = torch.zeros(1, 9, 3, 4, 4)
        current["observation.state"] = current["observation.state"][:, None].expand(-1, 4, -1).clone()
        current["observation.state"][:, 1:] = 777
        executed = None if step == 0 else torch.tensor([[float(step - 1), float("nan"), step + 19.0, 0]])
        sampler.update(current, control_step=step, executed_action=executed)
        current["observation.state"][:, 0, :2] += 500
        current["image_t0"].fill_(255)
        for _ in range(2):
            predicted = sampler.sample()
            assert not predicted[:, :, [1, 3]].any()
            assert "action" not in policy.seen and "video" not in policy.seen
            assert policy.seen["observation.state"][0, 0] == step
            assert policy.seen["image_t0"][0, 0, 0, 0] == step
            for name in ("state", "action"):
                if name not in policy.config.physical_history_modalities:
                    assert f"history_{name}" not in policy.seen
                    continue
                values, mask = policy.seen[f"history_{name}"], policy.seen[f"history_{name}_mask"]
                past = list(range(max(0, step - 3), step))
                assert not mask[:, : 3 - len(past)].any()
                assert not values[~mask].any()
                assert values[0, 3 - len(past) :, 0].tolist() == past
            if visual:
                assert policy.seen["context_images"][0, :, 0, 0, 0, 0].tolist() == [0, max(0, step - 2), step]
                assert not policy.seen["context_image_mask"][:, :, 1].any()
            else:
                assert "context_images" not in policy.seen


def test_only_the_executed_prefix_is_recorded_and_failed_updates_are_atomic():
    policy = RecordingPolicy("state_action", visual=True).eval()
    sampler = Qwen3VLMoTSampler(policy)
    sampler.update(observation(0), control_step=0)
    chunk = sampler.sample()
    for invalid in (None, chunk, torch.full((1, 4), float("nan"))):
        with pytest.raises(ValueError, match="executed|Executed"):
            sampler.update(observation(1), control_step=1, executed_action=invalid)
        sampler.sample()
        assert not policy.seen["history_action_mask"].any()
        assert policy.seen["context_images"][0, -1, 0, 0, 0, 0] == 0
    for step in (1, 2):
        sampler.update(observation(step), control_step=step, executed_action=chunk[:, step - 1])
    sampler.sample()
    torch.testing.assert_close(policy.seen["history_action"][:, -2:], chunk[:, :2])
    assert not (policy.seen["history_action"] == 99).any()
    sampler.reset()
    with pytest.raises(ValueError, match="initial observation"):
        sampler.sample()
    with pytest.raises(ValueError, match="previous episode"):
        sampler.update(observation(0), control_step=0, executed_action=chunk[:, 0])
    sampler.update(observation(0), control_step=0)
    sampler.sample()
    assert not policy.seen["history_state_mask"].any()
    assert not policy.seen["history_action_mask"].any()


def test_state_only_feedback_and_independent_robots_can_share_model_weights():
    policy = RecordingPolicy("state", visual=True).eval()
    first, second = Qwen3VLMoTSampler(policy), Qwen3VLMoTSampler(policy)
    first.update(observation(0), control_step=0)
    first.update(observation(1), control_step=1)
    second.update(observation(7), control_step=0)
    second.update(observation(8), control_step=1)
    second.sample()
    assert policy.seen["history_state"][0, -1, 0] == 7
    assert "history_action" not in policy.seen
    first.reset()
    second.sample()
    assert policy.seen["history_state"][0, -1, 0] == 7
    assert policy.seen["context_images"][0, 0, 0, 0, 0, 0] == 7


@pytest.mark.parametrize("step", [0, 2, -1, True, 1.0])
def test_duplicate_skipped_or_noninteger_control_steps_are_rejected(step):
    sampler = Qwen3VLMoTSampler(RecordingPolicy("state").eval())
    sampler.update(observation(0), control_step=0)
    with pytest.raises(ValueError, match="exactly one"):
        sampler.update(observation(1), control_step=step)
    sampler.update(observation(1), control_step=1)
    sampler.sample()
    assert sampler.policy.seen["history_state"][0, -1, 0] == 0


@pytest.mark.parametrize(
    "bad", ["state", "state_width", "state_mask", "image_view_mask", "view_count", "rgb", "external_history"]
)
def test_invalid_current_observations_do_not_replace_the_previous_frame(bad):
    sampler = Qwen3VLMoTSampler(RecordingPolicy("state", visual=True).eval())
    sampler.update(observation(0), control_step=0)
    current = observation(1)
    if bad == "state":
        current["observation.state"][0, 0] = float("nan")
    elif bad == "state_width":
        current["observation.state"] = current["observation.state"][:, :3]
    elif bad in ("state_mask", "image_view_mask"):
        current[bad] = current[bad].float()
        current[bad][0, 0] = float("nan")
    elif bad == "rgb":
        current["image_t0"] = current["image_t0"].float()
        current["image_t0"][0, 0, 0, 0] = float("nan")
    elif bad == "view_count":
        current["images_t0"] = current["images_t0"][:, :1]
    else:
        current["history_state"] = torch.zeros(1, 3, 4)
    with pytest.raises(ValueError):
        sampler.update(current, control_step=1)
    sampler.sample()
    assert sampler.policy.seen["observation.state"][0, 0] == 0


@pytest.mark.parametrize("bad", ["nonfinite_valid", "empty", "width"])
def test_invalid_prediction_is_not_returned_for_hardware_execution(bad):
    policy = RecordingPolicy().eval()
    sampler = Qwen3VLMoTSampler(policy)
    sampler.update(observation(0), control_step=0)
    if bad == "nonfinite_valid":
        policy.output[0, 0, 0] = float("nan")
    elif bad == "empty":
        policy.output = policy.output[:, :0]
    else:
        policy.output = policy.output[:, :, :3]
    with pytest.raises(ValueError, match="Sampled"):
        sampler.sample()
    policy.output = torch.ones(1, 4, 4)
    policy.output[:, :, 1] = float("nan")
    assert torch.isfinite(sampler.sample()).all()
