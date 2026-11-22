"""Per-robot causal inference sessions; normalization and hardware execution stay in the adapter."""

import copy
from numbers import Integral

import torch

from lerobot.common.datasets.physical_history import PhysicalHistoryBuffer
from lerobot.common.datasets.visual_history import VisualHistoryBuffer


class Qwen3VLMoTSampler:
    def __init__(self, policy):
        self.policy = policy
        self.reset()

    @property
    def control_step(self):
        return self._control_step

    def reset(self):
        config = self.policy.config
        self._physical = (
            PhysicalHistoryBuffer(config.physical_history_steps, config.physical_history_modalities)
            if getattr(config, "physical_history_steps", 0)
            else None
        )
        self._visual = (
            VisualHistoryBuffer(config.visual_history_offsets, config.visual_history_include_first)
            if getattr(config, "visual_history_enabled", False)
            else None
        )
        self._batch = None
        self._control_step = -1

    @staticmethod
    def _mask(values, shape, name):
        if not isinstance(values, torch.Tensor) or values.shape != shape:
            raise ValueError(f"{name} must be a canonical dimension/view mask with shape {shape}.")
        if not ((values == 0) | (values == 1)).all():
            raise ValueError(f"{name} must contain only finite zero/one values.")
        return values.bool()

    def _snapshot(self, batch):
        owned = {
            "history_state",
            "history_state_mask",
            "history_action",
            "history_action_mask",
            "context_images",
            "context_image_mask",
            "context_frame_offsets",
        }
        if owned.intersection(batch):
            raise ValueError(
                "Pass current observations, not precomputed history; the sampler owns its buffers."
            )
        targets = {"action", "video", "video_views", "image_t1", "images_t1"}
        snapshot = {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
            for key, value in batch.items()
            if key not in targets
        }
        state = snapshot.get("observation.state")
        if not isinstance(state, torch.Tensor) or state.ndim not in (2, 3):
            raise ValueError("observation.state must be normalized canonical (B,D) or (B,T,D).")
        if state.ndim == 3:
            if not state.shape[1]:
                raise ValueError("observation.state must include a current timestep.")
            state = state[:, 0]
        batch_size = state.shape[0]
        if batch_size < 1 or state.shape != (batch_size, self.policy.config.max_state_dim):
            raise ValueError("Current state width must match the checkpoint's canonical state layout.")
        if self._batch is not None and batch_size != self._batch["observation.state"].shape[0]:
            raise ValueError("Robot batch size changed; reset the sampler at episode boundaries.")
        state_mask = self._mask(snapshot.get("state_mask"), state.shape, "state_mask")
        action_mask = self._mask(
            snapshot.get("action_mask"), (batch_size, self.policy.config.max_action_dim), "action_mask"
        )
        if not action_mask.any(-1).all():
            raise ValueError("Every robot must declare valid canonical action dimensions.")
        state = torch.where(state_mask.to(state.device), state.float(), 0)
        if not torch.isfinite(state).all():
            raise ValueError("Valid normalized current state values must be finite.")
        snapshot.update({"observation.state": state, "state_mask": state_mask, "action_mask": action_mask})
        primary = self.policy._current_image(snapshot)
        if primary.ndim != 4 or primary.shape[:2] != (batch_size, 3) or min(primary.shape[-2:]) < 1:
            raise ValueError("Current RGB must be batched (B,3,H,W).")
        snapshot["image_t0"] = primary
        views = len(getattr(self.policy.config, "rgb_views", ("primary",)))
        if "image_view_mask" in snapshot:
            snapshot["image_view_mask"] = self._mask(
                snapshot["image_view_mask"], (batch_size, views), "image_view_mask"
            )
        images, mask = VisualHistoryBuffer.current(snapshot)
        if images.ndim != 5 or images.shape[:3] != (batch_size, views, 3):
            raise ValueError("Current RGB views must match the checkpoint's saved camera order.")
        self._mask(mask, (batch_size, views), "image_view_mask")
        if not torch.isfinite(primary).all() or not torch.isfinite(images).all():
            raise ValueError("Current RGB values must be finite; pad unavailable views and mask them.")
        return snapshot

    def update(self, batch, *, control_step, executed_action=None):
        """Observe step 0, then each executed step's normalized feedback before replanning."""
        if (
            isinstance(control_step, bool)
            or not isinstance(control_step, Integral)
            or control_step != self._control_step + 1
        ):
            raise ValueError("control_step must start at 0 and advance by exactly one executed step.")
        if self._batch is None and executed_action is not None:
            raise ValueError("The first observation must not include an action from a previous episode.")
        snapshot = self._snapshot(batch)
        if executed_action is not None:
            action = torch.as_tensor(executed_action).detach().float().clone()
            previous_mask = self._batch["action_mask"].to(action.device)
            if action.shape != previous_mask.shape:
                raise ValueError("Executed action must be one normalized canonical step (B,D), not a chunk.")
            if not torch.isfinite(action[previous_mask]).all():
                raise ValueError("Valid executed action values must be finite.")
            executed_action = torch.where(previous_mask, action, 0)
        # Fork containers, not old tensor snapshots, so rejected updates cannot advance either history.
        physical, visual = copy.copy(self._physical), copy.copy(self._visual)
        if visual is not None:
            visual.frames = visual.frames.copy()
            visual.update(snapshot)
        if physical is not None:
            physical.frames = {name: frames.copy() for name, frames in physical.frames.items()}
            physical.update(snapshot, executed_action=executed_action)
        self._physical, self._visual = physical, visual
        self._batch = snapshot
        self._control_step = int(control_step)

    @torch.no_grad()
    def sample(self, *, generator=None):
        """Return a normalized canonical chunk without adding any predicted action to history."""
        if self._batch is None:
            raise ValueError("Call update with the initial observation before sampling.")
        if self.policy.training:
            raise ValueError("Deployment sampling requires policy.eval().")
        batch = dict(self._batch)
        if self._visual is not None:
            batch = self._visual.attach(batch)
        if self._physical is not None:
            batch = self._physical.attach(batch)
        action = self.policy.sample_canonical_action(batch, generator=generator)
        expected = (
            batch["observation.state"].shape[0],
            self.policy.config.n_action_steps,
            self.policy.config.max_action_dim,
        )
        if not isinstance(action, torch.Tensor) or action.shape != expected:
            raise ValueError(f"Sampled canonical action must have shape {expected}.")
        mask = batch["action_mask"].to(action.device)[:, None].expand_as(action)
        if not torch.isfinite(action[mask]).all():
            raise ValueError("Sampled valid actions are nonfinite; do not send them to hardware.")
        return torch.where(mask, action, 0)
