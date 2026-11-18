"""Causal normalized state/action histories updated only after executed control steps."""

from collections import deque

import torch


class PhysicalHistoryBuffer:
    def __init__(self, steps, modalities):
        if steps < 1 or not modalities or set(modalities) - {"state", "action"}:
            raise ValueError("Physical history requires positive steps and state/action modalities.")
        self.steps = steps
        self.modalities = tuple(modalities)
        self.frames = {name: deque(maxlen=steps) for name in modalities}
        self.current = None
        self.step = -1

    @staticmethod
    def current_state(batch):
        state = batch["observation.state"]
        return state[:, 0] if state.ndim == 3 else state

    def update(self, batch, *, executed_action=None):
        state = self.current_state(batch).detach().cpu().clone()
        state_mask = batch["state_mask"].detach().cpu().bool().clone()
        action_mask = batch["action_mask"].detach().cpu().bool().clone()
        if state.ndim != 2 or state.shape != state_mask.shape:
            raise ValueError("Physical history needs canonical current state and dimension masks.")
        if self.current is not None:
            previous, previous_state_mask, previous_action_mask = self.current
            if previous.shape != state.shape:
                raise ValueError("Physical history batch shape changed; reset at episode boundaries.")
            if "action" in self.modalities:
                if executed_action is None:
                    raise ValueError(
                        "Action history requires the action actually executed before each observation."
                    )
                action = torch.as_tensor(executed_action).detach().float().cpu().clone()
                if action.shape != previous_action_mask.shape:
                    raise ValueError("Executed action must have canonical normalized shape (B,D).")
                if not torch.isfinite(action[previous_action_mask]).all():
                    raise ValueError("Executed action history must be finite.")
                self.frames["action"].append((action, previous_action_mask))
            if "state" in self.modalities:
                self.frames["state"].append((previous, previous_state_mask))
        elif executed_action is not None:
            raise ValueError("Reset history must not include an action from a previous episode.")
        self.current = (state, state_mask, action_mask)
        self.step += 1
        return self.attach(batch)

    def attach(self, batch):
        if self.current is None or not torch.equal(self.current_state(batch).detach().cpu(), self.current[0]):
            raise ValueError("Update physical history at every control observation before planning.")
        device = self.current_state(batch).device
        result = dict(batch)
        for name in self.modalities:
            width = self.current[0 if name == "state" else 2].shape[-1]
            values = torch.zeros(self.current[0].shape[0], self.steps, width)
            mask = torch.zeros_like(values, dtype=torch.bool)
            for index, (value, keep) in enumerate(self.frames[name], self.steps - len(self.frames[name])):
                values[:, index] = torch.where(keep, value, 0.0)
                mask[:, index] = keep
            result[f"history_{name}"] = values.to(device)
            result[f"history_{name}_mask"] = mask.to(device)
        return result
