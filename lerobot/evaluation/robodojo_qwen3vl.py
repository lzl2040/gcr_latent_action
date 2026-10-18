"""Training-space adapter for RoboDojo's dual ARX-X5 end-effector interface."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.nn import functional

from lerobot.evaluation.eval_qwen3vl_mot_libero import _finite_array, ort6d_to_controls

logger = logging.getLogger(__name__)
CAMERAS = {"primary": "cam_head", "secondary": "cam_left_wrist", "wrist": "cam_right_wrist"}
ARMS = ("left", "right")


def state_to_canonical(observation: dict) -> np.ndarray:
    parts = []
    for arm in ARMS:
        pose = _finite_array(observation["state"][f"{arm}_ee_pose"], (7,), f"{arm} pose")
        gripper = _finite_array(observation["state"][f"{arm}_ee_joint_state"], (1,), f"{arm} gripper")
        if not np.isclose(np.linalg.norm(pose[3:]), 1.0, atol=1e-4):
            raise ValueError(f"{arm} world-frame wxyz quaternion must have unit norm.")
        matrix = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
        parts.append(np.concatenate((pose[:3], matrix[:, :2].reshape(6), gripper)))
    return np.concatenate(parts).astype(np.float32)


def camera_tensor(observation: dict, name: str, size: int) -> torch.Tensor:
    image = np.asarray(observation["vision"][name]["color"])
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError(f"{name} must be decoded HWC uint8 RGB, got {image.shape}/{image.dtype}.")
    image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
    if image.shape[-2:] != (size, size):
        image = functional.interpolate(
            image[None].float(), size=(size, size), mode="bilinear", align_corners=False
        )[0].to(torch.uint8)
    return image


@dataclass
class RoboDojoAdapter:
    state_mean: torch.Tensor
    state_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor
    fps: int

    @classmethod
    def from_dataset(cls, root: Path) -> RoboDojoAdapter:
        info = json.loads((root / "meta/info.json").read_text())
        stats = json.loads((root / "meta/stats.json").read_text())
        values = []
        for key in ("observation.ee_ort6d_pos", "action.ee_ort6d_pos"):
            mean = _finite_array(stats[key]["mean"], (20,), f"{key} mean")
            std = _finite_array(stats[key]["std"], (20,), f"{key} std")
            if np.any(std < 0):
                raise ValueError(f"{key} standard deviations cannot be negative.")
            values.extend(
                (torch.tensor(mean, dtype=torch.float32), torch.tensor(std.clip(1e-3), dtype=torch.float32))
            )
        fps = info["fps"]
        if not isinstance(fps, int) or isinstance(fps, bool) or fps <= 0:
            raise ValueError("Training metadata must declare a positive integer source fps.")
        return cls(*values, fps=fps)

    def observation_batch(self, observation: dict, config, device: torch.device) -> dict:
        if config.max_state_dim != 40 or config.max_action_dim != 40:
            raise ValueError("RoboDojo requires the saved 40D canonical layout.")
        if config.use_tactile_conditioning:
            raise ValueError("This RoboDojo interface has no tactile observations.")
        if not config.rgb_views or config.rgb_views[0] != "primary":
            raise ValueError("The saved camera order must start with primary.")
        instruction = observation["instruction"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("RoboDojo must provide a nonempty task instruction.")
        state = torch.zeros(1, 40)
        state[0, :20] = (torch.from_numpy(state_to_canonical(observation)) - self.state_mean) / self.state_std
        mask = torch.zeros(1, 40)
        mask[:, :20] = 1
        # All saved views are mandatory: do not silently evaluate a three-camera model with one view.
        images = torch.stack(
            [camera_tensor(observation, CAMERAS[view], config.video_image_size) for view in config.rgb_views]
        )[None]
        batch = {
            "image_t0": images[:, 0],
            "observation.state": state,
            "state_mask": mask,
            "action_mask": mask.clone(),
            "sample_rate": torch.tensor([self.fps]),
            "tactile_image_mask": torch.zeros(1, config.max_tactile_views),
            "task": [instruction],
        }
        if len(config.rgb_views) > 1:
            batch["images_t0"] = images
            batch["image_view_mask"] = torch.ones(1, len(config.rgb_views), dtype=torch.bool)
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def native_actions(self, normalized: torch.Tensor) -> list[dict]:
        if normalized.ndim != 3 or normalized.shape[0] != 1 or normalized.shape[2] != 40:
            raise ValueError("Expected one canonical action chunk with shape (1,T,40).")
        if normalized.shape[1] == 0 or not torch.isfinite(normalized).all():
            raise ValueError("Predicted action chunks must be nonempty and finite.")
        raw = normalized[0, :, :20].detach().float().cpu() * self.action_std + self.action_mean
        if not torch.isfinite(raw).all():
            raise ValueError("Inverse-normalized actions contain nonfinite values.")
        raw = raw.numpy()
        chunks = [{} for _ in raw]
        for arm, offset in zip(ARMS, (0, 10), strict=True):
            # Reuse the existing interleaved-column Gram-Schmidt projection.
            rotation = Rotation.from_euler("xyz", ort6d_to_controls(raw[:, offset + 3 : offset + 9]))
            xyzw = rotation.as_quat()
            poses = np.concatenate((raw[:, offset : offset + 3], xyzw[:, [3, 0, 1, 2]]), axis=-1)
            grip = raw[:, offset + 9 : offset + 10]
            if np.any((grip < 0) | (grip > 1)):
                logger.warning("Clipping %s gripper predictions to RoboDojo's native [0,1] range.", arm)
            for action, pose, gripper in zip(chunks, poses, grip.clip(0, 1), strict=True):
                action[f"{arm}_ee_pose"] = pose.astype(np.float32)
                action[f"{arm}_ee_joint_state"] = gripper.astype(np.float32)
        return chunks


class RoboDojoModel:
    """XPolicyLab model interface; one model, independent latest observations per environment."""

    def __init__(self, policy, adapter: RoboDojoAdapter, *, device: str, replan_steps: int, output_dir: Path):
        if not 1 <= replan_steps <= policy.config.n_action_steps:
            raise ValueError("replan_steps must be in [1, saved n_action_steps].")
        self.policy = policy
        self.adapter = adapter
        self.device = torch.device(device)
        self.replan_steps = replan_steps
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.observations: dict[int, dict] = {}
        self.single_env_idx: int | None = None
        self.calls = 0
        self.reset_index = -1

    def reset(self):
        self.observations.clear()
        self.single_env_idx = None
        self.policy.reset()
        self.reset_index += 1

    def update_obs(self, obs):
        index = int(obs["env_idx"])
        self.observations[index] = obs
        self.single_env_idx = index

    def update_obs_batch(self, obs_list):
        indices = [int(obs["env_idx"]) for obs in obs_list]
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("Observation batches must have distinct environment indices.")
        for obs in obs_list:
            self.update_obs(obs)

    @torch.inference_mode()
    def _predict(self, index: int) -> list[dict]:
        observation = self.observations[index]
        batch = self.adapter.observation_batch(observation, self.policy.config, self.device)
        started = time.perf_counter()
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            predicted = self.policy.sample_canonical_action(batch)
        actions = self.adapter.native_actions(predicted)
        if len(actions) < self.replan_steps:
            raise ValueError("The sampled chunk is shorter than replan_steps.")
        seconds = time.perf_counter() - started
        path = self.output_dir / f"chunk_{self.calls:06d}_env_{index:03d}.npz"
        np.savez_compressed(
            path,
            normalized_actions=predicted.detach().float().cpu().numpy(),
            **{key: np.stack([action[key] for action in actions]) for key in actions[0]},
        )
        record = {
            "call": self.calls,
            "reset_index": self.reset_index,
            "env_idx": index,
            "instruction": observation["instruction"],
            "inference_seconds": seconds,
            "predicted_steps": len(actions),
            "executed_chunk_steps": self.replan_steps,
            "action_path": str(path.resolve()),
        }
        with (self.output_dir / "inference.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.calls += 1
        logger.info("env=%d chunk=%d steps=%d inference=%.3fs", index, self.calls, self.replan_steps, seconds)
        return actions[: self.replan_steps]

    def get_action(self):
        if self.single_env_idx is None:
            raise RuntimeError("update_obs must precede get_action after reset.")
        return self._predict(self.single_env_idx)

    def get_action_batch(self, env_idx_list=None):
        indices = list(self.observations) if env_idx_list is None else [int(index) for index in env_idx_list]
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("Action requests must contain distinct environment indices.")
        # Sequential per-environment inference bounds peak GPU memory for the multi-view world model.
        return [self._predict(index) for index in indices]
