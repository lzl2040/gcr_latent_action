"""Single-GPU LIBERO rollouts of a Stage 2 Qwen3-VL/Cosmos DCP checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import shutil
import sys
import time
from collections import deque
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.nn import functional

logger = logging.getLogger(__name__)

SUITE_DATASETS = {
    suite: f"{suite}_no_noops_lerobot"
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
}
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
CANONICAL_DIM = 40
DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def _finite_array(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {array.shape}.")
    return array


def state_to_ort6d(observation: dict) -> np.ndarray:
    position = _finite_array(observation["robot0_eef_pos"], (3,), "eef position")
    quat = _finite_array(observation["robot0_eef_quat"], (4,), "eef quaternion").copy()
    fingers = _finite_array(observation["robot0_gripper_qpos"], (2,), "gripper positions")
    if not np.isclose(np.linalg.norm(quat), 1.0, atol=1e-4):
        raise ValueError("LIBERO eef quaternion must have unit norm.")
    w = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - w * w)
    axes = np.zeros(3) if denominator < 1e-6 else quat[:3] * (2 * np.arccos(w) / denominator)
    # v20_to_v21.py treats the reference evaluator's axis-angle triple as Euler XYZ.
    # Reproduce that training representation, not the physically equivalent quaternion matrix.
    columns = Rotation.from_euler("xyz", axes).as_matrix()[:, :2].reshape(6)
    return np.concatenate((position, columns, fingers[-1:])).astype(np.float32)


def ort6d_to_controls(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.ndim != 2 or rotation.shape[1] != 6 or not np.isfinite(rotation).all():
        raise ValueError("Action rotations must be a finite (T,6) array.")
    columns = rotation.reshape(-1, 3, 2)
    first, second = columns[:, :, 0], columns[:, :, 1]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm < 1e-6):
        raise ValueError("Predicted 6D rotation has a degenerate first column.")
    first = first / first_norm
    second = second - (first * second).sum(-1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(second_norm < 1e-6):
        raise ValueError("Predicted 6D rotation has collinear columns.")
    second = second / second_norm
    matrix = np.stack((first, second, np.cross(first, second)), axis=-1)
    # Invert the dataset's Euler encoding to recover the original OSC control triple.
    return Rotation.from_matrix(matrix).as_euler("xyz")


def libero_image(observation: dict, key: str, size: int) -> torch.Tensor:
    image = np.asarray(observation[key])
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"{key} must be an HWC uint8 RGB image, got {image.shape}/{image.dtype}.")
    image = torch.from_numpy(np.ascontiguousarray(image[::-1, ::-1])).permute(2, 0, 1)
    if image.shape[-2:] != (size, size):
        image = functional.interpolate(
            image[None].float(), size=(size, size), mode="bilinear", align_corners=False
        )[0].to(torch.uint8)
    return image


@dataclass
class LiberoAdapter:
    state_mean: torch.Tensor
    state_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor
    fps: int

    @classmethod
    def from_dataset(cls, path: Path) -> LiberoAdapter:
        info = json.loads((path / "meta/info.json").read_text())
        stats = json.loads((path / "meta/stats.json").read_text())
        values = []
        for key in ("observation.state", "action"):
            if info["features"][key]["shape"] != [10]:
                raise ValueError(f"{path}: {key} must use xyz + interleaved ort6d + gripper (10D).")
            mean = _finite_array(stats[key]["mean"], (10,), f"{key} mean")
            std = _finite_array(stats[key]["std"], (10,), f"{key} std")
            if np.any(std < 0):
                raise ValueError(f"{key} standard deviations cannot be negative.")
            values.extend(
                (
                    torch.tensor(mean, dtype=torch.float32),
                    torch.tensor(np.maximum(std, 1e-3), dtype=torch.float32),
                )
            )
        fps = info["fps"]
        if not isinstance(fps, int) or fps <= 0:
            raise ValueError(f"{path}: expected a positive integer source fps.")
        return cls(*values, fps=fps)

    def observation_batch(self, observation: dict, instruction: str, config, device: torch.device) -> dict:
        if config.max_state_dim != CANONICAL_DIM or config.max_action_dim != CANONICAL_DIM:
            raise ValueError("This adapter requires the saved 40D canonical action/state layout.")
        state = torch.zeros(1, CANONICAL_DIM, dtype=torch.float32)
        state[0, :10] = (torch.from_numpy(state_to_ort6d(observation)) - self.state_mean) / self.state_std
        mask = torch.zeros(1, CANONICAL_DIM)
        mask[:, :10] = 1
        camera_keys = {"primary": "agentview_image", "wrist": "robot0_eye_in_hand_image"}
        images, view_mask = [], []
        for view in config.rgb_views:
            key = camera_keys.get(view)
            if key is None and view != "secondary":
                raise ValueError(f"Unsupported LIBERO RGB view {view!r}.")
            available = key is not None and key in observation
            if view == "primary" and not available:
                raise KeyError("LIBERO must provide agentview_image.")
            images.append(
                libero_image(observation, key, config.video_image_size)
                if available
                else torch.zeros(3, config.video_image_size, config.video_image_size, dtype=torch.uint8)
            )
            view_mask.append(available)
        if not images or config.rgb_views[0] != "primary":
            raise ValueError("The saved RGB view order must start with primary.")
        batch = {
            "image_t0": images[0][None].to(device),
            "observation.state": state.to(device),
            "state_mask": mask.to(device),
            "action_mask": mask.to(device),
            "sample_rate": torch.tensor([self.fps], device=device),
            "tactile_image_mask": torch.zeros(1, config.max_tactile_views, device=device),
            "task": [instruction],
        }
        if len(images) > 1:
            batch["images_t0"] = torch.stack(images)[None].to(device)
            batch["image_view_mask"] = torch.tensor([view_mask], device=device, dtype=torch.bool)
        return batch

    def native_actions(self, normalized: torch.Tensor) -> np.ndarray:
        if normalized.ndim != 3 or normalized.shape[0] != 1 or normalized.shape[2] != CANONICAL_DIM:
            raise ValueError("Expected one canonical action chunk shaped (1,T,40).")
        if normalized.shape[1] == 0 or not torch.isfinite(normalized).all():
            raise ValueError("The sampled action chunk is empty or nonfinite.")
        raw = normalized[0, :, :10].detach().float().cpu() * self.action_std + self.action_mean
        if not torch.isfinite(raw).all():
            raise ValueError("Inverse-normalized actions contain nonfinite values.")
        raw = raw.numpy()
        actions = np.concatenate(
            (raw[:, :3], ort6d_to_controls(raw[:, 3:9]), -np.sign(2 * raw[:, 9:10] - 1)), axis=-1
        )
        clipped = np.clip(actions, -1.0, 1.0)
        if np.any(clipped != actions):
            logger.warning(
                "Clipping %d predicted control values to the OSC [-1,1] action bounds.",
                np.count_nonzero(clipped != actions),
            )
        return clipped


class PredictionVideoWriter:
    def __init__(self, config, device: torch.device, source_fps: int):
        self.config = config
        self.device = device
        self.source_fps = source_fps
        self.decoder = None

    def _prepare_decoder(self) -> None:
        if self.decoder is not None:
            return
        from lerobot.common.policies.ace.cosmos3_encoders import build_cosmos3_vae

        decoder, _, _, _, _ = build_cosmos3_vae(self.config.cosmos3_dir, encoder_only=False)
        decoder.encoder = None
        decoder.quant_conv = None
        decoder.requires_grad_(False).eval()
        decoder.enable_tiling()
        self.decoder = decoder.to(device=self.device, dtype=torch.bfloat16)

    @torch.inference_mode()
    def save(
        self,
        prediction: dict,
        actions: np.ndarray,
        directory: Path,
        *,
        control_step: int,
        replan_index: int,
        seed: int,
        save_frames: bool = False,
    ) -> dict:
        import imageio.v2 as imageio

        from lerobot.common.datasets.dataset_fps import evenly_spaced_frame_timestamps

        views = tuple(self.config.rgb_views)
        if len(views) == 1:
            latents = prediction["video_latents"][:, None]
            valid = torch.ones(1, 1, dtype=torch.bool)
        else:
            latents = prediction["video_latents_views"]
            valid = prediction["video_view_mask"].bool().cpu()
        if latents.ndim != 6 or latents.shape[:2] != (1, len(views)):
            raise ValueError(
                "Predicted video latents must have shape (1,V,C,T,H,W) matching saved RGB views."
            )
        if valid.shape != (1, len(views)) or not bool(valid[0, 0]):
            raise ValueError("Predicted RGB masks must match saved views and include the primary camera.")
        if not torch.isfinite(latents).all():
            raise ValueError("Predicted video latents contain nonfinite values.")
        directory.mkdir(parents=True, exist_ok=False)
        self._prepare_decoder()
        frames_expected = self.config.world_video_frames
        times = evenly_spaced_frame_timestamps(self.config.chunk_size - 1, frames_expected, self.source_fps)
        fps = self.source_fps * (frames_expected - 1) / (self.config.chunk_size - 1)
        video_paths = {}
        frame_paths = {}
        for view_index, view in enumerate(views):
            if not bool(valid[0, view_index]):
                continue
            with torch.autocast(self.device.type, dtype=torch.bfloat16):
                decoded = self.decoder.decode(
                    latents[:, view_index].to(device=self.device, dtype=torch.bfloat16)
                ).sample
            expected = (1, 3, frames_expected, self.config.video_image_size, self.config.video_image_size)
            if tuple(decoded.shape) != expected or not torch.isfinite(decoded).all():
                raise ValueError(f"Decoded {view} video must be finite with shape {expected}.")
            frames = (
                ((decoded[0].float().clamp(-1, 1) + 1) * 127.5)
                .round()
                .to(torch.uint8)
                .permute(1, 2, 3, 0)
                .cpu()
                .numpy()
            )
            video_path = directory / f"{view}.mp4"
            imageio.mimsave(video_path, frames, fps=fps, codec="libx264", macro_block_size=1)
            video_paths[view] = str(video_path.resolve())
            if save_frames:
                frame_dir = directory / f"{view}_frames"
                frame_dir.mkdir()
                paths = []
                for frame_index, frame in enumerate(frames):
                    frame_path = frame_dir / f"frame_{frame_index:03d}.png"
                    imageio.imwrite(frame_path, frame)
                    paths.append(str(frame_path.resolve()))
                frame_paths[view] = paths
                imageio.imwrite(directory / f"{view}_filmstrip.png", np.concatenate(frames, axis=1))
        actions_path = directory / "action_chunk.npy"
        np.save(actions_path, actions.astype(np.float32))
        metadata = {
            "replan_index": replan_index,
            "control_step": control_step,
            "control_time_s": control_step / self.source_fps,
            "run_seed": seed,
            "sampling_rng": "run_seeded_continuous",
            "video_frames": frames_expected,
            "includes_current_frame": True,
            "video_fps": fps,
            "video_frame_times_s": times,
            "videos": video_paths,
            "action_chunk_path": str(actions_path.resolve()),
            "metadata_path": str((directory / "metadata.json").resolve()),
        }
        if save_frames:
            metadata["frames"] = frame_paths
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        return metadata


def checkpoint_paths(checkpoint: Path, policy_config: Path | None) -> tuple[Path, Path]:
    checkpoint = checkpoint.resolve()
    model_dir = checkpoint if (checkpoint / ".metadata").is_file() else checkpoint / "model"
    if not (model_dir / ".metadata").is_file():
        raise FileNotFoundError(
            f"Expected a DCP checkpoint at {checkpoint}/model/.metadata or {checkpoint}/.metadata."
        )
    if policy_config is None:
        candidates = (model_dir.parent / "config.json", model_dir.parent.parent / "config.json")
        policy_config = next((path for path in candidates if path.is_file()), None)
    if policy_config is None or not policy_config.is_file():
        raise FileNotFoundError("The saved policy config.json is required; pass --policy-config explicitly.")
    return model_dir, policy_config.resolve()


def load_dcp_model(model: torch.nn.Module, model_dir: Path) -> None:
    import torch.distributed.checkpoint as dcp

    reader = dcp.FileSystemReader(model_dir)
    stored = reader.read_metadata().state_dict_metadata
    state = model.state_dict()
    expected = {f"model.{name}": tensor for name, tensor in state.items()}
    if set(stored) != set(expected):
        missing = sorted(set(expected) - set(stored))
        unexpected = sorted(set(stored) - set(expected))
        raise ValueError(f"DCP model keys differ: missing={missing[:10]}, unexpected={unexpected[:10]}.")
    for name, tensor in expected.items():
        metadata = stored[name]
        if not isinstance(metadata, dcp.TensorStorageMetadata) or tuple(metadata.size) != tuple(tensor.shape):
            raise ValueError(
                f"DCP shape/type mismatch for {name}: expected {tuple(tensor.shape)}, got {metadata}."
            )
    dcp.load({"model": state}, storage_reader=reader, no_dist=True)
    model.load_state_dict(state, strict=True)


def load_policy(args):
    import draccus

    from lerobot.common.policies.qwen3vl_mot.configuration_qwen3vl_mot import Qwen3VLMoTConfig
    from lerobot.common.policies.qwen3vl_mot.modeling_qwen3vl_mot import Qwen3VLMoTPolicy
    from lerobot.common.policies.qwen3vl_mot.tasks import validate_saved_task_contract

    fp8_config = inference_fp8_config(args.precision, args.fp8_training_config, torch.device(args.device))
    model_dir, config_file = checkpoint_paths(args.checkpoint, args.policy_config)
    payload = json.loads(config_file.read_text())
    validate_saved_task_contract(payload)
    if payload.pop("type", None) != "qwen3vl_mot" or not payload.get("stage1_policy_config"):
        raise ValueError("Expected a self-contained qwen3vl_mot policy configuration.")
    for field, path in (("qwen3vl_dir", args.qwen3vl_dir), ("cosmos3_dir", args.cosmos3_dir)):
        if not path.is_dir():
            raise FileNotFoundError(f"Local pretrained asset directory does not exist: {path}.")
        payload[field] = str(path.resolve())
    if not args.stage1_text_dir.is_dir():
        raise FileNotFoundError(f"Stage 1 text asset directory does not exist: {args.stage1_text_dir}.")
    payload["stage1_policy_config"]["text_model_name"] = str(args.stage1_text_dir.resolve())
    payload["stage1_policy_config"]["tactile_pretrained"] = False
    payload["initialize_from_stage1"] = False
    if args.inference_steps is not None:
        payload["inference_steps"] = args.inference_steps
    config = draccus.decode(Qwen3VLMoTConfig, payload)
    if args.replan_steps > config.n_action_steps:
        raise ValueError("--replan-steps must not exceed the saved n_action_steps.")
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        policy = Qwen3VLMoTPolicy(config).to(dtype=torch.bfloat16)
    finally:
        torch.set_default_dtype(old_dtype)
    logger.info(
        "Restoring all model tensors from %s (no optimizer or distributed launch required).", model_dir
    )
    load_dcp_model(policy, model_dir)
    policy.requires_grad_(False).eval().to(args.device)
    precision_details = {"precision": args.precision, "converted_linears": 0}
    if fp8_config is not None:
        from lerobot.common.utils.fsdp_training import convert_policy_to_fp8

        converted = convert_policy_to_fp8(policy, fp8_config, torch.device(args.device))
        policy.requires_grad_(False).eval()
        precision_details.update(
            converted_linears=len(converted),
            recipe=fp8_config.fp8_recipe,
            scope=fp8_config.fp8_scope,
            min_features=fp8_config.fp8_min_features,
            emulate=False,
            training_config_sha256=hashlib.sha256(args.fp8_training_config.read_bytes()).hexdigest(),
        )
    policy.libero_precision_details = precision_details
    logger.info("Inference precision: %s", precision_details)
    logger.info(
        "Loaded %.3fB parameters; tasks=%s, RGB views=%s, denoising steps=%d.",
        sum(p.numel() for p in policy.parameters()) / 1e9,
        config.task_names,
        config.rgb_views,
        config.inference_steps,
    )
    return policy


def inference_fp8_config(precision: str, training_config: Path | None, device: torch.device):
    if precision == "bf16":
        return None
    if precision != "fp8" or training_config is None:
        raise ValueError("FP8 requires --fp8-training-config pointing to the saved train_config.json.")
    from lerobot.common.utils.fsdp_training import FSDPTrainingConfig, resolve_fp8_mode

    saved = json.loads(training_config.read_text())["fsdp"]
    required = {"fp8", "fp8_recipe", "fp8_scope", "fp8_emulate", "fp8_min_features"}
    if missing := required - saved.keys():
        raise ValueError(f"Saved training configuration lacks explicit FP8 settings: {sorted(missing)}.")
    config = FSDPTrainingConfig(**saved)
    if config.fp8 is not True or config.fp8_emulate:
        raise ValueError("The saved training configuration must enable native FP8, not emulation.")
    resolve_fp8_mode(config, device)
    import importlib.metadata

    from packaging.version import Version

    if not Version("0.15") <= Version(importlib.metadata.version("torchao")) < Version("0.16"):
        raise RuntimeError("Native FP8 evaluation requires the training-compatible torchao>=0.15,<0.16.")
    return config


def sampling_fingerprint(batch: dict, device: torch.device) -> dict:
    digest = hashlib.sha256()
    for key, value in sorted(batch.items()):
        digest.update(key.encode())
        if isinstance(value, torch.Tensor):
            digest.update(f"{value.dtype}:{tuple(value.shape)}".encode())
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True).encode())
    return {
        "batch_sha256": digest.hexdigest(),
        "cpu_rng_sha256": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        "cuda_rng_sha256": hashlib.sha256(
            torch.cuda.get_rng_state(device).cpu().numpy().tobytes()
        ).hexdigest(),
    }


@contextmanager
def policy_inference_context(precision: str):
    # TorchAO 0.15 rowwise reshape needs view tracking, which inference_mode disables.
    with torch.inference_mode(precision != "fp8"), torch.no_grad():
        yield


def load_libero_initial_states(suite, task_id: int):
    from libero.libero import get_libero_path

    task = suite.get_task(task_id)
    path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    # Official LIBERO assets contain NumPy pickles, not weights-only state dicts.
    return torch.load(path, map_location="cpu", weights_only=False)


def configure_libero(libero_root: Path, output_dir: Path, render_gpu_device_id: int) -> None:
    import yaml

    benchmark_root = libero_root.resolve() / "libero/libero"
    paths = {
        "benchmark_root": benchmark_root,
        "bddl_files": benchmark_root / "bddl_files",
        "init_states": benchmark_root / "init_files",
        "datasets": output_dir.resolve() / "libero_datasets_unused",
        "assets": benchmark_root / "assets",
    }
    for key in ("benchmark_root", "bddl_files", "init_states", "assets"):
        if not paths[key].is_dir():
            raise FileNotFoundError(f"LIBERO {key} directory is missing: {paths[key]}.")
    paths["datasets"].mkdir(exist_ok=True)
    config_dir = output_dir.resolve() / "libero_config"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "config.yaml").write_text(yaml.safe_dump({key: str(path) for key, path in paths.items()}))
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    if render_gpu_device_id >= 0:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(render_gpu_device_id)
    sys.path.insert(0, str(libero_root.resolve()))


def rollout_episode(
    env,
    initial_state,
    plan,
    *,
    max_steps: int,
    wait_steps: int,
    replan_steps: int,
    save_video: bool,
) -> tuple[dict, list]:
    env.reset()
    observation = env.set_init_state(initial_state)
    success, done = bool(env.check_success()), False
    for _ in range(wait_steps):
        if success or done:
            break
        observation, _, done, _ = env.step(DUMMY_ACTION)
        success = bool(env.check_success())
    queue: deque = deque()
    frames, actions = [], []
    steps, calls = 0, 0
    inference_seconds = 0.0
    started = time.perf_counter()
    while steps < max_steps and not done and not success:
        if save_video:
            frames.append(np.ascontiguousarray(observation["agentview_image"][::-1, ::-1]))
        if not queue:
            started_plan = time.perf_counter()
            chunk = np.asarray(plan(observation, control_step=steps, replan_index=calls))
            inference_seconds += time.perf_counter() - started_plan
            calls += 1
            if (
                chunk.ndim != 2
                or chunk.shape[1] != 7
                or len(chunk) < replan_steps
                or not np.isfinite(chunk).all()
            ):
                raise ValueError("The policy must return at least replan_steps finite 7D native actions.")
            queue.extend(chunk[:replan_steps])
        action = queue.popleft()
        actions.append(action.tolist())
        observation, _, done, _ = env.step(action.tolist())
        steps += 1
        success = bool(env.check_success())
    if save_video:
        frames.append(np.ascontiguousarray(observation["agentview_image"][::-1, ::-1]))
    return {
        "success": success,
        "steps": steps,
        "policy_calls": calls,
        "timeout": not success and not done and steps >= max_steps,
        "environment_done": bool(done),
        "rollout_seconds": time.perf_counter() - started,
        "inference_seconds": inference_seconds,
        "actions": actions,
    }, frames


def write_results(path: Path, settings: dict, episodes: list[dict]) -> dict:
    tasks = {}
    for episode in episodes:
        key = str(episode["task_id"])
        task = tasks.setdefault(key, {"task_name": episode["task_name"], "episodes": 0, "successes": 0})
        task["episodes"] += 1
        task["successes"] += int(episode["success"])
    for task in tasks.values():
        task["success_rate"] = task["successes"] / task["episodes"]
    successes = sum(int(episode["success"]) for episode in episodes)
    result = {
        "settings": settings,
        "episodes_completed": len(episodes),
        "evaluation_complete": len(episodes) == settings.get("expected_episodes"),
        "successes": successes,
        "success_rate": successes / len(episodes) if episodes else None,
        "tasks": tasks,
        "episodes": episodes,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    return result


def prepare_output_directory(directory: Path, *, protected_paths: tuple[Path, ...] = ()) -> Path:
    directory = directory.expanduser()
    if directory.is_symlink():
        raise ValueError(f"Refusing to overwrite a symlink output directory: {directory}")
    directory = directory.resolve()
    protected = (Path.home(), Path.cwd(), Path(__file__).resolve().parents[2], *protected_paths)
    if (
        len(directory.parents) < 2
        or directory.is_mount()
        or any(
            path.expanduser().absolute().is_relative_to(directory)
            or path.expanduser().resolve().is_relative_to(directory)
            for path in protected
        )
    ):
        raise ValueError(f"Refusing to overwrite a protected directory or evaluation inputs: {directory}")
    if directory.exists():
        if not directory.is_dir():
            raise FileExistsError(f"Evaluation output path exists but is not a directory: {directory}")
        logger.warning("Overwriting evaluation output directory; deleting previous contents: %s", directory)
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="stepNN/model DCP directory or its stepNN parent."
    )
    parser.add_argument(
        "--policy-config", type=Path, help="Defaults to config.json beside the step directories."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("/Data/lerobot_data_ort6d/v30"))
    parser.add_argument("--libero-root", type=Path, default=Path("/home/v-wangxiaofa/lzl/LIBERO"))
    parser.add_argument(
        "--qwen3vl-dir", type=Path, default=Path("/Data/lzl/huggingface/Qwen3-VL-4B-Instruct")
    )
    parser.add_argument("--cosmos3-dir", type=Path, default=Path("/Data/lzl/huggingface/Cosmos3-Edge"))
    parser.add_argument(
        "--stage1-text-dir", type=Path, default=Path("/Data/lzl/huggingface/siglip2-base-patch16-224")
    )
    parser.add_argument("--task-suite", choices=tuple(SUITE_DATASETS), default="libero_goal")
    parser.add_argument("--task-ids", type=int, nargs="+", help="Defaults to all tasks in benchmark order 0.")
    parser.add_argument("--trials-per-task", type=int, default=50)
    parser.add_argument(
        "--stop-on-success",
        action="store_true",
        help="Stop after saving the first successful episode, without running further tasks or trials.",
    )
    parser.add_argument(
        "--replan-steps",
        type=int,
        default=10,
        help="Execute this many consecutive commands before replanning.",
    )
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument(
        "--max-steps", type=int, help="Default: standard suite horizon; shorten only for smoke checks."
    )
    parser.add_argument(
        "--inference-steps", type=int, help="Defaults to the saved flow integration step count."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument(
        "--fp8-training-config", type=Path, help="Saved train_config.json; required for native FP8."
    )
    parser.add_argument(
        "--verify-fp8-kernels", action="store_true", help="Require real scaled_mm calls in the first plan."
    )
    parser.add_argument(
        "--capture-first-plan",
        action="store_true",
        help="Save inputs/RNG fingerprints and first action chunk.",
    )
    parser.add_argument(
        "--render-gpu-device-id",
        type=int,
        default=-1,
        help="Physical EGL device index, independent of CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Initialize RNGs once per run; do not reseed between tasks, episodes, or replans.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory; if it exists, delete its previous contents and start a fresh evaluation.",
    )
    rollout_video = parser.add_mutually_exclusive_group()
    rollout_video.add_argument("--no-save-video", action="store_true")
    rollout_video.add_argument(
        "--save-failed-videos-only",
        action="store_true",
        help="Save actual rollout videos only for failed episodes; does not control predicted videos.",
    )
    parser.add_argument(
        "--no-save-predicted-video",
        action="store_true",
        help="Disable future video decoding/export; independent of --no-save-video (actual rollout).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("trials_per_task", "replan_steps", "max_steps", "inference_steps"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.wait_steps < 0 or args.seed < 0 or args.render_gpu_device_id < -1:
        parser.error("wait-steps/seed must be nonnegative, and render-gpu-device-id must be >= -1.")
    if args.precision == "fp8" and args.fp8_training_config is None:
        parser.error("--precision fp8 requires --fp8-training-config.")
    if args.verify_fp8_kernels and args.precision != "fp8":
        parser.error("--verify-fp8-kernels requires --precision fp8.")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("A CUDA device is required for the full Qwen3-VL/Cosmos checkpoint.")
    torch.cuda.set_device(device)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = True
    output_dir = args.output_dir or Path("outputs/evaluation/libero") / datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    output_dir = prepare_output_directory(
        output_dir,
        protected_paths=tuple(
            value for name, value in vars(args).items() if name != "output_dir" and isinstance(value, Path)
        ),
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "evaluation.log")],
    )
    configure_libero(args.libero_root, output_dir, args.render_gpu_device_id)
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.task_suite](task_order_index=0)
    task_ids = list(range(suite.n_tasks)) if args.task_ids is None else args.task_ids
    if len(set(task_ids)) != len(task_ids) or any(
        task_id < 0 or task_id >= suite.n_tasks for task_id in task_ids
    ):
        parser.error("task-ids must be unique valid indices for the selected suite.")
    initial_states = {task_id: load_libero_initial_states(suite, task_id) for task_id in task_ids}
    if any(len(states) < args.trials_per_task for states in initial_states.values()):
        parser.error("trials-per-task exceeds the number of fixed benchmark initial states.")
    adapter = LiberoAdapter.from_dataset(args.dataset_root / SUITE_DATASETS[args.task_suite])
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    policy = load_policy(args)
    video_writer = (
        None if args.no_save_predicted_video else PredictionVideoWriter(policy.config, device, adapter.fps)
    )
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(
        output_dir=str(output_dir.resolve()),
        max_steps=args.max_steps or MAX_STEPS[args.task_suite],
        inference_steps=policy.config.inference_steps,
        rgb_views=list(policy.config.rgb_views),
        task_names=list(policy.config.task_names),
        task_ids=task_ids,
        expected_episodes=len(task_ids) * args.trials_per_task,
        source_fps=adapter.fps,
        rotation_encoding="Euler XYZ -> matrix[:, :2].reshape(6)",
        state_gripper="second finger",
        action_gripper="0=close, 1=open; native=-sign(2*x-1)",
        sampling_rng="run_seeded_continuous",
        precision_details=policy.libero_precision_details,
    )
    episodes: list[dict] = []
    write_results(output_dir / "results.json", settings, episodes)
    for task_id in task_ids:
        task = suite.get_task(task_id)
        env = OffScreenRenderEnv(
            bddl_file_name=suite.get_task_bddl_file_path(task_id),
            camera_heights=policy.config.video_image_size,
            camera_widths=policy.config.video_image_size,
            control_freq=adapter.fps,
            horizon=settings["max_steps"] + args.wait_steps + 1,
            render_gpu_device_id=args.render_gpu_device_id,
        )
        try:
            for trial in range(args.trials_per_task):
                policy.reset()
                stem = f"task_{task_id:02d}_trial_{trial:03d}"
                prediction_root = output_dir / "predictions" / stem
                prediction_records = []
                plan_timings = []

                def plan(
                    observation,
                    *,
                    control_step,
                    replan_index,
                    instruction=task.language,
                    seed=args.seed,
                    root=prediction_root,
                    records=prediction_records,
                    timings=plan_timings,
                    probe_stem=stem,
                ):
                    started_inference = time.perf_counter()
                    batch = adapter.observation_batch(observation, instruction, policy.config, device)
                    fingerprint = (
                        sampling_fingerprint(batch, device)
                        if args.capture_first_plan and replan_index == 0
                        else None
                    )
                    verify_kernels = args.verify_fp8_kernels and "fp8_kernel_evidence" not in settings
                    profile = (
                        torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])
                        if verify_kernels
                        else nullcontext()
                    )
                    with (
                        profile,
                        policy_inference_context(args.precision),
                        torch.autocast("cuda", dtype=torch.bfloat16),
                    ):
                        if video_writer is None:
                            sampled = policy.sample_canonical_action(batch)
                        else:
                            prediction = policy.generate_validation(batch, seed=None)
                            sampled = prediction["action_normalized"]
                    if verify_kernels:
                        calls = sum(
                            event.count for event in profile.key_averages() if "_scaled_mm" in event.key
                        )
                        if not calls:
                            raise RuntimeError("FP8 requested but no native scaled_mm calls were observed.")
                        settings["fp8_kernel_evidence"] = {"scaled_mm_calls": calls}
                    actions = adapter.native_actions(sampled)
                    if fingerprint is not None:
                        probe_dir = output_dir / "first_plans"
                        probe_dir.mkdir(exist_ok=True)
                        np.savez(
                            probe_dir / f"{probe_stem}.npz",
                            normalized_actions=sampled.float().cpu().numpy(),
                            action_mask=batch["action_mask"].cpu().numpy(),
                            native_actions=actions,
                        )
                        (probe_dir / f"{probe_stem}.json").write_text(
                            json.dumps(fingerprint, indent=2) + "\n"
                        )
                    inference_seconds = time.perf_counter() - started_inference
                    export_seconds = 0.0
                    if video_writer is not None:
                        started_export = time.perf_counter()
                        records.append(
                            video_writer.save(
                                prediction,
                                actions,
                                root / f"replan_{replan_index:04d}_step_{control_step:04d}",
                                control_step=control_step,
                                replan_index=replan_index,
                                seed=seed,
                            )
                        )
                        export_seconds = time.perf_counter() - started_export
                    timings.append((inference_seconds, export_seconds))
                    return actions

                episode, frames = rollout_episode(
                    env,
                    initial_states[task_id][trial],
                    plan,
                    max_steps=settings["max_steps"],
                    wait_steps=args.wait_steps,
                    replan_steps=args.replan_steps,
                    save_video=not args.no_save_video,
                )
                episode["planning_seconds"] = episode["inference_seconds"]
                episode["inference_seconds"] = sum(timing[0] for timing in plan_timings)
                episode["prediction_export_seconds"] = sum(timing[1] for timing in plan_timings)
                episode["prediction_videos"] = prediction_records
                actions_path = output_dir / f"{stem}_actions.npy"
                np.save(actions_path, np.asarray(episode.pop("actions"), dtype=np.float32).reshape(-1, 7))
                episode.update(
                    task_id=task_id,
                    task_name=task.name,
                    instruction=task.language,
                    trial=trial,
                    init_state_index=trial,
                    seed=args.seed,
                    actions_path=str(actions_path.resolve()),
                )
                if frames and (not args.save_failed_videos_only or not episode["success"]):
                    import imageio.v2 as imageio

                    video_path = output_dir / f"{stem}_{'success' if episode['success'] else 'failure'}.mp4"
                    imageio.mimsave(video_path, frames, fps=adapter.fps, codec="libx264")
                    episode["video_path"] = str(video_path.resolve())
                episodes.append(episode)
                result = write_results(output_dir / "results.json", settings, episodes)
                logger.info(
                    "Task %d trial %d: success=%s, steps=%d, plans=%d, inference=%.2fs; total=%d/%d (%.1f%%).",
                    task_id,
                    trial,
                    episode["success"],
                    episode["steps"],
                    episode["policy_calls"],
                    episode["inference_seconds"],
                    result["successes"],
                    result["episodes_completed"],
                    100 * result["success_rate"],
                )
                if args.stop_on_success and episode["success"]:
                    logger.info("Stopping after the first success. Results: %s", output_dir / "results.json")
                    return
        finally:
            env.close()
    logger.info("Results: %s", (output_dir / "results.json").resolve())


if __name__ == "__main__":
    main()
