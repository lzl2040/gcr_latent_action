"""Single-GPU LIBERO rollouts of a Stage 2 Qwen3-VL/Cosmos DCP checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import deque
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
    logger.info(
        "Loaded %.3fB parameters; tasks=%s, RGB views=%s, denoising steps=%d.",
        sum(p.numel() for p in policy.parameters()) / 1e9,
        config.task_names,
        config.rgb_views,
        config.inference_steps,
    )
    return policy


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
    seed: int,
    max_steps: int,
    wait_steps: int,
    replan_steps: int,
    save_video: bool,
) -> tuple[dict, list]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
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
            chunk = np.asarray(plan(observation))
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
        "--replan-steps",
        type=int,
        default=8,
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
    parser.add_argument(
        "--render-gpu-device-id",
        type=int,
        default=-1,
        help="Physical EGL device index, independent of CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-save-video", action="store_true")
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
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("A CUDA device is required for the full Qwen3-VL/Cosmos checkpoint.")
    torch.cuda.set_device(device)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = True
    output_dir = args.output_dir or Path("outputs/evaluation/libero") / datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
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
    initial_states = {task_id: suite.get_task_init_states(task_id) for task_id in task_ids}
    if any(len(states) < args.trials_per_task for states in initial_states.values()):
        parser.error("trials-per-task exceeds the number of fixed benchmark initial states.")
    adapter = LiberoAdapter.from_dataset(args.dataset_root / SUITE_DATASETS[args.task_suite])
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    policy = load_policy(args)
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

                def plan(observation, instruction=task.language):
                    batch = adapter.observation_batch(observation, instruction, policy.config, device)
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        sampled = policy.sample_canonical_action(batch)
                    return adapter.native_actions(sampled)

                episode, frames = rollout_episode(
                    env,
                    initial_states[task_id][trial],
                    plan,
                    seed=args.seed + task_id * 1000 + trial,
                    max_steps=settings["max_steps"],
                    wait_steps=args.wait_steps,
                    replan_steps=args.replan_steps,
                    save_video=not args.no_save_video,
                )
                stem = f"task_{task_id:02d}_trial_{trial:03d}"
                actions_path = output_dir / f"{stem}_actions.npy"
                np.save(actions_path, np.asarray(episode.pop("actions"), dtype=np.float32).reshape(-1, 7))
                episode.update(
                    task_id=task_id,
                    task_name=task.name,
                    instruction=task.language,
                    trial=trial,
                    init_state_index=trial,
                    seed=args.seed + task_id * 1000 + trial,
                    actions_path=str(actions_path.resolve()),
                )
                if frames:
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
        finally:
            env.close()
    logger.info("Results: %s", (output_dir / "results.json").resolve())


if __name__ == "__main__":
    main()
