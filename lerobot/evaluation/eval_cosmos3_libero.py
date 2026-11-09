"""Batched real LIBERO rollouts using native Cosmos3 checkpoints and processors."""

import argparse
import hashlib
import json
import logging
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.evaluation.eval_qwen3vl_mot_libero import (
    DUMMY_ACTION,
    MAX_STEPS,
    configure_libero,
    libero_image,
    load_dcp_model,
    load_libero_initial_states,
    write_results,
)

LOGGER = logging.getLogger(__name__)


def observation_batch(observations, instruction, device):
    batch = {"task": [instruction] * len(observations)}
    for destination, source in (
        ("observation.images.image", "agentview_image"),
        ("observation.images.image2", "robot0_eye_in_hand_image"),
    ):
        batch[destination] = torch.stack(
            [libero_image(observation, source, 256) for observation in observations]
        ).to(device)
    return batch


def rollout_batch(envs, initial_states, plan, *, max_steps, wait_steps=10, replan_steps=16):
    """Keep live episodes synchronized at chunk boundaries; never step a finished environment."""
    observations, records, frames = [], [], []
    started = time.perf_counter()
    for env, initial in zip(envs, initial_states, strict=True):
        env.reset()
        observation = env.set_init_state(initial)
        done = False
        settled = 0
        for _ in range(wait_steps):
            observation, _, done, _ = env.step(DUMMY_ACTION)
            settled += 1
            if done:
                break
        success = bool(env.check_success())
        observations.append(observation)
        records.append(
            {
                "success": success,
                "environment_done": bool(done),
                "steps": 0,
                "wait_steps": settled,
                "policy_calls": 0,
                "inference_seconds_shared": 0.0,
                "actions": [],
                "native_chunks": [],
            }
        )
        frames.append([np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])])
    while True:
        active = [
            i
            for i, record in enumerate(records)
            if not record["success"] and not record["environment_done"] and record["steps"] < max_steps
        ]
        if not active:
            break
        plan_started = time.perf_counter()
        controls, native = plan([observations[i] for i in active])
        duration = time.perf_counter() - plan_started
        controls, native = np.asarray(controls), np.asarray(native)
        if (
            controls.shape != (len(active), replan_steps, 7)
            or native.shape != (len(active), replan_steps, 10)
            or not np.isfinite(controls).all()
            or not np.isfinite(native).all()
        ):
            raise ValueError("Expected finite native (B,T,10) chunks and (B,T,7) LIBERO controls.")
        for row, i in enumerate(active):
            env, record = envs[i], records[i]
            record["policy_calls"] += 1
            record["inference_seconds_shared"] += duration
            record["native_chunks"].append(native[row])
            for action in controls[row, : min(replan_steps, max_steps - record["steps"])]:
                observation, _, done, _ = env.step(action.tolist())
                observations[i] = observation
                record["actions"].append(action)
                record["steps"] += 1
                record["environment_done"] = bool(done)
                record["success"] = bool(env.check_success())
                frames[i].append(np.ascontiguousarray(observation["agentview_image"][::-1, ::-1]))
                if record["success"] or done:
                    break
    for record in records:
        record["timeout"] = (
            not record["success"] and not record["environment_done"] and record["steps"] >= max_steps
        )
        record["batch_rollout_seconds"] = time.perf_counter() - started
    return list(zip(records, frames, strict=True))


def load_policy(model_dir, device, dcp_checkpoint=None):
    from lerobot.common.policies.cosmos3.modeling_cosmos3 import Cosmos3NanoPolicy
    from lerobot.configs.policies import PreTrainedConfig

    config = PreTrainedConfig.from_pretrained(model_dir)
    if config.type != "cosmos3_nano" or config.use_state:
        raise ValueError("This runner requires a native Cosmos3 Nano LIBERO policy with use_state=false.")
    if config.chunk_size != 16 or config.n_action_steps != 16 or config.fps != 20:
        raise ValueError("LIBERO evaluation requires the native 16-action/20Hz contract.")
    config.dtype = torch.bfloat16
    config.gradient_checkpointing = False
    config.compile_model = False
    config.save_predicted_video = False
    if dcp_checkpoint:
        # DCP restores every action tensor after bootstrapping the shared backbone.
        config.load_action_modules = False
    # Avoid the safetensors direct-CUDA path, which can fail during a new worker's startup.
    policy = Cosmos3NanoPolicy.from_pretrained(
        config.pretrained_path if dcp_checkpoint else model_dir,
        config=config,
        map_location="cpu",
    )
    policy.to(device)
    if dcp_checkpoint:
        load_dcp_model(policy, dcp_checkpoint / "model")
        policy.config.pretrained_path = str(model_dir)
    policy.get_processor()
    policy.requires_grad_(False).eval()
    return policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dcp-checkpoint", type=Path)
    parser.add_argument("--libero-root", type=Path, default=Path("/home/v-wangxiaofa/lzl/LIBERO"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-suite", choices=tuple(MAX_STEPS), required=True)
    parser.add_argument("--task-ids", nargs="+", type=int)
    parser.add_argument("--trials-per-task", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    parser.add_argument("--checkpoint-receipt", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.trials_per_task < 1:
        parser.error("batch-size and trials-per-task must be positive.")
    args.model_dir = args.model_dir.resolve()
    if not (args.model_dir / "config.json").is_file():
        parser.error("model-dir must contain the native config and processor artifacts.")
    if args.dcp_checkpoint and not (args.dcp_checkpoint / "model/.metadata").is_file():
        parser.error("dcp-checkpoint must be a complete native FSDP checkpoint directory.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output_dir = args.output_dir.expanduser().absolute()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    configure_libero(args.libero_root, args.output_dir, args.render_gpu_device_id)
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.task_suite](task_order_index=0)
    task_ids = list(range(suite.n_tasks)) if args.task_ids is None else args.task_ids
    if (
        not task_ids
        or len(set(task_ids)) != len(task_ids)
        or any(task < 0 or task >= suite.n_tasks for task in task_ids)
    ):
        parser.error("task-ids must be unique valid indices.")
    initial_states = {task: load_libero_initial_states(suite, task) for task in task_ids}
    if any(len(states) < args.trials_per_task for states in initial_states.values()):
        parser.error("Not enough fixed benchmark initial states.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    policy = load_policy(args.model_dir, device, args.dcp_checkpoint)
    from lerobot.common.policies.cosmos3.processor_cosmos3 import cosmos3_libero_action

    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(
        task_ids=task_ids,
        expected_episodes=len(task_ids) * args.trials_per_task,
        max_steps=MAX_STEPS[args.task_suite],
        wait_steps=10,
        fps=20,
        execute_steps=16,
        inference_steps=policy.config.num_inference_steps,
        sampler=policy.config.sampler,
        sampling_shift=policy.config.sampling_shift,
        guidance_scale=policy.config.guidance_scale,
        rotation_encoding="native column rot6d -> SVD rotation -> axis-angle",
        gripper_encoding="continuous clamp(1 - 2 * open_gripper, -1, 1)",
        normalization="native checkpoint QUANTILES, no clipping",
        sampling_rng="run_seeded_continuous; batch size shrinks as episodes finish",
        precision="BF16, native FP32 timestep/RoPE",
        image_rotation_degrees=180,
        config_sha256=hashlib.sha256((args.model_dir / "config.json").read_bytes()).hexdigest(),
        initial_states_sha256={
            str(task): hashlib.sha256(
                np.asarray(states[: args.trials_per_task], dtype=np.float64).tobytes()
            ).hexdigest()
            for task, states in initial_states.items()
        },
    )
    try:
        settings["implementation_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        settings["implementation_commit"] = None
    if args.checkpoint_receipt:
        settings["checkpoint_verification_receipt"] = json.loads(args.checkpoint_receipt.read_text())
    episodes = []
    write_results(args.output_dir / "results.json", settings, episodes)
    for task_id in task_ids:
        task = suite.get_task(task_id)
        envs = []
        try:
            for _ in range(min(args.batch_size, args.trials_per_task)):
                envs.append(
                    OffScreenRenderEnv(
                        bddl_file_name=suite.get_task_bddl_file_path(task_id),
                        camera_heights=256,
                        camera_widths=256,
                        control_freq=20,
                        horizon=MAX_STEPS[args.task_suite] + 11,
                        render_gpu_device_id=args.render_gpu_device_id,
                    )
                )
            for first in range(0, args.trials_per_task, len(envs)):
                count = min(len(envs), args.trials_per_task - first)
                policy.reset()

                @torch.inference_mode()
                def plan(observations, instruction=task.language, task_index=task_id, first_trial=first):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        native = policy.predict_action_chunk(
                            observation_batch(observations, instruction, device)
                        )
                        controls = cosmos3_libero_action(native)
                    LOGGER.info(
                        "Sampled real-environment chunk: task=%d first_trial=%d batch=%d finite=%s peak=%.2fGiB",
                        task_index,
                        first_trial,
                        len(observations),
                        bool(torch.isfinite(controls).all()),
                        torch.cuda.max_memory_allocated() / 2**30,
                    )
                    return controls.float().cpu().numpy(), native.float().cpu().numpy()

                rollouts = rollout_batch(
                    envs[:count],
                    initial_states[task_id][first : first + count],
                    plan,
                    max_steps=MAX_STEPS[args.task_suite],
                )
                for offset, (record, frames) in enumerate(rollouts):
                    trial = first + offset
                    stem = f"task_{task_id:02d}_trial_{trial:03d}"
                    action_file = args.output_dir / f"{stem}_actions.npz"
                    actions = np.asarray(record.pop("actions"), dtype=np.float32).reshape(-1, 7)
                    chunks = np.asarray(record.pop("native_chunks"), dtype=np.float32).reshape(-1, 16, 10)
                    np.savez_compressed(action_file, controls=actions, native_chunks=chunks)
                    record.update(
                        task_id=task_id,
                        task_name=task.name,
                        instruction=task.language,
                        trial=trial,
                        initial_state_index=trial,
                        actions_file=action_file.name,
                    )
                    if not record["success"]:
                        import imageio.v2 as imageio

                        video = args.output_dir / f"{stem}_failure.mp4"
                        imageio.mimwrite(video, frames, fps=20, codec="libx264", quality=7)
                        record["failure_video"] = video.name
                    episodes.append(record)
                    result = write_results(args.output_dir / "results.json", settings, episodes)
                    LOGGER.info(
                        "Task %d trial %d success=%s steps=%d; total=%d/%d (%.2f%%)",
                        task_id,
                        trial,
                        record["success"],
                        record["steps"],
                        result["successes"],
                        result["episodes_completed"],
                        100 * result["success_rate"],
                    )
        finally:
            for env in envs:
                env.close()
    LOGGER.info("Evaluation complete: %s", args.output_dir / "results.json")


if __name__ == "__main__":
    main()
