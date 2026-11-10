"""Native Cosmos3 on the pinned five-factor LIBERO-Pro fixed-state diagnostic."""

import argparse
import importlib.metadata
import json
import logging
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

from lerobot.evaluation import eval_cosmos3_libero as native
from lerobot.evaluation import eval_qwen3vl_mot_libero_pro as pro

LOGGER = logging.getLogger(__name__)
PROTOCOL = "libero_pro_classic_fixed_initial_states_cosmos3_v1"


def summarize(settings, episodes, catalog):
    result = pro.summarize(settings, episodes, catalog)
    result["environment_change"] = {}
    for changed in (True, False):
        records = [
            episode
            for episode in episodes
            if episode["category"] == "environment" and episode["environment_changed"] == changed
        ]
        wins = sum(episode["success"] for episode in records)
        result["environment_change"]["changed" if changed else "unchanged"] = {
            "episodes": len(records),
            "successes": wins,
            "success_rate": wins / len(records) if records else None,
        }
    return result


def merge_results(paths, catalog):
    common = None
    episodes, selected, inputs = [], set(), []
    local_fields = {
        "selected_tasks",
        "num_shards",
        "shard_index",
        "render_gpu_device_id",
        "implementation_commit",
        "mode",
    }
    for path in paths:
        payload = json.loads(path.read_text())
        settings = payload["settings"]
        if settings["protocol"] != PROTOCOL:
            raise ValueError("Expected native Cosmos3 Pro results.")
        checked = summarize(settings, payload["episodes"], catalog)
        if not checked["evaluation_complete"]:
            raise ValueError(f"Incomplete Pro shard: {path}")
        shared = {key: value for key, value in settings.items() if key not in local_fields}
        if common is not None and shared != common:
            raise ValueError("Cannot merge different native policies, code, assets or sampling protocols.")
        common = shared
        keys = {tuple(key) for key in settings["selected_tasks"]}
        if selected & keys:
            raise ValueError("Overlapping Pro tasks.")
        selected.update(keys)
        episodes.extend(payload["episodes"])
        inputs.append({"path": str(path.resolve()), "sha256": pro.sha256(path)})
    if common is None:
        raise ValueError("At least one complete Pro shard is required.")
    settings = {
        **common,
        "mode": "merged",
        "inputs": inputs,
        "selected_tasks": [pro.task_key(row) for row in catalog if pro.task_key(row) in selected],
        "num_shards": len(paths),
        "shard_index": "merged",
    }
    return summarize(settings, episodes, catalog)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/Data/lzl/huggingface/Cosmos3-Nano-Policy-LIBERO")
    )
    parser.add_argument("--libero-root", type=Path, default=Path("/home/v-wangxiaofa/lzl/LIBERO_PRO"))
    parser.add_argument("--environment-assets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-receipt", type=Path)
    parser.add_argument("--trials-per-task", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--task-indices", type=int, nargs="+", help="Canonical catalog indices for smoke/subsets."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    parser.add_argument("--render-backend", choices=("egl", "osmesa"), default="egl")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--merge", type=Path, nargs="+")
    args = parser.parse_args(argv)
    if (
        not 1 <= args.trials_per_task <= 50
        or args.batch_size < 1
        or args.seed < 0
        or args.num_shards < 1
        or not 0 <= args.shard_index < args.num_shards
    ):
        parser.error("Invalid trial count, batch size, seed or shard.")
    if not args.dry_run and not args.merge and not torch.cuda.is_available():
        parser.error("Native policy rollouts require CUDA.")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(4)
    pro.configure(args.libero_root, output, args.render_gpu_device_id, args.render_backend)
    catalog = pro.load_catalog(args.libero_root, args.environment_assets, args.trials_per_task)
    if args.merge:
        result = merge_results(args.merge, catalog)
        pro.write_json(output / "results.json", result)
        LOGGER.info(
            "Merged Pro: %d/%d, full coverage=%s",
            result["successes"],
            result["episodes_completed"],
            result["benchmark_task_coverage_complete"],
        )
        return
    indices = list(range(len(catalog)))
    if args.task_indices is not None:
        if len(set(args.task_indices)) != len(args.task_indices) or not set(args.task_indices) <= set(
            indices
        ):
            parser.error("task-indices must be unique valid canonical catalog indices.")
        indices = [index for index in indices if index in args.task_indices]
    selected = [catalog[index] for index in indices[args.shard_index :: args.num_shards]]
    if not selected:
        parser.error("Empty task selection.")
    model_dir = args.model_dir.resolve()
    config = json.loads((model_dir / "config.json").read_text())
    settings = {
        "protocol": PROTOCOL,
        "asset_protocol": pro.PROTOCOL,
        "source_revision": pro.REVISION,
        "evaluator_sha256": pro.sha256(Path(__file__)),
        "native_rollout_sha256": pro.sha256(Path(native.__file__)),
        "policy_code_sha256": {
            path.name: pro.sha256(path)
            for path in sorted((Path(__file__).resolve().parents[1] / "common/policies/cosmos3").glob("*.py"))
        },
        "environment_manifest_sha256": pro.sha256(args.environment_assets / "environment_manifest.json"),
        "policy_identity": {
            "model_dir": str(model_dir),
            "config_sha256": pro.sha256(model_dir / "config.json"),
        },
        "trials_per_task": args.trials_per_task,
        "batch_size": args.batch_size,
        "selected_tasks": [pro.task_key(row) for row in selected],
        "seed": args.seed,
        "environment_seed": 0,
        "wait_steps": 10,
        "replan_steps": 16,
        "max_steps": native.MAX_STEPS,
        "fps": 20,
        "inference_steps": config["num_inference_steps"],
        "sampler": config["sampler"],
        "sampling_shift": config["sampling_shift"],
        "guidance_scale": config["guidance_scale"],
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "render_gpu_device_id": args.render_gpu_device_id,
        "render_backend": args.render_backend,
        "env_resolution": 256,
        "image_rotation_degrees": 180,
        "precision": "BF16 with native FP32 timestep/RoPE",
        "sampling_rng": "torch_run_seeded_continuous_batched; numpy_env_seed0_before_each_reset",
        "normalization": "native checkpoint QUANTILES, no Pro trajectory statistics",
        "rotation": "native column rot6d -> SVD -> axis-angle",
        "gripper": "continuous clamp(1-2*g,-1,1)",
        "reference_settling": "full10 dummy steps; score only after first policy-controlled step",
        "mode": "dry_run" if args.dry_run else "policy",
        "environment_noop_tasks": sum(
            not row["environment_changed"] for row in catalog if row["category"] == "environment"
        ),
        "runtime_versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "diffusers", "numpy", "mujoco", "robosuite")
        },
    }
    if args.checkpoint_receipt:
        settings["policy_identity"]["verification_receipt"] = json.loads(args.checkpoint_receipt.read_text())
    try:
        settings["implementation_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        settings["implementation_commit"] = None
    pro.write_json(output / "task_manifest.json", {"settings": settings, "tasks": selected})
    if args.dry_run:
        LOGGER.info(
            "Validated %d Pro variants x%d fixed states; no model loaded.",
            len(selected),
            args.trials_per_task,
        )
        return
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    policy = native.load_policy(model_dir, device)
    from libero.libero.envs import OffScreenRenderEnv

    from lerobot.common.policies.cosmos3.processor_cosmos3 import cosmos3_libero_action

    episodes = []
    pro.write_json(output / "results.json", summarize(settings, episodes, catalog))
    for row in selected:
        states = pro.load_states(row, args.trials_per_task)
        envs = []
        try:
            for _ in range(min(args.batch_size, args.trials_per_task)):
                envs.append(
                    OffScreenRenderEnv(
                        bddl_file_name=row["bddl_file"],
                        camera_heights=256,
                        camera_widths=256,
                        control_freq=20,
                        horizon=native.MAX_STEPS[row["base_suite"]] + 11,
                        render_gpu_device_id=args.render_gpu_device_id,
                    )
                )
            for first in range(0, args.trials_per_task, len(envs)):
                count = min(len(envs), args.trials_per_task - first)
                policy.reset()

                @torch.inference_mode()
                def plan(observations, record=row, trial_start=first):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        raw = policy.predict_action_chunk(
                            native.observation_batch(
                                observations,
                                record["instruction"],
                                device,
                            )
                        )
                        controls = cosmos3_libero_action(raw)
                    LOGGER.info(
                        "Sampled Pro chunk: %s task%d trial_start=%d batch=%d finite=%s peak=%.2fGiB",
                        record["suite"],
                        record["task_id"],
                        trial_start,
                        len(observations),
                        bool(torch.isfinite(controls).all()),
                        torch.cuda.max_memory_allocated() / 2**30,
                    )
                    return controls.float().cpu().numpy(), raw.float().cpu().numpy()

                rollouts = native.rollout_batch(
                    envs[:count],
                    states[first : first + count],
                    plan,
                    max_steps=native.MAX_STEPS[row["base_suite"]],
                    environment_seed=0,
                    reference_settling=True,
                )
                for offset, (episode, frames) in enumerate(rollouts):
                    trial = first + offset
                    stem = f"{row['suite']}_task_{row['task_id']:02d}_trial_{trial:03d}"
                    actions_path = output / f"{stem}_actions.npy"
                    chunks_path = output / f"{stem}_native_chunks.npy"
                    np.save(actions_path, np.asarray(episode.pop("actions"), dtype=np.float32).reshape(-1, 7))
                    np.save(
                        chunks_path,
                        np.asarray(episode.pop("native_chunks"), dtype=np.float32).reshape(-1, 16, 10),
                    )
                    episode.update(
                        **row,
                        trial=trial,
                        init_state_index=trial,
                        seed=args.seed,
                        actions_path=str(actions_path),
                        actions_sha256=pro.sha256(actions_path),
                        native_chunks_path=str(chunks_path),
                        native_chunks_sha256=pro.sha256(chunks_path),
                        video_path=None,
                    )
                    if not episode["success"]:
                        import imageio.v2 as imageio

                        video = output / f"{stem}_failure.mp4"
                        imageio.mimwrite(video, frames, fps=20, codec="libx264", quality=7)
                        episode["video_path"] = str(video)
                    episodes.append(episode)
                    result = summarize(settings, episodes, catalog)
                    pro.write_json(output / "results.json", result)
                    LOGGER.info(
                        "%s task%d trial%d success=%s; completed=%d/%d",
                        row["suite"],
                        row["task_id"],
                        trial,
                        episode["success"],
                        result["episodes_completed"],
                        result["expected_episodes"],
                    )
        finally:
            for env in envs:
                env.close()
    LOGGER.info("Pro shard complete: %s", output / "results.json")


if __name__ == "__main__":
    main()
