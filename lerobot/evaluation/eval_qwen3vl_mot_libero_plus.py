"""Official LIBERO-Plus tasks with the existing Qwen3-VL/Cosmos policy adapter."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import logging
import os
import random
import subprocess
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from lerobot.evaluation import eval_qwen3vl_mot_libero as original

logger = logging.getLogger(__name__)
SOURCE_REVISION = "4976dc30028e805ff8094b55501d532c48fec182"
SUITE_COUNTS = {"libero_spatial": 2402, "libero_object": 2518, "libero_goal": 2591, "libero_10": 2519}
CATEGORIES = {
    "camera": "Camera Viewpoints",
    "robot": "Robot Initial States",
    "language": "Language Instructions",
    "light": "Light Conditions",
    "background": "Background Textures",
    "noise": "Sensor Noise",
    "layout": "Objects Layout",
}
PROTOCOL = "libero_plus_official_one_trial_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def configure_plus(root: Path, output: Path, render_gpu_device_id: int, backend: str) -> dict:
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "diff", "--name-only", "HEAD"], text=True)
    if revision != SOURCE_REVISION or dirty:
        raise ValueError("Use the clean, revision-pinned checkout from scripts/setup_libero_plus.sh.")
    original.configure_libero(root, output, render_gpu_device_id)
    os.environ["MUJOCO_GL"] = backend
    os.environ["PYOPENGL_PLATFORM"] = backend
    import libero.libero as libero

    if not Path(libero.__file__).resolve().is_relative_to(root.resolve()):
        raise RuntimeError("Another LIBERO checkout is already imported; start a fresh Python process.")
    return {
        "source_revision": revision,
        "classification_sha256": sha256(root / "libero/libero/benchmark/task_classification.json"),
        "evaluator_sha256": sha256(Path(__file__)),
        "module_path": str(Path(libero.__file__).resolve()),
        "runtime_versions": {
            name: importlib.metadata.version(name) for name in ("torch", "numpy", "mujoco", "robosuite")
        },
    }


def load_catalog(root: Path) -> tuple[dict, list[dict]]:
    from libero.libero import benchmark

    classification = json.loads((root / "libero/libero/benchmark/task_classification.json").read_text())
    if set(classification) != set(SUITE_COUNTS):
        raise ValueError("The official classification must contain exactly the four Plus suites.")
    suites, catalog = {}, []
    for name, count in SUITE_COUNTS.items():
        with contextlib.redirect_stdout(io.StringIO()):
            suite = benchmark.get_benchmark_dict()[name](task_order_index=0)
        entries = classification[name]
        by_name = {entry["name"]: entry for entry in entries}
        names = [suite.get_task(index).name for index in range(suite.n_tasks)]
        if (
            suite.n_tasks != count
            or len(entries) != count
            or len(by_name) != count
            or len(set(names)) != count
            or set(by_name) != set(names)
            or len({entry["id"] for entry in entries}) != count
        ):
            raise ValueError(f"{name}: classification and official task names/counts do not match.")
        suites[name] = suite
        for index, task_name in enumerate(names):
            entry = by_name[task_name]
            if entry["category"] not in CATEGORIES.values() or entry["difficulty_level"] not in (
                None,
                1,
                2,
                3,
                4,
                5,
            ):
                raise ValueError(f"Unknown official classification: {entry}")
            task = suite.get_task(index)
            catalog.append(
                {
                    "suite": name,
                    "task_id": index,
                    "classification_id": entry["id"],
                    "task_name": task_name,
                    "category": entry["category"],
                    "difficulty_level": entry["difficulty_level"],
                    "instruction": task.language,
                    "bddl_file": str(suite.get_task_bddl_file_path(index)),
                }
            )
    return suites, catalog


def select_tasks(catalog: list[dict], args) -> list[dict]:
    records = [row for row in catalog if args.task_suite == "all" or row["suite"] == args.task_suite]
    if args.task_ids is not None:
        if args.task_suite == "all":
            raise ValueError("--task-ids requires a single --task-suite.")
        valid = {row["task_id"] for row in records}
        if len(set(args.task_ids)) != len(args.task_ids) or not set(args.task_ids) <= valid:
            raise ValueError("--task-ids must be unique, valid zero-based suite indices.")
        records = [row for row in records if row["task_id"] in args.task_ids]
    if args.categories:
        categories = {CATEGORIES[name] for name in args.categories}
        records = [row for row in records if row["category"] in categories]
    if args.difficulty_levels:
        records = [row for row in records if row["difficulty_level"] in args.difficulty_levels]
    records = records[args.shard_index :: args.num_shards]
    if not records:
        raise ValueError("The requested filters/shard select no tasks.")
    return records


def load_initial_state(suite, task_id: int) -> np.ndarray:
    # Scope the compatibility flag to trusted upstream NumPy pickles, never model loading.
    key = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
    previous = os.environ.get(key)
    os.environ[key] = "1"
    try:
        states = suite.get_task_init_states(task_id)
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous
    state = np.array(states[0], copy=True)
    if state.ndim != 1 or not state.size or not np.isfinite(state).all():
        raise ValueError(f"Invalid official initial state 0 for task {task_id}.")
    return state


def task_key(row: dict) -> tuple[str, int]:
    return row["suite"], row["task_id"]


def group_results(catalog: list[dict], episodes: list[dict], field: str) -> dict:
    totals = Counter(str(row[field]) if row[field] is not None else "unrated" for row in catalog)
    result = {
        key: {"benchmark_tasks": count, "episodes_completed": 0, "successes": 0}
        for key, count in totals.items()
    }
    for row in episodes:
        group = result[str(row[field]) if row[field] is not None else "unrated"]
        group["episodes_completed"] += 1
        group["successes"] += int(row["success"])
    for group in result.values():
        count = group["episodes_completed"]
        group["success_rate"] = group["successes"] / count if count else None
        group["benchmark_complete"] = count == group["benchmark_tasks"]
    return result


def summarize(settings: dict, episodes: list[dict], catalog: list[dict]) -> dict:
    by_key = {task_key(row): row for row in catalog}
    requested = [(suite, index) for suite, indices in settings["selected_tasks"].items() for index in indices]
    requested_keys = set(requested)
    if len(requested_keys) != len(requested) or not requested_keys <= by_key.keys():
        raise ValueError("Invalid or duplicated tasks in the evaluation manifest.")
    seen = set()
    for episode in episodes:
        key = task_key(episode)
        if key in seen or key not in requested_keys:
            raise ValueError(f"Duplicate or unrequested episode: {key}")
        if (
            any(episode.get(field) != value for field, value in by_key[key].items())
            or episode.get("trial") != 0
            or episode.get("init_state_index") != 0
            or type(episode.get("success")) is not bool
        ):
            raise ValueError(f"Episode does not match the official task/one-trial manifest: {key}")
        seen.add(key)
    successes = sum(int(row["success"]) for row in episodes)
    complete = seen == by_key.keys()
    return {
        "settings": settings,
        "episodes_completed": len(episodes),
        "expected_episodes": len(requested),
        "evaluation_complete": seen == requested_keys,
        "benchmark_complete": complete,
        "successes": successes,
        "success_rate": successes / len(episodes) if episodes else None,
        "official_total_success_rate": successes / len(catalog) if complete else None,
        "per_suite": group_results(catalog, episodes, "suite"),
        "per_category": group_results(catalog, episodes, "category"),
        "per_difficulty": group_results(catalog, episodes, "difficulty_level"),
        "episodes": episodes,
    }


def merge_results(paths: list[Path], catalog: list[dict], identity: dict) -> dict:
    common = None
    selected, episodes, suite_inputs = {}, [], {}
    settings = {}
    for path in paths:
        payload = json.loads(path.read_text())
        current = payload["settings"]
        if current.get("mode") != "policy" or current.get("protocol") != PROTOCOL:
            raise ValueError("Only raw Plus policy evaluations using the same official protocol can merge.")
        for key in ("source_revision", "classification_sha256", "evaluator_sha256", "runtime_versions"):
            if current.get(key) != identity[key]:
                raise ValueError(f"Evaluation source/runtime mismatch: {path}: {key}")
        shared = {
            key: current[key]
            for key in (
                "protocol",
                "source_revision",
                "classification_sha256",
                "evaluator_sha256",
                "runtime_versions",
                "seed",
                "environment_seed",
                "trials_per_task",
                "init_state_index",
                "wait_steps",
                "max_steps",
                "env_resolution",
                "control_frequency",
                "replan_steps",
                "inference_steps",
                "precision",
                "rgb_views",
            )
        }
        if common is not None and common != shared:
            raise ValueError(f"Incompatible evaluation protocols: {path}")
        common = shared
        summarize(current, payload["episodes"], catalog)
        for suite, indices in current["selected_tasks"].items():
            fingerprint = {
                "policy": current["policy_identity"],
                "normalization": current["normalization"][suite],
            }
            if suite in suite_inputs and suite_inputs[suite] != fingerprint:
                raise ValueError(f"Different checkpoint/config/normalization within {suite}.")
            suite_inputs[suite] = fingerprint
            if set(selected.setdefault(suite, [])) & set(indices):
                raise ValueError(f"Overlapping task selections in {suite}; do not merge repeated runs.")
            selected[suite].extend(indices)
        episodes.extend(payload["episodes"])
        settings = {**shared, "mode": "merged", "selected_tasks": selected, "suite_inputs": suite_inputs}
    if not paths:
        raise ValueError("At least one raw results.json is required.")
    settings["inputs"] = [{"path": str(path.resolve()), "sha256": sha256(path)} for path in paths]
    return summarize(settings, episodes, catalog)


def rollout_episode(env, initial_state, plan, *, max_steps: int, replan_steps: int, save_video: bool, observe=None):
    env.reset()
    observation = env.set_init_state(initial_state)
    # Match the reference: always settle for 10 steps, then score policy-controlled steps.
    for _ in range(10):
        observation, _, _, _ = env.step(original.DUMMY_ACTION)
    queue, frames, actions = deque(), [], []
    success, done, calls, inference_seconds = False, False, 0, 0.0
    started = time.perf_counter()
    while len(actions) < max_steps and not done and not success:
        if observe is not None:
            observe(observation)
        if save_video:
            frames.append(np.ascontiguousarray(observation["agentview_image"][::-1, ::-1]))
        if not queue:
            start_plan = time.perf_counter()
            chunk = np.asarray(plan(observation, control_step=len(actions), replan_index=calls))
            inference_seconds += time.perf_counter() - start_plan
            calls += 1
            if (
                chunk.ndim != 2
                or chunk.shape[1] != 7
                or len(chunk) < replan_steps
                or not np.isfinite(chunk).all()
            ):
                raise ValueError("The policy must return at least replan_steps finite native 7D actions.")
            queue.extend(chunk[:replan_steps])
        action = queue.popleft().tolist()
        actions.append(action)
        observation, _, done, _ = env.step(action)
        success = bool(env.check_success())
    if save_video:
        frames.append(np.ascontiguousarray(observation["agentview_image"][::-1, ::-1]))
    return {
        "success": success,
        "steps": len(actions),
        "policy_calls": calls,
        "timeout": not success and not done and len(actions) >= max_steps,
        "environment_done": bool(done),
        "rollout_seconds": time.perf_counter() - started,
        "inference_seconds": inference_seconds,
        "actions": actions,
    }, frames


def evaluate_task(record: dict, suite, policy, adapter, args, output: Path, device: torch.device) -> dict:
    from libero.libero.envs import OffScreenRenderEnv

    initial_state = load_initial_state(suite, record["task_id"])
    env = OffScreenRenderEnv(
        bddl_file_name=record["bddl_file"],
        camera_heights=256,
        camera_widths=256,
        control_freq=20,
        render_gpu_device_id=args.render_gpu_device_id,
    )
    try:
        # The reference evaluator seeds each environment with 0 even with fixed init states.
        env.seed(0)
        policy.reset()
        history = (
            original.VisualHistoryBuffer(
                policy.config.visual_history_offsets, policy.config.visual_history_include_first
            ) if getattr(policy.config, "visual_history_enabled", False) else None
        )

        def observe(observation):
            history.update(adapter.observation_batch(
                observation, record["instruction"], policy.config, torch.device("cpu")
            ))

        def plan(observation, *, control_step, replan_index):
            batch = adapter.observation_batch(observation, record["instruction"], policy.config, device)
            if history is not None:
                batch = history.attach(batch)
            with (
                original.policy_inference_context(args.precision),
                torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"),
            ):
                sampled = policy.sample_canonical_action(batch)
            return adapter.native_actions(sampled)

        episode, frames = rollout_episode(
            env,
            initial_state,
            plan,
            max_steps=original.MAX_STEPS[record["suite"]],
            replan_steps=args.replan_steps,
            save_video=not args.no_save_video,
            observe=observe if history is not None else None,
        )
        stem = f"{record['suite']}_task_{record['task_id']:04d}_trial_000"
        actions_path = output / f"{stem}_actions.npy"
        np.save(actions_path, np.asarray(episode.pop("actions"), dtype=np.float32).reshape(-1, 7))
        episode.update(
            **record,
            trial=0,
            init_state_index=0,
            initial_state_sha256=hashlib.sha256(initial_state.tobytes()).hexdigest(),
            seed=args.seed,
            environment_seed=0,
            actions_path=str(actions_path.resolve()),
            actions_sha256=sha256(actions_path),
        )
        if frames and (args.save_all_videos or not episode["success"]):
            import imageio.v2 as imageio

            video_path = output / f"{stem}_{'success' if episode['success'] else 'failure'}.mp4"
            imageio.mimsave(video_path, frames, fps=20, codec="libx264")
            episode["video_path"] = str(video_path.resolve())
        return episode
    finally:
        env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, help="Required for policy rollouts; stepNN or stepNN/model."
    )
    parser.add_argument("--policy-config", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=Path("/Data/lerobot_data_ort6d/v30"))
    parser.add_argument("--libero-root", type=Path, default=Path("/home/v-wangxiaofa/lzl/LIBERO_PLUS"))
    parser.add_argument(
        "--qwen3vl-dir", type=Path, default=Path("/Data/lzl/huggingface/Qwen3-VL-4B-Instruct")
    )
    parser.add_argument("--cosmos3-dir", type=Path, default=Path("/Data/lzl/huggingface/Cosmos3-Edge"))
    parser.add_argument(
        "--stage1-text-dir", type=Path, default=Path("/Data/lzl/huggingface/siglip2-base-patch16-224")
    )
    parser.add_argument("--task-suite", choices=("all", *SUITE_COUNTS), default="all")
    parser.add_argument(
        "--task-ids", type=int, nargs="+", help="Zero-based suite indices, not classification IDs."
    )
    parser.add_argument("--categories", choices=tuple(CATEGORIES), nargs="+")
    parser.add_argument("--difficulty-levels", type=int, choices=range(1, 6), nargs="+")
    parser.add_argument("--trials-per-task", type=int, choices=(1,), default=1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--inference-steps", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--fp8-training-config", type=Path)
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    parser.add_argument("--render-backend", choices=("egl", "osmesa"), default="egl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, help="Must not exist; previous results are never deleted.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="Validate every selected official init; no policy/GPU."
    )
    mode.add_argument(
        "--merge", type=Path, nargs="+", help="Merge disjoint raw Plus results.json files; no policy."
    )
    video = parser.add_mutually_exclusive_group()
    video.add_argument("--no-save-video", action="store_true")
    video.add_argument("--save-all-videos", action="store_true", help="Default: failed rollout videos only.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (
        args.num_shards <= 0
        or not 0 <= args.shard_index < args.num_shards
        or args.replan_steps <= 0
        or args.seed < 0
        or args.render_gpu_device_id < -1
        or (args.inference_steps is not None and args.inference_steps <= 0)
    ):
        parser.error("Invalid shard, seed, device index, or inference/replan step count.")
    if not (args.dry_run or args.merge):
        if args.checkpoint is None:
            parser.error("--checkpoint is required for policy rollouts.")
        if args.precision == "fp8" and args.fp8_training_config is None:
            parser.error("--precision fp8 requires --fp8-training-config.")
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            parser.error("A CUDA device is required for policy rollouts; use --dry-run for CPU preflight.")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    output = args.output_dir or Path("outputs/evaluation/libero_plus") / datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "evaluation.log")],
    )
    identity = configure_plus(args.libero_root, output, args.render_gpu_device_id, args.render_backend)
    suites, catalog = load_catalog(args.libero_root)
    if args.merge:
        write_json(output / "results.json", merge_results(args.merge, catalog, identity))
        return
    selected = select_tasks(catalog, args)
    selections = {}
    for record in selected:
        selections.setdefault(record["suite"], []).append(record["task_id"])
    settings = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        **identity,
        "protocol": PROTOCOL,
        "mode": "dry_run" if args.dry_run else "policy",
        "selected_tasks": selections,
        "benchmark_tasks": len(catalog),
        "trials_per_task": 1,
        "init_state_index": 0,
        "wait_steps": 10,
        "max_steps": original.MAX_STEPS,
        "env_resolution": 256,
        "control_frequency": 20,
        "environment_seed": 0,
        "sampling_rng": "torch_run_seeded_continuous_numpy_env_seed_0_per_task",
    }
    if args.dry_run:
        for record in selected:
            initial_state = load_initial_state(suites[record["suite"]], record["task_id"])
            record["initial_state_sha256"] = hashlib.sha256(initial_state.tobytes()).hexdigest()
        write_json(
            output / "task_manifest.json", {"settings": settings, "tasks": selected, "validated": True}
        )
        logger.info(
            "Validated %d/%d tasks and initial states; no policy or benchmark scores.",
            len(selected),
            len(catalog),
        )
        return
    adapters, normalization = {}, {}
    for suite in selections:
        dataset = args.dataset_root / original.SUITE_DATASETS[suite]
        adapters[suite] = original.LiberoAdapter.from_dataset(dataset)
        if adapters[suite].fps != 20:
            raise ValueError(f"Official LIBERO-Plus requires 20 Hz normalization metadata: {dataset}")
        normalization[suite] = {
            "dataset": str(dataset.resolve()),
            "info_sha256": sha256(dataset / "meta/info.json"),
            "stats_sha256": sha256(dataset / "meta/stats.json"),
        }
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    policy = original.load_policy(args)
    model_dir, config_file = original.checkpoint_paths(args.checkpoint, args.policy_config)
    settings.update(
        inference_steps=policy.config.inference_steps,
        rgb_views=list(policy.config.rgb_views),
        task_names=list(policy.config.task_names),
        normalization=normalization,
        policy_identity={
            "checkpoint": str(model_dir),
            "dcp_metadata_sha256": sha256(model_dir / ".metadata"),
            "policy_config_sha256": sha256(config_file),
            "precision_details": policy.libero_precision_details,
        },
    )
    write_json(output / "task_manifest.json", {"settings": settings, "tasks": selected})
    episodes = []
    write_json(output / "results.json", summarize(settings, episodes, catalog))
    for record in selected:
        episode = evaluate_task(
            record, suites[record["suite"]], policy, adapters[record["suite"]], args, output, device
        )
        episodes.append(episode)
        result = summarize(settings, episodes, catalog)
        write_json(output / "results.json", result)
        logger.info(
            "%s task %d (%s, L%s): success=%s; %d/%d completed, success rate %.2f%%",
            record["suite"],
            record["task_id"],
            record["category"],
            record["difficulty_level"],
            episode["success"],
            len(episodes),
            len(selected),
            100 * result["success_rate"],
        )
    logger.info(
        "Results: %s; full benchmark complete=%s", output / "results.json", result["benchmark_complete"]
    )


if __name__ == "__main__":
    main()
