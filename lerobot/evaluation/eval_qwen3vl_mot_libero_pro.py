"""Classic five-factor LIBERO-Pro evaluation with pinned task-specific initial states."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

from lerobot.evaluation import eval_qwen3vl_mot_libero as original
from lerobot.evaluation.eval_qwen3vl_mot_libero_plus import rollout_episode, sha256, write_json

logger = logging.getLogger(__name__)
REVISION = "eafdb809426b13153aa1e4c42d6601844217dfec"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CATEGORIES = {"object": "object", "position": "swap", "semantic": "lan", "task": "task", "environment": "env"}
PROTOCOL = "libero_pro_classic_fixed_initial_states_v1"


def configure(root, output, gpu, backend):
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "diff", "--name-only", "HEAD"], text=True)
    if revision != REVISION or dirty:
        raise ValueError("Use the clean, revision-pinned checkout from setup_libero_pro.sh.")
    original.configure_libero(root, output, gpu)
    os.environ.update(MUJOCO_GL=backend, PYOPENGL_PLATFORM=backend)
    import libero.libero as libero

    if not Path(libero.__file__).resolve().is_relative_to(root.resolve()):
        raise RuntimeError("Another LIBERO checkout is imported; start a fresh process.")


def prepare_environment(root, output, trials, seed):
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from perturbation import BDDLParser, EnvironmentReplacePerturbator

    config_path = root / "libero_ood/ood_environment.yaml"
    entries = []
    for suite_index, name in enumerate(SUITES):
        with contextlib.redirect_stdout(io.StringIO()):
            suite = benchmark.get_benchmark_dict()[name](task_order_index=0)
        for task_id in range(suite.n_tasks):
            task = suite.get_task(task_id)
            source = Path(suite.get_task_bddl_file_path(task_id))
            perturbator = EnvironmentReplacePerturbator(BDDLParser(source.read_text()), str(config_path))
            changed = perturbator.perturb(task_suite_name=name, task_name=task.name, seed=seed)
            directory = output / name
            directory.mkdir(exist_ok=True)
            bddl = directory / source.name
            bddl.write_text(changed)
            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl),
                camera_heights=128,
                camera_widths=128,
                control_freq=20,
                render_gpu_device_id=-1,
            )
            states, seeds = [], []
            try:
                for trial in range(trials):
                    state_seed = seed + suite_index * 100000 + task_id * 1000 + trial
                    env.seed(state_seed)
                    env.reset()
                    state = np.asarray(env.get_sim_state()).copy()
                    if state.ndim != 1 or not np.isfinite(state).all():
                        raise ValueError(f"Invalid generated state: {name}/{task_id}/{trial}")
                    states.append(state)
                    seeds.append(state_seed)
            finally:
                env.close()
            path = directory / f"{source.stem}_initial_states.npy"
            np.save(path, np.stack(states))
            entries.append(
                {
                    "base_suite": name,
                    "task_id": task_id,
                    "task_name": task.name,
                    "bddl_file": str(bddl.resolve()),
                    "bddl_sha256": sha256(bddl),
                    "initial_state_file": str(path.resolve()),
                    "initial_states_sha256": sha256(path),
                    "original_bddl_sha256": sha256(source),
                    "initial_state_source": "locally_generated_environment_fixed_states",
                    "environment_changed": changed != source.read_text(),
                    "generation_seeds": seeds,
                    "original_environment": perturbator._extract_current_env(name, task.name),
                    "target_environment": "living_room_table",
                }
            )
            logger.info("Prepared environment %s task%d: %d fixed states", name, task_id, trials)
    write_json(
        output / "environment_manifest.json",
        {
            "source_revision": REVISION,
            "generator_sha256": sha256(root / "perturbation.py"),
            "environment_config_sha256": sha256(config_path),
            "generation_seed": seed,
            "trials_per_task": trials,
            "tasks": entries,
            "policy_loaded": False,
            "note": "Upstream hardcoded living_room_table is preserved; this is not a downloaded official init set.",
        },
    )


def load_states(row, trials):
    path = Path(row["initial_state_file"])
    if sha256(path) != row["initial_states_sha256"]:
        raise ValueError(f"Initial states changed: {path}")
    states = (
        np.load(path, allow_pickle=False)
        if path.suffix == ".npy"
        else torch.load(path, map_location="cpu", weights_only=False)
    )
    states = np.asarray(states)
    if states.ndim != 2 or len(states) < trials or not np.isfinite(states[:trials]).all():
        raise ValueError(f"Expected at least {trials} finite task-specific states: {path}")
    return states[:trials]


def load_catalog(root, environment_assets, trials):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.bddl_utils import get_problem_info

    environment_manifest = json.loads((environment_assets / "environment_manifest.json").read_text())
    if environment_manifest["source_revision"] != REVISION or len(environment_manifest["tasks"]) != 40:
        raise ValueError("Expected all40 environment tasks from the pinned Pro revision.")
    for key, path in (
        ("generator_sha256", root / "perturbation.py"),
        ("environment_config_sha256", root / "libero_ood/ood_environment.yaml"),
    ):
        if environment_manifest[key] != sha256(path):
            raise ValueError("Environment generation source changed.")
    environment = {(row["base_suite"], row["task_id"]): row for row in environment_manifest["tasks"]}
    if len(environment) != 40:
        raise ValueError("Duplicate environment tasks.")
    catalog = []
    for name in SUITES:
        for category, suffix in CATEGORIES.items():
            with contextlib.redirect_stdout(io.StringIO()):
                suite = benchmark.get_benchmark_dict()[
                    name if category == "environment" else f"{name}_{suffix}"
                ](task_order_index=0)
            if suite.n_tasks != 10:
                raise ValueError(f"Expected10 tasks: {name}/{category}")
            for task_id in range(10):
                task = suite.get_task(task_id)
                if category == "environment":
                    sources = dict(environment[name, task_id])
                    if sources["task_name"] != task.name:
                        raise ValueError("Environment task order/name differs.")
                else:
                    bddl = Path(suite.get_task_bddl_file_path(task_id))
                    initial = (
                        Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
                    )
                    sources = {
                        "bddl_file": str(bddl),
                        "bddl_sha256": sha256(bddl),
                        "initial_state_file": str(initial),
                        "initial_states_sha256": sha256(initial),
                        "initial_state_source": "official_50_state_file_first_requested_indices",
                    }
                bddl = Path(sources["bddl_file"])
                if sha256(bddl) != sources["bddl_sha256"]:
                    raise ValueError(f"BDDL changed: {bddl}")
                instruction = get_problem_info(str(bddl))["language_instruction"]
                if not isinstance(instruction, str) or not instruction.strip():
                    raise ValueError("Missing actual BDDL instruction.")
                row = {
                    **sources,
                    "base_suite": name,
                    "suite": f"{name}_{suffix}",
                    "category": category,
                    "task_id": task_id,
                    "task_name": task.name,
                    "instruction": instruction,
                }
                load_states(row, trials)
                catalog.append(row)
    return catalog


def task_key(row):
    return row["suite"], row["task_id"]


def summarize(settings, episodes, catalog):
    requested = {tuple(key) for key in settings["selected_tasks"]}
    by_key = {task_key(row): row for row in catalog}
    if len(requested) != len(settings["selected_tasks"]) or not requested <= by_key.keys():
        raise ValueError("Invalid selected task manifest.")
    trials = settings["trials_per_task"]
    expected = {(suite, task, trial) for suite, task in requested for trial in range(trials)}
    seen = set()
    for episode in episodes:
        key = (*task_key(episode), episode["trial"])
        if key in seen or key not in expected:
            raise ValueError("Duplicate/unrequested episode.")
        if episode["init_state_index"] != episode["trial"] or type(episode["success"]) is not bool:
            raise ValueError("Episode does not match its fixed initial-state index.")
        if any(episode.get(field) != value for field, value in by_key[task_key(episode)].items()):
            raise ValueError("Episode task/instruction/initial-state sources changed.")
        seen.add(key)
    rates = {}
    for field in ("category", "base_suite"):
        rates[field] = {}
        for value in sorted({row[field] for row in catalog}):
            records = [episode for episode in episodes if episode[field] == value]
            wins = sum(episode["success"] for episode in records)
            rates[field][value] = {
                "episodes": len(records),
                "successes": wins,
                "success_rate": wins / len(records) if records else None,
            }
    wins = sum(episode["success"] for episode in episodes)
    complete = seen == expected
    full = complete and requested == by_key.keys()
    return {
        "settings": settings,
        "episodes": episodes,
        "episodes_completed": len(episodes),
        "expected_episodes": len(expected),
        "evaluation_complete": complete,
        "benchmark_task_coverage_complete": full,
        "official_50_trials_per_task": full and trials == 50,
        "successes": wins,
        "success_rate": wins / len(episodes) if episodes else None,
        "per_category": rates["category"],
        "per_suite": rates["base_suite"],
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--policy-config", type=Path)
    parser.add_argument("--libero-root", type=Path, default=Path("/home/v-wangxiaofa/lzl/LIBERO_PRO"))
    parser.add_argument("--dataset-root", type=Path, default=Path("/Data/lerobot_data_ort6d/v30"))
    parser.add_argument(
        "--qwen3vl-dir", type=Path, default=Path("/Data/lzl/huggingface/Qwen3-VL-4B-Instruct")
    )
    parser.add_argument("--cosmos3-dir", type=Path, default=Path("/Data/lzl/huggingface/Cosmos3-Edge"))
    parser.add_argument(
        "--stage1-text-dir", type=Path, default=Path("/Data/lzl/huggingface/siglip2-base-patch16-224")
    )
    parser.add_argument("--trials-per-task", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--inference-steps", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16",), default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not exist.")
    parser.add_argument("--no-save-video", action="store_true")
    parser.set_defaults(fp8_training_config=None)
    parser.add_argument("--environment-assets", type=Path)
    parser.add_argument("--prepare-environment", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--render-backend", choices=("egl", "osmesa"), default="egl")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if (
        not 1 <= args.trials_per_task <= 50
        or args.num_shards <= 0
        or not 0 <= args.shard_index < args.num_shards
    ):
        parser.error("Invalid trial count/shard.")
    if args.replan_steps <= 0 or (args.inference_steps is not None and args.inference_steps <= 0):
        parser.error("Inference/replan steps must be positive.")
    if args.precision != "bf16":
        parser.error("This Pro evaluator currently supports BF16 only.")
    if args.seed < 0 or args.render_gpu_device_id < -1:
        parser.error("Invalid seed/render GPU.")
    preparation = args.prepare_environment or args.dry_run
    if not preparation and (args.checkpoint is None or not torch.cuda.is_available()):
        parser.error("Policy evaluation needs checkpoint and CUDA; use preparation modes for CPU.")
    if not args.prepare_environment and args.environment_assets is None:
        parser.error("--environment-assets must contain generated/pinned environment states.")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    configure(args.libero_root, output, args.render_gpu_device_id, args.render_backend)
    if args.prepare_environment:
        prepare_environment(args.libero_root, output, args.trials_per_task, args.seed)
        return
    catalog = load_catalog(args.libero_root, args.environment_assets, args.trials_per_task)
    selected = catalog[args.shard_index :: args.num_shards]
    if not selected:
        parser.error("Empty shard.")
    settings = {
        "protocol": PROTOCOL,
        "source_revision": REVISION,
        "evaluator_sha256": sha256(Path(__file__)),
        "environment_manifest_sha256": sha256(args.environment_assets / "environment_manifest.json"),
        "trials_per_task": args.trials_per_task,
        "selected_tasks": [task_key(row) for row in selected],
        "seed": args.seed,
        "environment_seed": 0,
        "wait_steps": 10,
        "replan_steps": args.replan_steps,
        "max_steps": original.MAX_STEPS,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "render_gpu_device_id": args.render_gpu_device_id,
        "precision": args.precision,
        "sampling_rng": "torch_run_seeded_continuous_numpy_env_seed0_per_trial",
        "mode": "dry_run" if args.dry_run else "policy",
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "environment_noop_tasks": sum(
            not row["environment_changed"] for row in catalog if row["category"] == "environment"
        ),
    }
    write_json(output / "task_manifest.json", {"settings": settings, "tasks": selected})
    if args.dry_run:
        logger.info(
            "Validated%d task variants x%d fixed initial states; no policy.",
            len(selected),
            args.trials_per_task,
        )
        return
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    policy = original.load_policy(args)
    adapters = {
        name: original.LiberoAdapter.from_dataset(args.dataset_root / original.SUITE_DATASETS[name])
        for name in SUITES
    }
    if any(adapter.fps != 20 for adapter in adapters.values()):
        raise ValueError("Pro requires20Hz normalization metadata.")
    model, config = original.checkpoint_paths(args.checkpoint, args.policy_config)
    settings.update(
        policy_config=str(config),
        policy_config_sha256=sha256(config),
        dcp_metadata_sha256=sha256(model / ".metadata"),
        inference_steps=policy.config.inference_steps,
        rgb_views=list(policy.config.rgb_views),
        task_names=list(policy.config.task_names),
    )
    from libero.libero.envs import OffScreenRenderEnv

    episodes = []
    write_json(output / "results.json", summarize(settings, episodes, catalog))
    for row in selected:
        states = load_states(row, args.trials_per_task)
        env = OffScreenRenderEnv(
            bddl_file_name=row["bddl_file"],
            camera_heights=256,
            camera_widths=256,
            control_freq=20,
            render_gpu_device_id=args.render_gpu_device_id,
        )
        try:
            for trial, initial_state in enumerate(states):
                env.seed(0)
                policy.reset()

                def plan(observation, record=row, **_):
                    batch = adapters[record["base_suite"]].observation_batch(
                        observation, record["instruction"], policy.config, device
                    )
                    with (
                        original.policy_inference_context(args.precision),
                        torch.autocast("cuda", dtype=torch.bfloat16),
                    ):
                        sampled = policy.sample_canonical_action(batch)
                    return adapters[record["base_suite"]].native_actions(sampled)

                episode, frames = rollout_episode(
                    env,
                    initial_state,
                    plan,
                    max_steps=original.MAX_STEPS[row["base_suite"]],
                    replan_steps=args.replan_steps,
                    save_video=not args.no_save_video,
                )
                stem = f"{row['suite']}_task_{row['task_id']:02d}_trial_{trial:03d}"
                action_path = output / f"{stem}_actions.npy"
                np.save(action_path, np.asarray(episode.pop("actions"), dtype=np.float32).reshape(-1, 7))
                episode.update(
                    **row,
                    trial=trial,
                    init_state_index=trial,
                    seed=args.seed,
                    actions_path=str(action_path),
                    actions_sha256=sha256(action_path),
                    video_path=None,
                )
                if frames and not episode["success"]:
                    import imageio.v2 as imageio

                    video = output / f"{stem}.mp4"
                    imageio.mimsave(video, frames, fps=20, codec="libx264")
                    episode["video_path"] = str(video)
                episodes.append(episode)
                write_json(output / "results.json", summarize(settings, episodes, catalog))
                logger.info(
                    "%s task%d trial%d success=%s completed=%d",
                    row["suite"],
                    row["task_id"],
                    trial,
                    episode["success"],
                    len(episodes),
                )
        finally:
            env.close()


if __name__ == "__main__":
    main()
