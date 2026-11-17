#!/usr/bin/env python
"""Export actual closed-loop outcomes and the futures sampled with their executed action chunks."""

import argparse
import csv
import difflib
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

LOGGER = logging.getLogger(__name__)
REVISION = "eafdb809426b13153aa1e4c42d6601844217dfec"
CAMERAS = {"primary": "agentview_image", "wrist": "robot0_eye_in_hand_image"}
STATE_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
CATEGORY_LABELS = {
    "environment": "Env",
    "object": "Object",
    "position": "Position",
    "semantic": "Semantic",
    "task": "Task",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def three_replans(count):
    if count < 3:
        raise ValueError("Three distinct replans are required; do not duplicate or interpolate frames.")
    return [0, (count - 1) // 2, count - 1]


def prediction_replans(count, maximum):
    if count < 1 or maximum < 3:
        raise ValueError("Require at least one executed plan and a snapshot limit of at least three.")
    selected = min(count, maximum)
    if selected == 1:
        return [0]
    return [(count - 1) * index // (selected - 1) for index in range(selected)]


def forecast_index(offsets, offset):
    if offset <= 0 or offset not in offsets:
        raise ValueError("The forecast must be an actual, strictly future decoded frame.")
    return offsets.index(offset)


def bddl_section(text, name):
    match = re.search(rf"\(:{re.escape(name)}\b", text)
    if match is None:
        return ""
    depth = 1
    for index in range(match.end(), len(text)):
        depth += (text[index] == "(") - (text[index] == ")")
        if depth == 0:
            return text[match.end() : index]
    raise ValueError(f"Unclosed BDDL {name} section.")


def object_assets(text):
    tokens = bddl_section(text, "objects").split()
    objects, names, index = {}, [], 0
    while index < len(tokens):
        if tokens[index] == "-":
            objects[tokens[index + 1]] = names
            names = []
            index += 2
        else:
            names.append(tokens[index])
            index += 1
    if names:
        raise ValueError("Object declarations require explicit asset types.")
    return objects


def describe_perturbation(row, libero_root):
    variant = Path(row["bddl_file"])
    original = libero_root / "libero/libero/bddl_files" / row["base_suite"] / variant.name
    before, after = original.read_text(), variant.read_text()
    if digest(variant) != row["bddl_sha256"]:
        raise ValueError("The selected task BDDL changed.")
    if row.get("original_bddl_sha256", digest(original)) != digest(original):
        raise ValueError("The ordinary LIBERO reference BDDL changed.")
    old_objects, new_objects = object_assets(before), object_assets(after)
    removed = {key: value for key, value in old_objects.items() if new_objects.get(key) != value}
    added = {key: value for key, value in new_objects.items() if old_objects.get(key) != value}
    old_problem = re.search(r"\(problem\s+([^\s)]+)", before).group(1)
    new_problem = re.search(r"\(problem\s+([^\s)]+)", after).group(1)
    placements = [
        dict(re.findall(r"\((?:On|In)\s+([^\s()]+)\s+([^\s()]+)\)", bddl_section(text, "init")))
        for text in (before, after)
    ]
    moved = {
        name: {"original_region": region, "variant_region": placements[1][name]}
        for name, region in placements[0].items()
        if name in placements[1] and region != placements[1][name]
    }
    category = row["category"]
    if category == "environment":
        summary = f"{row['original_environment']} -> {row['target_environment']}"
    elif category == "position" and moved:
        names = list(moved)
        summary = "Reassigned placements: " + ", ".join(names[:2])
        if len(names) > 2:
            summary += f" (+{len(names) - 2} others)"
    elif removed or added:
        summary = f"{', '.join(removed)} -> {', '.join(added)}"
    else:
        summary = f"{category} perturbation; see BDDL differences"
    return {
        "base_suite": row["base_suite"],
        "suite": row["suite"],
        "category": category,
        "label": CATEGORY_LABELS[category],
        "summary": summary,
        "original_bddl_file": str(original.resolve()),
        "original_bddl_sha256": digest(original),
        "variant_bddl_file": str(variant.resolve()),
        "variant_bddl_sha256": digest(variant),
        "original_problem": old_problem,
        "variant_problem": new_problem,
        "environment_changed": old_problem != new_problem,
        "original_environment": row.get("original_environment"),
        "target_environment": row.get("target_environment"),
        "objects_removed": removed,
        "objects_added": added,
        "original_objects": old_objects,
        "variant_objects": new_objects,
        "original_instruction": " ".join(bddl_section(before, "language").split()),
        "variant_instruction": " ".join(bddl_section(after, "language").split()),
        "original_goal": " ".join(bddl_section(before, "goal").split()),
        "variant_goal": " ".join(bddl_section(after, "goal").split()),
        "placement_changes": moved,
        "original_initial_predicates": " ".join(bddl_section(before, "init").split()),
        "variant_initial_predicates": " ".join(bddl_section(after, "init").split()),
        "original_regions": " ".join(bddl_section(before, "regions").split()),
        "variant_regions": " ".join(bddl_section(after, "regions").split()),
        "initial_state_source": row["initial_state_source"],
        "bddl_diff": "".join(
            difflib.unified_diff(
                before.splitlines(True), after.splitlines(True), fromfile=str(original), tofile=str(variant)
            )
        ),
        "asset_name_caveat": "Identifiers are asset names, not inferred visible colors; inspect rendered references.",
    }


def check_pair(ours, native, require_advantage=True):
    for key in ("case_id", "task", "trial", "initial_state_sha256"):
        if ours[key] != native[key]:
            raise ValueError(f"Different paired rollout {key}.")
    if ours.get("perturbation") != native.get("perturbation"):
        raise ValueError("Different paired rollout perturbation metadata.")
    if require_advantage and (not ours["success"] or native["success"]):
        raise ValueError("Require newly measured Ours success and Cosmos3 failure.")
    if not ours["outcome_from_environment"] or not native["outcome_from_environment"]:
        raise ValueError("Image-only or archived outcomes cannot label these new predictions.")


def check_snapshots(record, expected_count=3):
    snapshots = record["snapshots"]
    steps = [snapshot["control_step"] for snapshot in snapshots]
    if (
        not steps
        or (expected_count is not None and len(steps) != expected_count)
        or steps[0] < 0
        or not np.all(np.diff(steps) > 0)
    ):
        raise ValueError("Require chronological, distinct replanning observations and the expected count.")
    for snapshot in snapshots:
        index = forecast_index(snapshot["decoded_source_offsets"], 16)
        if (
            snapshot["display_frame_index"] != index
            or snapshot["forecast_start_s"] != snapshot["control_step"] / 20
            or snapshot["forecast_end_s"] != (snapshot["control_step"] + 16) / 20
            or not snapshot["jointly_sampled_with_executed_action_prefix"]
            or snapshot["executed_action_prefix_length"] <= 0
        ):
            raise ValueError("Forecast timing or correspondence to executed actions changed.")


def load_states(row, trial):
    import torch

    path = Path(row["initial_state_file"])
    if digest(path) != row["initial_states_sha256"]:
        raise ValueError("The task-specific initial states changed.")
    values = (
        np.load(path, allow_pickle=False)
        if path.suffix == ".npy"
        else torch.load(path, map_location="cpu", weights_only=False)
    )
    values = np.asarray(values)
    if values.ndim != 2 or not 0 <= trial < len(values) or not np.isfinite(values[trial]).all():
        raise ValueError("Invalid fixed initial state.")
    return values[trial]


def object_poses(env):
    return {
        name: {
            "world_position_m": env.env.sim.data.body_xpos[body_id].tolist(),
            "world_quaternion_wxyz": env.env.sim.data.body_xquat[body_id].tolist(),
        }
        for name, body_id in env.env.obj_body_id.items()
    }


def configure_simulator(args):
    sys.path.insert(0, str(args.repo_root.resolve()))
    from lerobot.evaluation import eval_qwen3vl_mot_libero as common

    revision = subprocess.check_output(
        ["git", "-C", str(args.libero_root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(args.libero_root), "diff", "--name-only", "HEAD"], text=True
    )
    if revision != REVISION or dirty:
        raise ValueError("Use the clean pinned LIBERO-PRO checkout.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    common.configure_libero(args.libero_root, args.output_dir, -1)
    os.environ.update(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa")
    from libero.libero.envs import OffScreenRenderEnv

    return common, OffScreenRenderEnv


def scenes(args):
    import imageio.v2 as imageio
    import torch

    torch.set_num_threads(4)
    common, env_type = configure_simulator(args)
    for case in json.loads(args.manifest.read_text())["cases"]:
        row, trial = case["task"], case["trial"]
        details = describe_perturbation(row, args.libero_root)
        directory = args.output_dir / case["case_id"]
        directory.mkdir()
        write_json(directory / "perturbation.json", details)
        for name in ("ordinary", "pro"):
            bddl = Path(details[f"{'original' if name == 'ordinary' else 'variant'}_bddl_file"])
            reference = (
                args.libero_root / "libero/libero/init_files" / row["base_suite"] / f"{bddl.stem}.pruned_init"
            )
            state_row = (
                {"initial_state_file": str(reference), "initial_states_sha256": digest(reference)}
                if name == "ordinary"
                else row
            )
            initial = load_states(state_row, trial)
            env = env_type(
                bddl_file_name=str(bddl),
                camera_heights=256,
                camera_widths=256,
                control_freq=20,
                render_gpu_device_id=-1,
            )
            try:
                env.seed(0)
                np.random.seed(0)
                env.reset()
                observation = env.set_init_state(initial)
                for _ in range(10):
                    observation, _, _, _ = env.step(common.DUMMY_ACTION)
                for view, key in CAMERAS.items():
                    imageio.imwrite(
                        directory / f"{name}_{view}.png",
                        common.libero_image(observation, key, 256).permute(1, 2, 0).numpy(),
                    )
                (directory / f"{name}_environment.xml").write_text(env.env.model.get_xml())
                write_json(
                    directory / f"{name}_reset.json",
                    {
                        "trial": trial,
                        "initial_state_file": state_row["initial_state_file"],
                        "initial_state_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
                        "problem": env.problem_name,
                        "object_poses": object_poses(env),
                        "reference_only": name == "ordinary",
                        "reset_caveat": "Ordinary and Pro use their own fixed reset states, not matched object poses.",
                    },
                )
            finally:
                env.close()
        LOGGER.info("SCENE %s %s: %s", case["case_id"], details["label"], details["summary"])


def rollout(args):
    import imageio.v2 as imageio
    import torch

    torch.set_num_threads(4)
    prediction_replans(1, args.prediction_snapshots)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    common, env_type = configure_simulator(args)
    manifest = json.loads(args.manifest.read_text())
    descriptions = {
        case["case_id"]: describe_perturbation(case["task"], args.libero_root) for case in manifest["cases"]
    }
    if args.backend == "ours":
        policy = common.load_policy(
            SimpleNamespace(
                checkpoint=args.checkpoint,
                policy_config=None,
                qwen3vl_dir=args.qwen3vl_dir,
                cosmos3_dir=args.cosmos3_dir,
                stage1_text_dir=Path("."),
                inference_steps=20,
                replan_steps=10,
                precision="bf16",
                fp8_training_config=None,
                device=args.device,
            )
        )
        if policy.config.physical_encoder_type != "mlp":
            raise ValueError("This runner explicitly compares the existing MLP baseline, not Stage I.")
        if (policy.config.chunk_size, policy.config.world_video_frames) != (32, 9):
            raise ValueError("Expected the original 32-action/9-frame baseline.")
        writer = common.PredictionVideoWriter(policy.config, device, 20)
        replan_steps = 10
        model_dir, config_path = common.checkpoint_paths(args.checkpoint, None)
        identity = {
            "checkpoint": str(args.checkpoint),
            "config_sha256": digest(config_path),
            "dcp_index_sha256": digest(model_dir / ".metadata"),
            "physical_encoder_type": "mlp",
            "stage1_grounding_causal_claim": False,
        }
    else:
        from lerobot.common.policies.cosmos3.processor_cosmos3 import cosmos3_libero_action

        from lerobot.evaluation import eval_cosmos3_libero as native

        policy = native.load_policy(args.model_dir, device)
        policy.config.save_predicted_video = True
        replan_steps = 16
        identity = {
            "checkpoint": str(args.model_dir),
            "config_sha256": digest(args.model_dir / "config.json"),
            "inference_chunk_size": 16,
            "trained_chunk_size": 16,
            "canvas_policy": "official_resolution_tier",
        }
    identity.update(
        backend=args.backend,
        seed=args.seed,
        replan_steps=replan_steps,
        source_fps=20,
        runtime_repository=str(args.repo_root.resolve()),
        runtime_commit=subprocess.check_output(
            ["git", "-C", str(args.repo_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        runner_sha256=digest(__file__),
        manifest_sha256=digest(args.manifest),
        future_ground_truth_conditioning=False,
        protocol="new_single_episode_closed_loop_joint_prediction",
        decoded_replan_limit=args.prediction_snapshots,
    )
    write_json(args.output_dir / "run.json", identity)
    results = []
    for case in manifest["cases"]:
        row, trial = case["task"], case["trial"]
        if digest(row["bddl_file"]) != row["bddl_sha256"]:
            raise ValueError("The selected task BDDL changed.")
        initial = load_states(row, trial)
        directory = args.output_dir / case["case_id"]
        directory.mkdir()
        details = descriptions[case["case_id"]]
        write_json(directory / "perturbation.json", details)
        shutil.copy2(details["original_bddl_file"], directory / "ordinary_task.bddl")
        shutil.copy2(details["variant_bddl_file"], directory / "pro_task.bddl")
        adapter = (
            common.LiberoAdapter.from_dataset(args.dataset_root / common.SUITE_DATASETS[row["base_suite"]])
            if args.backend == "ours"
            else None
        )
        env = env_type(
            bddl_file_name=row["bddl_file"],
            camera_heights=256,
            camera_widths=256,
            control_freq=20,
            horizon=MAX_STEPS[row["base_suite"]] + 11,
            render_gpu_device_id=-1,
        )
        started = time.perf_counter()
        plans, executed, actual = [], [], []
        try:
            env.seed(0)
            np.random.seed(0)
            env.reset()
            observation = env.set_init_state(initial)
            for _ in range(10):
                observation, _, done, _ = env.step(common.DUMMY_ACTION)
                if done:
                    raise ValueError("Task ended during the fixed ten-step settling period.")
            if env.check_success():
                raise ValueError("Do not select a task already successful before control.")
            model_xml = env.env.model.get_xml()
            (directory / "environment.xml").write_text(model_xml)
            environment_identity = {
                "problem": env.problem_name,
                "model_xml_sha256": digest(directory / "environment.xml"),
                "initial_object_poses": object_poses(env),
            }
            if env.problem_name.lower() != details["variant_problem"].lower():
                raise ValueError("The runtime environment does not match the Pro BDDL.")
            policy.reset()
            torch.manual_seed(args.seed)
            success = done = False
            while len(executed) < MAX_STEPS[row["base_suite"]] and not success and not done:
                step = len(executed)
                observation_path = directory / f"condition_step_{step:04d}.npz"
                np.savez_compressed(
                    observation_path,
                    **{key: np.asarray(observation[key]) for key in (*CAMERAS.values(), *STATE_KEYS)},
                )
                if step == 0:
                    for view, key in CAMERAS.items():
                        imageio.imwrite(
                            directory / f"initial_{view}.png",
                            common.libero_image(observation, key, 256).permute(1, 2, 0).numpy(),
                        )
                with (
                    torch.inference_mode(),
                    torch.autocast("cuda", dtype=torch.bfloat16),
                ):
                    if args.backend == "ours":
                        batch = adapter.observation_batch(
                            observation, row["instruction"], policy.config, device
                        )
                        prediction = policy.generate_validation(batch, seed=None)
                        actions = adapter.native_actions(prediction["action_normalized"])
                        latent = {
                            key: prediction[key].detach().cpu()
                            for key in ("video_latents", "video_latents_views", "video_view_mask")
                            if key in prediction
                        }
                    else:
                        chunk = policy.predict_action_chunk(
                            native.observation_batch([observation], row["instruction"], device)
                        )
                        actions = cosmos3_libero_action(chunk)[0].float().cpu().numpy()
                        if policy.last_predicted_latents is None:
                            raise ValueError("Missing video latents from the executed joint prediction.")
                        latent = policy.last_predicted_latents.detach().cpu().clone()
                actions = np.asarray(actions)
                if (
                    actions.ndim != 2
                    or actions.shape[1] != 7
                    or len(actions) < replan_steps
                    or not np.isfinite(actions).all()
                ):
                    raise ValueError("Require finite native 7D controls.")
                plans.append(
                    {
                        "control_step": step,
                        "condition_file": str(observation_path.resolve()),
                        "condition_sha256": digest(observation_path),
                        "latent": latent,
                        "actions": actions,
                    }
                )
                for action in actions[:replan_steps]:
                    actual.append(np.ascontiguousarray(observation["agentview_image"][::-1, ::-1]))
                    observation, _, done, _ = env.step(action.tolist())
                    executed.append(action)
                    success = bool(env.check_success())
                    if success or done or len(executed) >= MAX_STEPS[row["base_suite"]]:
                        break
                LOGGER.info("%s %s step=%d success=%s", args.backend, case["case_id"], len(executed), success)
            actual.append(np.ascontiguousarray(observation["agentview_image"][::-1, ::-1]))
            imageio.mimsave(
                directory / "actual_rollout.mp4", actual, fps=20, codec="libx264", macro_block_size=1
            )
            for view, key in CAMERAS.items():
                imageio.imwrite(
                    directory / f"actual_final_{view}.png",
                    common.libero_image(observation, key, 256).permute(1, 2, 0).numpy(),
                )
        finally:
            env.close()
        planned = np.concatenate([plan["actions"][:replan_steps] for plan in plans])
        np.testing.assert_array_equal(planned[: len(executed)], np.asarray(executed))
        np.save(directory / "executed_actions.npy", np.asarray(executed, dtype=np.float32))
        torch.save(plans, directory / "all_joint_plans.pt")
        selected = prediction_replans(len(plans), args.prediction_snapshots)
        snapshots = []
        for index in selected:
            plan = plans[index]
            destination = directory / f"prediction_step_{plan['control_step']:04d}"
            if args.backend == "ours":
                with torch.inference_mode():
                    exported = writer.save(
                        plan["latent"],
                        plan["actions"],
                        destination,
                        control_step=plan["control_step"],
                        replan_index=index,
                        seed=args.seed,
                        save_frames=True,
                    )
                views = {
                    view: np.stack([imageio.imread(p) for p in paths])
                    for view, paths in exported["frames"].items()
                }
                offsets = [round(31 * i / 8) for i in range(9)]
            else:
                destination.mkdir()
                with torch.inference_mode():
                    decoded = policy.decode_predicted_latents(plan["latent"]).numpy()
                if decoded.shape != (17, 160, 320, 3):
                    raise ValueError(f"Expected official native 17-frame/160px output, got {decoded.shape}.")
                views = {"primary": decoded[:, :, :160], "wrist": decoded[:, :, 160:]}
                offsets = list(range(17))
                np.save(destination / "native_canvas_all_frames.npy", decoded)
                np.save(destination / "action_chunk.npy", plan["actions"])
                for view, frames in views.items():
                    frame_dir = destination / f"{view}_frames"
                    frame_dir.mkdir()
                    for i, frame in enumerate(frames):
                        imageio.imwrite(frame_dir / f"frame_{i:03d}.png", frame)
            np.savez_compressed(destination / "frames.npz", **views)
            snapshot = {
                "replan_index": index,
                "control_step": plan["control_step"],
                "condition_file": plan["condition_file"],
                "condition_sha256": plan["condition_sha256"],
                "prediction_directory": str(destination.resolve()),
                "prediction_sha256": digest(destination / "frames.npz"),
                "decoded_source_offsets": offsets,
                "display_forecast_offset": 16,
                "display_frame_index": forecast_index(offsets, 16),
                "forecast_start_s": plan["control_step"] / 20,
                "forecast_end_s": (plan["control_step"] + 16) / 20,
                "jointly_sampled_with_executed_action_prefix": True,
                "executed_action_prefix_length": min(replan_steps, len(executed) - plan["control_step"]),
            }
            write_json(destination / "forecast.json", snapshot)
            snapshots.append(snapshot)
        result = {
            **identity,
            "case_id": case["case_id"],
            "task": row,
            "perturbation": details,
            "environment_identity": environment_identity,
            "trial": trial,
            "initial_state_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
            "success": success,
            "outcome_from_environment": True,
            "environment_done": bool(done),
            "timeout": not success and not done and len(executed) >= MAX_STEPS[row["base_suite"]],
            "steps": len(executed),
            "policy_calls": len(plans),
            "rollout_seconds": time.perf_counter() - started,
            "executed_actions_sha256": digest(directory / "executed_actions.npy"),
            "snapshots": snapshots,
        }
        write_json(directory / "result.json", result)
        results.append(result)
        write_json(args.output_dir / "results.json", {"episodes": results})
        LOGGER.info("COMPLETED %s %s success=%s", args.backend, case["case_id"], success)


def figure(args):
    import matplotlib.pyplot as plt
    from PIL import Image

    selection = json.loads(args.selection.read_text())
    if len(selection["cases"]) != 4:
        raise ValueError("The requested publication figure requires exactly four tasks.")
    tasks, metadata = set(), []
    for case in selection["cases"]:
        if case.get("display_view", "primary") not in CAMERAS:
            raise ValueError("Select an existing primary or wrist camera, not an edited view.")
        ours = json.loads((Path(case["ours_directory"]) / "result.json").read_text())
        native = json.loads((Path(case["cosmos3_directory"]) / "result.json").read_text())
        check_pair(ours, native)
        check_snapshots(ours)
        check_snapshots(native)
        if case["case_id"] != ours["case_id"]:
            raise ValueError("Figure case label and measured rollout do not match.")
        tasks.add((ours["task"]["base_suite"], ours["task"]["task_id"]))
        details = ours.get("perturbation", {})
        metadata.append(
            {
                "case_id": case["case_id"],
                "base_suite": ours["task"]["base_suite"],
                "suite": ours["task"].get("suite", ours["task"]["base_suite"]),
                "perturbation_type": details.get("label", "Unspecified"),
                "category": ours["task"].get("category", "unspecified"),
                "summary": details.get("summary", "No perturbation details in this older export"),
                "task_id": ours["task"]["task_id"],
                "trial": ours["trial"],
                "display_view": case.get("display_view", "primary"),
                "instruction": ours["task"].get("instruction", case["title"]),
                "initial_state_sha256": ours["initial_state_sha256"],
                "ours_success": ours["success"],
                "cosmos3_success": native["success"],
                "details": details,
            }
        )
    if len(tasks) != 4:
        raise ValueError("Use four different base tasks, not four variants of the same task.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "font.size": 9})
    output_views = list(CAMERAS)
    if any("display_view" in case for case in selection["cases"]):
        output_views.append("selected_views")
    for output_view in output_views:
        fig, axes = plt.subplots(4, 7, figsize=(14, 11))
        fig.subplots_adjust(left=0.025, right=0.995, bottom=0.08, top=0.85, wspace=0.035, hspace=0.65)
        fig.text(0.09, 0.965, "Initial observation", ha="center", fontsize=12)
        fig.text(0.365, 0.965, "Cosmos3-Nano: generated futures", ha="center", fontsize=12)
        fig.text(0.79, 0.965, "Ours: generated futures", ha="center", fontsize=12)
        for row_index, case in enumerate(selection["cases"]):
            view = case.get("display_view", "primary") if output_view == "selected_views" else output_view
            ours_dir, native_dir = Path(case["ours_directory"]), Path(case["cosmos3_directory"])
            ours = json.loads((ours_dir / "result.json").read_text())
            native = json.loads((native_dir / "result.json").read_text())
            check_pair(ours, native)
            initial = np.asarray(Image.open(ours_dir / f"initial_{view}.png"))
            np.testing.assert_array_equal(initial, np.asarray(Image.open(native_dir / f"initial_{view}.png")))
            directory = args.output_dir / case["case_id"]
            directory.mkdir(exist_ok=True)
            if output_view == "primary":
                write_json(directory / "metadata.json", metadata[row_index])
                for key, name in [
                    ("original_bddl_file", "ordinary_task.bddl"),
                    ("variant_bddl_file", "pro_task.bddl"),
                ]:
                    if key in metadata[row_index]["details"]:
                        shutil.copy2(metadata[row_index]["details"][key], directory / name)
            if "reference_directory" in case:
                reference = Path(case["reference_directory"])
                np.testing.assert_array_equal(initial, np.asarray(Image.open(reference / f"pro_{view}.png")))
                shutil.copy2(reference / f"ordinary_{view}.png", directory / f"ordinary_reference_{view}.png")
                if output_view == "primary":
                    shutil.copy2(
                        reference / "ordinary_reset.json", directory / "ordinary_reference_reset.json"
                    )
            initial_display = np.asarray(
                Image.fromarray(initial).resize((160, 160), Image.Resampling.LANCZOS)
            )
            images, titles = [initial_display], ["Input"]
            Image.fromarray(initial).save(directory / f"initial_{view}_original.png")
            for name, model_record, source in [("cosmos3", native, native_dir), ("ours", ours, ours_dir)]:
                for index, snapshot in enumerate(model_record["snapshots"]):
                    prediction = Path(snapshot["prediction_directory"])
                    if digest(prediction / "frames.npz") != snapshot["prediction_sha256"]:
                        raise ValueError("A decoded prediction changed.")
                    if digest(snapshot["condition_file"]) != snapshot["condition_sha256"]:
                        raise ValueError("A replanning observation changed.")
                    with np.load(prediction / "frames.npz", allow_pickle=False) as data:
                        raw = data[view][snapshot["display_frame_index"]]
                    if name == "cosmos3" and raw.shape != (160, 160, 3):
                        raise ValueError("Do not upscale or otherwise alter the native Cosmos3 result.")
                    display = (
                        raw
                        if name == "cosmos3"
                        else np.asarray(Image.fromarray(raw).resize((160, 160), Image.Resampling.LANCZOS))
                    )
                    Image.fromarray(raw).save(directory / f"{name}_{view}_{index + 1:02d}_original.png")
                    Image.fromarray(display).save(directory / f"{name}_{view}_{index + 1:02d}_display.png")
                    images.append(display)
                    titles.append(f"{snapshot['forecast_start_s']:.1f} -> {snapshot['forecast_end_s']:.1f} s")
                if output_view == "primary":
                    shutil.copy2(source / "result.json", directory / f"{name}_rollout_result.json")
                    shutil.copy2(source / "actual_rollout.mp4", directory / f"{name}_actual_rollout.mp4")
                    shutil.copy2(source / "actual_final_primary.png", directory / f"{name}_actual_final.png")
            for column, (image, title) in enumerate(zip(images, titles, strict=True)):
                axis = axes[row_index, column]
                axis.imshow(image, interpolation="nearest", vmin=0, vmax=255)
                axis.set_axis_off()
                axis.set_title(title, fontsize=8, pad=3)
            position = axes[row_index, 0].get_position()
            fig.text(0.025, position.y1 + 0.064, case["title"], fontsize=10, weight="bold")
            info = metadata[row_index]
            fig.text(
                0.025,
                position.y1 + 0.038,
                f"{info['suite']} | {info['perturbation_type']} | trial {info['trial']} | {view} | {info['summary']}",
                fontsize=8,
                color="#35485c",
            )
            axes[row_index, 2].text(
                0.5,
                -0.07,
                "Rollout: failure",
                color="#a63726",
                ha="center",
                transform=axes[row_index, 2].transAxes,
            )
            axes[row_index, 5].text(
                0.5,
                -0.07,
                "Rollout: success",
                color="#237441",
                ha="center",
                transform=axes[row_index, 5].transAxes,
            )
        fig.text(
            0.025,
            0.025,
            "Each generated image is a +0.8 s forecast from its own closed-loop observation; labels show start -> forecast time.\n"
            "Outcomes are environment-verified. Hand-selected system comparison, not a physical-grounding ablation. Native 160 px; Ours downsampled.",
            fontsize=8,
        )
        fig.savefig(args.output_dir / f"comparison_{output_view}.pdf", bbox_inches="tight", pad_inches=0.04)
        fig.savefig(
            args.output_dir / f"comparison_{output_view}.png", dpi=220, bbox_inches="tight", pad_inches=0.04
        )
        plt.close(fig)
    write_json(args.output_dir / "selection.json", selection)
    write_json(args.output_dir / "cases_metadata.json", {"cases": metadata})
    columns = [key for key in metadata[0] if key != "details"]
    with (args.output_dir / "cases_metadata.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metadata)
    if all("reference_directory" in case for case in selection["cases"]):
        fig, axes = plt.subplots(4, 2, figsize=(9, 14))
        fig.subplots_adjust(left=0.04, right=0.98, top=0.91, bottom=0.06, hspace=0.65, wspace=0.05)
        for index, case in enumerate(selection["cases"]):
            reference = Path(case["reference_directory"])
            for column, name in enumerate(("ordinary", "pro")):
                axes[index, column].imshow(Image.open(reference / f"{name}_primary.png"))
                axes[index, column].set_axis_off()
                axes[index, column].set_title(
                    "Ordinary LIBERO reference" if column == 0 else "LIBERO-PRO input"
                )
            pos = axes[index, 0].get_position()
            info = metadata[index]
            fig.text(0.04, pos.y1 + 0.055, f"{info['suite']} | {info['perturbation_type']}", weight="bold")
            fig.text(0.04, pos.y1 + 0.032, info["summary"], fontsize=8)
        fig.text(
            0.04,
            0.018,
            "Environment-rendered initial scenes, not model predictions. Independently initialized reference poses;\n"
            "this panel identifies the perturbation, not a matched-state causal comparison.",
            fontsize=8,
        )
        fig.savefig(args.output_dir / "perturbation_reference.pdf", bbox_inches="tight", pad_inches=0.04)
        fig.savefig(
            args.output_dir / "perturbation_reference.png", dpi=180, bbox_inches="tight", pad_inches=0.04
        )
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("rollout")
    worker.add_argument("--backend", choices=("ours", "cosmos3"), required=True)
    worker.add_argument("--repo-root", type=Path, required=True)
    worker.add_argument("--libero-root", type=Path, required=True)
    worker.add_argument("--manifest", type=Path, required=True)
    worker.add_argument("--output-dir", type=Path, required=True)
    worker.add_argument("--checkpoint", type=Path)
    worker.add_argument("--qwen3vl-dir", type=Path)
    worker.add_argument("--cosmos3-dir", type=Path)
    worker.add_argument("--dataset-root", type=Path)
    worker.add_argument("--model-dir", type=Path)
    worker.add_argument("--device", default="cuda:0")
    worker.add_argument("--seed", type=int, default=0)
    worker.add_argument("--prediction-snapshots", type=int, default=3)
    inspect = commands.add_parser("scenes")
    inspect.add_argument("--repo-root", type=Path, required=True)
    inspect.add_argument("--libero-root", type=Path, required=True)
    inspect.add_argument("--manifest", type=Path, required=True)
    inspect.add_argument("--output-dir", type=Path, required=True)
    export = commands.add_parser("figure")
    export.add_argument("--selection", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    {"rollout": rollout, "figure": figure, "scenes": scenes}[args.command](args)


if __name__ == "__main__":
    main()
