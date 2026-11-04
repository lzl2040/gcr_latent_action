"""Run resumable, sequential three-layout evaluations of non-random RoboDojo tasks."""

from __future__ import annotations

import ast
import csv
import fcntl
import hashlib
import json
import logging
import math
import os
import shutil
import signal
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from lerobot.evaluation.eval_qwen3vl_mot_robodojo import validate_result, validate_training_action_mode
from lerobot.evaluation.eval_qwen3vl_mot_robodojo_delta import build_parser as task_parser

logger = logging.getLogger(__name__)


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def task_specs(root: Path, seed: int, requested: list[str] | None = None) -> list[dict]:
    directory = root / "task/RoboDojo"
    available = sorted(
        p.stem
        for p in (directory / "config").glob("*.yml")
        if not p.stem.startswith("_") and "random" not in p.stem.lower()
    )
    names = available if requested is None else requested
    if not names or len(set(names)) != len(names) or set(names) - set(available):
        raise ValueError("Select distinct, existing non-random task names.")
    specs = []
    for name in names:
        source = directory / "tasks" / f"{name}.py"
        limits = [
            node.value.value
            for node in ast.walk(ast.parse(source.read_text()))
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and any(isinstance(t, ast.Attribute) and t.attr == "step_lim" for t in node.targets)
        ]
        if len(limits) != 1 or type(limits[0]) is not int or limits[0] <= 0:
            raise ValueError(f"Cannot determine the unmodified native horizon for {name}: {limits}.")
        files = [source, directory / "config" / f"{name}.yml", directory / "config/_task.yml"]
        files.extend(
            root / f"Assets/Eval_Layout/RoboDojo/arx_x5/{seed}/{name}_{layout}.json" for layout in range(3)
        )
        specs.append(
            {
                "task": name,
                "control_limit": limits[0],
                "native_files": [{"path": str(p.resolve()), "sha256": digest(p)} for p in files],
            }
        )
    return specs


def verify_sources(args, provenance: dict, *, full_hash: bool) -> None:
    if Path(provenance["checkpoint"]).resolve() != args.checkpoint.resolve():
        raise ValueError("Checkpoint provenance belongs to another checkpoint.")
    if Path(provenance["training_contract"]["dataset_root"]).resolve() != args.dataset_root.resolve():
        raise ValueError("Checkpoint provenance belongs to another dataset.")
    for root, records in (
        (args.checkpoint.resolve().parent, provenance["files"]),
        (args.dataset_root.resolve(), provenance["dataset_files"]),
    ):
        for row in records:
            path = root / row["relative_path"]
            before = path.stat()
            if (before.st_size, before.st_mtime_ns) != (row["bytes"], row["mtime_ns"]):
                raise ValueError(f"Source changed: {path}.")
            if (full_hash or path.suffix == ".json" or path.name == ".metadata") and digest(path) != row[
                "sha256"
            ]:
                raise ValueError(f"Source SHA256 changed: {path}.")
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"Source changed while being checked: {path}.")


def verify_native(spec: dict) -> None:
    for row in spec["native_files"]:
        if digest(Path(row["path"])) != row["sha256"]:
            raise ValueError(f"Native task or layout changed: {row['path']}.")


def validate_episode_profile(result: dict, profile: dict, control_limit: int) -> list[int]:
    batches = profile["batches"]
    if [b["layout_ids"] for b in batches] != [[0], [1], [2]]:
        raise ValueError("Expected the original three layouts, without skips or replacements.")
    controls = []
    for batch in batches:
        steps = batch["control_steps"]
        if batch["unstable_episodes"] != 0 or len(steps) != 1 or not 0 < steps[0] <= control_limit:
            raise ValueError(f"Invalid native episode: {batch}.")
        controls.append(steps[0])
    expected_score = sum(row["score"] for row in result["details"].values()) / 3 * 100
    if not math.isclose(result["score"], expected_score, abs_tol=1e-6):
        raise ValueError("Aggregate native score disagrees with episode scores.")
    return controls


def audit_task(root: Path, spec: dict, args) -> dict:
    import imageio_ffmpeg
    import numpy as np
    from PIL import Image, ImageDraw
    from scipy.spatial.transform import Rotation

    from lerobot.evaluation.robodojo_qwen3vl import RoboDojoAdapter, state_to_canonical

    summary = read_json(root / "summary.json")
    if summary["status"] != "completed" or not summary["evaluation_complete"]:
        raise RuntimeError(f"Task did not complete: {summary}.")
    result_path, result = validate_result(root / "simulator", spec["task"], 3)
    profile = read_json(result_path.parent / "_physics_profile.json")
    controls = validate_episode_profile(result, profile, spec["control_limit"])
    rows = [json.loads(line) for line in (root / "policy/chunks/inference.jsonl").read_text().splitlines()]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["reset_index"], row["env_idx"])].append(row)
    if len(groups) != 3:
        raise ValueError("Expected one uninterrupted chunk group per native layout.")
    adapter = RoboDojoAdapter.from_dataset(args.dataset_root, action_mode="delta")
    mean, std = adapter.action_mean.numpy(), adapter.action_std.numpy()
    maxima = {"denormalization": 0.0, "translation": 0.0, "rotation_matrix": 0.0, "gripper": 0.0}
    tracking = {"left": [], "right": []}

    def rotation(values):
        columns = np.asarray(values, dtype=np.float64).reshape(3, 2)
        norm = np.linalg.norm(columns[:, 0])
        if norm <= 1e-8:
            raise ValueError("Degenerate ort6d first column.")
        first = columns[:, 0] / norm
        second = columns[:, 1] - np.dot(first, columns[:, 1]) * first
        norm = np.linalg.norm(second)
        if norm <= 1e-8:
            raise ValueError("Degenerate ort6d second column.")
        second /= norm
        return np.column_stack((first, second, np.cross(first, second)))

    camera_path = root / "camera_audit/camera_observations.jsonl"
    cameras = (
        [json.loads(line) for line in camera_path.read_text().splitlines()] if camera_path.exists() else []
    )
    if args.camera_wrapper and not cameras:
        raise ValueError("Requested passive camera observations are missing.")
    scene_references = {}
    head_reference = None
    for capture in cameras:
        head = np.asarray(capture["cameras"]["cam_head"]["world_matrix"])
        if head_reference is None:
            head_reference = head
        if capture["head_matrix_drift"] != 0 or not np.array_equal(head, head_reference):
            raise ValueError("Head camera drift detected.")
        reference = scene_references.setdefault(capture["layout"], capture["scene_transforms"])
        if reference != capture["scene_transforms"]:
            raise ValueError("Static scene transform changed.")
        for camera in capture["cameras"].values():
            if camera["camera_targets"] != [camera["prim"]]:
                raise ValueError("Render product is bound to the wrong camera.")
    logged = {Path(row["action_path"]).resolve() for row in rows}
    if logged != set((root / "policy/chunks").glob("chunk_*.npz")):
        raise ValueError("Saved and logged policy chunks differ.")
    for layout, group in enumerate(groups.values()):
        if [row["control_step"] for row in group] != list(range(0, controls[layout], 10)):
            raise ValueError("Missing or repeated replans.")
        captures = [c for c in cameras if c["layout"] == layout]
        if cameras and [c["control_step"] for c in captures] != list(range(0, controls[layout] + 1, 50)):
            raise ValueError("Missing passive camera snapshots.")
        previous = None
        for index, row in enumerate(group):
            if row["action_mode"] != "delta" or row["executed_chunk_steps"] != 10:
                raise ValueError("Unexpected action semantics or execution prefix.")
            with np.load(row["action_path"]) as chunk:
                if any(
                    not np.isfinite(chunk[key]).all()
                    for key in chunk.files
                    if np.issubdtype(chunk[key].dtype, np.number)
                ):
                    raise ValueError("Nonfinite policy output.")
                raw, state = chunk["unnormalized_actions"], chunk["current_state"]
                if (
                    raw.shape != (32, 20)
                    or chunk["normalized_actions"].shape != (1, 32, 40)
                    or state.shape != (20,)
                ):
                    raise ValueError("Unexpected action chunk shape.")
                maxima["denormalization"] = max(
                    maxima["denormalization"],
                    float(np.abs(raw - (chunk["normalized_actions"][0, :, :20] * std + mean)).max()),
                )
                if index == 0 and cameras:
                    np.testing.assert_array_equal(
                        state, state_to_canonical({"state": captures[0]["robot_state"]})
                    )
                for arm, offset in (("left", 0), ("right", 10)):
                    position, orientation = (
                        state[offset : offset + 3].astype(float),
                        rotation(state[offset + 3 : offset + 9]),
                    )
                    grip = float(state[offset + 9])
                    for step, delta in enumerate(raw):
                        position += delta[offset : offset + 3]
                        orientation = rotation(delta[offset + 3 : offset + 9]) @ orientation
                        grip = float(np.clip(grip + delta[offset + 9], 0, 1))
                        pose = chunk[f"{arm}_ee_pose"][step]
                        if abs(np.linalg.norm(pose[3:]) - 1) >= 1e-5:
                            raise ValueError("Non-unit native target quaternion.")
                        decoded_rotation = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
                        for key, error in (
                            ("translation", np.abs(position - pose[:3]).max()),
                            ("rotation_matrix", np.abs(orientation - decoded_rotation).max()),
                            ("gripper", abs(grip - float(chunk[f"{arm}_ee_joint_state"][step, 0]))),
                        ):
                            maxima[key] = max(maxima[key], float(error))
                    if previous is not None:
                        tracking[arm].append(
                            float(np.linalg.norm(previous[arm] - state[offset : offset + 3]))
                        )
                previous = {arm: chunk[f"{arm}_ee_pose"][9, :3].copy() for arm in tracking}
                if args.save_predicted_video:
                    native = np.concatenate(
                        [chunk[f"{arm}_{key}"] for arm in tracking for key in ("ee_pose", "ee_joint_state")],
                        axis=-1,
                    )
                    np.testing.assert_array_equal(native, np.load(row["prediction"]["action_chunk_path"]))
    if any(not np.isfinite(value) or value >= 2e-6 for value in maxima.values()):
        raise ValueError(f"Independent delta integration failed: {maxima}.")
    if args.save_predicted_video:
        if len(list((root / "policy/predictions").rglob("*.mp4"))) != 3 * len(rows):
            raise ValueError("Missing predicted videos.")
        if len(list((root / "policy/predictions").rglob("*.png"))) != 33 * len(rows):
            raise ValueError("Missing prediction or conditioning frames.")
    videos = []
    sheet = Image.new("RGB", (1440, 1152), "white")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(sorted(result_path.parent.glob("episode_*.mp4"))):
        reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
        metadata = next(reader)
        count, last = 0, None
        for frame in reader:
            count, last = count + 1, frame
        layout = int(path.name.split("_")[1])
        if count != controls[layout] + 1 or metadata["fps"] != 25 or metadata["size"] != (640, 480):
            raise ValueError(f"Wrong native video frame geometry/count: {path}.")
        image = Image.frombytes("RGB", metadata["size"], last)
        image.thumbnail((480, 360))
        x, y = index % 3 * 480, index // 3 * 384
        sheet.paste(image, (x, y + 24))
        draw.text((x + 2, y + 2), path.stem, fill="black")
        videos.append({"path": str(path), "frames": count, "bytes": path.stat().st_size})
    if len(videos) != 9:
        raise ValueError("Expected three actual camera videos per layout.")
    sheet.save(root / "native_final_frames.jpg")
    audit = {
        "audit_passed": True,
        "task": spec["task"],
        "native_result": result,
        "result_sha256": digest(result_path),
        "control_steps": controls,
        "native_control_limit": spec["control_limit"],
        "chunks": len(rows),
        "integration_max_errors": maxima,
        "target_tracking_m": {
            arm: {"median": float(np.median(v)), "max": max(v)} if v else None for arm, v in tracking.items()
        },
        "camera_snapshots": len(cameras),
        "native_videos": videos,
        "visual_review_pending": True,
    }
    write_json(root / "suite_audit.json", audit)
    return audit


def completed_result(root: Path, spec: dict) -> dict:
    audit = read_json(root / "suite_audit.json")
    summary = read_json(root / "summary.json")
    path, result = validate_result(root / "simulator", spec["task"], 3)
    if (
        not audit["audit_passed"]
        or summary["status"] != "completed"
        or not summary["evaluation_complete"]
        or audit["native_result"] != result
        or audit["result_sha256"] != digest(path)
        or audit["native_control_limit"] != spec["control_limit"]
        or len(audit["native_videos"]) != 9
        or any(Path(v["path"]).stat().st_size != v["bytes"] for v in audit["native_videos"])
    ):
        raise ValueError("Previously completed task artifacts changed.")
    return audit


def update_progress(root: Path, records: list[dict], status: str, **extra) -> None:
    completed = [row for row in records if row["status"] == "completed"]
    successes = sum(sum(v["success"] for v in row["result"]["details"].values()) for row in completed)
    count = 3 * len(completed)
    value = {
        "status": status,
        "suite_complete": status == "completed",
        "planned_tasks": len(records),
        "planned_episodes": 3 * len(records),
        "completed_tasks": len(completed),
        "error_tasks": sum(row["status"] == "error" for row in records),
        "evaluated_episodes": count,
        "successes": successes,
        "success_rate_completed_only": successes / count if count else None,
        "mean_score_completed_only": (
            sum(row["result"]["score"] for row in completed) / len(completed) if completed else None
        ),
        "tasks": records,
        **extra,
    }
    write_json(root / "progress.json", value)
    temporary = root / ".results.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["task", "status", "successes", "episodes", "success_rate", "score", "output", "error"]
        )
        for row in records:
            result = row.get("result") if row["status"] == "completed" else None
            writer.writerow(
                [
                    row["task"],
                    row["status"],
                    sum(v["success"] for v in result["details"].values()) if result else "",
                    3 if result else "",
                    result["success_rate"] if result else "",
                    result["score"] if result else "",
                    row.get("output", ""),
                    row.get("error", ""),
                ]
            )
    temporary.replace(root / "results.csv")


def task_command(args, spec: dict, output: Path) -> list[str]:
    command = [sys.executable, "-u"]
    command.extend(
        [str(args.camera_wrapper.resolve())]
        if args.camera_wrapper
        else ["-m", "lerobot.evaluation.eval_qwen3vl_mot_robodojo_delta"]
    )
    command.append("eval")
    values = {
        "checkpoint": args.checkpoint,
        "dataset-root": args.dataset_root,
        "qwen3vl-dir": args.qwen3vl_dir,
        "cosmos3-dir": args.cosmos3_dir,
        "stage1-text-dir": args.stage1_text_dir,
        "robodojo-root": args.robodojo_root,
        "assets": args.assets,
        "sim-python": args.sim_python,
        "output-dir": output,
        "task": spec["task"],
        "eval-num": 3,
        "num-envs": 1,
        "gpu": args.gpu,
        "sim-gpu": args.sim_gpu,
        "port": args.port,
        "seed": args.seed,
        "host": args.host,
        "device": args.device,
        "inference-steps": 10,
        "replan-steps": 10,
        "timeout": args.timeout,
        "startup-timeout": args.startup_timeout,
    }
    for key in ("policy_config", "training_config"):
        if getattr(args, key):
            values[key.replace("_", "-")] = getattr(args, key)
    for name, value in values.items():
        command.extend((f"--{name}", str(value.resolve() if isinstance(value, Path) else value)))
    if args.save_predicted_video:
        command.append("--save-predicted-video")
    return command


def run_child(command: list[str], log: Path, env: dict, timeout: float) -> None:
    with log.open("w") as stream:
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=timeout)
            if code:
                raise RuntimeError(f"Evaluation exited with {code}; see {log}.")
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def run_suite(args) -> None:
    if args.mode != "eval" or args.action_mode != "delta":
        raise ValueError("This suite requires eval mode and delta actions.")
    if args.eval_num != "3" or args.num_envs != 1 or args.inference_steps != 10 or args.replan_steps != 10:
        raise ValueError("This suite fixes3layouts,1environment,sampling10 and execution10.")
    if min(args.timeout, args.startup_timeout) <= 0 or min(args.seed, args.gpu, args.sim_gpu) < 0:
        raise ValueError("Timeouts must be positive and seed/GPU indices nonnegative.")
    if args.output_dir is None:
        raise ValueError("Specify a persistent --output-dir for the suite.")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    with (root / ".suite.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        settings = {
            key: str(v.resolve()) if isinstance(v, Path) else v
            for key, v in vars(args).items()
            if key not in ("resume", "task")
        }
        specs = task_specs(args.robodojo_root, args.seed, args.tasks)
        provenance = read_json(args.checkpoint_provenance)
        contract = validate_training_action_mode(args)
        evaluation = Path(__file__).resolve().parent
        runtime_paths = [
            evaluation / name
            for name in (
                "eval_qwen3vl_mot_robodojo_suite.py",
                "eval_qwen3vl_mot_robodojo.py",
                "eval_qwen3vl_mot_robodojo_delta.py",
                "eval_qwen3vl_mot_libero.py",
                "robodojo_qwen3vl.py",
                "robodojo_deploy.py",
            )
        ]
        policies = evaluation.parent / "common/policies"
        runtime_paths.extend((policies / "qwen3vl_mot").glob("*.py"))
        runtime_paths.extend(
            policies / "ace" / name for name in ("modeling_robo_contrast.py", "cosmos3_encoders.py")
        )
        if getattr(args, "camera_wrapper", None):
            runtime_paths.append(args.camera_wrapper.resolve())
        runtime = {
            "native_files": [{"path": str(path), "sha256": digest(path)} for path in sorted(runtime_paths)]
        }
        manifest = {
            "settings": settings,
            "task_specs": specs,
            "provenance": provenance,
            "contract": contract,
            "runtime_files": runtime,
        }
        if args.resume:
            if read_json(root / "manifest.json") != manifest:
                raise ValueError(
                    "Resume must use identical settings, source provenance and native task files."
                )
            records = read_json(root / "progress.json")["tasks"]
        else:
            write_json(root / "manifest.json", manifest)
            records = [{"task": s["task"], "status": "pending", "attempts": []} for s in specs]
        update_progress(root, records, "preflight")
        env = dict(os.environ, ROBODOJO_EVAL_MODULE="lerobot.evaluation.eval_qwen3vl_mot_robodojo_delta")
        env.pop("ROBODOJO_STOP_AFTER_FIRST_FAILURE", None)
        try:
            verify_sources(args, provenance, full_hash=True)
            for spec, row in zip(specs, records, strict=True):
                verify_native(runtime)
                verify_sources(args, provenance, full_hash=False)
                verify_native(spec)
                if row["status"] == "completed":
                    completed_result(Path(row["output"]), spec)
                    continue
                if shutil.disk_usage(root).free < 10 * 1024**3:
                    raise OSError("Less than10GiB free; stopping without deleting any files.")
                directory = root / "tasks" / spec["task"]
                directory.mkdir(parents=True, exist_ok=True)
                output = directory / f"attempt_{len(row['attempts']) + 1:03d}"
                row["attempts"].append(str(output))
                row.update(status="running", output=str(output))
                row.pop("error", None)
                row.pop("result", None)
                update_progress(root, records, "running")
                try:
                    run_child(
                        task_command(args, spec, output),
                        output.with_suffix(".log"),
                        env,
                        args.timeout + args.startup_timeout + 300,
                    )
                    audit = audit_task(output, spec, args)
                    verify_native(runtime)
                    verify_native(spec)
                    verify_sources(args, provenance, full_hash=False)
                    row.update(status="completed", result=audit["native_result"])
                except Exception as exc:
                    logger.exception("Task %s failed; keeping its attempt and continuing.", spec["task"])
                    row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                verify_sources(args, provenance, full_hash=False)
                update_progress(root, records, "running")
            verify_sources(args, provenance, full_hash=True)
            complete = all(row["status"] == "completed" for row in records)
            update_progress(
                root, records, "completed" if complete else "incomplete", source_sha256_verified=True
            )
        except BaseException as exc:
            for row in records:
                if row["status"] == "running":
                    row.update(status="interrupted", error=f"{type(exc).__name__}: {exc}")
            update_progress(root, records, "interrupted", error=f"{type(exc).__name__}: {exc}")
            raise


def main() -> None:
    parser = task_parser()
    parser.description = __doc__
    parser.set_defaults(eval_num="3", num_envs=1, inference_steps=10, timeout=14400)
    parser.add_argument("--tasks", nargs="+", help="Defaults to all native task definitions except random.")
    parser.add_argument("--checkpoint-provenance", type=Path, required=True)
    parser.add_argument("--camera-wrapper", type=Path, help="Optional passive native camera instrumentation.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt(
            "Suite interrupted; child evaluation cleanup and resumable state are retained."
        )

    signal.signal(signal.SIGTERM, interrupt)
    run_suite(args)
    if read_json(args.output_dir / "progress.json")["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
