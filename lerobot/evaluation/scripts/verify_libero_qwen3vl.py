"""Validate EGL and compare one fixed BF16/FP8 plan; not a benchmark success-rate test."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from lerobot.evaluation.eval_qwen3vl_mot_libero import (
    CANONICAL_DIM,
    DUMMY_ACTION,
    SUITE_DATASETS,
    LiberoAdapter,
    checkpoint_paths,
    configure_libero,
    inference_fp8_config,
    load_libero_initial_states,
)

NORMALIZATION_SNAPSHOT = Path(__file__).resolve().parents[1] / "assets/libero_ort6d_normalization.json"


def prepare_dataset_metadata(args, output: Path, *, exist_ok: bool = False) -> tuple[Path, dict]:
    if args.dataset_root is not None:
        root = args.dataset_root.expanduser().resolve()
        LiberoAdapter.from_dataset(root / SUITE_DATASETS[args.task_suite])
        return root, {"source": "explicit_dataset_root", "path": str(root)}
    payload = NORMALIZATION_SNAPSHOT.read_bytes()
    snapshot = json.loads(payload)
    if (
        snapshot["format_version"] != 1
        or snapshot["layout"] != "xyz_interleaved_ort6d_gripper"
        or snapshot["feature_dim"] != 10
    ):
        raise ValueError("Unsupported bundled LIBERO normalization metadata.")
    suite = snapshot["suites"][args.task_suite]
    if suite["dataset"] != SUITE_DATASETS[args.task_suite]:
        raise ValueError("Bundled LIBERO normalization suite/dataset mismatch.")
    root = output / "dataset_metadata"
    meta = root / suite["dataset"] / "meta"
    meta.mkdir(parents=True, exist_ok=exist_ok)
    info = {
        "fps": snapshot["fps"],
        "features": {key: {"shape": [snapshot["feature_dim"]]} for key in ("observation.state", "action")},
    }
    with tempfile.TemporaryDirectory(prefix=".normalization-", dir=meta) as temporary:
        for filename, content in (("info.json", info), ("stats.json", suite["stats"])):
            path = Path(temporary) / filename
            path.write_text(json.dumps(content, allow_nan=False) + "\n")
            path.replace(meta / filename)
    LiberoAdapter.from_dataset(root / suite["dataset"])
    return root, {
        "source": "bundled_libero_ort6d_v30",
        "snapshot_sha256": hashlib.sha256(payload).hexdigest(),
        "original_metadata_sha256": suite["source_sha256"],
    }


def check_environment(args, output: Path) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the node driver/container CUDA compatibility.")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must select a CUDA device.")
    torch.cuda.set_device(device)
    matrix = torch.ones((16, 16), dtype=torch.bfloat16, device=device)
    if not torch.equal(matrix @ matrix, torch.full_like(matrix, 16)):
        raise RuntimeError("CUDA BF16 matrix multiplication produced an incorrect result.")
    packages = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "torchvision",
            "transformers",
            "diffusers",
            "peft",
            "numpy",
            "mujoco",
            "robosuite",
        )
    }
    if "fp8" in args.precisions:
        inference_fp8_config("fp8", args.fp8_training_config, device)
        packages["torchao"] = importlib.metadata.version("torchao")
    configure_libero(args.libero_root, output, args.render_gpu_device_id)
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.task_suite](task_order_index=0)
    if args.task_id < 0 or args.task_id >= suite.n_tasks:
        raise ValueError("task-id is outside the selected suite.")
    states = load_libero_initial_states(suite, args.task_id)
    env = OffScreenRenderEnv(
        bddl_file_name=suite.get_task_bddl_file_path(args.task_id),
        camera_heights=256,
        camera_widths=256,
        render_gpu_device_id=args.render_gpu_device_id,
    )
    try:
        env.seed(args.seed)
        env.reset()
        observation = env.set_init_state(states[0])
        observation, _, _, _ = env.step(DUMMY_ACTION)
        views = {}
        for key in ("agentview_image", "robot0_eye_in_hand_image"):
            image = np.asarray(observation[key])
            if image.shape != (256, 256, 3) or image.dtype != np.uint8 or image.std() == 0:
                raise RuntimeError(f"EGL camera output is invalid: {key}: {image.shape}/{image.dtype}")
            views[key] = {"shape": list(image.shape), "std": float(image.std())}
        if np.array_equal(observation["agentview_image"], observation["robot0_eye_in_hand_image"]):
            raise RuntimeError("Primary and wrist images unexpectedly match.")
        from OpenGL import GL

        renderer = GL.glGetString(GL.GL_RENDERER)
        vendor = GL.glGetString(GL.GL_VENDOR)
        if renderer is None or vendor is None or b"NVIDIA" not in vendor:
            raise RuntimeError(
                f"NVIDIA EGL rendering is required; got renderer={renderer!r}, vendor={vendor!r}."
            )
        return {
            "packages": packages,
            "dependency_scope": "evaluation runtime, not unrelated packages in an existing environment",
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "cuda_bf16_matmul_verified": True,
            "egl_renderer": renderer.decode(),
            "egl_vendor": vendor.decode(),
            "views": views,
        }
    finally:
        env.close()


def read_smoke_run(output: Path, precision: str, task_id: int) -> tuple[dict, dict, dict]:
    stem = f"task_{task_id:02d}_trial_000"
    root = output / precision
    result = json.loads((root / "results.json").read_text())
    if not result["evaluation_complete"] or result["episodes_completed"] != 1 or len(result["episodes"]) != 1:
        raise RuntimeError(f"Incomplete {precision} smoke rollout.")
    episode = result["episodes"][0]
    if episode["policy_calls"] != 1:
        raise RuntimeError(
            "Precision comparison must use exactly one plan, not diverging replan observations."
        )
    if result["settings"]["precision"] != precision:
        raise RuntimeError("Precision label does not match the requested mode.")
    if (
        precision == "fp8"
        and result["settings"].get("fp8_kernel_evidence", {}).get("scaled_mm_calls", 0) <= 0
    ):
        raise RuntimeError("Native FP8 kernel execution was not verified.")
    fingerprint = json.loads((root / "first_plans" / f"{stem}.json").read_text())
    required_hashes = {"batch_sha256", "cpu_rng_sha256", "cuda_rng_sha256"}
    if set(fingerprint) != required_hashes or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        for value in fingerprint.values()
    ):
        raise RuntimeError("Missing or invalid first-plan input/RNG fingerprints.")
    with np.load(root / "first_plans" / f"{stem}.npz", allow_pickle=False) as data:
        probe = {key: data[key] for key in data.files}
    normalized, native, mask = (probe[key] for key in ("normalized_actions", "native_actions", "action_mask"))
    if (
        normalized.ndim != 3
        or normalized.shape[0] != 1
        or normalized.shape[2] != CANONICAL_DIM
        or normalized.shape[1] < 10
        or native.shape != (normalized.shape[1], 7)
        or mask.shape != (1, CANONICAL_DIM)
        or not mask.any()
        or not np.isin(mask, [0, 1]).all()
        or not all(np.isfinite(array).all() for array in (normalized, native, mask))
    ):
        raise RuntimeError("Invalid or nonfinite first-plan action shapes/mask.")
    executed = np.load(episode["actions_path"], allow_pickle=False)
    steps = episode["steps"]
    if (
        not 0 < steps <= 10
        or executed.shape != (steps, 7)
        or not np.array_equal(executed, native[:steps].astype(np.float32))
    ):
        raise RuntimeError("Executed actions do not match the first-plan prefix.")
    return result, fingerprint, probe


def compare_results(output: Path, task_id: int) -> dict:
    probes = [read_smoke_run(output, precision, task_id) for precision in ("bf16", "fp8")]
    if probes[0][1] != probes[1][1]:
        raise RuntimeError(
            "BF16/FP8 initial inputs or RNG states differ; this is not a controlled comparison."
        )
    left, right = probes[0][2], probes[1][2]
    if not np.array_equal(left["action_mask"], right["action_mask"]):
        raise RuntimeError("Precision runs used different action masks.")
    mask = left["action_mask"].astype(bool)[:, None, :]
    normalized = (
        left["normalized_actions"].astype(np.float64) - right["normalized_actions"].astype(np.float64)
    )[np.broadcast_to(mask, left["normalized_actions"].shape)]
    native = left["native_actions"].astype(np.float64) - right["native_actions"].astype(np.float64)
    if not normalized.size or not np.isfinite(normalized).all() or not np.isfinite(native).all():
        raise RuntimeError("Missing or nonfinite action comparison.")
    return {
        "initial_inputs_and_rng_identical": True,
        "normalized_valid_action_rmse": float(np.sqrt(np.mean(normalized**2))),
        "normalized_valid_action_max_abs": float(np.abs(normalized).max()),
        "native_xyz_control_rmse": float(np.sqrt(np.mean(native[:, :3] ** 2))),
        "native_rotation_control_rmse": float(np.sqrt(np.mean(native[:, 3:6] ** 2))),
        "native_gripper_disagreements": int(np.count_nonzero(native[:, 6])),
        "units": "normalized actions and native OSC controls, not meters",
        "note": "No pass/fail accuracy threshold; a short rollout cannot measure benchmark success rate.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory; if it exists, create a fresh timestamped run subdirectory. Never overwrite.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--render-gpu-device-id", type=int, required=True, help="Physical EGL device index.")
    parser.add_argument("--precisions", nargs="+", choices=("bf16", "fp8"), default=["bf16", "fp8"])
    parser.add_argument("--environment-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--policy-config", type=Path)
    parser.add_argument("--fp8-training-config", type=Path)
    parser.add_argument("--dataset-root", type=Path, help="Override the bundled LIBERO v3 normalization.")
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=Path(os.environ.get("MODEL_LOCAL_CACHE", "/tmp/libero_qwen3vl_models")),
        help="Node-local SSD/scratch cache for pretrained model downloads.",
    )
    parser.add_argument("--offline", action="store_true", help="Require existing complete local assets.")
    parser.add_argument("--qwen3vl-dir", type=Path)
    parser.add_argument("--cosmos3-dir", type=Path)
    parser.add_argument("--stage1-text-dir", type=Path)
    parser.add_argument("--task-suite", choices=tuple(SUITE_DATASETS), default="libero_goal")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.seed < 0 or args.render_gpu_device_id < 0 or len(set(args.precisions)) != len(args.precisions):
        parser.error("Use a nonnegative seed/physical EGL device and unique precision modes.")
    if not args.environment_only and args.checkpoint is None:
        parser.error("Model validation requires --checkpoint pointing to cluster-local fine-tuned weights.")
    if "fp8" in args.precisions and args.fp8_training_config is None:
        parser.error("FP8 validation requires --fp8-training-config from the same training run.")
    requested_output = args.output_dir.expanduser().resolve()
    try:
        requested_output.mkdir(parents=True, exist_ok=False)
        output = requested_output
    except FileExistsError:
        if not requested_output.is_dir():
            raise
        prefix = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_")
        output = Path(tempfile.mkdtemp(prefix=prefix, dir=requested_output))
    print(f"Verification output directory: {output}", flush=True)
    report = {
        "status": "running",
        "purpose": "environment_and_precision_smoke_not_benchmark",
        "precisions_requested": args.precisions,
        "model_validation_requested": not args.environment_only,
        "native_fp8_model_validated": False,
        "requested_output_dir": str(requested_output),
        "output_dir": str(output),
    }
    try:
        report["environment"] = check_environment(args, output)
        if not args.environment_only:
            from lerobot.common.utils.stage2_model_cache import resolve_evaluation_model_paths

            checkpoint_paths(args.checkpoint, args.policy_config)
            args.dataset_root, report["normalization"] = prepare_dataset_metadata(args, output)
            logging.basicConfig(level=logging.INFO, format="evaluation setup: %(message)s")
            paths = resolve_evaluation_model_paths(
                args.qwen3vl_dir,
                args.cosmos3_dir,
                args.stage1_text_dir,
                args.model_cache_dir.expanduser().resolve(),
                auto_download=not args.offline,
            )
            args.qwen3vl_dir, args.cosmos3_dir, args.stage1_text_dir = map(Path, paths)
            report["pretrained_assets"] = dict(
                zip(
                    ("qwen3vl", "cosmos3_vae", "stage1_text"),
                    paths,
                    strict=True,
                )
            )
            child_env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            report["runs"] = {}
            for precision in args.precisions:
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "lerobot.evaluation.eval_qwen3vl_mot_libero",
                    "--precision",
                    precision,
                    "--checkpoint",
                    str(args.checkpoint),
                    "--dataset-root",
                    str(args.dataset_root),
                    "--libero-root",
                    str(args.libero_root),
                    "--qwen3vl-dir",
                    str(args.qwen3vl_dir),
                    "--cosmos3-dir",
                    str(args.cosmos3_dir),
                    "--stage1-text-dir",
                    str(args.stage1_text_dir),
                    "--task-suite",
                    args.task_suite,
                    "--task-ids",
                    str(args.task_id),
                    "--trials-per-task",
                    "1",
                    "--max-steps",
                    "10",
                    "--replan-steps",
                    "10",
                    "--seed",
                    str(args.seed),
                    "--device",
                    args.device,
                    "--render-gpu-device-id",
                    str(args.render_gpu_device_id),
                    "--no-save-predicted-video",
                    "--save-failed-videos-only",
                    "--capture-first-plan",
                    "--output-dir",
                    str(output / precision),
                ]
                if args.policy_config is not None:
                    command.extend(["--policy-config", str(args.policy_config)])
                if precision == "fp8":
                    command.extend(
                        [
                            "--fp8-training-config",
                            str(args.fp8_training_config),
                            "--verify-fp8-kernels",
                        ]
                    )
                with (output / f"{precision}.log").open("x") as log:
                    print(f"Running {precision} smoke; log: {log.name}", flush=True)
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, env=child_env)
                result, _, _ = read_smoke_run(output, precision, args.task_id)
                report["runs"][precision] = {
                    "steps": result["episodes"][0]["steps"],
                    "policy_calls": result["episodes"][0]["policy_calls"],
                    "precision_details": result["settings"]["precision_details"],
                }
                if precision == "fp8":
                    report["native_fp8_model_validated"] = True
            if set(args.precisions) == {"bf16", "fp8"}:
                report["comparison"] = compare_results(output, args.task_id)
        report["status"] = "passed"
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        if report["status"] != "passed":
            report.update(status="failed", error=repr(sys.exc_info()[1]))
        (output / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Verification {report['status']}: {output / 'verification.json'}", flush=True)


if __name__ == "__main__":
    main()
