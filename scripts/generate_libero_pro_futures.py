#!/usr/bin/env python
"""Generate time-aligned LIBERO-Pro futures in separate MoT and native Cosmos3 environments."""

import argparse
import hashlib
import json
import logging
import math
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

LOGGER = logging.getLogger(__name__)
CAMERAS = {"primary": "agentview_image", "wrist": "robot0_eye_in_hand_image"}
OBSERVATION_KEYS = (*CAMERAS.values(), "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def future_offsets(chunk_size=32, video_frames=9):
    if chunk_size < video_frames or video_frames < 2:
        raise ValueError("The sampling grid must contain distinct current/future source frames.")
    return [round((chunk_size - 1) * index / (video_frames - 1)) for index in range(1, video_frames)]


def load_input(directory):
    metadata = json.loads((directory / "metadata.json").read_text())
    if digest(directory / "observation.npz") != metadata["observation_sha256"]:
        raise ValueError("The saved input observation changed.")
    with np.load(directory / "observation.npz", allow_pickle=False) as archive:
        observation = {key: archive[key] for key in OBSERVATION_KEYS}
    return observation, metadata


def prepare(args):
    import torch

    from lerobot.evaluation import eval_qwen3vl_mot_libero_pro as pro
    from lerobot.evaluation.eval_qwen3vl_mot_libero import DUMMY_ACTION, libero_image

    catalog = json.loads(args.task_manifest.read_text())
    record = catalog["tasks"][args.task_index]
    if digest(record["bddl_file"]) != record["bddl_sha256"]:
        raise ValueError("The LIBERO-Pro task changed.")
    pro.configure(args.libero_root, args.output_dir, args.render_gpu_device_id, args.render_backend)
    from libero.libero.envs import OffScreenRenderEnv

    initial = pro.load_states(record, args.trial + 1)[args.trial]
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = OffScreenRenderEnv(
        bddl_file_name=record["bddl_file"],
        camera_heights=256,
        camera_widths=256,
        control_freq=args.source_fps,
        render_gpu_device_id=args.render_gpu_device_id,
    )
    try:
        env.seed(args.seed)
        env.reset()
        observation = env.set_init_state(initial)
        for _ in range(args.wait_steps):
            observation, _, done, _ = env.step(DUMMY_ACTION)
            if done:
                raise ValueError("The episode ended during settling.")
        values = {key: np.asarray(observation[key]) for key in OBSERVATION_KEYS}
        np.savez_compressed(args.output_dir / "observation.npz", **values)
        np.save(args.output_dir / "initial_state.npy", initial)
        import imageio.v2 as imageio

        for view, key in CAMERAS.items():
            image = libero_image(values, key, 256).permute(1, 2, 0).numpy()
            imageio.imwrite(args.output_dir / f"input_{view}.png", image)
        write_json(
            args.output_dir / "metadata.json",
            {
                "task": record,
                "trial": args.trial,
                "seed": args.seed,
                "wait_steps": args.wait_steps,
                "source_fps": args.source_fps,
                "chunk_size": 32,
                "video_frames": 9,
                "future_source_offsets": future_offsets(),
                "observation_sha256": digest(args.output_dir / "observation.npz"),
                "task_manifest_sha256": digest(args.task_manifest),
                "libero_revision": pro.REVISION,
                "ground_truth_future_supplied": False,
            },
        )
    finally:
        env.close()


def export_frames(directory, views, offsets, source_fps):
    import imageio.v2 as imageio

    fps = len(offsets) * source_fps / offsets[-1]
    for view, frames in views.items():
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("Decoded frames must be THWC uint8 RGB.")
        future = frames[1:]
        if len(future) != len(offsets):
            raise ValueError("The generated future length does not match the alignment grid.")
        frame_dir = directory / f"{view}_frames"
        frame_dir.mkdir()
        for index, frame in enumerate(frames):
            imageio.imwrite(frame_dir / f"frame_{index:03d}.png", frame)
        imageio.mimsave(directory / f"{view}.mp4", frames, fps=fps, codec="libx264", macro_block_size=1)
        imageio.mimsave(
            directory / f"{view}_future8.mp4", future, fps=fps, codec="libx264", macro_block_size=1
        )
        imageio.imwrite(directory / f"{view}_future8.png", np.concatenate(future, axis=1))
    np.savez_compressed(directory / "frames.npz", **views)


def generation_metadata(args, source, backend, offsets, elapsed, **details):
    import torch
    import transformers

    write_json(
        args.output_dir / "metadata.json",
        {
            "backend": backend,
            "input_sha256": source["observation_sha256"],
            "instruction": source["task"]["instruction"],
            "seed": args.seed,
            "future_source_offsets": offsets,
            "future_times_s": [index / source["source_fps"] for index in offsets],
            "source_fps": source["source_fps"],
            "generation_seconds": elapsed,
            "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "runtime_repository": str(args.repo_root.resolve()),
            "generation_script_sha256": digest(__file__),
            "runtime_commit": subprocess.check_output(
                ["git", "-C", str(args.repo_root), "rev-parse", "HEAD"], text=True
            ).strip(),
            "preview_timing": "constant-FPS preview; exact source times are recorded above",
            "qualitative_generation_only": True,
            **details,
        },
    )


def baseline(args):
    import torch

    from lerobot.evaluation.eval_qwen3vl_mot_libero import (
        LiberoAdapter,
        PredictionVideoWriter,
        checkpoint_paths,
        load_policy,
        policy_inference_context,
    )

    observation, source = load_input(args.input_dir)
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    adapter = LiberoAdapter.from_dataset(args.dataset_root)
    if adapter.fps != source["source_fps"]:
        raise ValueError("Baseline normalization dataset and source FPS differ.")
    policy = load_policy(
        SimpleNamespace(
            checkpoint=args.checkpoint,
            policy_config=None,
            qwen3vl_dir=args.qwen3vl_dir,
            cosmos3_dir=args.cosmos3_dir,
            stage1_text_dir=Path("."),
            inference_steps=args.steps,
            replan_steps=1,
            precision="bf16",
            fp8_training_config=None,
            device=args.device,
        )
    )
    if policy.config.physical_encoder_type != "mlp":
        raise ValueError("This comparison expects the no-Stage1 MLP baseline.")
    if (policy.config.chunk_size, policy.config.world_video_frames) != (32, 9):
        raise ValueError("Expected the saved baseline's 32-action/9-frame generation contract.")
    if getattr(policy.config, "visual_history_enabled", False):
        raise ValueError("Use the original baseline checkpoint, not a history-conditioned checkpoint.")
    batch = adapter.observation_batch(
        observation, source["task"]["instruction"], policy.config, torch.device(args.device)
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with (
        torch.inference_mode(),
        policy_inference_context("bf16"),
        torch.autocast("cuda", dtype=torch.bfloat16),
    ):
        prediction = policy.generate_validation(batch, seed=args.seed)
        actions = adapter.native_actions(prediction["action_normalized"])
        writer = PredictionVideoWriter(policy.config, torch.device(args.device), adapter.fps)
        exported = writer.save(
            prediction,
            actions,
            args.output_dir / "decoded",
            control_step=0,
            replan_index=0,
            seed=args.seed,
            save_frames=True,
        )
    torch.cuda.synchronize()
    import imageio.v2 as imageio

    views = {
        view: np.stack([imageio.imread(path) for path in paths]) for view, paths in exported["frames"].items()
    }
    offsets = future_offsets(policy.config.chunk_size, policy.config.world_video_frames)
    export_frames(args.output_dir, views, offsets, adapter.fps)
    model_dir, config_path = checkpoint_paths(args.checkpoint, None)
    generation_metadata(
        args,
        source,
        "baseline_mlp",
        offsets,
        time.perf_counter() - started,
        checkpoint=str(args.checkpoint.resolve()),
        saved_video_frames=9,
        sampling_steps=policy.config.inference_steps,
        current_state_conditioning=True,
        config_sha256=digest(config_path),
        checkpoint_index_sha256=digest(model_dir / ".metadata"),
        normalization_stats_sha256=digest(args.dataset_root / "meta/stats.json"),
    )


def split_native_canvas(frames, content_size):
    height, width = content_size
    if frames.ndim != 4 or frames.shape[1] < height or frames.shape[2] < width or width % 2:
        raise ValueError("The native prediction does not contain the expected two-camera canvas.")
    return {
        "primary": frames[:, :height, : width // 2],
        "wrist": frames[:, :height, width // 2 : width],
    }


def cosmos3(args):
    import torch
    from lerobot.common.policies.cosmos3.modeling_cosmos3 import Cosmos3NanoPolicy
    from lerobot.evaluation.eval_cosmos3_libero import observation_batch

    from lerobot.configs.policies import PreTrainedConfig

    observation, source = load_input(args.input_dir)
    torch.cuda.set_device(args.device)
    offsets = source["future_source_offsets"]
    config = PreTrainedConfig.from_pretrained(args.model_dir)
    if config.type != "cosmos3_nano" or config.use_state:
        raise ValueError("Expected the official native Nano policy with use_state=false.")
    trained_chunk_size = config.chunk_size
    config.chunk_size = (
        math.ceil(offsets[-1] / config.vae.scale_factor_temporal) * config.vae.scale_factor_temporal
    )
    config.n_action_steps = config.chunk_size
    config.fps = float(source["source_fps"])
    config.dtype = torch.bfloat16
    config.gradient_checkpointing = False
    config.compile_model = False
    config.save_predicted_video = True
    if args.steps is not None:
        config.num_inference_steps = args.steps
    torch.manual_seed(args.seed)
    policy = Cosmos3NanoPolicy.from_pretrained(args.model_dir, config=config, map_location="cpu")
    policy.to(args.device).requires_grad_(False).eval()
    policy.model.network.vae.enable_tiling()
    processor = policy.get_processor()
    batch = observation_batch([observation], source["task"]["instruction"], torch.device(args.device))
    with torch.inference_mode():
        prepared = processor(batch)
        if prepared["pixel_values_videos_gen"].shape[2] != 1:
            raise ValueError("Native inference must receive only the current frame, never future RGB.")
        if (
            int(prepared["video_grid_thw_gen"][0, 0])
            != 1 + config.chunk_size // config.vae.scale_factor_temporal
        ):
            raise ValueError("Native processor did not adopt the extended generation horizon.")
        view_h, view_w = config.view_size
        _, content_size = processor.canvas.fit_canvas(
            torch.zeros(1, 1, 3, view_h, 2 * view_w, dtype=torch.uint8)
        )
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        actions = policy.predict_action_chunk(prepared)
        if (
            not torch.isfinite(actions).all()
            or policy.last_predicted_latents is None
            or not torch.isfinite(policy.last_predicted_latents).all()
        ):
            raise ValueError("Native generation produced invalid actions or no video latents.")
        decoded = policy.decode_predicted_latents(policy.last_predicted_latents).numpy()
    torch.cuda.synchronize()
    if len(decoded) != config.chunk_size + 1:
        raise ValueError("Native video length does not match its requested horizon.")
    np.save(args.output_dir / "native_canvas_all_frames.npy", decoded)
    np.save(args.output_dir / "action_chunk.npy", actions.float().cpu().numpy())
    selected = decoded[[0, *offsets]]
    views = split_native_canvas(selected, content_size)
    export_frames(args.output_dir, views, offsets, source["source_fps"])
    generation_metadata(
        args,
        source,
        "cosmos3_nano_native",
        offsets,
        time.perf_counter() - started,
        model_dir=str(args.model_dir.resolve()),
        config_sha256=digest(args.model_dir / "config.json"),
        trained_chunk_size=trained_chunk_size,
        inference_chunk_size=config.chunk_size,
        generated_video_frames=len(decoded),
        selected_video_indices=[0, *offsets],
        current_state_conditioning=False,
        content_size=list(content_size),
        sampling_steps=config.num_inference_steps,
        horizon_changed_at_inference=config.chunk_size != trained_chunk_size,
    )


def compare(args):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    baseline_meta = json.loads((args.baseline_dir / "metadata.json").read_text())
    native_meta = json.loads((args.cosmos3_dir / "metadata.json").read_text())
    for key in ("input_sha256", "instruction", "future_source_offsets", "source_fps", "future_times_s"):
        if baseline_meta[key] != native_meta[key]:
            raise ValueError(f"Cannot compare predictions with different {key}.")
    camera_display = {}
    with (
        np.load(args.baseline_dir / "frames.npz", allow_pickle=False) as ours,
        np.load(args.cosmos3_dir / "frames.npz", allow_pickle=False) as native,
    ):
        if not set(CAMERAS) <= set(ours.files) & set(native.files):
            raise ValueError("Both primary and wrist predictions are required.")
        for view in CAMERAS:
            if ours[view].ndim != 4 or native[view].ndim != 4 or len(ours[view]) != len(native[view]):
                raise ValueError("Camera predictions must have the same number of THWC frames.")
            if len(ours[view]) != len(baseline_meta["future_times_s"]) + 1:
                raise ValueError("Predictions must contain the current frame and all aligned future frames.")
            height, width = ours[view].shape[1:3]
            resize_native = ours[view].shape[1:3] != native[view].shape[1:3]
            camera_display[view] = {
                "baseline_resolution": list(ours[view].shape[1:3]),
                "cosmos3_resolution": list(native[view].shape[1:3]),
                "cosmos3_spatial_resampling": "bilinear" if resize_native else "none",
                "display_panel_resolution": [height, width],
            }
            frames = []
            times = [0, *baseline_meta["future_times_s"]]
            for index, seconds in enumerate(times):
                native_frame = Image.fromarray(native[view][index])
                if resize_native:
                    native_frame = native_frame.resize((width, height), resample=Image.Resampling.BILINEAR)
                row = np.concatenate([np.asarray(native_frame), ours[view][index]], axis=1)
                canvas = Image.new("RGB", (row.shape[1], row.shape[0] + 24), "white")
                canvas.paste(Image.fromarray(row), (0, 24))
                draw = ImageDraw.Draw(canvas)
                draw.text((4, 5), f"Cosmos3-Nano {native[view].shape[1]}px  t={seconds:.2f}s", fill="black")
                draw.text((width + 4, 5), f"MLP baseline {height}px  t={seconds:.2f}s", fill="black")
                frames.append(np.asarray(canvas))
            fps = 8 / times[-1]
            imageio.mimsave(
                args.output_dir / f"{view}_comparison.mp4",
                frames,
                fps=fps,
                codec="libx264",
                macro_block_size=1,
            )
            imageio.imwrite(
                args.output_dir / f"{view}_comparison_future8.png", np.concatenate(frames[1:], axis=0)
            )
    write_json(
        args.output_dir / "metadata.json",
        {
            "baseline": baseline_meta,
            "cosmos3": native_meta,
            "camera_display": camera_display,
        },
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    common.add_argument("--output-dir", type=Path, required=True)
    common.add_argument("--seed", type=int, default=0)
    prepare_parser = commands.add_parser("prepare", parents=[common])
    prepare_parser.add_argument("--libero-root", type=Path, required=True)
    prepare_parser.add_argument("--task-manifest", type=Path, required=True)
    prepare_parser.add_argument("--task-index", type=int, default=0)
    prepare_parser.add_argument("--trial", type=int, default=0)
    prepare_parser.add_argument("--wait-steps", type=int, default=10)
    prepare_parser.add_argument("--source-fps", type=int, default=20)
    prepare_parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    prepare_parser.add_argument("--render-backend", choices=("osmesa", "egl"), default="osmesa")
    for backend in ("baseline", "cosmos3"):
        worker = commands.add_parser(backend, parents=[common])
        worker.add_argument("--input-dir", type=Path, required=True)
        worker.add_argument("--device", default="cuda:0")
        worker.add_argument("--steps", type=int)
        if backend == "baseline":
            worker.set_defaults(steps=20)
            worker.add_argument("--checkpoint", type=Path, required=True)
            worker.add_argument("--dataset-root", type=Path, required=True)
            worker.add_argument("--qwen3vl-dir", type=Path, required=True)
            worker.add_argument("--cosmos3-dir", type=Path, required=True)
        else:
            worker.add_argument("--model-dir", type=Path, required=True)
    comparison = commands.add_parser("compare", parents=[common])
    comparison.add_argument("--baseline-dir", type=Path, required=True)
    comparison.add_argument("--cosmos3-dir", type=Path, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.repo_root.resolve()))
    LOGGER.info("Running %s; artifacts: %s", args.command, args.output_dir)
    {"prepare": prepare, "baseline": baseline, "cosmos3": cosmos3, "compare": compare}[args.command](args)
    LOGGER.info("Completed %s", args.command)


if __name__ == "__main__":
    main()
