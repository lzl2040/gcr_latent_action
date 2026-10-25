"""Compare authentic training samples, deployment inputs and free-running sampling."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from lerobot.evaluation.eval_qwen3vl_mot_libero import checkpoint_paths, ort6d_to_controls
from lerobot.evaluation.robodojo_qwen3vl import ARMS, CAMERAS, RoboDojoAdapter

logger = logging.getLogger(__name__)


def matching_training_config(args) -> Path:
    _, policy_path = checkpoint_paths(args.checkpoint, args.policy_config)
    path = args.training_config or policy_path.with_name("train_config.json")
    saved = json.loads(path.read_text())
    if saved["policy"] != json.loads(policy_path.read_text()):
        raise ValueError(
            f"Training config {path} does not match the selected saved policy config {policy_path}."
        )
    return path.resolve()


def training_dataset(args, policy_config):
    from lerobot.common.datasets.contrastive_dataset import MultiModalContrastiveDataset

    saved = json.loads(args.training_config.read_text())
    mix = saved["data_mix"]
    if mix not in ("robodojo", "delta_robodojo_stack_bowl") or saved["stage"] != "finetune":
        raise ValueError("This diagnostic requires the saved RoboDojo finetuning configuration.")
    data = dict(saved["dataset"])
    data["image_transforms"] = SimpleNamespace(**data["image_transforms"])
    for name in ("parent_dir_v21", "parent_dir_v30", "parent_dir_extra"):
        data[name] = str(args.dataset_root.resolve().parents[2])
    cfg = SimpleNamespace(policy=policy_config, dataset=SimpleNamespace(**data), stage=saved["stage"])
    generation = saved["generation_eval"]
    dataset = MultiModalContrastiveDataset(
        cfg,
        seed=saved["seed"],
        data_mix=saved["data_mix"],
        vla2root_json=str(Path(__file__).resolve().parents[2] / "vla2root.json"),
        dataset_size_one_epoch=data["dataset_size_one_epoch"],
        heldout_eval_dataset_names=(mix,),
        heldout_eval_episodes_per_dataset=generation["episodes_per_dataset"],
        heldout_eval_seed=generation["seed"],
    )
    if dataset.dataset_names != [mix]:
        raise ValueError("Expected exactly the supplied RoboDojo dataset.")
    if Path(dataset.datasets[0].root).resolve() != args.dataset_root.resolve():
        raise ValueError("Training loader resolved a different dataset root.")
    return dataset


def validation_observation(item: dict, adapter: RoboDojoAdapter, views: tuple[str, ...]) -> dict:
    raw = item["observation.state"][0, :20].float().cpu() * adapter.state_std + adapter.state_mean
    state = {}
    for arm, offset in zip(ARMS, (0, 10), strict=True):
        xyzw = Rotation.from_euler(
            "xyz", ort6d_to_controls(raw[None, offset + 3 : offset + 9].numpy())
        ).as_quat()[0]
        state[f"{arm}_ee_pose"] = np.r_[raw[offset : offset + 3].numpy(), xyzw[[3, 0, 1, 2]]]
        state[f"{arm}_ee_joint_state"] = raw[offset + 9 : offset + 10].numpy()
    images = item.get("images_t0", item["image_t0"][None])
    return {
        "env_idx": 0,
        "instruction": item["task"],
        "state": state,
        "vision": {
            CAMERAS[view]: {"color": image.permute(1, 2, 0).cpu().numpy()}
            for view, image in zip(views, images, strict=True)
        },
    }


def input_parity(validation: dict, deployment: dict) -> dict:
    errors = {}
    for key in ("image_t0", "images_t0", "image_view_mask", "state_mask", "action_mask", "sample_rate"):
        if key not in deployment:
            continue
        if not torch.equal(validation[key], deployment[key]):
            raise ValueError(f"Training/deployment input mismatch in {key}.")
        errors[key] = 0.0
    if validation["task"] != deployment["task"]:
        raise ValueError("Training/deployment instructions differ.")
    current = validation["observation.state"][:, 0].float()
    error = (current - deployment["observation.state"].float()).abs().max().item()
    if error > 1e-5:
        raise ValueError(f"Training/deployment normalized state mismatch: {error}.")
    errors["normalized_current_state_max_error"] = error
    return errors


def action_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    adapter: RoboDojoAdapter,
    prefix: int,
    observation: dict | None = None,
) -> dict:
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 40:
        raise ValueError("Prediction and ground truth must have matching (1,T,40) shapes.")
    if not 1 <= prefix <= prediction.shape[1]:
        raise ValueError("The executed prefix must fit the predicted chunk.")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Diagnostic actions must be finite.")
    error = (prediction[..., :20].float() - target[..., :20].float()).square()
    predicted_native = adapter.native_actions(prediction, observation)
    target_native = adapter.native_actions(target, observation)
    result = {"normalized_mse": error.mean().item(), "prefix_normalized_mse": error[:, :prefix].mean().item()}
    for arm in ARMS:
        p = np.stack([a[f"{arm}_ee_pose"] for a in predicted_native])
        t = np.stack([a[f"{arm}_ee_pose"] for a in target_native])
        angle = np.degrees(
            (
                Rotation.from_quat(p[:, [4, 5, 6, 3]]).inv() * Rotation.from_quat(t[:, [4, 5, 6, 3]])
            ).magnitude()
        )
        grip = np.abs(
            np.stack([a[f"{arm}_ee_joint_state"] for a in predicted_native])
            - np.stack([a[f"{arm}_ee_joint_state"] for a in target_native])
        )
        result[arm] = {
            "first_predicted_xyz_m": p[0, :3].tolist(),
            "first_target_xyz_m": t[0, :3].tolist(),
            "first_xyz_l2_m": float(np.linalg.norm(p[0, :3] - t[0, :3])),
            "first_rotation_deg": float(angle[0]),
            "xyz_mean_l2_m": float(np.linalg.norm(p[:, :3] - t[:, :3], axis=1).mean()),
            "prefix_xyz_mean_l2_m": float(np.linalg.norm(p[:prefix, :3] - t[:prefix, :3], axis=1).mean()),
            "rotation_mean_deg": float(angle.mean()),
            "prefix_rotation_mean_deg": float(angle[:prefix].mean()),
            "gripper_mae_clipped": float(grip.mean()),
        }
    return result


def clean_action_autoencoding(policy, batch: dict) -> torch.Tensor:
    """Isolate frozen-encoder/decoder reconstruction from the generation sampler."""
    valid = batch["state_mask"].bool().any(-1) & batch["action_mask"].bool().any(-1)
    clean, _, _, _ = policy._clean_physical_embedding(batch, "action", valid)
    decoded = policy._decode_physical_embedding("action", clean).reshape_as(batch["action"])
    return decoded.float() * batch["action_mask"][:, None]


@torch.inference_mode()
def conditional_losses(policy, batch: dict, seed: int, device: torch.device) -> list[dict]:
    rows = []
    for task in ("joint_prediction", "inverse_dynamics"):
        if task not in policy.config.task_names:
            continue
        for sigma in (0.1, 0.5, 0.9):
            with (
                torch.random.fork_rng(devices=[device] if device.type == "cuda" else []),
                patch.object(
                    policy,
                    "_sample_sigma",
                    lambda size, dev, sigma=sigma: torch.full((size,), sigma, device=dev),
                ),
            ):
                torch.manual_seed(seed)
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    _, losses = policy(batch, task_type=task)
            rows.append(
                {
                    "task": task,
                    "sigma": sigma,
                    **{key: value for key, value in losses.items() if key.startswith("action_")},
                }
            )
    return rows


def diagnostic_samples(dataset, episodes: list[int], frames: list[int]) -> list[dict]:
    samples = [
        dict(sample, source="configured_generation_validation")
        for sample in dataset.heldout_validation_samples()
    ]
    ranges = dataset.all_episode_ranges[0]
    for episode in episodes:
        if not 0 <= episode < len(ranges):
            raise ValueError(f"Unknown diagnostic episode {episode}.")
        start, end = map(int, ranges[episode])
        for frame in frames:
            if frame < 0 or start + frame >= end:
                raise ValueError(f"Frame {frame} is outside episode {episode}.")
            samples.append(
                {
                    "dataset_index": 0,
                    "episode_index": episode,
                    "frame_index": start + frame,
                    "source": "additional_training_diagnostic",
                }
            )
    return samples


@torch.inference_mode()
def diagnose(args) -> None:
    from lerobot.common.datasets.contrastive_dataset import contrastive_collate_fn
    from lerobot.evaluation.eval_qwen3vl_mot_robodojo import load_model

    if any(step <= 0 for step in args.diagnostic_inference_steps):
        raise ValueError("Diagnostic inference steps must be positive.")
    args.training_config = matching_training_config(args)
    model = load_model(args)
    policy, adapter, device = model.policy, model.adapter, model.device
    precision = "bfloat16; training FP8 transformation is not enabled"
    if args.diagnostic_fp8:
        from lerobot.common.utils.fsdp_training import FSDPTrainingConfig, convert_policy_to_fp8

        saved = json.loads(args.training_config.read_text())
        config = FSDPTrainingConfig(**saved["fsdp"])
        if config.fp8 is not True or config.fp8_emulate:
            raise ValueError("The native FP8 diagnostic requires saved native FP8 training settings.")
        converted = convert_policy_to_fp8(policy, config, device)
        policy.requires_grad_(False).eval()
        precision = f"native FP8 {config.fp8_recipe}, scope={config.fp8_scope}, converted={len(converted)}"
    dataset = training_dataset(args, policy.config)
    samples = diagnostic_samples(dataset, args.diagnostic_episodes, args.diagnostic_frames)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "training_config": str(args.training_config.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "action_mode": adapter.action_mode,
        "action_feature": adapter.action_feature,
        "validation_split": dataset.validation_split,
        "saved_inference_steps": policy.config.inference_steps,
        "window_mode": dataset.window_mode,
        "frame_horizon": dataset.frame_horizons[0],
        "true_fps": dataset.true_fps[0],
        "video_backend": dataset._video_backend,
        "precision": precision,
        "sampler_parity_note": "Uses identical normalized current state after separately measuring adapter roundtrip.",
        "samples": [],
    }
    path = args.output_dir / "diagnostics.json"
    saved_steps = policy.config.inference_steps
    for index, info in enumerate(samples):
        item = dataset.get_heldout_item(0, info["frame_index"])
        if bool(item["data_read_fallback"]) or int(item["frame_index"]) != info["frame_index"]:
            raise RuntimeError("Diagnostic sample was replaced by a training-loader fallback.")
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in contrastive_collate_fn([item]).items()
        }
        observation = validation_observation(item, adapter, tuple(policy.config.rgb_views))
        deployment = adapter.observation_batch(observation, policy.config, device)
        entry = {**info, "instruction": item["task"], "input_parity": input_parity(batch, deployment)}
        deployment["observation.state"] = batch["observation.state"][:, 0].clone()
        seed = args.seed + index
        with torch.autocast(device.type, dtype=torch.bfloat16):
            validation = policy.generate_validation(batch, seed=seed)
            deployed = policy.generate_validation(deployment, seed=seed)
        entry["sampler_parity_max_action_error"] = (
            (validation["action_normalized"] - deployed["action_normalized"]).abs().max().item()
        )
        target = batch["action"][:, : policy.config.n_action_steps]
        entry["free_running"] = {}
        predictions = {"ground_truth": target.float().cpu().numpy()}
        if policy.physical_embedding_flow:
            with torch.autocast(device.type, dtype=torch.bfloat16):
                decoded = clean_action_autoencoding(policy, batch)[:, : policy.config.n_action_steps]
            entry["clean_action_autoencoding"] = action_metrics(
                decoded, target, adapter, args.replan_steps, observation
            )
            predictions["decoded_ground_truth_embedding"] = decoded.float().cpu().numpy()
        for steps in dict.fromkeys([saved_steps, *args.diagnostic_inference_steps]):
            policy.config.inference_steps = steps
            if steps == saved_steps:
                predicted = deployed["action_normalized"]
            else:
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    predicted = policy.generate_validation(deployment, seed=seed)["action_normalized"]
            entry["free_running"][str(steps)] = action_metrics(
                predicted, target, adapter, args.replan_steps, observation
            )
            predictions[f"prediction_steps_{steps}"] = predicted.float().cpu().numpy()
        policy.config.inference_steps = saved_steps
        current_raw = (
            batch["observation.state"][:, 0, :20].float().cpu() * adapter.state_std + adapter.state_mean
        )
        predictions["current_state_raw"] = current_raw.numpy()
        hold_raw = current_raw
        if adapter.action_mode == "delta":
            hold_raw = torch.zeros_like(current_raw)
            hold_raw[:, [3, 6, 13, 16]] = 1
        hold = torch.zeros_like(target)
        hold[..., :20] = ((hold_raw - adapter.action_mean) / adapter.action_std).to(device)[:, None]
        entry["hold_current_baseline"] = action_metrics(hold, target, adapter, args.replan_steps, observation)
        entry["ground_truth_conditioned_losses"] = conditional_losses(policy, batch, seed, device)
        directory = (
            args.output_dir
            / f"sample_{index:03d}_episode_{info['episode_index']:04d}_frame_{int(item['frame_index'])}"
        )
        directory.mkdir()
        import imageio.v2 as imageio

        for view in policy.config.rgb_views:
            imageio.imwrite(
                directory / f"conditioning_{view}.png", observation["vision"][CAMERAS[view]]["color"]
            )
        np.savez_compressed(directory / "actions.npz", **predictions)
        (directory / "metrics.json").write_text(json.dumps(entry, indent=2, allow_nan=False) + "\n")
        report["samples"].append(entry)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        logger.info(
            "Diagnostic %d/%d episode=%s frame=%s sampled MSE=%.6f, sampler parity=%.6g",
            index + 1,
            len(samples),
            info["episode_index"],
            item["frame_index"],
            entry["free_running"][str(saved_steps)]["normalized_mse"],
            entry["sampler_parity_max_action_error"],
        )
    report["diagnostics_complete"] = True
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
