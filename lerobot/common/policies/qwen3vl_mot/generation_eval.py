"""Fixed held-out generation validation and human-readable artifact export."""

from __future__ import annotations

import html
import json
import logging
import os
import shutil
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image, ImageDraw
from torch import distributed as dist

from lerobot.common.datasets.contrastive_dataset import contrastive_collate_fn

logger = logging.getLogger(__name__)


@dataclass
class GenerationEvalConfig:
    enabled: bool = False
    datasets: str = "open_neo_arx5,ms_data_xdof_1,interna1_dual_arm_1,ftp_1_sharpa"
    episodes_per_dataset: int = 2
    seed: int = 20_260_927
    output_subdir: str = "generation_eval"
    video_fps: int = 8
    batch_size: int = 8

    @property
    def dataset_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(name.strip() for name in self.datasets.split(",") if name.strip()))

    def validate(self, eval_freq: int) -> None:
        if not self.enabled:
            return
        if eval_freq <= 0:
            raise ValueError("generation_eval.enabled=true requires a positive eval_freq.")
        if not self.dataset_names:
            raise ValueError("generation_eval.datasets must name at least one dataset.")
        if self.episodes_per_dataset <= 0:
            raise ValueError("generation_eval.episodes_per_dataset must be positive.")
        if self.video_fps <= 0:
            raise ValueError("generation_eval.video_fps must be positive.")
        if self.batch_size <= 0:
            raise ValueError("generation_eval.batch_size must be positive.")
        subdir = Path(self.output_subdir)
        if subdir.is_absolute() or ".." in subdir.parts:
            raise ValueError("generation_eval.output_subdir must be a relative path inside output_dir.")


_ACTION_DIM_NAMES = (
    "eef0_x",
    "eef0_y",
    "eef0_z",
    "eef0_rot6d_0",
    "eef0_rot6d_1",
    "eef0_rot6d_2",
    "eef0_rot6d_3",
    "eef0_rot6d_4",
    "eef0_rot6d_5",
    "eef0_gripper",
    "eef1_x",
    "eef1_y",
    "eef1_z",
    "eef1_rot6d_0",
    "eef1_rot6d_1",
    "eef1_rot6d_2",
    "eef1_rot6d_3",
    "eef1_rot6d_4",
    "eef1_rot6d_5",
    "eef1_gripper",
    "joint0_0",
    "joint0_1",
    "joint0_2",
    "joint0_3",
    "joint0_4",
    "joint0_5",
    "joint0_6",
    "joint0_gripper",
    "joint1_0",
    "joint1_1",
    "joint1_2",
    "joint1_3",
    "joint1_4",
    "joint1_5",
    "joint1_6",
    "joint1_gripper",
    "reserved_0",
    "reserved_1",
    "reserved_2",
    "reserved_3",
)

_ACTION_GROUPS = {
    "eef0_xyz": range(0, 3),
    "eef0_rot6d": range(3, 9),
    "eef0_gripper": range(9, 10),
    "eef1_xyz": range(10, 13),
    "eef1_rot6d": range(13, 19),
    "eef1_gripper": range(19, 20),
    "joint0": range(20, 28),
    "joint1": range(28, 36),
}


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def persist_generation_manifest(output_root: Path, manifest: dict) -> Path:
    path = output_root / "split_manifest.json"
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        existing = path.read_text(encoding="utf-8")
        if existing != encoded:
            raise RuntimeError(
                f"Held-out validation split changed relative to {path}. Use a new "
                "output directory rather than mixing evaluation populations."
            )
        return path
    _atomic_write_text(path, encoded)
    return path


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _resize_video(video: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if video.shape[-2:] == size:
        return video
    frames = video.shape[0]
    return functional.interpolate(
        video.float(),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).reshape(frames, 3, *size)


def _labeled_triptych(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    difference: np.ndarray,
) -> np.ndarray:
    panels = [ground_truth, prediction, difference]
    labels = ("Ground truth", "Prediction", "Absolute difference")
    height, width = ground_truth.shape[:2]
    canvas = Image.new("RGB", (3 * width, height + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(zip(panels, labels, strict=True)):
        canvas.paste(Image.fromarray(panel), (index * width, 24))
        draw.text((index * width + 4, 5), label, fill="black")
    return np.asarray(canvas)


def _save_video_artifacts(
    output_dir: Path,
    ground_truth: torch.Tensor,
    prediction: torch.Tensor,
    fps: int,
) -> dict[str, float]:
    ground_truth = ground_truth.float().clamp(0, 255) / 255.0
    prediction = prediction.float()
    if not torch.isfinite(prediction).all():
        raise RuntimeError("Generated video contains non-finite pixels.")
    prediction = prediction.clamp(0, 1)
    frames = min(ground_truth.shape[0], prediction.shape[0])
    ground_truth = ground_truth[:frames]
    prediction = prediction[:frames]
    prediction = _resize_video(prediction, ground_truth.shape[-2:])
    difference = (ground_truth - prediction).abs()
    comparison_frames = []
    for gt, pred, diff in zip(
        ground_truth,
        prediction,
        difference,
        strict=True,
    ):
        comparison_frames.append(
            _labeled_triptych(
                (gt.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8),
                (pred.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8),
                (diff.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8),
            )
        )
    imageio.mimsave(
        output_dir / "video_comparison.mp4",
        comparison_frames,
        fps=fps,
        macro_block_size=1,
    )

    selected = np.linspace(0, frames - 1, min(frames, 6), dtype=np.int64)
    rows = [Image.fromarray(comparison_frames[int(index)]) for index in selected]
    sheet = Image.new(
        "RGB",
        (max(row.width for row in rows), sum(row.height for row in rows)),
        "white",
    )
    y = 0
    for frame_index, row in zip(selected, rows, strict=True):
        draw = ImageDraw.Draw(row)
        draw.text((4, row.height - 16), f"frame {int(frame_index)}", fill="white")
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(output_dir / "video_contact_sheet.png")
    return {"video_pixel_mae": float(difference.mean().item())}


def _denormalize_action(
    normalized: torch.Tensor,
    stats: dict,
    *,
    clip: bool,
) -> tuple[torch.Tensor, float]:
    normalized = normalized.float().cpu()
    mean = stats["mean"].float().cpu()
    std = stats["std"].float().cpu()
    raw = normalized * std + mean
    if not clip:
        return raw, 0.0
    minimum = stats["min"].float().cpu()
    maximum = stats["max"].float().cpu()
    finite = torch.isfinite(minimum) & torch.isfinite(maximum)
    if not finite.any():
        return raw, 0.0
    before = raw.clone()
    raw[..., finite] = torch.maximum(
        torch.minimum(raw[..., finite], maximum[finite]),
        minimum[finite],
    )
    changed = before[..., finite] != raw[..., finite]
    return raw, float(changed.float().mean().item())


def _heat_color(value: float, scale: float, *, difference: bool) -> tuple[int, int, int]:
    ratio = min(1.0, abs(value) / max(scale, 1e-8))
    if difference:
        return (255, int(255 * (1.0 - ratio)), int(255 * (1.0 - ratio)))
    if value >= 0:
        return (255, int(255 * (1.0 - ratio)), int(255 * (1.0 - ratio)))
    return (int(255 * (1.0 - ratio)), int(255 * (1.0 - ratio)), 255)


def _action_heatmap_panel(
    values: np.ndarray,
    names: list[str],
    title: str,
    scale: float,
    *,
    difference: bool,
) -> Image.Image:
    cell_width = 12
    cell_height = 18
    label_width = 118
    title_height = 28
    image = Image.new(
        "RGB",
        (
            label_width + values.shape[0] * cell_width,
            title_height + values.shape[1] * cell_height,
        ),
        "white",
    )
    draw = ImageDraw.Draw(image)
    draw.text((4, 6), f"{title} (scale={scale:.3g})", fill="black")
    for row, name in enumerate(names):
        y = title_height + row * cell_height
        draw.text((3, y + 2), name, fill="black")
        for column, value in enumerate(values[:, row]):
            x = label_width + column * cell_width
            draw.rectangle(
                (x, y, x + cell_width - 1, y + cell_height - 1),
                fill=_heat_color(float(value), scale, difference=difference),
            )
    return image


def _save_action_artifacts(
    output_dir: Path,
    ground_truth_normalized: torch.Tensor,
    prediction_normalized: torch.Tensor,
    stats: dict,
    valid_mask: torch.Tensor,
) -> dict[str, Any]:
    valid_indices = valid_mask.nonzero(as_tuple=True)[0].tolist()
    if not valid_indices:
        return {"action_valid_dimensions": 0}
    ground_truth, _ = _denormalize_action(
        ground_truth_normalized,
        stats,
        clip=False,
    )
    prediction, clip_fraction = _denormalize_action(
        prediction_normalized,
        stats,
        clip=True,
    )
    ground_truth = ground_truth[:, valid_indices]
    prediction = prediction[:, valid_indices]
    if not torch.isfinite(prediction).all():
        raise RuntimeError("Generated action contains non-finite values.")
    difference = (ground_truth - prediction).abs()
    names = [_ACTION_DIM_NAMES[index] for index in valid_indices]
    signed_scale = float(
        torch.cat([ground_truth.abs().reshape(-1), prediction.abs().reshape(-1)]).max().clamp_min(1e-8).item()
    )
    diff_scale = float(difference.max().clamp_min(1e-8).item())
    panels = [
        _action_heatmap_panel(
            ground_truth.numpy(),
            names,
            "Ground truth action",
            signed_scale,
            difference=False,
        ),
        _action_heatmap_panel(
            prediction.numpy(),
            names,
            "Predicted action",
            signed_scale,
            difference=False,
        ),
        _action_heatmap_panel(
            difference.numpy(),
            names,
            "Absolute error",
            diff_scale,
            difference=True,
        ),
    ]
    canvas = Image.new(
        "RGB",
        (sum(panel.width for panel in panels), max(panel.height for panel in panels)),
        "white",
    )
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    canvas.save(output_dir / "action_comparison.png")

    per_dimension = difference.mean(dim=0)
    valid_set = set(valid_indices)
    group_mae = {}
    for group, group_indices in _ACTION_GROUPS.items():
        selected = [valid_indices.index(index) for index in group_indices if index in valid_set]
        if selected:
            group_mae[group] = float(difference[:, selected].mean().item())
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{float(per_dimension[index]):.6g}</td>"
        f"<td>{float(ground_truth[:, index].min()):.6g}</td>"
        f"<td>{float(ground_truth[:, index].max()):.6g}</td>"
        f"<td>{float(prediction[:, index].min()):.6g}</td>"
        f"<td>{float(prediction[:, index].max()):.6g}</td>"
        "</tr>"
        for index, name in enumerate(names)
    )
    groups = "\n".join(
        f"<li><code>{html.escape(name)}</code>: {value:.6g}</li>" for name, value in group_mae.items()
    )
    action_html = f"""<!doctype html>
<meta charset="utf-8">
<title>Action comparison</title>
<style>
body {{ font-family: sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; }}
th, td {{ border: 1px solid #aaa; padding: 4px 8px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
img {{ max-width: 100%; }}
</style>
<h1>Denormalized valid action dimensions</h1>
<p>Overall MAE: <strong>{float(difference.mean()):.6g}</strong>;
prediction clip fraction: <strong>{clip_fraction:.2%}</strong>.</p>
<img src="action_comparison.png" alt="Action heatmaps">
<h2>Semantic group MAE</h2><ul>{groups}</ul>
<h2>Per-dimension summary</h2>
<table>
<tr><th>Dimension</th><th>MAE</th><th>GT min</th><th>GT max</th>
<th>Prediction min</th><th>Prediction max</th></tr>
{rows}
</table>
"""
    _atomic_write_text(output_dir / "action_summary.html", action_html)
    return {
        "action_mae": float(difference.mean().item()),
        "action_clip_fraction": clip_fraction,
        "action_valid_dimensions": len(valid_indices),
        "action_group_mae": group_mae,
    }


def _save_tactile_artifacts(
    output_dir: Path,
    ground_truth: torch.Tensor,
    prediction_z: torch.Tensor,
    mask: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> dict[str, float | int]:
    valid_views = mask.nonzero(as_tuple=True)[0].tolist()
    if not valid_views or prediction_z.numel() == 0:
        return {"tactile_valid_pads": 0}
    ground_truth = ground_truth[valid_views].float() / 255.0
    prediction_z = prediction_z[valid_views].float()
    if not torch.isfinite(prediction_z).all():
        raise RuntimeError("Generated tactile image contains non-finite values.")
    mean_01 = (mean[valid_views].float() + 1.0) * 0.5
    std_01 = (std[valid_views].float() * 0.5).clamp_min(1e-3)
    prediction = (prediction_z * std_01[:, :, None, None] + mean_01[:, :, None, None]).clamp(0, 1)
    if prediction.shape[-2:] != ground_truth.shape[-2:]:
        prediction = functional.interpolate(
            prediction,
            size=ground_truth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    difference = (ground_truth - prediction).abs()
    rows = []
    for pad, gt, pred, diff in zip(
        valid_views,
        ground_truth,
        prediction,
        difference,
        strict=True,
    ):
        row = Image.fromarray(
            _labeled_triptych(
                (gt.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8),
                (pred.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8),
                (diff.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8),
            )
        )
        draw = ImageDraw.Draw(row)
        draw.text((4, row.height - 16), f"pad {pad}", fill="white")
        rows.append(row)
    sheet = Image.new(
        "RGB",
        (max(row.width for row in rows), sum(row.height for row in rows)),
        "white",
    )
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(output_dir / "tactile_comparison.png")
    return {
        "tactile_pixel_mae": float(difference.mean().item()),
        "tactile_valid_pads": len(valid_views),
    }


def _write_sample_index(
    output_dir: Path,
    sample: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    action = (
        '<p><a href="action_summary.html">Open denormalized action summary</a></p>'
        '<img src="action_comparison.png">'
        if (output_dir / "action_comparison.png").is_file()
        else "<p>No valid action dimension.</p>"
    )
    tactile = (
        '<h2>Tactile image prediction</h2><img src="tactile_comparison.png">'
        if (output_dir / "tactile_comparison.png").is_file()
        else "<h2>Tactile image prediction</h2><p>No valid tactile image pad.</p>"
    )
    document = f"""<!doctype html>
<meta charset="utf-8">
<title>{html.escape(str(sample["dataset"]))} generation validation</title>
<style>
body {{ font-family: sans-serif; margin: 24px; }}
img, video {{ max-width: 100%; height: auto; }}
code, pre {{ background: #f4f4f4; padding: 2px 4px; }}
</style>
<h1>{html.escape(str(sample["dataset"]))}</h1>
<p>Episode {sample["episode_index"]}, frame {sample["frame_index"]},
{sample["fps"]:.3g} FPS.</p>
<h2>RGB video prediction</h2>
<video controls loop muted src="video_comparison.mp4"></video>
<p><a href="video_contact_sheet.png">Open RGB contact sheet</a></p>
<h2>Action prediction</h2>
{action}
{tactile}
<h2>Metrics</h2><pre>{html.escape(json.dumps(metrics, indent=2, sort_keys=True))}</pre>
"""
    _atomic_write_text(output_dir / "index.html", document)


class GenerationEvaluator:
    """Run deterministic generation through an FSDP wrapper and export on rank zero."""

    def __init__(
        self,
        *,
        config: GenerationEvalConfig,
        dataset,
        output_dir: Path,
        policy_config,
        device: torch.device,
        rank: int,
        local_rank: int,
        process_group: dist.ProcessGroup | None = None,
    ):
        self.config = config
        self.dataset = dataset
        self.root = Path(output_dir) / config.output_subdir
        self.policy_config = policy_config
        self.device = device
        self.rank = rank
        self.local_rank = local_rank
        self.process_group = process_group
        self.samples = dataset.heldout_validation_samples()
        if not self.samples:
            raise RuntimeError("Generation validation is enabled but no held-out windows were built.")
        self.video_decoder = None
        manifest_error = None
        if rank == 0:
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                persist_generation_manifest(
                    self.root,
                    dataset.heldout_validation_manifest(),
                )
            except Exception as exc:  # noqa: BLE001 - synchronize failure across ranks
                manifest_error = exc
        self._raise_rank0_error(manifest_error, "persist validation split")

    def _control_device(self) -> torch.device:
        if self.process_group is not None and dist.get_backend(self.process_group) == dist.Backend.GLOO:
            return torch.device("cpu")
        return self.device

    def _raise_rank0_error(
        self,
        error: Exception | None,
        phase: str,
    ) -> None:
        failed = torch.tensor(
            int(error is not None),
            device=self._control_device(),
            dtype=torch.int32,
        )
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(failed, src=0, group=self.process_group)
        if failed.item():
            if error is not None:
                raise RuntimeError(f"Generation validation failed during {phase}: {error}") from error
            raise RuntimeError(f"Generation validation rank 0 failed during {phase}; see rank 0 logs.")

    def _prepare_video_decoder(self) -> None:
        if self.rank != 0 or self.video_decoder is not None:
            return
        from lerobot.common.policies.ace.cosmos3_encoders import build_cosmos3_vae

        decoder, _, _, _, _ = build_cosmos3_vae(
            self.policy_config.cosmos3_dir,
            encoder_only=False,
        )
        decoder.encoder = None
        decoder.quant_conv = None
        for parameter in decoder.parameters():
            parameter.requires_grad_(False)
        decoder.eval()
        if hasattr(decoder, "enable_tiling"):
            decoder.enable_tiling()
        self.video_decoder = decoder

    @torch.no_grad()
    def _decode_video(self, latents: torch.Tensor) -> torch.Tensor:
        self._prepare_video_decoder()
        decoder = self.video_decoder.to(
            device=self.device,
            dtype=torch.bfloat16,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            decoded = decoder.decode(latents.to(device=self.device, dtype=torch.bfloat16)).sample
        return ((decoded.float().clamp(-1, 1) + 1.0) * 0.5).permute(
            0,
            2,
            1,
            3,
            4,
        )

    def _write_step_index(
        self,
        output_dir: Path,
        entries: list[dict[str, Any]],
    ) -> None:
        rows = "\n".join(
            "<tr>"
            f"<td>{html.escape(entry['dataset'])}</td>"
            f"<td>{entry['episode_index']}</td>"
            f"<td>{entry['frame_index']}</td>"
            f'<td><a href="{html.escape(entry["relative_path"])}/index.html">open</a></td>'
            f"<td>{entry['metrics'].get('video_pixel_mae', float('nan')):.6g}</td>"
            f"<td>{entry['metrics'].get('action_mae', float('nan')):.6g}</td>"
            f"<td>{entry['metrics'].get('tactile_pixel_mae', float('nan')):.6g}</td>"
            "</tr>"
            for entry in entries
        )
        document = f"""<!doctype html>
<meta charset="utf-8">
<title>Generation validation</title>
<style>
body {{ font-family: sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; }}
th, td {{ border: 1px solid #aaa; padding: 5px 9px; }}
</style>
<h1>Generation validation</h1>
<table>
<tr><th>Dataset</th><th>Episode</th><th>Frame</th><th>Artifacts</th>
<th>Video MAE</th><th>Action MAE</th><th>Tactile MAE</th></tr>
{rows}
</table>
"""
        _atomic_write_text(output_dir / "index.html", document)

    def run(self, model, step: int) -> None:
        final_dir = self.root / f"step_{step:08d}"
        directory_error = None
        if self.rank == 0 and final_dir.exists():
            if (final_dir / "metrics.json").is_file():
                logger.info("Generation artifacts already exist for step %d; skipping.", step)
                skip = torch.ones(
                    (),
                    device=self._control_device(),
                    dtype=torch.int32,
                )
            else:
                directory_error = RuntimeError(
                    f"Incomplete generation artifact directory already exists: {final_dir}"
                )
                skip = torch.zeros((), device=self.device, dtype=torch.int32)
        else:
            skip = torch.zeros(
                (),
                device=self._control_device(),
                dtype=torch.int32,
            )
        self._raise_rank0_error(directory_error, "inspect artifact directory")
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(skip, src=0, group=self.process_group)
        if skip.item():
            return
        decoder_error = None
        if self.rank == 0:
            try:
                self._prepare_video_decoder()
            except Exception as exc:  # noqa: BLE001 - synchronize failure across ranks
                decoder_error = exc
        self._raise_rank0_error(decoder_error, "load video decoder")

        temporary_dir = None
        if self.rank == 0:
            temporary_dir = Path(
                tempfile.mkdtemp(
                    prefix=f".step_{step:08d}.tmp-",
                    dir=self.root,
                )
            )
        was_training = model.training
        model.eval()
        entries = []
        try:
            for batch_start in range(0, len(self.samples), self.config.batch_size):
                batch_infos = self.samples[batch_start : batch_start + self.config.batch_size]
                samples = [
                    self.dataset.get_heldout_item(
                        int(sample_info["dataset_index"]),
                        int(sample_info["frame_index"]),
                    )
                    for sample_info in batch_infos
                ]
                batch = _move_batch(
                    contrastive_collate_fn(samples),
                    self.device,
                )
                autocast = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if self.device.type == "cuda"
                    else nullcontext()
                )
                with autocast:
                    prediction = model(
                        batch,
                        task_type="generate_validation",
                        step=self.config.seed + batch_start * 10,
                    )
                artifact_error = None
                if self.rank == 0:
                    try:
                        for local_index, sample_info in enumerate(batch_infos):
                            sample_dir = (
                                temporary_dir
                                / str(sample_info["dataset"])
                                / (
                                    f"episode_{int(sample_info['episode_index']):06d}"
                                    f"_frame_{int(sample_info['frame_index']):09d}"
                                )
                            )
                            sample_dir.mkdir(parents=True, exist_ok=False)
                            predicted_video = self._decode_video(
                                prediction["video_latents"][local_index : local_index + 1]
                            )[0].cpu()
                            metrics = _save_video_artifacts(
                                sample_dir,
                                batch["video"][local_index].cpu(),
                                predicted_video,
                                self.config.video_fps,
                            )
                            dataset_index = int(sample_info["dataset_index"])
                            metrics.update(
                                _save_action_artifacts(
                                    sample_dir,
                                    batch["action"][
                                        local_index,
                                        : self.policy_config.n_action_steps,
                                    ].cpu(),
                                    prediction["action_normalized"][local_index].cpu(),
                                    self.dataset.norm_stats[dataset_index]["action"],
                                    batch["action_mask"][local_index].cpu(),
                                )
                            )
                            metrics.update(
                                _save_tactile_artifacts(
                                    sample_dir,
                                    batch["tactile_image"][local_index, :, -1].cpu(),
                                    prediction["tactile_decoded_z"][local_index].cpu(),
                                    prediction["tactile_mask"][local_index].cpu(),
                                    batch["tactile_img_mean"][local_index].cpu(),
                                    batch["tactile_img_std"][local_index].cpu(),
                                )
                            )
                            _write_sample_index(sample_dir, sample_info, metrics)
                            relative_path = sample_dir.relative_to(temporary_dir).as_posix()
                            entries.append(
                                {
                                    **sample_info,
                                    "relative_path": relative_path,
                                    "metrics": metrics,
                                }
                            )
                    except Exception as exc:  # noqa: BLE001 - synchronize failure
                        artifact_error = exc
                del prediction, batch
                self._raise_rank0_error(
                    artifact_error,
                    f"write artifacts for batch {batch_start}",
                )

            if self.rank == 0:
                try:
                    payload = {
                        "step": step,
                        "split_fingerprint": self.dataset.heldout_eval_fingerprint,
                        "samples": entries,
                    }
                    _atomic_write_text(
                        temporary_dir / "metrics.json",
                        json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    )
                    self._write_step_index(temporary_dir, entries)
                    temporary_dir.rename(final_dir)
                    _atomic_write_text(
                        self.root / "latest_generation_eval",
                        final_dir.name + "\n",
                    )
                    logger.info("Generation artifacts saved to %s", final_dir)
                    temporary_dir = None
                    finalize_error = None
                except Exception as exc:  # noqa: BLE001 - synchronize failure
                    finalize_error = exc
            else:
                finalize_error = None
            self._raise_rank0_error(finalize_error, "publish artifacts")
        finally:
            model.train(was_training)
            if self.rank == 0 and self.video_decoder is not None:
                self.video_decoder.to("cpu")
                torch.cuda.empty_cache()
            if temporary_dir is not None and temporary_dir.exists():
                shutil.rmtree(temporary_dir)
