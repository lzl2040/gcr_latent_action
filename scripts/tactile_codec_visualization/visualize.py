#!/usr/bin/env python
"""Visualize a Stage 1 tactile encoder/decoder on local LeRobot v3 datasets."""

from __future__ import annotations

import argparse
import csv
import gc
import html
import json
import logging
import math
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lerobot.common.datasets.canonical_space import get_spec, tactile_image_keys  # noqa: E402
from lerobot.common.datasets_v30.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.common.policies.ace.configuration_robo_contrast import RoboContrastConfig  # noqa: E402
from lerobot.common.policies.ace.ftp1_tactile import tactile_image_sensors  # noqa: E402
from lerobot.common.policies.ace.modeling_robo_contrast import (  # noqa: E402
    TactileImageEncoder,
    TactilePatchDecoder,
)

LOGGER = logging.getLogger("tactile_codec_visualization")

DEFAULT_DATA_ROOTS = (
    Path("/Data/lerobot_data_ort6d/v30/FTP-1"),
    Path("/media/v-wangxiaofa/新加卷/lerobot_data/OpenNeoData"),
)


@dataclass(frozen=True)
class DatasetSource:
    name: str
    root: Path
    version: str
    tactile_keys: tuple[str, ...]
    view_indices: tuple[int, ...]


@dataclass
class TactileSample:
    dataset: str
    dataset_root: str
    episode_index: int
    frame_index: int
    view_index: int
    view_key: str
    image: torch.Tensor
    mean_01: torch.Tensor
    std_01: torch.Tensor
    spatial_std: float


@dataclass
class EvaluationRecord:
    dataset: str
    dataset_root: str
    episode_index: int
    frame_index: int
    view_index: int
    view_key: str
    spatial_std: float
    pixel_mae: float
    pixel_mse: float
    psnr_db: float
    z_mse: float
    mean_baseline_mse: float
    mse_improvement: float
    output_clip_fraction: float
    latent_rms_mean: float
    latent_rms_std: float
    panel: str


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def _strip_distributed_prefix(name: str) -> str:
    previous = None
    while name != previous:
        previous = name
        for prefix in ("module.", "_forward_module."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
    return name


def _step_number(path: Path) -> int:
    match = re.search(r"(\d+)([kKmM]?)", path.parent.name)
    if match is None:
        return -1
    value = int(match.group(1))
    suffix = match.group(2).lower()
    if suffix == "k":
        value *= 1_000
    elif suffix == "m":
        value *= 1_000_000
    return value


def resolve_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    latest = path / "latest"
    if latest.is_file():
        candidate = path / latest.read_text(encoding="utf-8").strip() / "mp_rank_00_model_states.pt"
        if candidate.is_file():
            return candidate
    direct = path / "mp_rank_00_model_states.pt"
    if direct.is_file():
        return direct
    candidates = list(path.glob("*/mp_rank_00_model_states.pt"))
    if not candidates:
        raise FileNotFoundError(f"No DeepSpeed mp_rank_00_model_states.pt was found under {path}.")
    return max(candidates, key=lambda candidate: (_step_number(candidate), candidate.stat().st_mtime_ns))


def resolve_config(checkpoint_arg: Path, checkpoint_file: Path, config_arg: Path | None) -> Path:
    if config_arg is not None:
        if not config_arg.is_file():
            raise FileNotFoundError(f"Stage 1 config does not exist: {config_arg}")
        return config_arg
    candidates = []
    if checkpoint_arg.is_dir():
        candidates.append(checkpoint_arg / "config.json")
    candidates.extend(parent / "config.json" for parent in checkpoint_file.parents[:4])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find config.json next to the Stage 1 checkpoint. Pass --config explicitly."
    )


def load_stage1_config(path: Path) -> RoboContrastConfig:
    import draccus

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("type", None)
    config = draccus.decode(RoboContrastConfig, payload)
    if config.tactile_backbone != "resnet18":
        raise ValueError(
            "A decodable spatial tactile codec requires tactile_backbone='resnet18'; "
            f"the checkpoint config uses {config.tactile_backbone!r}."
        )
    if config.tactile_recon_weight <= 0:
        raise ValueError("The checkpoint config has tactile_recon_weight <= 0, so it has no trained decoder.")
    return config


def _extract_module_state(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    result = {}
    for raw_name, value in state_dict.items():
        name = _strip_distributed_prefix(raw_name)
        if name.startswith(prefix):
            result[name[len(prefix) :]] = value
    if not result:
        raise KeyError(f"Checkpoint has no parameters under {prefix!r}.")
    return result


def load_codec(
    checkpoint_file: Path,
    config: RoboContrastConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[TactileImageEncoder, TactilePatchDecoder, dict[str, Any]]:
    LOGGER.info("Loading checkpoint index from %s", checkpoint_file)
    try:
        payload = torch.load(
            checkpoint_file,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    state_dict = payload.get("module", payload)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected checkpoint state_dict to be a mapping, got {type(state_dict).__name__}.")

    encoder = TactileImageEncoder(config.tactile_feat_dim, pretrained=False)
    decoder = TactilePatchDecoder(
        config.tactile_feat_dim,
        config.tactile_recon_size,
    )
    encoder_state = _extract_module_state(
        state_dict,
        "physical_encoder.tactile_cnn.",
    )
    decoder_state = _extract_module_state(
        state_dict,
        "physical_encoder.tactile_recon.",
    )
    encoder.load_state_dict(encoder_state, strict=True)
    decoder.load_state_dict(decoder_state, strict=True)

    checkpoint_meta = {
        "checkpoint": str(checkpoint_file),
        "global_steps": int(payload.get("global_steps", payload.get("step", -1))),
        "encoder_tensors": len(encoder_state),
        "decoder_tensors": len(decoder_state),
    }
    del state_dict, payload, encoder_state, decoder_state
    gc.collect()

    encoder.to(device=device, dtype=dtype).eval()
    decoder.to(device=device, dtype=dtype).eval()
    LOGGER.info(
        "Loaded tactile codec: encoder=%d tensors, decoder=%d tensors, device=%s, dtype=%s",
        checkpoint_meta["encoder_tensors"],
        checkpoint_meta["decoder_tensors"],
        device,
        dtype,
    )
    return encoder, decoder, checkpoint_meta


def infer_dataset_name(path: Path) -> str:
    parent_names = {parent.name for parent in path.parents}
    if "FTP-1" in parent_names:
        return f"ftp_1_{path.name}"
    if "OpenNeoData" in parent_names:
        return f"open_neo_{path.name}"
    return path.name


def _dataset_candidates(root: Path) -> list[Path]:
    if (root / "meta" / "info.json").is_file():
        return [root]
    if not root.is_dir():
        LOGGER.warning("Data root does not exist: %s", root)
        return []
    return sorted(
        child for child in root.iterdir() if child.is_dir() and (child / "meta" / "info.json").is_file()
    )


def discover_sources(
    roots: list[Path],
    requested_datasets: set[str],
) -> list[DatasetSource]:
    sources: dict[str, DatasetSource] = {}
    for root in roots:
        for dataset_root in _dataset_candidates(root):
            name = infer_dataset_name(dataset_root)
            if requested_datasets and name not in requested_datasets:
                continue
            info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
            version = str(info.get("codebase_version", ""))
            if not version.startswith("v3"):
                LOGGER.warning(
                    "Skipping %s: this visualizer currently expects LeRobot v3, got %s.",
                    dataset_root,
                    version or "unknown",
                )
                continue
            configured_keys = tactile_image_keys(get_spec(name))
            available = info.get("features", {})
            pairs = [
                (index, key)
                for index, key in enumerate(configured_keys)
                if key in available and available[key].get("dtype") in {"video", "image"}
            ]
            if not pairs:
                LOGGER.info("Skipping %s: no configured tactile image stream.", name)
                continue
            sources[name] = DatasetSource(
                name=name,
                root=dataset_root,
                version=version,
                tactile_keys=tuple(key for _, key in pairs),
                view_indices=tuple(index for index, _ in pairs),
            )

    missing = requested_datasets - set(sources)
    if missing:
        raise FileNotFoundError(
            "Requested tactile datasets were not discovered under the supplied roots: "
            + ", ".join(sorted(missing))
        )
    if not sources:
        raise RuntimeError("No LeRobot v3 dataset with configured tactile images was found.")
    return [sources[name] for name in sorted(sources)]


def _evenly_spaced_indices(total: int, count: int) -> list[int]:
    if total <= 0 or count <= 0:
        return []
    return sorted(set(np.linspace(0, total - 1, min(total, count), dtype=np.int64).tolist()))


def _episode_frame_indices(start: int, end: int, count: int) -> list[int]:
    length = end - start
    if length <= 0 or count <= 0:
        return []
    if count == 1:
        return [start + (length - 1) // 2]
    positions = np.linspace(start, end - 1, min(length, count) + 2, dtype=np.int64)[1:-1]
    return sorted({int(position) for position in positions})


def _as_tactile_uint8(image: torch.Tensor, size: int) -> torch.Tensor:
    image = image.detach().cpu()
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(f"Expected one decoded frame, got shape {tuple(image.shape)}.")
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Expected a CHW tactile image, got shape {tuple(image.shape)}.")
    if image.shape[0] not in (1, 3) and image.shape[-1] in (1, 3):
        image = image.permute(2, 0, 1)
    if image.shape[0] == 1:
        image = image.expand(3, -1, -1)
    if image.is_floating_point():
        image = (image.clamp(0, 1) * 255.0).round()
    image = image.to(torch.uint8)
    if image.shape[-2:] != (size, size):
        image = (
            functional.interpolate(
                image.unsqueeze(0).float(),
                size=(size, size),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
        )
    return image


def collect_samples(
    source: DatasetSource,
    *,
    image_size: int,
    episodes_per_dataset: int,
    frames_per_episode: int,
    max_images: int,
    dead_std: float,
    include_dead: bool,
    video_backend: str | None,
) -> tuple[list[TactileSample], dict[str, int]]:
    LOGGER.info(
        "Opening %s (%s) with %d tactile view(s)",
        source.name,
        source.root,
        len(source.tactile_keys),
    )
    dataset = LeRobotDataset(
        f"local-{source.name}",
        root=source.root,
        dataset_name=source.name,
        video_return_type="uint8",
        video_backend=video_backend,
        download_videos=False,
    )
    dataset.video_keys_to_decode = list(source.tactile_keys)

    configured_view_count = max(source.view_indices) + 1
    _, all_means, all_stds = tactile_image_sensors(source.name, configured_view_count)
    samples: list[TactileSample] = []
    counters = {
        "decoded_frames": 0,
        "dead_images_skipped": 0,
        "decode_failures": 0,
    }
    episode_positions = _evenly_spaced_indices(
        len(dataset.meta.episodes),
        episodes_per_dataset,
    )
    for episode_position in episode_positions:
        episode = dataset.meta.episodes[episode_position]
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        for frame_index in _episode_frame_indices(start, end, frames_per_episode):
            try:
                item = dataset[frame_index]
            except Exception as exc:  # noqa: BLE001 - continue to other fixed samples
                counters["decode_failures"] += 1
                LOGGER.warning(
                    "Failed to decode %s episode %d frame %d: %s",
                    source.name,
                    episode_position,
                    frame_index,
                    exc,
                )
                continue
            counters["decoded_frames"] += 1
            episode_index = int(item["episode_index"])
            for view_index, view_key in zip(
                source.view_indices,
                source.tactile_keys,
                strict=True,
            ):
                if view_key not in item:
                    continue
                image = _as_tactile_uint8(item[view_key], image_size)
                spatial_std = float((image.float() / 255.0).reshape(3, -1).std(dim=-1).max().item())
                if spatial_std < dead_std and not include_dead:
                    counters["dead_images_skipped"] += 1
                    continue
                mean = (torch.tensor(all_means[view_index], dtype=torch.float32) + 1.0) * 0.5
                std = (torch.tensor(all_stds[view_index], dtype=torch.float32) * 0.5).clamp_min(1e-3)
                samples.append(
                    TactileSample(
                        dataset=source.name,
                        dataset_root=str(source.root),
                        episode_index=episode_index,
                        frame_index=frame_index,
                        view_index=view_index,
                        view_key=view_key,
                        image=image,
                        mean_01=mean,
                        std_01=std,
                        spatial_std=spatial_std,
                    )
                )
                if len(samples) >= max_images:
                    break
            if len(samples) >= max_images:
                break
        if len(samples) >= max_images:
            break

    del dataset
    gc.collect()
    LOGGER.info(
        "%s: collected %d images, skipped %d dead images, %d decode failures",
        source.name,
        len(samples),
        counters["dead_images_skipped"],
        counters["decode_failures"],
    )
    return samples, counters


def _to_rgb_uint8(image: torch.Tensor) -> np.ndarray:
    return image.detach().float().clamp(0, 1).permute(1, 2, 0).mul(255).round().to(torch.uint8).cpu().numpy()


def _latent_heatmap(latent_rms: torch.Tensor, size: int) -> np.ndarray:
    values = latent_rms.detach().float().cpu().numpy()
    minimum = float(values.min())
    maximum = float(values.max())
    ratio = (values - minimum) / max(maximum - minimum, 1e-8)
    red = np.clip(1.5 * ratio, 0, 1)
    green = np.clip(1.5 - 1.5 * np.abs(2 * ratio - 1), 0, 1)
    blue = np.clip(1.5 * (1 - ratio), 0, 1)
    rgb = np.stack([red, green, blue], axis=-1)
    heatmap = Image.fromarray((rgb * 255).round().astype(np.uint8))
    return np.asarray(heatmap.resize((size, size), Image.Resampling.NEAREST))


def _labeled_image(array: np.ndarray, title: str, subtitle: str = "") -> Image.Image:
    image = Image.fromarray(array)
    header = 38
    panel = Image.new("RGB", (image.width, image.height + header), "white")
    panel.paste(image, (0, header))
    draw = ImageDraw.Draw(panel)
    draw.text((4, 4), title, fill="black")
    if subtitle:
        draw.text((4, 20), subtitle, fill="black")
    return panel


def save_panel(
    output_path: Path,
    ground_truth: torch.Tensor,
    reconstruction: torch.Tensor,
    latent_rms: torch.Tensor,
    record: EvaluationRecord,
) -> None:
    size = ground_truth.shape[-1]
    difference = (ground_truth - reconstruction).abs()
    panels = [
        _labeled_image(
            _to_rgb_uint8(ground_truth),
            "Ground truth",
            f"spatial std={record.spatial_std:.4f}",
        ),
        _labeled_image(
            _latent_heatmap(latent_rms, size),
            "Encoder latent RMS",
            f"{latent_rms.min():.3f} .. {latent_rms.max():.3f}",
        ),
        _labeled_image(
            _to_rgb_uint8(reconstruction),
            "Reconstruction",
            f"PSNR={record.psnr_db:.2f} dB",
        ),
        _labeled_image(
            _to_rgb_uint8((difference * 4.0).clamp(0, 1)),
            "Absolute error x4",
            f"MAE={record.pixel_mae:.4f}",
        ),
    ]
    footer = 24
    canvas = Image.new(
        "RGB",
        (sum(panel.width for panel in panels), max(panel.height for panel in panels) + footer),
        "white",
    )
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (4, canvas.height - 18),
        (
            f"{record.dataset} | episode {record.episode_index} | frame {record.frame_index} | "
            f"{record.view_key} | mean-baseline improvement {record.mse_improvement:+.1%}"
        ),
        fill="black",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def evaluate_samples(
    samples: list[TactileSample],
    *,
    encoder: TactileImageEncoder,
    decoder: TactilePatchDecoder,
    device: torch.device,
    batch_size: int,
    output_root: Path,
) -> list[EvaluationRecord]:
    records = []
    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        images_u8 = torch.stack([sample.image for sample in batch_samples])
        means = torch.stack([sample.mean_01 for sample in batch_samples])
        stds = torch.stack([sample.std_01 for sample in batch_samples])
        with torch.inference_mode():
            _, patch_grid = encoder(images_u8.unsqueeze(1).to(device, non_blocking=True))
            patches = patch_grid[:, 0]
            decoded_z = decoder(patches).float()

        ground_truth = images_u8.float().to(device) / 255.0
        if ground_truth.shape[-2:] != decoded_z.shape[-2:]:
            ground_truth = functional.interpolate(
                ground_truth,
                size=decoded_z.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        means_device = means.to(device)[:, :, None, None]
        stds_device = stds.to(device)[:, :, None, None]
        target_z = (ground_truth - means_device) / stds_device
        reconstruction_unclipped = decoded_z * stds_device + means_device
        reconstruction = reconstruction_unclipped.clamp(0, 1)
        baseline = means_device.expand_as(ground_truth).clamp(0, 1)
        difference = ground_truth - reconstruction
        pixel_mse = difference.square().mean(dim=(1, 2, 3))
        pixel_mae = difference.abs().mean(dim=(1, 2, 3))
        z_mse = (decoded_z - target_z).square().mean(dim=(1, 2, 3))
        baseline_mse = (ground_truth - baseline).square().mean(dim=(1, 2, 3))
        improvement = 1.0 - pixel_mse / baseline_mse.clamp_min(1e-12)
        clip_fraction = (
            ((reconstruction_unclipped < 0) | (reconstruction_unclipped > 1)).float().mean(dim=(1, 2, 3))
        )
        latent_rms = patches.float().square().mean(dim=1).sqrt()

        for local_index, sample in enumerate(batch_samples):
            mse = float(pixel_mse[local_index].item())
            panel_relative = Path(_slug(sample.dataset)) / (
                f"episode_{sample.episode_index:06d}_frame_{sample.frame_index:09d}"
                f"_view_{sample.view_index:02d}_{_slug(sample.view_key)}.png"
            )
            record = EvaluationRecord(
                dataset=sample.dataset,
                dataset_root=sample.dataset_root,
                episode_index=sample.episode_index,
                frame_index=sample.frame_index,
                view_index=sample.view_index,
                view_key=sample.view_key,
                spatial_std=sample.spatial_std,
                pixel_mae=float(pixel_mae[local_index].item()),
                pixel_mse=mse,
                psnr_db=float(-10.0 * math.log10(max(mse, 1e-12))),
                z_mse=float(z_mse[local_index].item()),
                mean_baseline_mse=float(baseline_mse[local_index].item()),
                mse_improvement=float(improvement[local_index].item()),
                output_clip_fraction=float(clip_fraction[local_index].item()),
                latent_rms_mean=float(latent_rms[local_index].mean().item()),
                latent_rms_std=float(latent_rms[local_index].std().item()),
                panel=panel_relative.as_posix(),
            )
            save_panel(
                output_root / panel_relative,
                ground_truth[local_index].cpu(),
                reconstruction[local_index].cpu(),
                latent_rms[local_index].cpu(),
                record,
            )
            records.append(record)
    return records


def _mean(records: list[EvaluationRecord], field: str) -> float:
    return float(np.mean([getattr(record, field) for record in records]))


def summarize(records: list[EvaluationRecord]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        grouped[record.dataset].append(record)
    summary = {}
    for dataset, dataset_records in sorted(grouped.items()):
        summary[dataset] = {
            "images": len(dataset_records),
            "pixel_mae": _mean(dataset_records, "pixel_mae"),
            "pixel_mse": _mean(dataset_records, "pixel_mse"),
            "psnr_db": _mean(dataset_records, "psnr_db"),
            "z_mse": _mean(dataset_records, "z_mse"),
            "mean_baseline_mse": _mean(dataset_records, "mean_baseline_mse"),
            "mse_improvement": _mean(dataset_records, "mse_improvement"),
            "output_clip_fraction": _mean(dataset_records, "output_clip_fraction"),
        }
    return summary


def write_csv(path: Path, records: list[EvaluationRecord]) -> None:
    fieldnames = list(asdict(records[0]))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def write_contact_sheet(
    output_path: Path,
    panel_paths: list[Path],
    *,
    max_panels: int = 24,
    columns: int = 2,
) -> None:
    selected = panel_paths[:max_panels]
    if not selected:
        return
    images = [Image.open(path).convert("RGB") for path in selected]
    try:
        width = max(image.width for image in images)
        height = max(image.height for image in images)
        rows = math.ceil(len(images) / columns)
        sheet = Image.new("RGB", (columns * width, rows * height), "white")
        for index, image in enumerate(images):
            x = (index % columns) * width
            y = (index // columns) * height
            sheet.paste(image, (x, y))
        sheet.save(output_path)
    finally:
        for image in images:
            image.close()


def write_dataset_pages(
    output_root: Path,
    records: list[EvaluationRecord],
    summary: dict[str, dict[str, float | int]],
) -> None:
    grouped: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        grouped[record.dataset].append(record)
    for dataset, dataset_records in grouped.items():
        dataset_dir = output_root / _slug(dataset)
        panel_paths = [output_root / record.panel for record in dataset_records]
        write_contact_sheet(dataset_dir / "contact_sheet.png", panel_paths)
        cards = "\n".join(
            f"""
<article>
  <img loading="lazy" src="{html.escape(Path(record.panel).name)}">
  <p><code>{html.escape(record.view_key)}</code><br>
  episode {record.episode_index}, frame {record.frame_index}<br>
  MSE {record.pixel_mse:.6f}, PSNR {record.psnr_db:.2f} dB,
  baseline improvement {record.mse_improvement:+.1%}</p>
</article>
"""
            for record in dataset_records
        )
        dataset_summary = html.escape(json.dumps(summary[dataset], indent=2, sort_keys=True))
        document = f"""<!doctype html>
<meta charset="utf-8">
<title>{html.escape(dataset)} tactile codec</title>
<style>
body {{ font-family: sans-serif; margin: 24px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 16px; }}
article {{ border: 1px solid #ccc; padding: 8px; }}
img {{ width: 100%; height: auto; }}
pre {{ background: #f4f4f4; padding: 12px; }}
</style>
<h1>{html.escape(dataset)}</h1>
<p><a href="../index.html">Back to all datasets</a> |
<a href="contact_sheet.png">Open contact sheet</a></p>
<pre>{dataset_summary}</pre>
<div class="grid">{cards}</div>
"""
        (dataset_dir / "index.html").write_text(document, encoding="utf-8")


def write_root_index(
    output_root: Path,
    summary: dict[str, dict[str, float | int]],
    checkpoint_meta: dict[str, Any],
) -> None:
    rows = "\n".join(
        f"""
<tr>
  <td><a href="{_slug(dataset)}/index.html">{html.escape(dataset)}</a></td>
  <td>{metrics["images"]}</td>
  <td>{metrics["pixel_mae"]:.6f}</td>
  <td>{metrics["pixel_mse"]:.6f}</td>
  <td>{metrics["psnr_db"]:.2f}</td>
  <td>{metrics["z_mse"]:.6f}</td>
  <td>{metrics["mse_improvement"]:+.1%}</td>
  <td>{metrics["output_clip_fraction"]:.1%}</td>
</tr>
"""
        for dataset, metrics in summary.items()
    )
    checkpoint = html.escape(json.dumps(checkpoint_meta, indent=2, sort_keys=True))
    document = f"""<!doctype html>
<meta charset="utf-8">
<title>Stage 1 tactile codec visualization</title>
<style>
body {{ font-family: sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; }}
th, td {{ border: 1px solid #aaa; padding: 5px 9px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
pre {{ background: #f4f4f4; padding: 12px; }}
</style>
<h1>Stage 1 tactile encoder/decoder</h1>
<p><strong>MSE improvement</strong> compares the learned codec against a decoder that emits
only the registered per-dataset mean image. Positive values mean the encoder/decoder preserves
more sample-specific contact structure than that fixed-image baseline.</p>
<table>
<tr><th>Dataset</th><th>Images</th><th>MAE</th><th>MSE</th><th>PSNR</th>
<th>Z-MSE</th><th>Mean-baseline improvement</th><th>Output clipped</th></tr>
{rows}
</table>
<h2>Checkpoint</h2><pre>{checkpoint}</pre>
<p><a href="metrics.csv">Per-image CSV</a> |
<a href="metrics.json">Full JSON report</a></p>
"""
    (output_root / "index.html").write_text(document, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/Data/lzl/ace_stage1"),
        help="Stage 1 root, step directory, or mp_rank_00_model_states.pt.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="RoboContrast config.json; auto-discovered by default.",
    )
    parser.add_argument(
        "--data-root",
        action="append",
        type=Path,
        default=None,
        help="Dataset root or parent containing V3 datasets; repeatable.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Restrict to a canonical dataset name, e.g. ftp_1_sharpa; repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="New report directory. Defaults to CHECKPOINT/tactile_codec_visualization.",
    )
    parser.add_argument("--episodes-per-dataset", type=int, default=4)
    parser.add_argument("--frames-per-episode", type=int, default=3)
    parser.add_argument("--max-images-per-dataset", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--dead-std",
        type=float,
        default=None,
        help="Spatial std threshold in [0,1]; defaults to the Stage 1 config.",
    )
    parser.add_argument(
        "--include-dead",
        action="store_true",
        help="Include spatially flat pads that Stage 1 normally masks out.",
    )
    parser.add_argument(
        "--video-backend",
        choices=("auto", "torchcodec", "pyav", "video_reader"),
        default="auto",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or an explicit CUDA device such as cuda:1.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float32"),
        default="auto",
    )
    return parser.parse_args()


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {value!r} was requested but CUDA is unavailable.")
    return device


def _resolve_dtype(value: str, device: torch.device) -> torch.dtype:
    if value == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    dtype = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[value]
    if device.type == "cpu" and dtype == torch.bfloat16:
        LOGGER.warning("Using bfloat16 on CPU may be slow or unsupported for some kernels.")
    return dtype


def main() -> int:
    args = _parse_args()
    if args.episodes_per_dataset <= 0 or args.frames_per_episode <= 0:
        raise ValueError("Episode and frame sample counts must be positive.")
    if args.max_images_per_dataset <= 0 or args.batch_size <= 0:
        raise ValueError("Image and batch sizes must be positive.")

    checkpoint_file = resolve_checkpoint(args.checkpoint)
    config_file = resolve_config(args.checkpoint, checkpoint_file, args.config)
    config = load_stage1_config(config_file)
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)
    encoder, decoder, checkpoint_meta = load_codec(
        checkpoint_file,
        config,
        device,
        dtype,
    )

    roots = list(args.data_root or DEFAULT_DATA_ROOTS)
    sources = discover_sources(roots, set(args.dataset))
    LOGGER.info(
        "Discovered tactile datasets: %s",
        ", ".join(source.name for source in sources),
    )

    output_root = args.output_dir
    if output_root is None:
        base = args.checkpoint if args.checkpoint.is_dir() else args.checkpoint.parent
        output_root = base / "tactile_codec_visualization"
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Output directory already exists: {output_root}. Choose a new --output-dir.")
    output_root.mkdir(parents=True)

    dead_std = config.tactile_dead_std if args.dead_std is None else args.dead_std
    all_records: list[EvaluationRecord] = []
    sampling_report = {}
    for source in sources:
        samples, counters = collect_samples(
            source,
            image_size=config.tactile_img_size,
            episodes_per_dataset=args.episodes_per_dataset,
            frames_per_episode=args.frames_per_episode,
            max_images=args.max_images_per_dataset,
            dead_std=dead_std,
            include_dead=args.include_dead,
            video_backend=None if args.video_backend == "auto" else args.video_backend,
        )
        sampling_report[source.name] = {
            **counters,
            "images_collected": len(samples),
            "root": str(source.root),
            "tactile_keys": list(source.tactile_keys),
        }
        if not samples:
            LOGGER.warning("%s produced no usable tactile image.", source.name)
            continue
        all_records.extend(
            evaluate_samples(
                samples,
                encoder=encoder,
                decoder=decoder,
                device=device,
                batch_size=args.batch_size,
                output_root=output_root,
            )
        )
        del samples
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not all_records:
        raise RuntimeError("No tactile image was evaluated.")
    summary = summarize(all_records)
    checkpoint_meta.update(
        {
            "config": str(config_file),
            "tactile_img_size": config.tactile_img_size,
            "tactile_recon_size": config.tactile_recon_size,
            "tactile_feat_dim": config.tactile_feat_dim,
            "dead_std": dead_std,
            "device": str(device),
            "dtype": str(dtype),
        }
    )
    payload = {
        "checkpoint": checkpoint_meta,
        "data_roots": [str(root) for root in roots],
        "sampling": sampling_report,
        "summary": summary,
        "records": [asdict(record) for record in all_records],
    }
    (output_root / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(output_root / "metrics.csv", all_records)
    write_dataset_pages(output_root, all_records, summary)
    write_root_index(output_root, summary, checkpoint_meta)
    (output_root / "COMPLETE").write_text("ok\n", encoding="utf-8")

    LOGGER.info("Wrote %d tactile reconstructions to %s", len(all_records), output_root)
    for dataset, metrics in summary.items():
        LOGGER.info(
            "%s: images=%d MSE=%.6f PSNR=%.2f dB baseline improvement=%+.1f%%",
            dataset,
            metrics["images"],
            metrics["pixel_mse"],
            metrics["psnr_db"],
            100.0 * metrics["mse_improvement"],
        )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
