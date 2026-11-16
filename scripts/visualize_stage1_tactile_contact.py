#!/usr/bin/env python
"""Find chronological contact onsets and export Stage-1 tactile codec reconstructions.

The decoder receives the spatial encoding of each GT tactile image, not RGB or a
predicted future latent. Contact scores measure image deformation, not calibrated force.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image
from torch.nn import functional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOGGER = logging.getLogger(__name__)
matplotlib.use("Agg")
SCRIPT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
DEFAULT_DATASETS = {
    **{
        f"open_neo_{name}": f"/mnt/msra-datahub/OpenNeoData/{name}" for name in ("ur", "arx5_single", "aloha")
    },
    **{
        f"ftp_1_{name}": f"/Data/lerobot_data_ort6d/v30/FTP-1/{name}"
        for name in ("VisuoTactile_D-WHEEL_split_0", "sharpa_split_0", "RDP")
    },
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def slug(value):
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")[:110]


def resize_tactile(images, size=112):
    """Match ContrastiveDataset._build_tactile_images, including uint8 truncation."""
    images = torch.as_tensor(np.asarray(images).copy()).permute(0, 3, 1, 2)
    return (
        functional.interpolate(images.float(), size=(size, size), mode="bilinear", align_corners=False)
        .clamp(0, 255)
        .to(torch.uint8)
    )


def pixel_statistics(dataset_name, keys, key):
    from lerobot.common.policies.ace.ftp1_tactile import FTP1_TACTILE_DATASETS, tactile_image_sensors

    if dataset_name not in FTP1_TACTILE_DATASETS:
        raise ValueError(f"No training-time tactile statistics for {dataset_name}.")
    _, means, stds = tactile_image_sensors(dataset_name, len(keys))
    slot = keys.index(key)
    return (np.asarray(means[slot]) + 1) / 2, np.maximum(np.asarray(stds[slot]) / 2, 1e-3)


def denormalize_tactile(prediction, mean, std):
    mean = torch.as_tensor(mean, dtype=torch.float32, device=prediction.device)[None, :, None, None]
    std = torch.as_tensor(std, dtype=torch.float32, device=prediction.device)[None, :, None, None]
    return prediction.float() * std + mean


class TactileCodec:
    def __init__(self, checkpoint, image_size=112, device="cpu"):
        from lerobot.common.policies.ace.modeling_robo_contrast import (
            TactileImageEncoder,
            TactilePatchDecoder,
        )

        checkpoint = Path(checkpoint)
        if checkpoint.is_dir():
            checkpoint /= "mp_rank_00_model_states.pt"
        # DeepSpeed includes non-tensor training metadata. Only use trusted training checkpoints.
        saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
        weights = saved["module"]
        encoder_prefix = "physical_encoder.tactile_cnn."
        decoder_prefix = "physical_encoder.tactile_recon."
        encoder_state = {
            k.removeprefix(encoder_prefix): v for k, v in weights.items() if k.startswith(encoder_prefix)
        }
        decoder_state = {
            k.removeprefix(decoder_prefix): v for k, v in weights.items() if k.startswith(decoder_prefix)
        }
        if not encoder_state or not decoder_state:
            raise ValueError(
                "Checkpoint must contain both trained ResNet tactile encoder and spatial decoder."
            )
        channels = encoder_state["norm.weight"].numel()
        self.encoder = TactileImageEncoder(channels, pretrained=False)
        self.decoder = TactilePatchDecoder(channels, image_size)
        self.encoder.load_state_dict(encoder_state, strict=True)
        self.decoder.load_state_dict(decoder_state, strict=True)
        digest = hashlib.sha256()
        for prefix, state in ((encoder_prefix, encoder_state), (decoder_prefix, decoder_state)):
            for name, value in sorted(state.items()):
                digest.update(f"{prefix}{name}:{value.dtype}:{tuple(value.shape)}".encode())
                digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
        self.provenance = {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
            "checkpoint_step": int(saved.get("step", saved.get("global_steps", -1))),
            "codec_sha256": digest.hexdigest(),
            "strictly_loaded_encoder_tensors": len(encoder_state),
            "strictly_loaded_decoder_tensors": len(decoder_state),
            "image_size": image_size,
            "image_size_source": "Stage-1 spatial ResNet codec contract, explicitly configurable",
            "inference_precision": "float32",
            "encoder_normalization": "ImageNet, inside TactileImageEncoder",
            "decoder_denormalization": "prediction * (registry_std / 2) + (registry_mean + 1) / 2",
            "reconstruction_not_future_prediction": True,
            "encoder_skip_connections": False,
            "visualization_script_sha256": SCRIPT_SHA256,
        }
        self.device = torch.device(device)
        self.encoder.to(self.device).eval().requires_grad_(False)
        self.decoder.to(self.device).eval().requires_grad_(False)
        del saved, weights, encoder_state, decoder_state

    @torch.inference_mode()
    def reconstruct(self, images, mean, std):
        _, patches = self.encoder(images[:, None].to(self.device))
        normalized = self.decoder(patches[:, 0])
        return denormalize_tactile(normalized, mean, std).cpu()


def video_location(root, info, episode, key):
    prefix = f"videos/{key}/"
    path = root / info["video_path"].format(
        video_key=key,
        chunk_index=int(episode[prefix + "chunk_index"]),
        file_index=int(episode[prefix + "file_index"]),
    )
    return path, float(episode[prefix + "from_timestamp"])


def episode_frame_times(root, info, episode, frame_indices):
    import pyarrow.parquet as pq

    path = root / info["data_path"].format(
        chunk_index=int(episode["data/chunk_index"]),
        file_index=int(episode["data/file_index"]),
    )
    rows = pq.read_table(
        path,
        columns=["episode_index", "frame_index", "timestamp"],
        filters=[
            ("episode_index", "=", int(episode["episode_index"])),
            ("frame_index", "in", frame_indices),
        ],
    ).to_pylist()
    rows.sort(key=lambda row: row["frame_index"])
    if [row["frame_index"] for row in rows] != frame_indices:
        raise ValueError(f"Data parquet does not contain exactly the requested episode frames: {path}")
    times = np.asarray([row["timestamp"] for row in rows], dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Dataset timestamps must be finite and strictly increasing.")
    error = float(np.abs(times - np.asarray(frame_indices) / info["fps"]).max())
    if error > 0.55 / info["fps"]:
        raise ValueError(f"Dataset timestamps differ from the nominal search timeline by {error}s.")
    return {
        "data_parquet": str(path),
        "frame_indices": frame_indices,
        "timestamp_s": times.tolist(),
        "max_nominal_timestamp_error_s": error,
    }


def read_frames(root, info, episode, key, frame_indices, size=None, episode_times=None):
    """Read nearest actual PTS frames using each view's v3 file and episode offset."""
    import av

    if any(b <= a for a, b in zip(frame_indices[:-1], frame_indices[1:], strict=True)):
        raise ValueError("Frame indices must be strictly increasing.")
    if not frame_indices or frame_indices[0] < 0 or frame_indices[-1] >= episode["length"]:
        raise ValueError("Requested frames cross the episode boundary.")
    path, offset = video_location(root, info, episode, key)
    times = np.asarray(frame_indices) / info["fps"] if episode_times is None else np.asarray(episode_times)
    if times.shape != (len(frame_indices),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Video target times must be finite, increasing, and match the requested frames.")
    targets = offset + times
    tolerance = 0.55 / info["fps"] + 1e-4
    images, actual = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        container.seek(max(0, int((targets[0] - 1 / info["fps"]) / stream.time_base)), stream=stream)
        previous = None
        position = 0
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            seconds = float(frame.pts * stream.time_base)
            while position < len(targets) and seconds >= targets[position] - 1e-6:
                chosen = frame
                chosen_time = seconds
                if previous is not None and abs(previous[0] - targets[position]) < abs(
                    seconds - targets[position]
                ):
                    chosen_time, chosen = previous
                if abs(chosen_time - targets[position]) > tolerance:
                    raise ValueError(
                        f"Unsynchronized {key}: requested {targets[position]}, got {chosen_time}."
                    )
                image = chosen.to_ndarray(format="rgb24")
                if size is not None:
                    image = resize_tactile(image[None], size)[0].permute(1, 2, 0).numpy()
                images.append(image)
                actual.append(chosen_time)
                position += 1
            if position == len(targets):
                break
            previous = seconds, frame
        # Float32 dataset timestamps can lie slightly past the final video PTS.
        while (
            position < len(targets)
            and previous is not None
            and abs(previous[0] - targets[position]) <= tolerance
        ):
            image = previous[1].to_ndarray(format="rgb24")
            if size is not None:
                image = resize_tactile(image[None], size)[0].permute(1, 2, 0).numpy()
            images.append(image)
            actual.append(previous[0])
            position += 1
        if position != len(targets):
            raise ValueError(f"Video ended before all requested {key} frames were decoded.")
    if any(b <= a for a, b in zip(actual[:-1], actual[1:], strict=True)):
        raise ValueError("Distinct requested times resolved to repeated video frames.")
    return np.stack(images), {
        "video": str(path),
        "episode_video_offset_s": offset,
        "requested_absolute_times_s": targets.tolist(),
        "actual_absolute_pts_s": actual,
        "actual_episode_times_s": (np.asarray(actual) - offset).tolist(),
        "max_timestamp_error_s": float(np.max(np.abs(targets - actual))),
    }


def contact_curve(images):
    values = images.float() / 255
    values = functional.avg_pool2d(values, 5, stride=1, padding=2)
    values = values[..., 10:-10, 10:-10]
    reference = values[: min(5, len(values))].median(dim=0).values
    delta = values - reference
    delta = delta - delta.flatten(2).median(dim=-1).values[:, :, None, None]
    return delta.square().mean(dim=(1, 2, 3)).sqrt().numpy()


def onset_windows(curve, sampled_indices, gaps=(3, 6, 9, 12)):
    """Select chronological, equally spaced weak-to-strong appearance changes."""
    positions = {frame: index for index, frame in enumerate(sampled_indices)}
    windows = []
    for gap in gaps:
        for start in sampled_indices:
            frames = [start + gap * i for i in range(5)]
            if any(frame not in positions for frame in frames):
                continue
            indices = [positions[frame] for frame in frames]
            energies = np.asarray(curve)[indices]
            gain = float(energies[-1] - energies[0])
            if gain < 0.018 or energies[0] > max(0.012, energies[-1] * 0.35):
                continue
            if np.diff(energies).min() < -0.08 * gain:
                continue
            if energies[1] < energies[0] + 0.03 * gain or energies[2] < energies[0] + 0.2 * gain:
                continue
            windows.append(
                {
                    "frame_indices": frames,
                    "sample_positions": indices,
                    "contact_proxy": energies.tolist(),
                    "contact_gain": gain,
                    "frame_gap": gap,
                    "onset_score": gain - float(energies[0]) - float(np.maximum(-np.diff(energies), 0).sum()),
                }
            )
    return sorted(windows, key=lambda x: x["onset_score"], reverse=True)


def pre_post_windows(images, sampled_indices, gaps=(3, 6, 9, 12)):
    """Propose two stable reference frames followed by three persistently changed frames.

    These are appearance onsets; physical contact still requires RGB verification.
    """
    values = functional.avg_pool2d(images.float() / 255, 5, stride=1, padding=2)
    values = values[..., 10:-10, 10:-10]
    positions = {frame: index for index, frame in enumerate(sampled_indices)}
    windows = []
    for gap in gaps:
        for start in sampled_indices:
            frames = [start + gap * i for i in range(5)]
            if any(frame not in positions for frame in frames):
                continue
            indices = [positions[frame] for frame in frames]
            selected = values[indices]
            reference = selected[:2].mean(0)
            before_noise = float((selected[1] - selected[0]).square().mean().sqrt())
            changes = (selected - reference).square().mean((1, 2, 3)).sqrt().numpy()
            after = changes[2:]
            if after.min() < 0.025 or before_noise > min(0.008, float(after.min()) / 3):
                continue
            if after[1:].min() < after[0] * 0.8:
                continue
            windows.append(
                {
                    "frame_indices": frames,
                    "sample_positions": indices,
                    "frame_gap": gap,
                    "contact_proxy": changes.tolist(),
                    "contact_gain": float(after.mean() - changes[:2].mean()),
                    "onset_score": float(after.min() - 3 * before_noise),
                    "pre_contact_variation": before_noise,
                    "contact_boundary_interval_frames": frames[1:3],
                    "frame_phases": ["Before change"] * 2 + ["After change"] * 3,
                    "contact_label_status": "appearance_candidate_requires_rgb_verification",
                }
            )
    return sorted(windows, key=lambda row: row["onset_score"], reverse=True)


def candidate_rank(record):
    quality = (record["metrics"]["nonconstant_contact_reconstruction"], record["quality_score"])
    if record.get("selection_mode") == "pre-post":
        return (record["onset_score"], *quality)
    return quality


def reconstruction_metrics(gt, prediction):
    from skimage.metrics import structural_similarity

    gt = gt.float() / 255
    prediction = prediction.float().clamp(0, 1)
    mse = (gt - prediction).square().mean(dim=(1, 2, 3)).numpy()
    delta_gt, delta_prediction = gt[-1] - gt[0], prediction[-1] - prediction[0]
    mask = delta_gt.abs().mean(0) > 0.025
    a, b = delta_gt[:, mask].flatten(), delta_prediction[:, mask].flatten()
    correlation = float(functional.cosine_similarity(a[None], b[None]).item()) if a.numel() else 0.0
    ratio = float(b.norm() / a.norm().clamp_min(1e-8)) if a.numel() else 0.0
    psnr = (-10 * np.log10(np.maximum(mse, 1e-12))).tolist()
    ssim = [
        float(
            structural_similarity(
                x.permute(1, 2, 0).numpy(), y.permute(1, 2, 0).numpy(), data_range=1, channel_axis=2
            )
        )
        for x, y in zip(gt, prediction, strict=True)
    ]
    return {
        "psnr_db": psnr,
        "mean_psnr_db": float(np.mean(psnr)),
        "ssim": ssim,
        "mean_ssim": float(np.mean(ssim)),
        "changing_pixel_fraction": float(mask.float().mean()),
        "contact_delta_cosine": correlation,
        "reconstructed_delta_norm_ratio": ratio,
        "nonconstant_contact_reconstruction": correlation > 0.2 and 0.15 < ratio < 3,
    }


def episode_records(root, task_pattern, episodes_per_task, max_episodes):
    import pyarrow.parquet as pq

    counts = Counter()
    result = []
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        parquet = pq.ParquetFile(path)
        columns = [
            key
            for key in parquet.schema_arrow.names
            if key in ("episode_index", "tasks", "length", "dataset_from_index", "dataset_to_index")
            or key.startswith(("videos/", "data/"))
        ]
        for episode in parquet.read(columns=columns).to_pylist():
            task = "; ".join(episode["tasks"])
            if not task_pattern.search(task) or counts[task] >= episodes_per_task:
                continue
            counts[task] += 1
            result.append(episode)
            if len(result) == max_episodes:
                return result
    return result


def rgb_keys(info, tactile_key):
    keys = [
        key
        for key, feature in info["features"].items()
        if feature["dtype"] in ("video", "image") and "tactile" not in key
    ]
    side = "right" if "right" in tactile_key and "left" not in tactile_key.split("wrist")[0] else "left"
    wrist = next((key for key in keys if side in key and "wrist" in key), None)
    primary = next((key for key in keys if "third_view" in key or key.endswith(".ego")), None)
    return list(dict.fromkeys(key for key in (wrist, primary, *keys[:1]) if key))


def case_directory(root, record):
    return (
        root / record["dataset"] / slug(record["task"]) / f"episode_{record['episode']['episode_index']:06d}"
    )


def save_images(directory, images, frame_indices):
    directory.mkdir()
    for index, (image, frame) in enumerate(zip(images, frame_indices, strict=True)):
        Image.fromarray(image).save(directory / f"{index:02d}_frame_{frame:06d}.png")


def crop_rgb(images, rectangle):
    if rectangle is None:
        return images
    x0, y0, x1, y1 = rectangle
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("RGB crop must be normalized x0 y0 x1 y1 within [0,1].")
    height, width = images.shape[1:3]
    left, right = round(x0 * width), round(x1 * width)
    top, bottom = round(y0 * height), round(y1 * height)
    if left == right or top == bottom:
        raise ValueError("RGB crop is empty at this resolution.")
    return images[:, top:bottom, left:right]


def make_figure(record, rgb, gt, prediction, native_gt=None):
    import matplotlib.pyplot as plt

    rectangle = record.get("figure_rgb_crop")
    rgb = crop_rgb(rgb, rectangle)
    tactile_rectangle = record.get("figure_tactile_crop")
    gt = crop_rgb(gt, tactile_rectangle)
    prediction = crop_rgb(prediction, tactile_rectangle)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    rows = [("RGB\n(crop)" if rectangle else "RGB", rgb)]
    if native_gt is not None:
        rows.append(("GT tactile\n(native)", native_gt))
    rows.extend(
        [
            (
                "GT tactile\n(fixed ROI)"
                if tactile_rectangle
                else "GT tactile\n(model input)"
                if native_gt is not None
                else "GT tactile",
                gt,
            ),
            ("Ours\n(ROI reconstruction)" if tactile_rectangle else "Ours\n(reconstruction)", prediction),
        ]
    )
    ratios = [images.shape[1] / images.shape[2] for _, images in rows]
    figure, axes = plt.subplots(
        len(rows),
        5,
        figsize=(7.1, 1.3 * sum(ratios)),
        gridspec_kw={"hspace": 0.04, "wspace": 0.035, "height_ratios": ratios},
    )
    times = np.asarray(
        record.get("dataset_frame_timestamps_s", np.asarray(record["frame_indices"]) / record["fps"])
    )
    times = times - times[0]
    phases = record.get("frame_phases")
    for row, (label, images) in enumerate(rows):
        for column, image in enumerate(images):
            axis = axes[row, column]
            axis.imshow(image, interpolation="nearest", vmin=0, vmax=255)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if row == 0:
                title = f"$t={times[column]:.2f}$ s"
                if phases:
                    title = phases[column] + "\n" + title
                axis.set_title(
                    title,
                    fontsize=8,
                    pad=4,
                    color=("#245b8a" if column < 2 else "#a84d14") if phases else "black",
                )
            if column == 0:
                axis.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=8)
    figure.subplots_adjust(left=0.17, right=0.995, bottom=0.015, top=0.88 if phases else 0.92)
    return figure


def export_case(record, codec, output_dir, figure_rgb_key=None, rgb_crop=None, tactile_crop=None):
    if figure_rgb_key is None:
        figure_rgb_key = record.get("figure_rgb_key")
    if rgb_crop is None:
        rgb_crop = record.get("figure_rgb_crop")
    root = Path(record["dataset_root"])
    info = json.loads((root / "meta/info.json").read_text())
    episode, indices = record["episode"], record["frame_indices"]
    data_sync = episode_frame_times(root, info, episode, indices)
    times = data_sync["timestamp_s"]
    record = {
        **record,
        "dataset_frame_timestamps_s": times,
        "figure_rgb_crop": rgb_crop,
        "figure_tactile_crop": tactile_crop
        if tactile_crop is not None
        else record.get("figure_tactile_crop"),
    }
    tactile, tactile_sync = read_frames(
        root, info, episode, record["tactile_key"], indices, episode_times=times
    )
    gt_tensor = resize_tactile(tactile, codec.provenance["image_size"])
    prediction_float = codec.reconstruct(gt_tensor, record["pixel_mean_01"], record["pixel_std_01"])
    prediction = prediction_float.clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
    gt = gt_tensor.permute(0, 2, 3, 1).numpy()
    directory = case_directory(output_dir, record)
    directory.mkdir(parents=True)
    save_images(directory / "gt_tactile_native", tactile, indices)
    save_images(directory / "gt_tactile", gt, indices)
    save_images(directory / "ours_tactile", prediction, indices)
    if record["figure_tactile_crop"] is not None:
        save_images(directory / "gt_tactile_figure", crop_rgb(gt, record["figure_tactile_crop"]), indices)
        save_images(
            directory / "ours_tactile_figure", crop_rgb(prediction, record["figure_tactile_crop"]), indices
        )
    synchronization = {record["tactile_key"]: tactile_sync}
    rgbs = {}
    import matplotlib.pyplot as plt

    keys = rgb_keys(info, record["tactile_key"])
    if figure_rgb_key:
        if figure_rgb_key not in keys:
            raise ValueError(f"Figure RGB key must be one of {keys}, got {figure_rgb_key}.")
        keys = [figure_rgb_key, *[key for key in keys if key != figure_rgb_key]]
    for index, key in enumerate(keys):
        rgb, sync = read_frames(root, info, episode, key, indices, episode_times=times)
        rgbs[key] = rgb
        synchronization[key] = sync
        save_images(directory / ("rgb" if index == 0 else f"rgb_{slug(key)}"), rgb, indices)
        if index == 0 and rgb_crop is not None:
            save_images(directory / "rgb_figure", crop_rgb(rgb, rgb_crop), indices)
        figure_record = record if index == 0 else {**record, "figure_rgb_crop": None}
        figure = make_figure(figure_record, rgb, gt, prediction)
        suffix = "" if index == 0 else f"_{slug(key)}"
        figure.savefig(directory / f"comparison{suffix}.pdf", bbox_inches="tight", pad_inches=0.025)
        figure.savefig(directory / f"comparison{suffix}.png", dpi=220, bbox_inches="tight", pad_inches=0.025)
        plt.close(figure)
        if index == 0 and record.get("frame_phases"):
            figure = make_figure(record, rgb, gt, prediction, native_gt=tactile)
            figure.savefig(directory / "contact_detail.pdf", bbox_inches="tight", pad_inches=0.025)
            figure.savefig(directory / "contact_detail.png", dpi=220, bbox_inches="tight", pad_inches=0.025)
            plt.close(figure)
    np.savez_compressed(
        directory / "frames.npz",
        gt_tactile=gt,
        gt_tactile_native=tactile,
        ours_tactile=prediction,
        ours_unclipped=prediction_float.numpy(),
        **rgbs,
    )
    metadata = {
        **record,
        "codec": codec.provenance,
        "metrics": reconstruction_metrics(gt_tensor, prediction_float),
        "synchronization": synchronization,
        "dataset_synchronization": data_sync,
        "figure_rgb_key": next(iter(rgbs)),
        "display": "GT and Ours use the same fixed RGB range; no per-frame contrast adjustment.",
        "interpretation": "Hand-selected tactile autoencoder reconstruction; not future tactile prediction, calibrated pressure, or a dataset-wide benchmark.",
    }
    write_json(directory / "metadata.json", metadata)
    LOGGER.info("Saved %s", directory)
    return directory


def search(args, codec):
    datasets = dict(pair.split("=", 1) for pair in args.dataset) if args.dataset else DEFAULT_DATASETS
    records, errors, scope = [], [], []
    for name, directory in datasets.items():
        root = Path(directory)
        info = json.loads((root / "meta/info.json").read_text())
        keys = sorted(
            key for key, value in info["features"].items() if "tactile" in key and value["dtype"] == "video"
        )
        episodes = episode_records(
            root, re.compile(args.task_pattern, re.I), args.episodes_per_task, args.max_episodes
        )
        scope.append(
            {
                "dataset": name,
                "episodes": [episode["episode_index"] for episode in episodes],
                "tactile_keys": keys,
            }
        )
        for episode in episodes:
            LOGGER.info("Scanning %s episode %s: %s", name, episode["episode_index"], episode["tasks"])
            sampled = list(range(0, episode["length"], args.scan_stride))
            for key in keys:
                try:
                    frames, _ = read_frames(
                        root, info, episode, key, sampled, size=codec.provenance["image_size"]
                    )
                    images = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2)
                    windows = (
                        pre_post_windows(images, sampled, args.gaps)
                        if args.selection_mode == "pre-post"
                        else onset_windows(contact_curve(images), sampled, args.gaps)
                    )
                    if not windows:
                        continue
                    mean, std = pixel_statistics(name, keys, key)
                    best = None
                    for window in windows[: args.windows_per_pad]:
                        gt = images[window["sample_positions"]]
                        prediction = codec.reconstruct(gt, mean, std)
                        metrics = reconstruction_metrics(gt, prediction)
                        score = (
                            metrics["mean_psnr_db"]
                            + 10 * metrics["mean_ssim"]
                            + 5 * metrics["contact_delta_cosine"]
                        )
                        record = {
                            "candidate_id": f"{name}_ep{episode['episode_index']:06d}_{slug(key)}_f{window['frame_indices'][0]}_g{window['frame_gap']}",
                            "dataset": name,
                            "dataset_root": str(root.resolve()),
                            "task": "; ".join(episode["tasks"]),
                            "episode": episode,
                            "tactile_key": key,
                            "fps": info["fps"],
                            "selection_mode": args.selection_mode,
                            "pixel_mean_01": mean.tolist(),
                            "pixel_std_01": std.tolist(),
                            **window,
                            "metrics": metrics,
                            "quality_score": score,
                            "contact_proxy_not_calibrated_force": True,
                        }
                        if best is None or candidate_rank(record) > candidate_rank(best):
                            best = record
                    records.append(best)
                except (OSError, ValueError, RuntimeError) as error:
                    LOGGER.warning("%s episode %s %s: %s", name, episode["episode_index"], key, error)
                    errors.append(
                        {
                            "dataset": name,
                            "episode": episode["episode_index"],
                            "key": key,
                            "error": str(error),
                        }
                    )
    records.sort(key=candidate_rank, reverse=True)
    manifest = {
        "codec": codec.provenance,
        "scope": scope,
        "search_arguments": vars(args)
        | {"checkpoint": str(args.checkpoint), "output_dir": str(args.output_dir)},
        "selection_bias": "Candidates explicitly favor contact onset and reconstruction quality; no dataset-wide performance claim.",
        "candidates": records,
        "errors": errors,
    }
    write_json(args.output_dir / "candidates.json", manifest)
    used = set()
    for record in records:
        identity = (record["dataset"], record["episode"]["episode_index"])
        if identity in used:
            continue
        export_case(
            record, codec, args.output_dir / "previews", args.figure_rgb_key, args.rgb_crop, args.tactile_crop
        )
        used.add(identity)
        if len(used) >= args.previews:
            break
    LOGGER.info("Scored %d candidates, exported %d distinct-episode previews", len(records), len(used))


def export(args, codec):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    manifest = json.loads(args.manifest.read_text())
    if manifest["codec"]["codec_sha256"] != codec.provenance["codec_sha256"]:
        raise ValueError("The search and export checkpoints have different tactile codec weights.")
    if manifest["codec"]["image_size"] != codec.provenance["image_size"]:
        raise ValueError("Search and export must use the same tactile image size.")
    by_id = {record["candidate_id"]: record for record in manifest["candidates"]}
    selected = [by_id[identity] for identity in args.candidate_ids]
    identities = [(record["dataset"], record["episode"]["episode_index"]) for record in selected]
    if len(set(identities)) != len(identities):
        raise ValueError("Choose only one tactile pad/window per episode.")
    directories = [
        export_case(record, codec, args.output_dir, args.figure_rgb_key, args.rgb_crop, args.tactile_crop)
        for record in selected
    ]
    with PdfPages(args.output_dir / "selected_examples.pdf") as pdf:
        for directory in directories:
            record = json.loads((directory / "metadata.json").read_text())
            with np.load(directory / "frames.npz", allow_pickle=False) as frames:
                figure = make_figure(
                    record, frames[record["figure_rgb_key"]], frames["gt_tactile"], frames["ours_tactile"]
                )
            pdf.savefig(figure, bbox_inches="tight", pad_inches=0.025)
            plt.close(figure)
    write_json(
        args.output_dir / "selection.json",
        {
            "source_manifest": str(args.manifest.resolve()),
            "candidate_ids": args.candidate_ids,
            "example_directories": [str(path) for path in directories],
            "codec": codec.provenance,
            "interpretation": manifest["selection_bias"],
        },
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Trusted Stage-1 DeepSpeed checkpoint."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--figure-rgb-key", help="Prefer a particular exported RGB view for the main figure.")
    parser.add_argument(
        "--rgb-crop",
        type=float,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Fixed normalized crop for main-figure RGB only; raw frames are preserved.",
    )
    parser.add_argument(
        "--tactile-crop",
        type=float,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Identical fixed model-space ROI for GT/Ours figures only; full images and metrics are preserved.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("search")
    scan.add_argument("--selection-mode", choices=["progressive", "pre-post"], default="progressive")
    scan.add_argument("--dataset", action="append", default=[], metavar="REGISTRY_NAME=ROOT")
    scan.add_argument(
        "--task-pattern", default="rubber|fragile|paper cup|fruit|press|trash|shampoo|plates|balloon"
    )
    scan.add_argument("--episodes-per-task", type=int, default=2)
    scan.add_argument("--max-episodes", type=int, default=24, help="Maximum per dataset.")
    scan.add_argument("--scan-stride", type=int, default=3)
    scan.add_argument("--gaps", type=int, nargs="+", default=[3, 6, 9, 12])
    scan.add_argument("--windows-per-pad", type=int, default=3)
    scan.add_argument("--previews", type=int, default=20)
    selection = commands.add_parser("export")
    selection.add_argument("--manifest", type=Path, required=True)
    selection.add_argument("--candidate-ids", nargs="+", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.image_size < 32 or args.threads < 1:
        raise ValueError("Image size must be at least 32 and threads must be positive.")
    if args.command == "search" and (
        args.scan_stride < 1
        or any(gap < 1 or gap % args.scan_stride for gap in args.gaps)
        or min(args.episodes_per_task, args.max_episodes, args.windows_per_pad, args.previews) < 1
    ):
        raise ValueError("Search counts must be positive; gaps must be multiples of scan-stride.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    codec = TactileCodec(args.checkpoint, args.image_size, args.device)
    {"search": search, "export": export}[args.command](args, codec)


if __name__ == "__main__":
    main()
