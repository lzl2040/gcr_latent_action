#!/usr/bin/env python
"""Export synchronized RGB contact sheets for manual object-deformation screening.

These are candidates, not detected deformation or calibrated force. Inspect object
boundaries rather than relying on tactile-patch growth, motion, or hand occlusion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def evenly_spaced_episodes(episodes, count):
    if count < 1:
        raise ValueError("Episode count must be positive.")
    if len(episodes) <= count:
        return episodes
    positions = np.linspace(0, len(episodes) - 1, count).round().astype(int)
    return [episodes[position] for position in positions]


def sample_frames(length, fractions):
    if length < 1 or not fractions or any(not 0 <= fraction < 1 for fraction in fractions):
        raise ValueError("Need a nonempty episode and fractions in [0,1).")
    return sorted({min(length - 1, round(fraction * (length - 1))) for fraction in fractions})


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default="/Data/lerobot_data_ort6d/v30/FTP-1/sharpa")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task-pattern", default="balloon")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--episode-ids", nargs="+", type=int)
    parser.add_argument("--exclude-episodes", nargs="*", type=int, default=[])
    parser.add_argument("--rgb-key", default="observation.images.ego")
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0, 0.025, 0.05, 0.08, 0.12, 0.17, 0.23, 0.32, 0.45, 0.6, 0.75, 0.9],
    )
    parser.add_argument("--rows-per-page", type=int, default=6)
    parser.add_argument("--rgb-crop", nargs=4, type=float, default=[0, 0.15, 1, 1])
    return parser


def main(argv=None):
    from scripts import visualize_stage1_tactile_contact as contact

    args = build_parser().parse_args(argv)
    if args.rows_per_page < 1:
        raise ValueError("Rows per page must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    info = json.loads((args.dataset_root / "meta/info.json").read_text())
    episodes = contact.episode_records(
        args.dataset_root, re.compile(args.task_pattern, re.I), 100_000, 100_000
    )
    episodes.sort(key=lambda episode: episode["episode_index"])
    excluded = set(args.exclude_episodes)
    episodes = [episode for episode in episodes if episode["episode_index"] not in excluded]
    if args.episode_ids:
        requested = set(args.episode_ids)
        selected = [episode for episode in episodes if episode["episode_index"] in requested]
        if {episode["episode_index"] for episode in selected} != requested:
            raise ValueError("Requested episode IDs do not all match the task and exclusion filters.")
    else:
        selected = evenly_spaced_episodes(episodes, args.episodes)
    if not selected:
        raise ValueError("No matching episodes.")
    contact.write_json(
        args.output_dir / "matching_episodes.json",
        {"dataset_root": str(args.dataset_root.resolve()), "episodes": episodes},
    )
    records, previews, errors = [], [], []
    for episode in selected:
        indices = sample_frames(episode["length"], args.fractions)
        try:
            data_sync = contact.episode_frame_times(args.dataset_root, info, episode, indices)
            images, sync = contact.read_frames(
                args.dataset_root,
                info,
                episode,
                args.rgb_key,
                indices,
                episode_times=data_sync["timestamp_s"],
            )
            filename = f"episode_{episode['episode_index']:06d}.npz"
            np.savez_compressed(
                args.output_dir / filename,
                rgb_original=images,
                frame_indices=indices,
                timestamp_s=data_sync["timestamp_s"],
            )
            records.append(
                {
                    "episode": episode,
                    "task": "; ".join(episode["tasks"]),
                    "rgb_key": args.rgb_key,
                    "frames_file": filename,
                    "rgb_frames_sha256": hashlib.sha256(images.tobytes()).hexdigest(),
                    "frame_indices": indices,
                    "dataset_synchronization": data_sync,
                    "synchronization": sync,
                }
            )
            previews.append(contact.crop_rgb(images, args.rgb_crop))
            print(f"Scouted ep{episode['episode_index']}: {len(indices)} source RGB frames", flush=True)
        except (ValueError, RuntimeError, OSError) as error:
            errors.append({"episode_index": episode["episode_index"], "error": str(error)})
            print(f"Skipped ep{episode['episode_index']}: {error}", flush=True)
    if not records:
        raise ValueError("No episode could be decoded.")
    font = ImageFont.truetype("DejaVuSans.ttf", 13)
    tile_width, label_width, title_height, row_height = 168, 96, 38, 174
    columns = max(len(record["frame_indices"]) for record in records)
    for start in range(0, len(records), args.rows_per_page):
        batch = list(
            zip(
                records[start : start + args.rows_per_page],
                previews[start : start + args.rows_per_page],
                strict=True,
            )
        )
        canvas = Image.new(
            "RGB", (label_width + columns * tile_width, title_height + len(batch) * row_height), "white"
        )
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (8, 8),
            f"SHARPA RGB-only deformation screening | {args.rgb_key} | "
            f"fixed crop={args.rgb_crop} | not a deformation/force detector",
            font=font,
            fill="black",
        )
        for row, (record, images) in enumerate(batch):
            top = title_height + row * row_height
            draw.text(
                (4, top + 50),
                f"ep{record['episode']['episode_index']}\nN={record['episode']['length']}",
                font=font,
                fill="black",
            )
            for column, (image, index, seconds) in enumerate(
                zip(
                    images,
                    record["frame_indices"],
                    record["dataset_synchronization"]["timestamp_s"],
                    strict=True,
                )
            ):
                left = label_width + column * tile_width
                draw.text((left + 2, top), f"f{index} | {seconds:.2f}s", font=font, fill="black")
                tile = Image.fromarray(image)
                tile.thumbnail((tile_width - 3, row_height - 22), Image.Resampling.LANCZOS)
                canvas.paste(tile, (left, top + 20))
        canvas.save(args.output_dir / f"rgb_page_{start // args.rows_per_page + 1:03d}.jpg", quality=94)
    contact.write_json(
        args.output_dir / "rgb_scout.json",
        {
            "dataset_root": str(args.dataset_root.resolve()),
            "matching_episode_count": len(episodes),
            "rgb_key": args.rgb_key,
            "display_crop": args.rgb_crop,
            "sampling_fractions": args.fractions,
            "episodes": records,
            "errors": errors,
            "interpretation": "RGB-only manual screening candidates. Motion, viewpoint, size and hand occlusion "
            "can change the silhouette without deforming the object; inspect before accepting.",
        },
    )


if __name__ == "__main__":
    main()
