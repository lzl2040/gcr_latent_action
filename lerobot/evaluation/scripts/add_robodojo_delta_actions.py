"""Export RoboDojo v3 with X-WAM-style measured-state increments in two new columns."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lerobot.evaluation.scripts.robodojo_dataset_utils import (
    DEFAULT_DATASET,
    V30Source,
    copy_video,
    matrix,
    new_dataset,
    numeric_stats,
    put_episode_stats,
    state_increments,
    validate_destination,
    vector_array,
    write_episodes,
    write_json,
)

logger = logging.getLogger(__name__)
DELTA_COLUMNS = ("delta_action", "delta_action.ee_ort6d_pos")
DELTA_CONTRACT = {
    "version": 1,
    "reference": "next_measured_state",
    "translation": "p_state[t+1] - p_state[t], world axes",
    "rotation": "R_state[t+1] @ R_state[t].T; left-multiplied on integration",
    "gripper": "g_state[t+1] - g_state[t]",
    "terminal": "zero translation, identity rotation, zero gripper increment",
    "quaternion_order": "wxyz; canonical nonnegative w",
    "ort6d_order": ["R00", "R01", "R10", "R11", "R20", "R21"],
    "normalization": "new columns have their own unnormalized statistics",
    "absolute_columns_preserved": True,
}


def convert(source_root: Path, destination: Path, video_mode: str = "symlink") -> dict:
    source = V30Source(source_root)
    destination = validate_destination(source.root, destination)
    if any(key in source.info["features"] for key in DELTA_COLUMNS):
        raise ValueError("Source already declares delta columns; refusing to overwrite them.")
    for path in source.files:
        if set(DELTA_COLUMNS) & set(pq.read_schema(path).names):
            raise ValueError(f"Source already has delta columns: {path}")
    states_table = pq.read_table(source.files, columns=["observation.state"])
    states = matrix(states_table["observation.state"], 16)
    if len(states) != source.info["total_frames"]:
        raise ValueError("State row count differs from info.json.")
    delta, delta_ort6d = state_increments(states, source.ends)
    arrays = dict(zip(DELTA_COLUMNS, (delta, delta_ort6d), strict=True))
    info = copy.deepcopy(source.info)
    for name, original in zip(DELTA_COLUMNS, ("action", "action.ee_ort6d_pos"), strict=True):
        info["features"][name] = copy.deepcopy(info["features"][original])
        width = len(arrays[name][0])
        info["features"][name]["names"] = [
            f"{arm}_{component}"
            for arm in ("left", "right")
            for component in (
                ["delta_x", "delta_y", "delta_z"]
                + (
                    ["delta_qw", "delta_qx", "delta_qy", "delta_qz"]
                    if width == 16
                    else ["delta_R00", "delta_R01", "delta_R10", "delta_R11", "delta_R20", "delta_R21"]
                )
                + ["delta_gripper"]
            )
        ]
    info["delta_action_contract"] = DELTA_CONTRACT
    stats = json.loads((source.root / "meta/stats.json").read_text())
    for name, values in arrays.items():
        stats[name] = numeric_stats(values)
    episodes = copy.deepcopy(source.episodes)
    videos = set()
    for episode in episodes:
        videos.update(source.video_relative(episode, key) for key in source.videos)
        for name, values in arrays.items():
            put_episode_stats(
                episode,
                name,
                numeric_stats(values[episode["dataset_from_index"] : episode["dataset_to_index"]]),
            )
        episode["meta/episodes/chunk_index"] = 0
        episode["meta/episodes/file_index"] = 0
    with new_dataset(source.root, destination) as output:
        total = 0
        for path in source.files:
            table = pq.read_table(path)
            source.validate_rows(table, total)
            if not np.array_equal(
                matrix(table["observation.state"], 16), states[total : total + table.num_rows]
            ):
                raise ValueError("Source states changed while converting.")
            for name, values in arrays.items():
                table = table.append_column(name, vector_array(values[total : total + table.num_rows]))
            target = output / path.relative_to(source.root)
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, target, compression="zstd")
            total += table.num_rows
            logger.info("Wrote %s (%d rows)", target.name, table.num_rows)
        if total != len(states):
            raise ValueError("Source changed while converting: inconsistent row counts.")
        write_episodes(output, episodes, source.episode_schema)
        shutil.copy2(source.root / "meta/tasks.parquet", output / "meta/tasks.parquet")
        for video in sorted(videos):
            copy_video(source.root / video, output / video, video_mode)
        manifest = {
            "source": str(source.root),
            "total_frames": total,
            "total_episodes": len(episodes),
            "new_columns": list(DELTA_COLUMNS),
            "video_mode": video_mode,
            "video_files": len(videos),
            "source_video_interval_warnings": source.video_warnings,
            "contract": DELTA_CONTRACT,
            "complete": True,
        }
        write_json(output / "meta/stats.json", stats)
        write_json(output / "meta/delta_action_manifest.json", manifest)
        write_json(output / "meta/info.json", info)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-mode", choices=("symlink", "copy"), default="symlink")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    convert(args.source, args.output, args.video_mode)


if __name__ == "__main__":
    main()
