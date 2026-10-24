"""Export complete tasks as a reindexed, standalone LeRobot v3 RoboDojo subset."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from lerobot.evaluation.scripts.robodojo_dataset_utils import (
    DEFAULT_DATASET,
    V30Source,
    copy_video,
    matrix,
    new_dataset,
    numeric_stats,
    put_episode_stats,
    relative_path,
    validate_destination,
    write_episodes,
    write_json,
)

logger = logging.getLogger(__name__)
DEFAULT_ROBODOJO = Path("/data2/guoqi/robodojo_pi05_repro")


def task_aliases(source: V30Source, robodojo_root: Path | None) -> dict[str, int]:
    instructions = {text.casefold(): index for index, text in source.tasks.items()}
    aliases = dict(instructions)
    if robodojo_root is None or not robodojo_root.is_dir():
        return aliases
    # Read literal instruction templates without importing simulator/task code.
    for path in sorted((robodojo_root / "task/RoboDojo/tasks").glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        ids = {
            instructions[value.value.casefold()]
            for method in ast.walk(tree)
            if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
            and method.name == "gen_instruction"
            for value in ast.walk(method)
            if isinstance(value, ast.Constant)
            and isinstance(value.value, str)
            and value.value.casefold() in instructions
        }
        if len(ids) == 1:
            aliases[path.stem.casefold()] = ids.pop()
    return aliases


def resolve_tasks(source: V30Source, requested: list[str], robodojo_root: Path | None) -> set[int]:
    aliases = task_aliases(source, robodojo_root)
    selected = set()
    for value in requested:
        if value.isdecimal() and int(value) in source.tasks:
            selected.add(int(value))
        elif value.casefold() in aliases:
            selected.add(aliases[value.casefold()])
        else:
            raise ValueError(
                f"Unknown task {value!r}. Use --list-tasks, exact quoted instructions or numeric task IDs; "
                "--robodojo-root enables task-name aliases from literal instruction templates."
            )
    if not selected:
        raise ValueError("At least one task must be selected.")
    return selected


def subset(
    source_root: Path,
    destination: Path,
    requested: list[str],
    *,
    robodojo_root: Path | None = DEFAULT_ROBODOJO,
    video_mode: str = "copy",
) -> dict:
    from lerobot.common.datasets_v30.compute_stats import aggregate_feature_stats

    source = V30Source(source_root)
    destination = validate_destination(source.root, destination)
    selected = resolve_tasks(source, requested, robodojo_root)
    instructions = {source.tasks[index] for index in selected}
    episodes = [copy.deepcopy(row) for row in source.episodes if instructions.intersection(row["tasks"])]
    if not episodes:
        raise ValueError("No complete episodes matched the requested tasks.")
    ids = [row["episode_index"] for row in episodes]
    data = pq.read_table(source.files, filters=[("episode_index", "in", ids)])
    if data.num_rows != sum(row["length"] for row in episodes):
        raise ValueError("Selected episode lengths disagree with physical parquet rows.")
    retained_tasks = sorted(set(matrix(data["task_index"], 1)[:, 0].tolist()))
    if set(retained_tasks) != selected:
        raise ValueError(
            "Selected episodes contain missing or additional tasks; mixed-task episodes are unsupported."
        )
    task_map = {old: new for new, old in enumerate(retained_tasks)}
    tasks = source.tasks_table.filter(pc.is_in(source.tasks_table["task_index"], pa.array(retained_tasks)))
    tasks = tasks.set_column(
        tasks.schema.get_field_index("task_index"),
        "task_index",
        pa.array([task_map[value] for value in tasks["task_index"].to_pylist()], type=pa.int64()),
    )
    info = copy.deepcopy(source.info)
    info.update(
        total_episodes=len(episodes),
        total_frames=data.num_rows,
        total_tasks=len(retained_tasks),
        splits={"train": f"0:{len(episodes)}"},
    )
    numeric_features = [
        key
        for key, feature in info["features"].items()
        if feature["dtype"] != "video" and key in data.column_names
    ]
    image_stats = {}
    for key in source.videos:
        rows = [
            {
                name.removeprefix(f"stats/{key}/"): np.asarray(value)
                for name, value in episode.items()
                if name.startswith(f"stats/{key}/")
            }
            for episode in episodes
        ]
        if any(not row for row in rows):
            raise ValueError(f"Missing per-episode image statistics for {key}.")
        image_stats[key] = {name: value.tolist() for name, value in aggregate_feature_stats(rows).items()}
    episode_map = []
    with new_dataset(source.root, destination) as output:
        offset, file_number, file_bytes = 0, 0, 0
        buffered, completed, copied_videos = [], [], set()
        for new_index, episode in enumerate(episodes):
            original_index = episode["episode_index"]
            length = episode["length"]
            table = data.slice(offset, length)
            source.validate_rows(table, episode["dataset_from_index"])
            table = table.set_column(
                table.schema.get_field_index("index"),
                "index",
                pa.array(np.arange(offset, offset + length), type=pa.int64()),
            )
            table = table.set_column(
                table.schema.get_field_index("episode_index"),
                "episode_index",
                pa.array(np.full(length, new_index), type=pa.int64()),
            )
            table = table.set_column(
                table.schema.get_field_index("task_index"),
                "task_index",
                pa.array([task_map[value] for value in table["task_index"].to_pylist()], type=pa.int64()),
            )
            if buffered and file_bytes + table.nbytes > source.info["data_files_size_in_mb"] * 1024**2:
                target = output / relative_path(
                    info["data_path"],
                    chunk_index=file_number // info["chunks_size"],
                    file_index=file_number % info["chunks_size"],
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.concat_tables(buffered), target, compression="zstd")
                completed.extend(buffered)
                buffered, file_bytes = [], 0
                file_number += 1
            buffered.append(table)
            file_bytes += table.nbytes
            episode_map.append({"source": original_index, "subset": new_index})
            episode.update(
                episode_index=new_index,
                dataset_from_index=offset,
                dataset_to_index=offset + length,
            )
            episode["data/chunk_index"] = file_number // info["chunks_size"]
            episode["data/file_index"] = file_number % info["chunks_size"]
            episode["meta/episodes/chunk_index"] = 0
            episode["meta/episodes/file_index"] = 0
            for key in numeric_features:
                put_episode_stats(episode, key, numeric_stats(matrix(table[key])))
            for key in source.videos:
                relative = source.video_relative(source.episodes[original_index], key)
                if relative not in copied_videos:
                    copy_video(source.root / relative, output / relative, video_mode)
                    copied_videos.add(relative)
            offset += length
        target = output / relative_path(
            info["data_path"],
            chunk_index=file_number // info["chunks_size"],
            file_index=file_number % info["chunks_size"],
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.concat_tables(buffered), target, compression="zstd")
        completed.extend(buffered)
        combined = pa.concat_tables(completed)
        stats = {key: numeric_stats(matrix(combined[key])) for key in numeric_features}
        stats.update(image_stats)
        write_episodes(output, episodes, source.episode_schema)
        pq.write_table(tasks, output / "meta/tasks.parquet")
        manifest = {
            "source": str(source.root),
            "requested_tasks": requested,
            "task_map": {str(old): new for old, new in task_map.items()},
            "episode_map": episode_map,
            "total_frames": offset,
            "total_episodes": len(episodes),
            "video_mode": video_mode,
            "video_files": len(copied_videos),
            "source_video_interval_warnings": source.video_warnings,
            "video_boundaries": "original encoded files and timestamps preserved; unused video frames may remain",
            "image_statistics": "aggregated from selected source episode statistics, without video re-encoding",
            "numeric_quantiles": "exact over selected rows",
            "complete": True,
        }
        if (source.root / "meta/delta_action_manifest.json").is_file():
            write_json(
                output / "meta/delta_action_manifest.json",
                {
                    "source_manifest": json.loads(
                        (source.root / "meta/delta_action_manifest.json").read_text()
                    ),
                    "subset_manifest": "subset_manifest.json",
                    "complete": True,
                },
            )
        write_json(output / "meta/stats.json", stats)
        write_json(output / "meta/subset_manifest.json", manifest)
        write_json(output / "meta/info.json", info)
    logger.info("Exported %d episodes, %d frames, %d video files", len(episodes), offset, len(copied_videos))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--task", "--tasks", dest="tasks", nargs="+")
    parser.add_argument("--robodojo-root", type=Path, default=DEFAULT_ROBODOJO)
    parser.add_argument("--video-mode", choices=("copy", "symlink"), default="copy")
    parser.add_argument("--list-tasks", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.list_tasks:
        source = V30Source(args.source)
        aliases = task_aliases(source, args.robodojo_root)
        for index, text in sorted(source.tasks.items()):
            names = [
                name for name, task_id in aliases.items() if task_id == index and name != text.casefold()
            ]
            print(f"{index}\t{','.join(names)}\t{text}")
        return
    if not args.tasks or args.output is None:
        parser.error("--tasks and --output are required unless --list-tasks is used.")
    subset(
        args.source,
        args.output,
        args.tasks,
        robodojo_root=args.robodojo_root,
        video_mode=args.video_mode,
    )


if __name__ == "__main__":
    main()
