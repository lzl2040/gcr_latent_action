"""Shared I/O and geometry for non-destructive RoboDojo v3 dataset tools."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

DEFAULT_DATASET = Path(
    "/data2/lizuolei/data/lerobot_data/simulation_data/RoboDojo/data/RoboDojo_ee_lerobot_v30_video"
)
POSE_FEATURES = {
    "action": "action.ee_ort6d_pos",
    "observation.state": "observation.ee_ort6d_pos",
}
QUANTILES = {"q01": 0.01, "q10": 0.1, "q50": 0.5, "q90": 0.9, "q99": 0.99}
logger = logging.getLogger(__name__)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def relative_path(template: str, **values) -> Path:
    path = Path(template.format(**values))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Dataset path must be relative without '..': {path}")
    return path


def validate_destination(source: Path, destination: Path) -> Path:
    source, destination = source.resolve(), destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}.")
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Source and destination must not contain one another.")
    return destination


@contextmanager
def new_dataset(source: Path, destination: Path):
    destination = validate_destination(source, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent))
    logger.info("Building in %s; failures leave this directory for inspection.", staging)
    yield staging
    if destination.exists():
        raise FileExistsError(f"Destination appeared while building: {destination}.")
    staging.rename(destination)
    logger.info("Completed dataset: %s", destination)


def matrix(column: pa.ChunkedArray | pa.Array, width: int | None = None) -> np.ndarray:
    array = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    if array.null_count:
        raise ValueError("Dataset numeric columns must not contain null rows.")
    if pa.types.is_fixed_size_list(array.type):
        if array.values.null_count:
            raise ValueError("Dataset vectors must not contain null elements.")
        result = array.flatten().to_numpy(zero_copy_only=False).reshape(len(array), array.type.list_size)
    elif pa.types.is_integer(array.type) or pa.types.is_floating(array.type):
        result = array.to_numpy(zero_copy_only=False).reshape(-1, 1)
    else:
        raise ValueError(f"Unsupported numeric column type: {array.type}.")
    if width is not None and result.shape[1] != width:
        raise ValueError(f"Expected width {width}, got {result.shape[1]}.")
    return result


def vector_array(values: np.ndarray) -> pa.FixedSizeListArray:
    values = np.asarray(values, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), values.shape[1])


def numeric_stats(values: np.ndarray) -> dict:
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Cannot compute statistics for empty or nonfinite values.")
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0, dtype=np.float64).tolist(),
        "std": values.std(axis=0, dtype=np.float64).tolist(),
        "count": [len(values)],
        **{
            key: value.tolist()
            for key, value in zip(
                QUANTILES, np.quantile(values, list(QUANTILES.values()), axis=0), strict=True
            )
        },
    }


def put_episode_stats(episode: dict, feature: str, stats: dict) -> None:
    for name, value in stats.items():
        episode[f"stats/{feature}/{name}"] = value


def write_episodes(root: Path, rows: list[dict], original_schema: pa.Schema) -> None:
    names = list(original_schema.names)
    names.extend(sorted({name for row in rows for name in row} - set(names)))
    arrays = [
        pa.array(
            [row.get(name) for row in rows],
            type=original_schema.field(name).type if name in original_schema.names else None,
        )
        for name in names
    ]
    path = root / "meta/episodes/chunk-000/file-000.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_arrays(arrays, names=names), path)


def pose_rotations(values: np.ndarray, offset: int) -> Rotation:
    quaternion = values[:, offset + 3 : offset + 7].astype(np.float64)
    if not np.isfinite(quaternion).all() or np.any(np.linalg.norm(quaternion, axis=1) < 1e-8):
        raise ValueError("Nonfinite or zero-norm source quaternion.")
    return Rotation.from_quat(quaternion[:, [1, 2, 3, 0]])


def pose_to_ort6d(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] != 16 or not np.isfinite(values).all():
        raise ValueError("Expected finite dual-arm xyz/wxyz/gripper values with shape (N,16).")
    output = np.empty((len(values), 20), dtype=np.float32)
    for old, new in ((0, 0), (8, 10)):
        output[:, new : new + 3] = values[:, old : old + 3]
        output[:, new + 3 : new + 9] = pose_rotations(values, old).as_matrix()[:, :, :2].reshape(-1, 6)
        output[:, new + 9] = values[:, old + 7]
    return output


def state_increments(states: np.ndarray, episode_ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """X-WAM RoboDojo semantics: next measured state minus current, world-axis/left rotation."""
    if states.ndim != 2 or states.shape[1] != 16 or not np.isfinite(states).all():
        raise ValueError("Expected finite (N,16) states.")
    if (
        episode_ends.ndim != 1
        or not np.issubdtype(episode_ends.dtype, np.integer)
        or not len(episode_ends)
        or episode_ends[0] <= 0
        or episode_ends[-1] != len(states)
        or np.any(np.diff(episode_ends) <= 0)
    ):
        raise ValueError("Episode ends must strictly partition all states.")
    following = np.arange(1, len(states) + 1)
    following[episode_ends - 1] = episode_ends - 1
    output = np.empty_like(states, dtype=np.float32)
    for offset in (0, 8):
        output[:, offset : offset + 3] = (
            states[following, offset : offset + 3].astype(np.float64) - states[:, offset : offset + 3]
        )
        rotations = pose_rotations(states, offset)
        delta = rotations[following] * rotations.inv()
        output[:, offset + 3 : offset + 7] = delta.as_quat(canonical=True)[:, [3, 0, 1, 2]]
        output[:, offset + 7] = states[following, offset + 7].astype(np.float64) - states[:, offset + 7]
    output[episode_ends - 1] = np.array([0, 0, 0, 1, 0, 0, 0, 0] * 2, dtype=np.float32)
    return output, pose_to_ort6d(output)


def copy_video(source: Path, destination: Path, mode: str) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Missing source video: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source.resolve())
    else:
        raise ValueError(f"Unsupported video mode: {mode}")


class V30Source:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.info = json.loads((self.root / "meta/info.json").read_text())
        if self.info["codebase_version"] != "v3.0":
            raise ValueError("These tools require a LeRobot v3.0 dataset.")
        self.files = sorted((self.root / "data").glob("chunk-*/file-*.parquet"))
        episode_files = sorted((self.root / "meta/episodes").glob("chunk-*/file-*.parquet"))
        if not self.files or not episode_files:
            raise ValueError("Dataset data/episode parquet files are missing.")
        episodes = pq.read_table(episode_files)
        self.episode_schema = episodes.schema
        self.episodes = sorted(episodes.to_pylist(), key=lambda row: row["episode_index"])
        if not self.episodes:
            raise ValueError("Dataset must contain at least one episode.")
        self.ends = np.array([row["dataset_to_index"] for row in self.episodes], dtype=np.int64)
        expected_start = 0
        for index, row in enumerate(self.episodes):
            if (
                row["episode_index"] != index
                or row["dataset_from_index"] != expected_start
                or row["length"] <= 0
                or row["dataset_to_index"] != expected_start + row["length"]
            ):
                raise ValueError(f"Inconsistent episode metadata at episode {index}.")
            expected_start = row["dataset_to_index"]
        if len(self.episodes) != self.info["total_episodes"] or expected_start != self.info["total_frames"]:
            raise ValueError("Episode metadata and info.json counts disagree.")
        self.tasks_table = pq.read_table(self.root / "meta/tasks.parquet")
        text_fields = [
            field.name
            for field in self.tasks_table.schema
            if pa.types.is_string(field.type) or pa.types.is_large_string(field.type)
        ]
        if len(text_fields) != 1:
            raise ValueError("tasks.parquet must contain one instruction/index string column.")
        self.task_text_column = text_fields[0]
        self.tasks = {row["task_index"]: row[self.task_text_column] for row in self.tasks_table.to_pylist()}
        if sorted(self.tasks) != list(range(self.info["total_tasks"])):
            raise ValueError("Task indices and info.json disagree.")
        self.videos = [key for key, feature in self.info["features"].items() if feature["dtype"] == "video"]
        self.video_warnings = []

    def video_relative(self, episode: dict, key: str) -> Path:
        prefix = f"videos/{key}"
        start, end = (episode[f"{prefix}/{suffix}_timestamp"] for suffix in ("from", "to"))
        if not np.isfinite([start, end]).all() or start < 0 or end <= start:
            raise ValueError(f"Invalid video interval for episode {episode['episode_index']}, {key}.")
        interval_frames = (end - start) * self.info["fps"]
        if abs(interval_frames - episode["length"]) > 1e-4:
            warning = {
                "source_episode_index": episode["episode_index"],
                "video_key": key,
                "data_frames": episode["length"],
                "interval_frames": interval_frames,
            }
            if warning not in self.video_warnings:
                self.video_warnings.append(warning)
                logger.warning("Preserving mismatched source video interval: %s", warning)
        return relative_path(
            self.info["video_path"],
            video_key=key,
            chunk_index=episode[f"{prefix}/chunk_index"],
            file_index=episode[f"{prefix}/file_index"],
        )

    def validate_rows(self, table: pa.Table, start: int) -> None:
        indices = np.arange(start, start + table.num_rows)
        if len(indices) and indices[-1] >= self.info["total_frames"]:
            raise ValueError("Parquet rows exceed the declared frame count.")
        episode_ids = np.searchsorted(self.ends, indices, side="right")
        starts = np.r_[0, self.ends[:-1]]
        for key, expected in (
            ("index", indices),
            ("episode_index", episode_ids),
            ("frame_index", indices - starts[episode_ids]),
        ):
            if not np.array_equal(matrix(table[key], 1)[:, 0], expected):
                raise ValueError(f"Inconsistent {key} at source row range starting {start}.")
        if not np.isin(matrix(table["task_index"], 1), list(self.tasks)).all():
            raise ValueError("Parquet rows contain unknown task indices.")
