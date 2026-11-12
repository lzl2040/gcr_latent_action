#!/usr/bin/env python
"""Decimate v3.0 Parquet and videos together; rebuild known RoboDojo measured-state deltas.

Only integer source/target FPS ratios are supported. Absolute action targets are sampled,
not differenced. Unknown delta conventions are rejected. Outputs are published atomically.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot.evaluation.scripts.add_robodojo_delta_actions import DELTA_COLUMNS, DELTA_CONTRACT
from lerobot.evaluation.scripts.robodojo_dataset_utils import (
    QUANTILES,
    V30Source,
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
STATS_SIZE = (32, 24)


def sampling_stride(source_fps: float, target_fps: float) -> int:
    if not math.isfinite(target_fps) or not 0 < target_fps < source_fps:
        raise ValueError("Target FPS must be finite, positive and smaller than source FPS.")
    ratio = source_fps / target_fps
    if not math.isclose(ratio, round(ratio), abs_tol=1e-8, rel_tol=0):
        raise ValueError("Only integer FPS ratios are supported; e.g.25->12.5, not25->10.")
    return round(ratio)


def frame_number(seconds: float, fps: float) -> int:
    value = seconds * fps
    if not math.isfinite(value) or value < 0 or abs(value - round(value)) > 1e-4:
        raise ValueError(f"Video timestamp {seconds} is not on the {fps} FPS frame grid.")
    return round(value)


def plan_episodes(source: V30Source, stride: int, short_video_policy: str, limit: int | None):
    rows = source.episodes if limit is None else source.episodes[:limit]
    plans, jobs = [], {}
    output_start = 0
    for row in rows:
        length = row["length"]
        intervals = {}
        for key in source.videos:
            relative = source.video_relative(row, key)
            prefix = f"videos/{key}"
            start = frame_number(row[f"{prefix}/from_timestamp"], source.info["fps"])
            end = frame_number(row[f"{prefix}/to_timestamp"], source.info["fps"])
            intervals[key] = (relative, start, end - start)
        usable = min([length, *(count for _, _, count in intervals.values())])
        if usable < length and short_video_policy == "error":
            raise ValueError(
                f"Episode {row['episode_index']} has a short video; use --short-video-policy trim."
            )
        if usable <= 0:
            raise ValueError(f"Episode {row['episode_index']} has no common valid frames.")
        indices = np.arange(0, usable, stride, dtype=np.int64)
        plan = {
            "episode_index": row["episode_index"],
            "source_length": length,
            "common_valid_length": usable,
            "trimmed_source_rows": length - usable,
            "output_length": len(indices),
            "output_start": output_start,
            "source_start": row["dataset_from_index"],
        }
        plans.append(plan)
        output_start += len(indices)
        for key, (relative, start, count) in intervals.items():
            job = jobs.setdefault((key, relative), {"key": key, "relative": relative, "segments": []})
            packed = sum(segment["length"] for segment in job["segments"])
            job["segments"].append(
                {
                    "episode_index": row["episode_index"],
                    "start": start,
                    "last": start + int(indices[-1]),
                    "length": len(indices),
                    "output_start": packed,
                    "source_interval_length": count,
                }
            )
    return plans, list(jobs.values())


def histogram_stats(histogram: np.ndarray) -> dict:
    counts = histogram.sum(axis=1)
    if not np.all(counts == counts[0]) or counts[0] <= 0:
        raise ValueError("Invalid RGB histogram counts.")
    count = int(counts[0])
    levels = np.arange(256, dtype=np.float64) / 255
    mean = (histogram * levels).sum(axis=1) / count
    variance = (histogram * (levels[None] - mean[:, None]) ** 2).sum(axis=1) / count
    cumulative = histogram.cumsum(axis=1)

    def at_rank(rank):
        return np.array([np.searchsorted(channel, rank, side="right") for channel in cumulative]) / 255

    def image_shape(value):
        return np.asarray(value).reshape(3, 1, 1).tolist()

    result = {
        "min": image_shape(at_rank(0)),
        "max": image_shape(at_rank(count - 1)),
        "mean": image_shape(mean),
        "std": image_shape(np.sqrt(variance)),
        "count": [count],
    }
    for name, quantile in QUANTILES.items():
        rank = quantile * (count - 1)
        low, high = math.floor(rank), math.ceil(rank)
        result[name] = (
            image_shape(at_rank(low) * (high - rank) + at_rank(high) * (rank - low))
            if high != low
            else image_shape(at_rank(low))
        )
    return result


def probe_video(path: Path, ffprobe: str) -> dict:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,avg_frame_rate,codec_name",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    stream = json.loads(result.stdout)["streams"][0]
    if stream.get("nb_frames") in (None, "N/A"):
        raise ValueError(f"Video has no declared frame count: {path}")
    stream["frames"] = int(stream["nb_frames"])
    stream["fps"] = float(Fraction(stream["avg_frame_rate"]))
    return stream


def read_exact(stream, size: int) -> bytes:
    chunks, remaining = [], size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise ValueError(f"Decoded video ended with {remaining} bytes missing.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def convert_video(
    job: dict,
    source: Path,
    output: Path,
    fps: float,
    stride: int,
    ffmpeg: str,
    ffprobe: str,
    threads: int,
    crf: int,
) -> dict:
    original, target = source / job["relative"], output / job["relative"]
    before = probe_video(original, ffprobe)
    if not math.isclose(before["fps"], fps * stride, abs_tol=1e-6):
        raise ValueError(f"Source video FPS disagrees with data: {original}")
    segments = job["segments"]
    if any(s["last"] >= before["frames"] for s in segments):
        raise ValueError(f"Video metadata references missing physical frames: {original}")
    if any(a["last"] >= b["start"] for a, b in zip(segments, segments[1:], strict=False)):
        raise ValueError(f"Overlapping source episode video intervals: {original}")
    expression = "+".join(
        f"between(n,{s['start']},{s['last']})*not(mod(n-{s['start']},{stride}))" for s in segments
    )
    expected = sum(s["length"] for s in segments)
    rate = str(Fraction(str(fps)))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".partial.mp4")
    command = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-threads",
        str(threads),
        "-i",
        str(original),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        f"select='{expression}',setpts=N/({rate}*TB)",
        "-filter_threads",
        "1",
        "-frames:v",
        str(expected),
        "-r",
        rate,
        "-fps_mode",
        "cfr",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        str(crf),
        "-g",
        "2",
        "-threads",
        str(threads),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=7200)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"FFmpeg failed for {original}: {exc.stderr.decode(errors='replace')}") from exc
    after = probe_video(temporary, ffprobe)
    if (
        after["frames"] != expected
        or not math.isclose(after["fps"], fps, abs_tol=1e-6)
        or any(after[key] != before[key] for key in ("height", "width"))
    ):
        raise ValueError(f"Output video count/FPS/resolution mismatch: {temporary}")
    histogram = np.zeros((3, 256), dtype=np.int64)
    per_episode = {}
    width, height = STATS_SIZE
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            [
                ffmpeg,
                "-nostdin",
                "-v",
                "error",
                "-threads",
                str(threads),
                "-i",
                str(temporary),
                "-vf",
                f"scale={width}:{height}:flags=area",
                "-filter_threads",
                "1",
                "-an",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=errors,
        )
        try:
            for segment in segments:
                pixels = np.frombuffer(
                    read_exact(process.stdout, segment["length"] * width * height * 3), dtype=np.uint8
                ).reshape(-1, 3)
                part = np.stack([np.bincount(pixels[:, channel], minlength=256) for channel in range(3)])
                histogram += part
                per_episode[segment["episode_index"]] = histogram_stats(part)
            if process.stdout.read(1):
                raise ValueError("Decoded video has unexpected extra frames.")
            if process.wait(timeout=120):
                errors.seek(0)
                raise RuntimeError(errors.read().decode(errors="replace"))
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
    temporary.rename(target)
    return {"job": job, "episodes": per_episode, "histogram": histogram, "frames": expected}


def convert(
    source_root: Path,
    destination: Path,
    fps: float,
    *,
    workers: int = 4,
    threads: int = 2,
    short_video_policy: str = "error",
    max_episodes: int | None = None,
    ffmpeg: str | None = None,
    crf: int = 18,
) -> dict:
    source = V30Source(source_root)
    destination = validate_destination(source.root, destination)
    stride = sampling_stride(float(source.info["fps"]), fps)
    if workers < 1 or threads < 1 or (max_episodes is not None and max_episodes < 1):
        raise ValueError("Workers, threads and max-episodes must be positive.")
    if short_video_policy not in ("error", "trim") or not 0 <= crf <= 51:
        raise ValueError("Invalid short-video policy or H.264 CRF.")
    plans, jobs = plan_episodes(source, stride, short_video_policy, max_episodes)
    ffmpeg = ffmpeg or shutil.which("ffmpeg") or str(Path(sys.executable).parent / "ffmpeg")
    ffprobe = str(Path(ffmpeg).with_name("ffprobe"))
    if jobs and not (Path(ffmpeg).is_file() and Path(ffprobe).is_file()):
        raise FileNotFoundError("FFmpeg/FFprobe are required; supply --ffmpeg /path/to/ffmpeg.")
    source_paths = {
        *source.files,
        *source.root.joinpath("meta").rglob("*.parquet"),
        source.root / "meta/info.json",
        source.root / "meta/stats.json",
    }
    source_paths.update(source.root / job["relative"] for job in jobs)
    snapshots = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in source_paths}
    old_end = source.episodes[len(plans) - 1]["dataset_to_index"]
    table = pq.read_table(source.files, filters=[("index", "<", old_end)])
    source.validate_rows(table, 0)
    if table.num_rows != old_end:
        raise ValueError("Data row count differs from episode metadata.")
    timestamps = matrix(table["timestamp"], 1)[:, 0]
    if not np.allclose(timestamps, matrix(table["frame_index"], 1)[:, 0] / source.info["fps"], atol=1e-4):
        raise ValueError("Source data timestamps are not a regular per-episode FPS grid.")
    selected = np.concatenate(
        [p["source_start"] + np.arange(0, p["common_valid_length"], stride, dtype=np.int64) for p in plans]
    )
    output_table = table.take(pa.array(selected))
    del table
    names = output_table.column_names
    delta_columns = set(names) & set(DELTA_COLUMNS)
    declared_delta = [
        key
        for key, feature in source.info["features"].items()
        if "delta" in key.lower() or "delta" in str(feature.get("names", "")).lower()
    ]
    if set(declared_delta) - set(DELTA_COLUMNS):
        raise ValueError("Unknown delta action representation; refusing naive subsampling.")
    if delta_columns:
        contract = source.info.get("delta_action_contract", {})
        if any(
            contract.get(key) != DELTA_CONTRACT[key]
            for key in ("reference", "rotation", "translation", "gripper", "quaternion_order", "ort6d_order")
        ):
            raise ValueError(
                "Unsupported delta contract; expected RoboDojo next-measured-state/world/left rotation."
            )
        values = state_increments(
            matrix(output_table["observation.state"], 16),
            np.cumsum([p["output_length"] for p in plans]),
        )
        for key, value in zip(DELTA_COLUMNS, values, strict=True):
            if key in delta_columns:
                output_table = output_table.set_column(names.index(key), key, vector_array(value))
    for key, values in {
        "index": np.arange(len(selected)),
        "frame_index": np.concatenate([np.arange(p["output_length"]) for p in plans]),
        "timestamp": np.concatenate([np.arange(p["output_length"]) / fps for p in plans]),
    }.items():
        output_table = output_table.set_column(
            names.index(key), key, pa.array(values, type=output_table.schema.field(key).type)
        )
    episodes = copy.deepcopy(source.episodes[: len(plans)])
    for row, plan in zip(episodes, plans, strict=True):
        row.update(
            length=plan["output_length"],
            dataset_from_index=plan["output_start"],
            dataset_to_index=plan["output_start"] + plan["output_length"],
        )
        row["meta/episodes/chunk_index"] = row["meta/episodes/file_index"] = 0
    info = copy.deepcopy(source.info)
    info.update(fps=fps, total_frames=len(selected), total_episodes=len(episodes))
    info["splits"] = {
        name: f"{start}:{min(end, len(episodes))}"
        for name, bounds in info["splits"].items()
        for start, end in [tuple(map(int, bounds.split(":")))]
        if start < len(episodes)
    }
    stats = {}
    for key in names:
        feature = info["features"][key]
        if feature["dtype"] == "string":
            continue
        values = matrix(output_table[key])
        stats[key] = numeric_stats(values)
        for row in episodes:
            put_episode_stats(
                row, key, numeric_stats(values[row["dataset_from_index"] : row["dataset_to_index"]])
            )
    manifest = {
        "source": str(source.root),
        "destination": str(destination),
        "source_fps": source.info["fps"],
        "fps": fps,
        "stride": stride,
        "source_total_frames": source.info["total_frames"],
        "total_frames": len(selected),
        "total_episodes": len(episodes),
        "video_files": len(jobs),
        "short_video_policy": short_video_policy,
        "episode_frame_mapping": plans,
        "source_video_interval_warnings": source.video_warnings,
        "delta_columns_recomputed": sorted(delta_columns),
        "absolute_action_columns": "Selected unchanged at the retained source timestamps.",
        "image_stats": {
            "source": "Every decoded output frame, area-resized RGB spatial sample.",
            "width": STATS_SIZE[0],
            "height": STATS_SIZE[1],
            "normalized": True,
        },
        "codec": "h264",
        "crf": crf,
        "complete": False,
        "completed_video_files": 0,
    }
    with new_dataset(source.root, destination) as output:
        progress = output / "meta/downsample_manifest.json"
        write_json(progress, manifest)
        offset = 0
        for path in source.files:
            count = pq.ParquetFile(path).metadata.num_rows
            begin, end = np.searchsorted(selected, [offset, offset + count])
            if end > begin:
                target = output / path.relative_to(source.root)
                target.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(output_table.slice(int(begin), int(end - begin)), target, compression="zstd")
            offset += count
        histograms = {key: np.zeros((3, 256), dtype=np.int64) for key in source.videos}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    convert_video, job, source.root, output, fps, stride, ffmpeg, ffprobe, threads, crf
                )
                for job in jobs
            ]
            try:
                for future in as_completed(futures):
                    result = future.result()
                    job, key = result["job"], result["job"]["key"]
                    histograms[key] += result["histogram"]
                    for segment in job["segments"]:
                        row = episodes[segment["episode_index"]]
                        prefix = f"videos/{key}"
                        row[f"{prefix}/from_timestamp"] = segment["output_start"] / fps
                        row[f"{prefix}/to_timestamp"] = (segment["output_start"] + segment["length"]) / fps
                        put_episode_stats(row, key, result["episodes"][segment["episode_index"]])
                    manifest["completed_video_files"] += 1
                    write_json(progress, manifest)
                    logger.info(
                        "Video %d/%d: %s (%d frames)",
                        manifest["completed_video_files"],
                        len(jobs),
                        job["relative"],
                        result["frames"],
                    )
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        for key in source.videos:
            stats[key] = histogram_stats(histograms[key])
            video_info = info["features"][key].setdefault("info", {})
            video_info.update({"video.fps": fps, "video.codec": "h264", "video.pix_fmt": "yuv420p"})
        for path, before in snapshots.items():
            if (path.stat().st_size, path.stat().st_mtime_ns) != before:
                raise ValueError(f"Source changed while converting: {path}")
        write_episodes(output, episodes, source.episode_schema)
        shutil.copy2(source.root / "meta/tasks.parquet", output / "meta/tasks.parquet")
        write_json(output / "meta/stats.json", stats)
        write_json(output / "meta/info.json", info)
        verified = V30Source(output)
        verified.validate_rows(output_table, 0)
        if sum(row["length"] for row in verified.episodes) != len(selected):
            raise ValueError("Output episode/frame counts disagree.")
        manifest.update(complete=True, source_stat_snapshots_unchanged=True)
        write_json(progress, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2, help="FFmpeg threads per worker.")
    parser.add_argument("--short-video-policy", choices=("error", "trim"), default="error")
    parser.add_argument(
        "--max-episodes", type=int, help="Convert only the first N episodes for a smoke test."
    )
    parser.add_argument("--ffmpeg", help="Defaults to PATH or FFmpeg beside the current Python.")
    parser.add_argument("--crf", type=int, default=18)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = convert(
        args.source,
        args.output,
        args.fps,
        workers=args.workers,
        threads=args.threads,
        short_video_policy=args.short_video_policy,
        max_episodes=args.max_episodes,
        ffmpeg=args.ffmpeg,
        crf=args.crf,
    )
    logger.info(
        "Complete: %d episodes, %d frames -> %s",
        manifest["total_episodes"],
        manifest["total_frames"],
        args.output,
    )


if __name__ == "__main__":
    main()
