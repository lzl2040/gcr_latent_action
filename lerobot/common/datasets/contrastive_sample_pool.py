"""Compact, persistent sample-start pools for temporal pre-training."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch.distributed as dist

SAMPLE_POOL_FORMAT_VERSION = 1
_EPISODE_TABLE_NAME = "episodes.npy"
_MANIFEST_NAME = "manifest.json"


def sample_anchor_stride(
    fps: float,
    *,
    keep_all_below_fps: float = 10.0,
    target_hz: float = 5.0,
) -> int:
    """Return the stride between sample starts without changing a sample's inner timeline."""
    fps = float(fps)
    keep_all_below_fps = float(keep_all_below_fps)
    target_hz = float(target_hz)
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}.")
    if keep_all_below_fps <= 0:
        raise ValueError(f"keep_all_below_fps must be positive, got {keep_all_below_fps!r}.")
    if target_hz <= 0:
        raise ValueError(f"target_hz must be positive, got {target_hz!r}.")
    if fps <= keep_all_below_fps:
        return 1
    return max(2, int(math.floor(fps / target_hz + 0.5)))


@dataclass(frozen=True)
class CompactSamplePool:
    """Logical frame-index pool represented by one row per usable episode."""

    episode_table: np.ndarray
    dataset_episode_offsets: np.ndarray
    dataset_anchor_counts: np.ndarray
    anchor_strides: np.ndarray
    fingerprint: str
    manifest: dict[str, Any]
    storage_dir: Path | None = None

    @property
    def total_anchors(self) -> int:
        return int(self.dataset_anchor_counts.sum(dtype=np.int64))

    @property
    def num_datasets(self) -> int:
        return int(self.dataset_anchor_counts.shape[0])

    def _dataset_rows(self, dataset_idx: int) -> np.ndarray:
        if dataset_idx < 0 or dataset_idx >= self.num_datasets:
            raise IndexError(f"dataset_idx={dataset_idx} is outside [0, {self.num_datasets}).")
        start = int(self.dataset_episode_offsets[dataset_idx])
        end = int(self.dataset_episode_offsets[dataset_idx + 1])
        return self.episode_table[start:end]

    def sample_frame(self, rng: np.random.Generator, dataset_idx: int) -> int:
        """Draw one start frame uniformly over all logical anchors in a dataset."""
        return int(self.sample_frames(rng, dataset_idx, 1)[0])

    def sample_frames(
        self,
        rng: np.random.Generator,
        dataset_idx: int,
        count: int,
    ) -> np.ndarray:
        """Vectorized uniform draw over all logical anchors in one dataset."""
        total = int(self.dataset_anchor_counts[dataset_idx])
        if total <= 0:
            raise RuntimeError(f"Dataset {dataset_idx} has no valid sample anchors.")
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}.")
        if count == 0:
            return np.empty(0, dtype=np.int64)
        rows = self._dataset_rows(dataset_idx)
        logical_indices = rng.integers(total, size=count, dtype=np.int64)
        row_indices = np.searchsorted(
            rows[:, 2],
            logical_indices,
            side="right",
        )
        previous_ends = np.zeros(count, dtype=np.int64)
        nonzero = row_indices > 0
        previous_ends[nonzero] = rows[row_indices[nonzero] - 1, 2]
        local_indices = logical_indices - previous_ends
        return (rows[row_indices, 0] + local_indices * int(self.anchor_strides[dataset_idx])).astype(
            np.int64, copy=False
        )

    def sample_frame_excluding(
        self,
        rng: np.random.Generator,
        dataset_idx: int,
        excluded_frame: int,
    ) -> int:
        """Draw a valid replacement, avoiding one failed anchor when possible."""
        total = int(self.dataset_anchor_counts[dataset_idx])
        candidate = self.sample_frame(rng, dataset_idx)
        if total <= 1 or candidate != excluded_frame:
            return candidate
        for _ in range(15):
            candidate = self.sample_frame(rng, dataset_idx)
            if candidate != excluded_frame:
                return candidate
        stride = int(self.anchor_strides[dataset_idx])
        for start, count, _ in self._dataset_rows(dataset_idx):
            start = int(start)
            if start != excluded_frame:
                return start
            if int(count) > 1:
                return start + stride
        return candidate

    def sample_episode_group(
        self,
        rng: np.random.Generator,
        dataset_idx: int,
        count: int,
        min_frame_gap: int,
    ) -> list[int]:
        """Draw starts from one episode with a guaranteed pairwise frame gap."""
        rows = self._dataset_rows(dataset_idx)
        if rows.shape[0] == 0:
            raise RuntimeError(f"Dataset {dataset_idx} has no usable episodes.")
        row = rows[int(rng.integers(rows.shape[0]))]
        anchor_count = int(row[1])
        stride = int(self.anchor_strides[dataset_idx])
        gap_in_anchors = max(1, math.ceil(min_frame_gap / stride))

        phase_limit = min(gap_in_anchors, anchor_count)
        phase = int(rng.integers(phase_limit)) if phase_limit > 1 else 0
        candidates = np.arange(phase, anchor_count, gap_in_anchors, dtype=np.int64)
        if candidates.size == 0:
            candidates = np.asarray([0], dtype=np.int64)
        if candidates.size < min(count, math.ceil(anchor_count / gap_in_anchors)):
            candidates = np.arange(0, anchor_count, gap_in_anchors, dtype=np.int64)

        group_size = min(int(count), int(candidates.size))
        selected = rng.choice(candidates, size=group_size, replace=False)
        start = int(row[0])
        return [start + int(index) * stride for index in selected]


def _normalized_episode_ranges(episode_ranges: np.ndarray) -> np.ndarray:
    ranges = np.asarray(episode_ranges, dtype=np.int64)
    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise ValueError(f"Episode ranges must have shape (N,2), got {tuple(ranges.shape)}.")
    if np.any(ranges[:, 1] < ranges[:, 0]):
        raise ValueError("Episode ranges contain an end before their start.")
    return np.ascontiguousarray(ranges)


def _pool_fingerprint(
    *,
    data_mix: str,
    dataset_names: list[str],
    dataset_sizes: list[int],
    episode_ranges: list[np.ndarray],
    true_fps: list[float],
    horizons: list[int],
    anchor_strides: np.ndarray,
    keep_all_below_fps: float,
    target_hz: float,
) -> tuple[str, list[dict[str, Any]]]:
    datasets = []
    range_digests = []
    for name, size, ranges, fps, horizon, stride in zip(
        dataset_names,
        dataset_sizes,
        episode_ranges,
        true_fps,
        horizons,
        anchor_strides,
        strict=True,
    ):
        normalized = _normalized_episode_ranges(ranges)
        range_digest = hashlib.sha256(normalized.tobytes()).hexdigest()
        range_digests.append(range_digest)
        datasets.append(
            {
                "name": name,
                "frames": int(size),
                "episodes": int(normalized.shape[0]),
                "episode_ranges_sha256": range_digest,
                "true_fps": float(fps),
                "horizon": int(horizon),
                "anchor_stride": int(stride),
            }
        )
    description = {
        "format_version": SAMPLE_POOL_FORMAT_VERSION,
        "data_mix": data_mix,
        "policy": {
            "keep_all_below_fps": float(keep_all_below_fps),
            "target_hz": float(target_hz),
            "phase": 0,
        },
        "datasets": datasets,
    }
    digest = hashlib.sha256(json.dumps(description, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for range_digest in range_digests:
        digest.update(range_digest.encode("ascii"))
    return digest.hexdigest()[:20], datasets


def _build_episode_table(
    episode_ranges: list[np.ndarray],
    horizons: list[int],
    anchor_strides: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chunks = []
    offsets = [0]
    dataset_anchor_counts = []
    for ranges, horizon, stride in zip(
        episode_ranges,
        horizons,
        anchor_strides,
        strict=True,
    ):
        normalized = _normalized_episode_ranges(ranges)
        usable_spans = normalized[:, 1] - int(horizon) - normalized[:, 0]
        counts = np.where(
            usable_spans > 0,
            (usable_spans + int(stride) - 1) // int(stride),
            0,
        ).astype(np.int64)
        keep = counts > 0
        kept_counts = counts[keep]
        if kept_counts.size:
            prefix_ends = np.cumsum(kept_counts, dtype=np.int64)
            chunks.append(
                np.column_stack(
                    (
                        normalized[keep, 0],
                        kept_counts,
                        prefix_ends,
                    )
                ).astype(np.int64, copy=False)
            )
            dataset_anchor_counts.append(int(prefix_ends[-1]))
            offsets.append(offsets[-1] + int(kept_counts.shape[0]))
        else:
            dataset_anchor_counts.append(0)
            offsets.append(offsets[-1])
    table = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 3), dtype=np.int64)
    return (
        table,
        np.asarray(offsets, dtype=np.int64),
        np.asarray(dataset_anchor_counts, dtype=np.int64),
    )


def _safe_mix_name(data_mix: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", data_mix).strip("._")
    return safe or "mixture"


def resolve_sample_pool_paths(cfg) -> tuple[Path | None, Path | None]:
    """Resolve shared persistence and node-local mmap cache paths from a train config."""
    pool_root = getattr(cfg.dataset, "sample_pool_root", None)
    if pool_root is None and getattr(cfg, "output_dir", None) is not None:
        pool_root = Path(cfg.output_dir) / "_sample_pools"
    elif pool_root == "":
        pool_root = None

    cache_root = getattr(cfg.dataset, "sample_pool_cache_dir", None)
    if cache_root is None:
        cache_root = Path(os.environ.get("TMPDIR", "/tmp")) / "robo_contrast_sample_pools"
    elif cache_root == "":
        cache_root = None

    return (
        Path(pool_root).expanduser() if pool_root is not None else None,
        Path(cache_root).expanduser() if cache_root is not None else None,
    )


def sample_pool_checkpoint_state(
    sample_pool: CompactSamplePool | None,
) -> dict[str, str]:
    if sample_pool is None:
        return {}
    return {"sample_pool_fingerprint": sample_pool.fingerprint}


def validate_sample_pool_resume(
    sample_pool: CompactSamplePool | None,
    loaded_state: dict[str, Any],
) -> None:
    saved_fingerprint = loaded_state.get("sample_pool_fingerprint")
    if sample_pool is not None:
        if saved_fingerprint is None:
            raise ValueError(
                "This checkpoint predates persistent sample pools. Resume it with "
                "`dataset.sample_pool_enabled=false`, or start a fresh run to use sparse "
                "temporal anchors."
            )
        if saved_fingerprint != sample_pool.fingerprint:
            raise ValueError(
                "The temporal sample pool changed across resume: "
                f"checkpoint={saved_fingerprint}, current={sample_pool.fingerprint}."
            )
    elif saved_fingerprint is not None:
        raise ValueError(
            "The checkpoint was trained with temporal sample pool "
            f"{saved_fingerprint}, but the current run disabled the pool."
        )


def _manifest_path(pool_dir: Path) -> Path:
    return pool_dir / _MANIFEST_NAME


def _table_path(pool_dir: Path) -> Path:
    return pool_dir / _EPISODE_TABLE_NAME


def _validate_loaded_pool(
    pool_dir: Path,
    expected_fingerprint: str,
    expected_datasets: int,
) -> dict[str, Any]:
    manifest_file = _manifest_path(pool_dir)
    table_file = _table_path(pool_dir)
    if not manifest_file.is_file() or not table_file.is_file():
        raise FileNotFoundError(f"Sample pool is incomplete: expected {manifest_file} and {table_file}.")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("format_version") != SAMPLE_POOL_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported sample pool format {manifest.get('format_version')!r} in "
            f"{manifest_file}; expected {SAMPLE_POOL_FORMAT_VERSION}."
        )
    if manifest.get("fingerprint") != expected_fingerprint:
        raise ValueError(
            f"Sample pool fingerprint mismatch in {manifest_file}: "
            f"{manifest.get('fingerprint')!r} != {expected_fingerprint!r}."
        )
    if len(manifest.get("datasets", [])) != expected_datasets:
        raise ValueError(
            f"Sample pool has {len(manifest.get('datasets', []))} datasets, expected {expected_datasets}."
        )
    expected_size = manifest.get("episode_table_bytes")
    if expected_size is not None and table_file.stat().st_size != int(expected_size):
        raise ValueError(
            f"Sample pool table size mismatch for {table_file}: "
            f"{table_file.stat().st_size} != {expected_size}."
        )
    return manifest


def _copy_to_local_cache(
    source: Path,
    *,
    cache_root: Path | None,
    fingerprint: str,
) -> Path:
    if cache_root is None:
        return source
    cache_dir = cache_root / fingerprint
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / source.name
    lock_path = cache_dir / ".copy.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not destination.is_file() or destination.stat().st_size != source.stat().st_size:
            temporary = cache_dir / f".{source.name}.tmp-{os.getpid()}"
            try:
                shutil.copyfile(source, temporary)
                os.replace(temporary, destination)
            finally:
                if temporary.exists():
                    temporary.unlink()
    return destination


def _load_pool(
    *,
    pool_dir: Path,
    fingerprint: str,
    expected_datasets: int,
    cache_root: Path | None,
) -> CompactSamplePool:
    manifest = _validate_loaded_pool(pool_dir, fingerprint, expected_datasets)
    table_file = _copy_to_local_cache(
        _table_path(pool_dir),
        cache_root=cache_root,
        fingerprint=fingerprint,
    )
    episode_table = np.load(table_file, mmap_mode="r", allow_pickle=False)
    if episode_table.dtype != np.int64 or episode_table.ndim != 2 or episode_table.shape[1] != 3:
        raise ValueError(
            f"Sample pool table {table_file} must be int64 with shape (N,3), got "
            f"dtype={episode_table.dtype}, shape={episode_table.shape}."
        )
    offsets = np.asarray(manifest["dataset_episode_offsets"], dtype=np.int64)
    counts = np.asarray(
        [entry["anchor_count"] for entry in manifest["datasets"]],
        dtype=np.int64,
    )
    strides = np.asarray(
        [entry["anchor_stride"] for entry in manifest["datasets"]],
        dtype=np.int64,
    )
    if offsets.shape != (expected_datasets + 1,):
        raise ValueError(
            f"Sample pool offsets have shape {offsets.shape}, expected {(expected_datasets + 1,)}."
        )
    if int(offsets[-1]) != int(episode_table.shape[0]):
        raise ValueError(
            f"Sample pool offsets end at {int(offsets[-1])}, but the table has {episode_table.shape[0]} rows."
        )
    return CompactSamplePool(
        episode_table=episode_table,
        dataset_episode_offsets=offsets,
        dataset_anchor_counts=counts,
        anchor_strides=strides,
        fingerprint=fingerprint,
        manifest=manifest,
        storage_dir=pool_dir,
    )


def _load_pool_with_retry(
    *,
    pool_dir: Path,
    fingerprint: str,
    expected_datasets: int,
    cache_root: Path | None,
    visibility_timeout_seconds: float = 600.0,
) -> CompactSamplePool:
    deadline = time.monotonic() + visibility_timeout_seconds
    while True:
        try:
            return _load_pool(
                pool_dir=pool_dir,
                fingerprint=fingerprint,
                expected_datasets=expected_datasets,
                cache_root=cache_root,
            )
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(1.0)


def _build_pool_payload(
    *,
    data_mix: str,
    dataset_descriptions: list[dict[str, Any]],
    episode_ranges: list[np.ndarray],
    horizons: list[int],
    anchor_strides: np.ndarray,
    fingerprint: str,
    keep_all_below_fps: float,
    target_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    table, offsets, counts = _build_episode_table(
        episode_ranges,
        horizons,
        anchor_strides,
    )
    manifest = {
        "format_version": SAMPLE_POOL_FORMAT_VERSION,
        "fingerprint": fingerprint,
        "data_mix": data_mix,
        "policy": {
            "keep_all_below_fps": float(keep_all_below_fps),
            "target_hz": float(target_hz),
            "phase": 0,
        },
        "dataset_episode_offsets": offsets.tolist(),
        "total_anchors": int(counts.sum(dtype=np.int64)),
        "datasets": [
            {
                **description,
                "anchor_count": int(count),
                "effective_anchor_hz": float(description["true_fps"]) / int(description["anchor_stride"]),
            }
            for description, count in zip(
                dataset_descriptions,
                counts,
                strict=True,
            )
        ],
    }
    return table, offsets, counts, manifest


def _persist_pool(
    *,
    pool_dir: Path,
    episode_table: np.ndarray,
    manifest: dict[str, Any],
) -> None:
    if pool_dir.is_dir():
        _validate_loaded_pool(
            pool_dir,
            str(manifest["fingerprint"]),
            len(manifest["datasets"]),
        )
        return
    pool_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{pool_dir.name}.tmp-",
            dir=pool_dir.parent,
        )
    )
    try:
        with _table_path(temporary).open("wb") as handle:
            np.save(handle, episode_table, allow_pickle=False)
        manifest["episode_table_bytes"] = _table_path(temporary).stat().st_size
        _manifest_path(temporary).write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        try:
            temporary.rename(pool_dir)
        except OSError:
            if not pool_dir.is_dir():
                raise
            _validate_loaded_pool(
                pool_dir,
                str(manifest["fingerprint"]),
                len(manifest["datasets"]),
            )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def build_or_load_sample_pool(
    *,
    data_mix: str,
    dataset_names: list[str],
    dataset_sizes: list[int],
    episode_ranges: list[np.ndarray],
    true_fps: list[float],
    horizons: list[int],
    keep_all_below_fps: float,
    target_hz: float,
    pool_root: str | Path | None,
    local_cache_root: str | Path | None,
) -> CompactSamplePool:
    """Build once on rank zero, then memory-map the compact episode table on every rank."""
    num_datasets = len(dataset_names)
    lengths = {
        len(dataset_sizes),
        len(episode_ranges),
        len(true_fps),
        len(horizons),
        num_datasets,
    }
    if len(lengths) != 1:
        raise ValueError(
            "Sample pool inputs must have one entry per dataset: "
            f"names={num_datasets}, sizes={len(dataset_sizes)}, "
            f"ranges={len(episode_ranges)}, fps={len(true_fps)}, "
            f"horizons={len(horizons)}."
        )
    anchor_strides = np.asarray(
        [
            sample_anchor_stride(
                fps,
                keep_all_below_fps=keep_all_below_fps,
                target_hz=target_hz,
            )
            for fps in true_fps
        ],
        dtype=np.int64,
    )
    fingerprint, dataset_descriptions = _pool_fingerprint(
        data_mix=data_mix,
        dataset_names=dataset_names,
        dataset_sizes=dataset_sizes,
        episode_ranges=episode_ranges,
        true_fps=true_fps,
        horizons=horizons,
        anchor_strides=anchor_strides,
        keep_all_below_fps=keep_all_below_fps,
        target_hz=target_hz,
    )
    root = Path(pool_root).expanduser() if pool_root else None
    cache_root = Path(local_cache_root).expanduser() if local_cache_root else None
    pool_dir = root / _safe_mix_name(data_mix) / fingerprint if root is not None else None

    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    if pool_dir is not None:
        if rank == 0 and not pool_dir.is_dir():
            table, _, _, manifest = _build_pool_payload(
                data_mix=data_mix,
                dataset_descriptions=dataset_descriptions,
                episode_ranges=episode_ranges,
                horizons=horizons,
                anchor_strides=anchor_strides,
                fingerprint=fingerprint,
                keep_all_below_fps=keep_all_below_fps,
                target_hz=target_hz,
            )
            _persist_pool(
                pool_dir=pool_dir,
                episode_table=table,
                manifest=manifest,
            )
        return _load_pool_with_retry(
            pool_dir=pool_dir,
            fingerprint=fingerprint,
            expected_datasets=num_datasets,
            cache_root=cache_root,
        )

    table, offsets, counts, manifest = _build_pool_payload(
        data_mix=data_mix,
        dataset_descriptions=dataset_descriptions,
        episode_ranges=episode_ranges,
        horizons=horizons,
        anchor_strides=anchor_strides,
        fingerprint=fingerprint,
        keep_all_below_fps=keep_all_below_fps,
        target_hz=target_hz,
    )
    return CompactSamplePool(
        episode_table=table,
        dataset_episode_offsets=offsets,
        dataset_anchor_counts=counts,
        anchor_strides=anchor_strides,
        fingerprint=fingerprint,
        manifest=manifest,
    )
