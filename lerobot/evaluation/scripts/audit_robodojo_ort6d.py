"""Check every RoboDojo ort6d row against its original xyz/wxyz/gripper column."""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.evaluation.scripts.robodojo_dataset_utils import (
    DEFAULT_DATASET,
    POSE_FEATURES,
    V30Source,
    matrix,
    pose_to_ort6d,
    write_json,
)

logger = logging.getLogger(__name__)


def audit(root: Path, report_path: Path, tolerance: float = 2e-6) -> dict:
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Tolerance must be positive and finite.")
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite audit report: {report_path}")
    source = V30Source(root)
    report = {
        "source": str(source.root),
        "tolerance": tolerance,
        "layout": "left/right xyz + [R00,R01,R10,R11,R20,R21] + gripper",
        "features": {
            key: {
                "converted": converted,
                "checked_rows": 0,
                "invalid_rows": 0,
                "mismatch_rows": 0,
                "max_abs_error": 0.0,
                "max_abs_error_per_dimension": [0.0] * 20,
                "max_quaternion_norm_error": 0.0,
                "nonunit_quaternion_rows": 0,
                "max_rotation_column_norm_error": 0.0,
                "max_rotation_column_dot": 0.0,
                "examples": [],
            }
            for key, converted in POSE_FEATURES.items()
        },
        "files": [],
    }
    for key, converted in POSE_FEATURES.items():
        if source.info["features"][key]["shape"] != [16] or source.info["features"][converted]["shape"] != [
            20
        ]:
            raise ValueError(f"Unexpected feature shapes for {key}/{converted}.")
    total = 0
    columns = [*POSE_FEATURES, *POSE_FEATURES.values(), "index", "episode_index", "frame_index", "task_index"]
    for path in source.files:
        rows_before = total
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=columns):
            table = pa.Table.from_batches([batch])
            source.validate_rows(table, total)
            for key, converted in POSE_FEATURES.items():
                raw, stored = matrix(table[key], 16), matrix(table[converted], 20)
                norms = np.stack(
                    [
                        np.linalg.norm(raw[:, offset + 3 : offset + 7].astype(np.float64), axis=1)
                        for offset in (0, 8)
                    ],
                    axis=1,
                )
                valid = (
                    np.isfinite(raw).all(axis=1)
                    & np.isfinite(stored).all(axis=1)
                    & np.isfinite(norms).all(axis=1)
                    & (norms > 1e-8).all(axis=1)
                )
                entry = report["features"][key]
                entry["checked_rows"] += len(raw)
                entry["invalid_rows"] += int((~valid).sum())
                errors = np.zeros(len(raw))
                if valid.any():
                    expected = pose_to_ort6d(raw[valid]).astype(np.float64)
                    error = np.abs(expected - stored[valid].astype(np.float64))
                    errors[valid] = error.max(axis=1)
                    entry["max_abs_error"] = max(entry["max_abs_error"], float(error.max()))
                    entry["max_abs_error_per_dimension"] = np.maximum(
                        entry["max_abs_error_per_dimension"], error.max(axis=0)
                    ).tolist()
                    norm_error = np.abs(norms[valid] - 1)
                    entry["max_quaternion_norm_error"] = max(
                        entry["max_quaternion_norm_error"], float(norm_error.max())
                    )
                    entry["nonunit_quaternion_rows"] += int((norm_error > 1e-4).any(axis=1).sum())
                    for offset in (0, 10):
                        columns_r = (
                            stored[valid, offset + 3 : offset + 9].astype(np.float64).reshape(-1, 3, 2)
                        )
                        entry["max_rotation_column_norm_error"] = max(
                            entry["max_rotation_column_norm_error"],
                            float(np.abs(np.linalg.norm(columns_r, axis=1) - 1).max()),
                        )
                        entry["max_rotation_column_dot"] = max(
                            entry["max_rotation_column_dot"],
                            float(np.abs((columns_r[:, :, 0] * columns_r[:, :, 1]).sum(axis=1)).max()),
                        )
                mismatch = valid & (errors > tolerance)
                entry["mismatch_rows"] += int(mismatch.sum())
                for index in np.flatnonzero(~valid | mismatch)[: max(0, 20 - len(entry["examples"]))]:
                    entry["examples"].append(
                        {
                            "index": total + int(index),
                            "reason": "conversion_mismatch"
                            if valid[index]
                            else "nonfinite_or_zero_quaternion",
                            "max_abs_error": float(errors[index]) if valid[index] else None,
                        }
                    )
            total += table.num_rows
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        report["files"].append(
            {
                "path": str(path.relative_to(source.root)),
                "rows": total - rows_before,
                "sha256": digest,
            }
        )
        logger.info("Checked %s: %d rows", path.name, total - rows_before)
    if total != source.info["total_frames"]:
        raise ValueError("Physical parquet row count differs from info.json.")
    report["total_frames"] = total
    report["total_episodes"] = len(source.episodes)
    report["complete"] = True
    report["passed"] = all(
        not entry["invalid_rows"]
        and not entry["mismatch_rows"]
        and not entry["nonunit_quaternion_rows"]
        and entry["max_rotation_column_norm_error"] <= tolerance
        and entry["max_rotation_column_dot"] <= tolerance
        for entry in report["features"].values()
    )
    write_json(report_path, report)
    logger.info("Audit passed=%s, report=%s", report["passed"], report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=2e-6)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not audit(args.source, args.report, args.tolerance)["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
