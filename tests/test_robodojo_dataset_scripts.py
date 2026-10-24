import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.spatial.transform import Rotation

from lerobot.evaluation.scripts.add_robodojo_delta_actions import DELTA_COLUMNS, convert
from lerobot.evaluation.scripts.robodojo_dataset_utils import (
    V30Source,
    matrix,
    numeric_stats,
    pose_to_ort6d,
    state_increments,
    vector_array,
    write_json,
)


def make_dataset(root: Path) -> Path:
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    lengths, total = [4, 3, 1], 8
    episode_ids = np.repeat(np.arange(3), lengths)
    frame_ids = np.concatenate([np.arange(length) for length in lengths])
    states = np.empty((total, 16), dtype=np.float32)
    for offset in (0, 8):
        states[:, offset : offset + 3] = np.arange(total)[:, None] * np.array([0.03, -0.02, 0.01]) + offset
        rotations = Rotation.from_euler(
            "xyz", np.arange(total)[:, None] * np.array([0.07, 0.11, -0.13]) + [0.3, -0.2, 0.4]
        )
        states[:, offset + 3 : offset + 7] = rotations.as_quat()[:, [3, 0, 1, 2]]
        states[::2, offset + 3 : offset + 7] *= -1
        states[:, offset + 7] = [0.2, 0.4, 0.6, 0.8, 0.3, 0.5, 0.7, 0.9]
    ends = np.cumsum(lengths)
    next_ids = np.arange(1, total + 1)
    next_ids[ends - 1] = ends - 1
    actions = states[next_ids].copy()
    actions[:, 0] += 0.02  # Deliberately distinguish measured-next-state from raw action targets.
    data = pa.table(
        {
            "observation.state": vector_array(states),
            "action": vector_array(actions),
            "timestamp": pa.array(frame_ids / 5, type=pa.float32()),
            "frame_index": pa.array(frame_ids, type=pa.int64()),
            "episode_index": pa.array(episode_ids, type=pa.int64()),
            "index": pa.array(np.arange(total), type=pa.int64()),
            "task_index": pa.array(np.repeat([0, 1, 0], lengths), type=pa.int64()),
            "observation.ee_ort6d_pos": vector_array(pose_to_ort6d(states)),
            "action.ee_ort6d_pos": vector_array(pose_to_ort6d(actions)),
        }
    )
    # Both episode 0 and episode 1 cross physical parquet boundaries.
    for index, (start, end) in enumerate(((0, 2), (2, 6), (6, 8))):
        pq.write_table(data.slice(start, end - start), root / f"data/chunk-000/file-{index:03d}.parquet")
    texts = ["Stack the three bowls together.", "Build a block tower."]
    pd.DataFrame({"task_index": [0, 1]}, index=pd.Index(texts, name="task")).to_parquet(
        root / "meta/tasks.parquet"
    )
    features = {
        name: {
            "dtype": "float32"
            if pa.types.is_fixed_size_list(data[name].type) or name == "timestamp"
            else "int64",
            "shape": [data[name].type.list_size] if pa.types.is_fixed_size_list(data[name].type) else [1],
            "names": None,
        }
        for name in data.column_names
    }
    video_keys = [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ]
    for key in video_keys:
        features[key] = {"dtype": "video", "shape": [3, 2, 2], "names": ["channels", "height", "width"]}
        for file in (0, 1):
            path = root / f"videos/{key}/chunk-000/file-{file:03d}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"opaque-video-copy-fixture-{key}-{file}".encode())
    info = {
        "codebase_version": "v3.0",
        "robot_type": "unified_robot",
        "total_episodes": 3,
        "total_frames": total,
        "total_tasks": 2,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": 5,
        "splits": {"train": "0:3"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    write_json(root / "meta/info.json", info)
    write_json(
        root / "meta/stats.json", {name: numeric_stats(matrix(data[name])) for name in data.column_names}
    )
    episodes, start = [], 0
    for index, length in enumerate(lengths):
        row = {
            "episode_index": index,
            "length": length,
            "tasks": [texts[int(index == 1)]],
            "dataset_from_index": start,
            "dataset_to_index": start + length,
            "data/chunk_index": 0,
            "data/file_index": (0, 1, 2)[index],
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for key in video_keys:
            prefix = f"videos/{key}"
            video_start = (0, 4 / 5, 0)[index]
            row.update(
                {
                    f"{prefix}/chunk_index": 0,
                    f"{prefix}/file_index": int(index == 2),
                    f"{prefix}/from_timestamp": video_start,
                    f"{prefix}/to_timestamp": video_start + length / 5,
                }
            )
            for metric, value in {
                "min": [[[0.0]]] * 3,
                "max": [[[1.0]]] * 3,
                "mean": [[[0.2 + index * 0.1]]] * 3,
                "std": [[[0.1]]] * 3,
                "count": [length * 4],
            }.items():
                row[f"stats/{key}/{metric}"] = value
        episodes.append(row)
        start += length
    pq.write_table(pa.Table.from_pylist(episodes), root / "meta/episodes/chunk-000/file-000.parquet")
    return root


@pytest.fixture
def dataset(tmp_path):
    return make_dataset(tmp_path / "source")


def test_state_delta_roundtrip_full_episodes_and_mid_episode_replanning(dataset):
    source = V30Source(dataset)
    states = matrix(pq.read_table(source.files, columns=["observation.state"])["observation.state"], 16)
    delta, ort6d = state_increments(states, source.ends)
    for start, end in ((0, 4), (1, 4), (4, 7), (5, 7), (7, 8)):
        for offset in (0, 8):
            position = states[start, offset : offset + 3].astype(np.float64)
            rotation = Rotation.from_quat(states[start, offset + 3 : offset + 7][[1, 2, 3, 0]])
            gripper = float(states[start, offset + 7])
            for index in range(start, end):
                position += delta[index, offset : offset + 3]
                step = Rotation.from_quat(delta[index, offset + 3 : offset + 7][[1, 2, 3, 0]])
                rotation = step * rotation
                gripper += float(delta[index, offset + 7])
                target = states[min(index + 1, end - 1)]
                np.testing.assert_allclose(position, target[offset : offset + 3], atol=2e-6)
                np.testing.assert_allclose(
                    rotation.as_matrix(),
                    Rotation.from_quat(target[offset + 3 : offset + 7][[1, 2, 3, 0]]).as_matrix(),
                    atol=2e-6,
                )
                assert gripper == pytest.approx(target[offset + 7], abs=1e-7)
    np.testing.assert_array_equal(delta[source.ends - 1], np.tile([0, 0, 0, 1, 0, 0, 0, 0] * 2, (3, 1)))
    np.testing.assert_array_equal(ort6d, pose_to_ort6d(delta))
    assert delta[0, 0] == pytest.approx(0.03)
    assert delta[0, 7] == pytest.approx(0.2)


def test_delta_export_preserves_absolute_columns_and_updates_all_metadata(dataset, tmp_path):
    output = tmp_path / "delta"
    before = {path.relative_to(dataset): path.read_bytes() for path in dataset.rglob("*.parquet")}
    manifest = convert(dataset, output)
    original, converted = V30Source(dataset), V30Source(output)
    assert manifest["complete"] and manifest["total_frames"] == 8
    old = pq.read_table(original.files)
    new = pq.read_table(converted.files)
    for name in old.column_names:
        assert old[name].equals(new[name])
    for name, width in zip(DELTA_COLUMNS, (16, 20), strict=True):
        assert converted.info["features"][name]["shape"] == [width]
        assert new[name].type.list_size == width
        stats = json.loads((output / "meta/stats.json").read_text())[name]
        assert stats == numeric_stats(matrix(new[name]))
        for episode in converted.episodes:
            start, end = episode["dataset_from_index"], episode["dataset_to_index"]
            expected = numeric_stats(matrix(new[name])[start:end])
            for metric, value in expected.items():
                assert episode[f"stats/{name}/{metric}"] == value
    assert converted.info["delta_action_contract"]["gripper"] == "g_state[t+1] - g_state[t]"
    for relative, content in before.items():
        assert (dataset / relative).read_bytes() == content
    for video in (output / "videos").rglob("*.mp4"):
        assert video.is_symlink() and video.read_bytes() == (dataset / video.relative_to(output)).read_bytes()
    with pytest.raises(FileExistsError):
        convert(dataset, output)
    with pytest.raises(ValueError, match="already"):
        convert(output, tmp_path / "again")


@pytest.mark.parametrize("invalid", ["nan", "zero_quaternion"])
def test_delta_rejects_invalid_source_geometry(invalid):
    states = np.array([[0, 0, 0, 1, 0, 0, 0, 1] * 2], dtype=np.float32)
    if invalid == "nan":
        states[0, 0] = np.nan
    else:
        states[0, 3:7] = 0
    with pytest.raises(ValueError):
        state_increments(states, np.array([1]))


@pytest.mark.parametrize("ends", [[0, 1], [-1, 1], [1, 1], [2], [1.0], []])
def test_delta_rejects_invalid_episode_boundaries(ends):
    states = np.array([[0, 0, 0, 1, 0, 0, 0, 1] * 2], dtype=np.float32)
    with pytest.raises(ValueError, match="partition"):
        state_increments(states, np.asarray(ends))


def test_delta_preserves_and_reports_source_video_interval_discrepancies(dataset, tmp_path, caplog):
    path = dataset / "meta/episodes/chunk-000/file-000.parquet"
    episodes = pq.read_table(path).to_pylist()
    key = "videos/observation.images.cam_high/to_timestamp"
    episodes[0][key] -= 0.2
    pq.write_table(pa.Table.from_pylist(episodes), path)
    manifest = convert(dataset, tmp_path / "delta")
    assert manifest["source_video_interval_warnings"] == [
        {
            "source_episode_index": 0,
            "video_key": "observation.images.cam_high",
            "data_frames": 4,
            "interval_frames": pytest.approx(3),
        }
    ]
    assert V30Source(tmp_path / "delta").episodes[0][key] == episodes[0][key]
    assert "mismatched source video interval" in caplog.text


def test_subset_reindexes_noncontiguous_episodes_and_preserves_video_offsets(dataset, tmp_path):
    from lerobot.evaluation.scripts.subset_robodojo_tasks import subset

    output = tmp_path / "bowls"
    manifest = subset(dataset, output, ["0"], robodojo_root=None)
    result = V30Source(output)
    assert manifest["episode_map"] == [{"source": 0, "subset": 0}, {"source": 2, "subset": 1}]
    assert manifest["total_frames"] == 5
    assert result.info["splits"] == {"train": "0:2"}
    assert result.info["total_tasks"] == 1
    assert result.tasks == {0: "Stack the three bowls together."}
    data = pq.read_table(result.files)
    assert matrix(data["index"])[:, 0].tolist() == list(range(5))
    assert matrix(data["episode_index"])[:, 0].tolist() == [0, 0, 0, 0, 1]
    assert matrix(data["frame_index"])[:, 0].tolist() == [0, 1, 2, 3, 0]
    assert matrix(data["task_index"])[:, 0].tolist() == [0] * 5
    old = pq.read_table(V30Source(dataset).files).take(pa.array([0, 1, 2, 3, 7]))
    for key in ("action", "observation.state", "action.ee_ort6d_pos", "timestamp"):
        assert old[key].equals(data[key])
    stats = json.loads((output / "meta/stats.json").read_text())
    for key in data.column_names:
        assert stats[key] == numeric_stats(matrix(data[key]))
    for original_id, episode in zip((0, 2), result.episodes, strict=True):
        original = V30Source(dataset).episodes[original_id]
        for key in original:
            if key.startswith("videos/"):
                assert episode[key] == original[key]
    for video in (output / "videos").rglob("*.mp4"):
        assert not video.is_symlink()
        assert video.read_bytes() == (dataset / video.relative_to(output)).read_bytes()
    assert pd.read_parquet(output / "meta/tasks.parquet").index.tolist() == [
        "Stack the three bowls together."
    ]
    with pytest.raises(FileExistsError):
        subset(dataset, output, ["0"], robodojo_root=None)


def test_subset_selects_aliases_multiple_tasks_and_composes_with_delta_export(dataset, tmp_path):
    from lerobot.evaluation.scripts.subset_robodojo_tasks import resolve_tasks, subset

    robodojo = tmp_path / "RoboDojo"
    tasks = robodojo / "task/RoboDojo/tasks"
    tasks.mkdir(parents=True)
    (tasks / "stack_bowls.py").write_text(
        "raise RuntimeError('Task files must never be imported')\n"
        "def gen_instruction(self):\n"
        "    return ['Stack the three bowls together.']\n"
    )
    source = V30Source(dataset)
    assert resolve_tasks(source, ["stack_bowls", "1"], robodojo) == {0, 1}
    with pytest.raises(ValueError, match="Unknown task"):
        resolve_tasks(source, ["not_a_task"], robodojo)
    convert(dataset, tmp_path / "delta")
    output = tmp_path / "subset"
    manifest = subset(
        tmp_path / "delta",
        output,
        ["stack_bowls", "Build a block tower."],
        robodojo_root=robodojo,
        video_mode="symlink",
    )
    assert manifest["total_episodes"] == 3 and manifest["total_frames"] == 8
    result = V30Source(output)
    assert result.info["delta_action_contract"]["reference"] == "next_measured_state"
    assert (output / "meta/delta_action_manifest.json").is_file()
    expected = pq.read_table(V30Source(tmp_path / "delta").files)
    actual = pq.read_table(result.files)
    for key in DELTA_COLUMNS:
        assert expected[key].equals(actual[key])
    assert all(path.is_symlink() for path in (output / "videos").rglob("*.mp4"))


def test_subset_reindexes_selected_nonzero_task_id_and_shards(dataset, tmp_path):
    from lerobot.evaluation.scripts.subset_robodojo_tasks import subset

    info_path = dataset / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["data_files_size_in_mb"] = 0.0001
    write_json(info_path, info)
    output = tmp_path / "all"
    subset(dataset, output, ["0", "1"], robodojo_root=None)
    result = V30Source(output)
    assert len(result.files) == 3
    for index, episode in enumerate(result.episodes):
        assert episode["data/file_index"] == index
    subset(dataset, tmp_path / "task1", ["1"], robodojo_root=None)
    result = V30Source(tmp_path / "task1")
    assert result.tasks == {0: "Build a block tower."}
    assert matrix(pq.read_table(result.files)["task_index"])[:, 0].tolist() == [0, 0, 0]


def test_ort6d_known_rotation_uses_interleaved_columns_and_wxyz():
    values = np.array([[1, 2, 3, 1, 0, 0, 1, 0.5] * 2], dtype=np.float32)
    expected = np.array([[1, 2, 3, 0, -1, 1, 0, 0, 0, 0.5] * 2], dtype=np.float32)
    np.testing.assert_allclose(pose_to_ort6d(values), expected, atol=1e-7)
    values[:, 3:7] *= -1
    np.testing.assert_allclose(pose_to_ort6d(values), expected, atol=1e-7)


def test_complete_ort6d_audit_reports_all_rows_and_refuses_report_overwrite(dataset, tmp_path):
    from lerobot.evaluation.scripts.audit_robodojo_ort6d import audit

    report = tmp_path / "audit.json"
    result = audit(dataset, report)
    assert result["passed"] and result["complete"] and result["total_frames"] == 8
    assert len(result["files"]) == 3
    for key in ("action", "observation.state"):
        entry = result["features"][key]
        assert entry["checked_rows"] == 8 and entry["invalid_rows"] == entry["mismatch_rows"] == 0
        assert entry["max_abs_error"] == 0
    assert json.loads(report.read_text()) == result
    with pytest.raises(FileExistsError):
        audit(dataset, report)


@pytest.mark.parametrize("corruption", ["ort6d", "nan", "zero_quaternion", "nonunit_quaternion", "index"])
def test_ort6d_audit_detects_corrupted_rows(dataset, tmp_path, corruption):
    from lerobot.evaluation.scripts.audit_robodojo_ort6d import audit

    path = dataset / "data/chunk-000/file-001.parquet"
    table = pq.read_table(path)
    key = "action.ee_ort6d_pos" if corruption in ("ort6d", "nan") else "action"
    if corruption == "index":
        table = table.set_column(
            table.schema.get_field_index("index"), "index", pa.array([999] * len(table), type=pa.int64())
        )
    else:
        values = matrix(table[key]).copy()
        if corruption == "ort6d":
            values[0, 5] += 0.2
        elif corruption == "nan":
            values[0, 5] = np.nan
        elif corruption == "zero_quaternion":
            values[0, 3:7] = 0
        else:
            values[0, 3:7] *= 2
        table = table.set_column(table.schema.get_field_index(key), key, vector_array(values))
    pq.write_table(table, path)
    report = tmp_path / "audit.json"
    if corruption == "index":
        with pytest.raises(ValueError, match="Inconsistent index"):
            audit(dataset, report)
        assert not report.exists()
        return
    result = audit(dataset, report)
    assert not result["passed"]
    assert result["features"]["observation.state"]["mismatch_rows"] == 0
    entry = result["features"]["action"]
    if corruption == "ort6d":
        assert entry["mismatch_rows"] == 1
    elif corruption == "nonunit_quaternion":
        assert entry["nonunit_quaternion_rows"] == 1
    else:
        assert entry["invalid_rows"] == 1
    if corruption != "nonunit_quaternion":
        assert entry["examples"][0]["index"] == 2


@pytest.mark.parametrize("tolerance", [0, -1, float("nan"), float("inf")])
def test_ort6d_audit_rejects_invalid_tolerance(dataset, tmp_path, tolerance):
    from lerobot.evaluation.scripts.audit_robodojo_ort6d import audit

    with pytest.raises(ValueError, match="Tolerance"):
        audit(dataset, tmp_path / "audit.json", tolerance)
