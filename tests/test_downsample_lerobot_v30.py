import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from scipy.spatial.transform import Rotation

from lerobot.evaluation.scripts.add_robodojo_delta_actions import convert as add_deltas
from lerobot.evaluation.scripts.robodojo_dataset_utils import V30Source, matrix, numeric_stats
from scripts.downsample_lerobot_v30 import convert, histogram_stats, probe_video, sampling_stride
from tests.test_robodojo_dataset_scripts import make_dataset

FFMPEG = Path(sys.executable).parent / "ffmpeg"
FFPROBE = FFMPEG.with_name("ffprobe")


def decode(path):
    return np.frombuffer(
        subprocess.check_output(
            [str(FFMPEG), "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        ),
        dtype=np.uint8,
    ).reshape(-1, 2, 2, 3)


def make_real_videos(source):
    keys = V30Source(source).videos
    for camera, key in enumerate(keys):
        for file_index, frames in ((0, np.arange(-1, 7)), (1, np.array([7]))):
            pixels = np.broadcast_to(
                (frames[:, None, None, None] * 24 + camera * 5).astype(np.uint8),
                (len(frames), 2, 2, 3),
            )
            path = source / f"videos/{key}/chunk-000/file-{file_index:03d}.mp4"
            subprocess.run(
                [
                    str(FFMPEG),
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    "2x2",
                    "-r",
                    "5",
                    "-i",
                    "pipe:0",
                    "-c:v",
                    "libx264",
                    "-threads",
                    "1",
                    "-crf",
                    "0",
                    "-pix_fmt",
                    "yuv420p",
                    str(path),
                ],
                input=pixels.tobytes(),
                check=True,
            )


@pytest.mark.parametrize("delta", [False, True])
def test_downsample_real_videos_and_cross_file_episodes_with_common_tail_trim(tmp_path, delta):
    if not FFMPEG.is_file():
        pytest.skip("FFmpeg is required for real-video integration.")
    source = make_dataset(tmp_path / "source")
    make_real_videos(source)
    episodes_path = source / "meta/episodes/chunk-000/file-000.parquet"
    episodes = pq.read_table(episodes_path)
    rows = episodes.to_pylist()
    import pyarrow as pa

    for row in rows[:2]:
        for key in V30Source(source).videos:
            for bound in ("from", "to"):
                row[f"videos/{key}/{bound}_timestamp"] += 1 / 5
    pq.write_table(pa.Table.from_pylist(rows, schema=episodes.schema), episodes_path)
    if delta:
        add_deltas(source, tmp_path / "delta")
        source = tmp_path / "delta"
    episodes_path = source / "meta/episodes/chunk-000/file-000.parquet"
    episodes = pq.read_table(episodes_path)
    rows = episodes.to_pylist()
    key = V30Source(source).videos[1]
    rows[1][f"videos/{key}/to_timestamp"] -= 1 / 5
    pq.write_table(pa.Table.from_pylist(rows, schema=episodes.schema), episodes_path)
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="short video"):
        convert(source, tmp_path / "strict", 2.5)
    assert not (tmp_path / "strict").exists()
    output = tmp_path / "output"
    manifest = convert(source, output, 2.5, short_video_policy="trim", workers=2, crf=0)
    assert manifest["complete"] and manifest["total_frames"] == 4
    assert [row["trimmed_source_rows"] for row in manifest["episode_frame_mapping"]] == [0, 1, 0]
    result = V30Source(output)
    original = pq.read_table(V30Source(source).files)
    actual = pq.read_table(result.files)
    indices = [0, 2, 4, 7]
    for name in ("observation.state", "action", "observation.ee_ort6d_pos", "action.ee_ort6d_pos"):
        assert actual[name].equals(original[name].take(indices))
    np.testing.assert_array_equal(matrix(actual["frame_index"])[:, 0], [0, 1, 0, 0])
    np.testing.assert_allclose(matrix(actual["timestamp"])[:, 0], [0, 0.4, 0, 0])
    result.validate_rows(actual, 0)
    assert [row["length"] for row in result.episodes] == [2, 1, 1]
    stats = json.loads((output / "meta/stats.json").read_text())
    for name in actual.column_names:
        assert stats[name] == numeric_stats(matrix(actual[name]))
    if delta:
        old_delta, new_delta = matrix(original["delta_action"]), matrix(actual["delta_action"])
        rotations = Rotation.from_quat(old_delta[:2, 3:7][:, [1, 2, 3, 0]])
        combined = Rotation.from_quat(new_delta[0, 3:7][[1, 2, 3, 0]])
        np.testing.assert_allclose(combined.as_matrix(), (rotations[1] * rotations[0]).as_matrix(), atol=1e-6)
        assert np.max(np.abs(combined.as_matrix() - (rotations[0] * rotations[1]).as_matrix())) > 1e-5
        states = matrix(actual["observation.state"])
        np.testing.assert_allclose(new_delta[0, :3], states[1, :3] - states[0, :3], atol=1e-7)
        assert new_delta[0, 7] == pytest.approx(states[1, 7] - states[0, 7])
        np.testing.assert_array_equal(new_delta[[1, 2, 3]], np.tile([0, 0, 0, 1, 0, 0, 0, 0] * 2, (3, 1)))
        from lerobot.evaluation.scripts.robodojo_dataset_utils import pose_to_ort6d

        np.testing.assert_allclose(matrix(actual["delta_action.ee_ort6d_pos"]), pose_to_ort6d(new_delta))
    for key in result.videos:
        first = output / f"videos/{key}/chunk-000/file-000.mp4"
        original_first = source / first.relative_to(output)
        # Both intervals start at odd video frames: do not decimate the whole file at phase0.
        np.testing.assert_allclose(decode(first), decode(original_first)[[1, 3, 5]], atol=2)
        assert probe_video(first, str(FFPROBE))["fps"] == 2.5
        assert stats[key]["count"] == [4 * 32 * 24]
        for row in result.episodes:
            assert (row[f"videos/{key}/to_timestamp"] - row[f"videos/{key}/from_timestamp"]) == (
                pytest.approx(row["length"] / 2.5)
            )
    for relative, content in before.items():
        assert (source / relative).read_bytes() == content
    with pytest.raises(FileExistsError):
        convert(source, output, 2.5)


def test_rgb_histogram_stats_match_direct_quantiles():
    pixels = np.array([[0, 2, 100], [255, 5, 100], [128, 5, 111], [1, 200, 200]], dtype=np.uint8)
    histogram = np.stack([np.bincount(pixels[:, channel], minlength=256) for channel in range(3)])
    result = histogram_stats(histogram)
    expected = numeric_stats(pixels / 255)
    for name in expected:
        np.testing.assert_allclose(np.asarray(result[name]).reshape(-1), expected[name], atol=1e-15)


def test_downsample_refuses_unknown_delta_contract_and_noninteger_rate(tmp_path):
    source = make_dataset(tmp_path / "source")
    add_deltas(source, tmp_path / "delta")
    source = tmp_path / "delta"
    path = source / "meta/info.json"
    info = json.loads(path.read_text())
    info["delta_action_contract"]["rotation"] = "right multiplied"
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="Unsupported delta contract"):
        convert(source, tmp_path / "output", 2.5)
    assert not (tmp_path / "output").exists()
    with pytest.raises(ValueError, match="integer FPS"):
        sampling_stride(25, 10)
