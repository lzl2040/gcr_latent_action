import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import generate_libero_pro_futures as futures


def test_future_grid_matches_baseline_training_and_native_source_indices():
    offsets = futures.future_offsets()
    assert offsets == [4, 8, 12, 16, 19, 23, 27, 31]
    with pytest.raises(ValueError, match="distinct"):
        futures.future_offsets(chunk_size=8, video_frames=9)


def test_native_canvas_splits_side_by_side_views_without_padding():
    frames = np.full((2, 6, 12, 3), 255, dtype=np.uint8)
    frames[:, :4, :4] = 10
    frames[:, :4, 4:8] = 20
    views = futures.split_native_canvas(frames, (4, 8))
    assert views["primary"].shape == (2, 4, 4, 3)
    assert (views["primary"] == 10).all()
    assert (views["wrist"] == 20).all()
    with pytest.raises(ValueError, match="canvas"):
        futures.split_native_canvas(frames, (8, 8))


@pytest.mark.parametrize(
    "changed", ["input_sha256", "instruction", "future_source_offsets", "source_fps", "future_times_s"]
)
def test_comparison_rejects_mismatched_conditions_or_time_grid(tmp_path, changed):
    baseline = tmp_path / "baseline"
    native = tmp_path / "native"
    baseline.mkdir()
    native.mkdir()
    metadata = {
        "input_sha256": "same-observation",
        "instruction": "pick up the bowl",
        "future_source_offsets": futures.future_offsets(),
        "future_times_s": [index / 20 for index in futures.future_offsets()],
        "source_fps": 20,
    }
    (baseline / "metadata.json").write_text(json.dumps(metadata))
    metadata[changed] = "different"
    (native / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=changed):
        futures.compare(
            SimpleNamespace(baseline_dir=baseline, cosmos3_dir=native, output_dir=tmp_path / "comparison")
        )


def test_changed_saved_observation_cannot_be_used(tmp_path):
    (tmp_path / "observation.npz").write_bytes(b"changed input")
    (tmp_path / "metadata.json").write_text(json.dumps({"observation_sha256": "original"}))
    with pytest.raises(ValueError, match="observation changed"):
        futures.load_input(tmp_path)


@pytest.mark.parametrize("native_size", [2, 4])
def test_comparison_requires_matched_resolution_and_preserves_time(tmp_path, monkeypatch, native_size):
    import imageio.v2 as imageio

    baseline, native, output = (tmp_path / name for name in ("baseline", "native", "comparison"))
    for directory in (baseline, native, output):
        directory.mkdir()
    metadata = {
        "input_sha256": "same-observation",
        "instruction": "pick up the bowl",
        "future_source_offsets": futures.future_offsets(),
        "future_times_s": [index / 20 for index in futures.future_offsets()],
        "source_fps": 20,
    }
    for directory in (baseline, native):
        (directory / "metadata.json").write_text(json.dumps(metadata))
    native_frames = np.stack(
        [np.full((native_size, native_size, 3), index * 20, dtype=np.uint8) for index in range(9)]
    )
    ours = np.zeros((9, 4, 4, 3), dtype=np.uint8)
    np.savez(baseline / "frames.npz", primary=ours, wrist=ours)
    np.savez(native / "frames.npz", primary=native_frames, wrist=native_frames)
    captured = {}

    def save_video(path, frames, **kwargs):
        captured[path.name] = (np.stack(frames), kwargs)

    monkeypatch.setattr(imageio, "mimsave", save_video)
    args = SimpleNamespace(baseline_dir=baseline, cosmos3_dir=native, output_dir=output)
    if native_size != 4:
        with pytest.raises(ValueError, match="same decoded resolution"):
            futures.compare(args)
        assert not captured
        return
    futures.compare(args)
    for frames, options in captured.values():
        assert frames.shape == (9, 28, 8, 3)
        np.testing.assert_array_equal(frames[:, 24:, :4], native_frames)
        assert options["macro_block_size"] == 1
    with np.load(native / "frames.npz") as original:
        np.testing.assert_array_equal(original["primary"], native_frames)
    saved = json.loads((output / "metadata.json").read_text())
    assert saved["camera_display"]["primary"]["cosmos3_spatial_resampling"] == "none"
