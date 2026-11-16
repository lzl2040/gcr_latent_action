import json
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import visualize_stage1_tactile_contact as visualization


def test_onset_windows_keep_chronological_equally_spaced_times():
    indices = [30, 33, 36, 39, 42]
    windows = visualization.onset_windows([0.001, 0.01, 0.025, 0.045, 0.07], indices)
    assert len(windows) == 1
    assert windows[0]["frame_indices"] == indices
    assert windows[0]["frame_gap"] == 3


@pytest.mark.parametrize(
    "curve",
    [[0.02] * 5, [0.07, 0.05, 0.03, 0.01, 0], [0, 0.04, 0.01, 0.06, 0.07]],
)
def test_onset_windows_reject_static_releasing_or_nonmonotonic_contact(curve):
    assert visualization.onset_windows(curve, [0, 3, 6, 9, 12]) == []


def test_pre_post_windows_require_two_stable_before_and_three_changed_after_frames():
    images = torch.full((5, 3, 32, 32), 100, dtype=torch.uint8)
    for index in range(2, 5):
        images[index, :, 10:22, 10:22] += 30 + index * 10
    windows = visualization.pre_post_windows(images, [0, 3, 6, 9, 12], gaps=[3])
    assert len(windows) == 1
    assert windows[0]["frame_indices"] == [0, 3, 6, 9, 12]
    assert windows[0]["frame_phases"] == ["Before change"] * 2 + ["After change"] * 3
    assert windows[0]["contact_boundary_interval_frames"] == [3, 6]
    assert windows[0]["pre_contact_variation"] == 0
    images[1] = images[2]
    assert visualization.pre_post_windows(images, [0, 3, 6, 9, 12], gaps=[3]) == []


def test_pre_post_windows_reject_static_contact_or_a_single_transient_change():
    images = torch.full((5, 3, 32, 32), 100, dtype=torch.uint8)
    assert visualization.pre_post_windows(images, [0, 3, 6, 9, 12], gaps=[3]) == []
    images[2, :, 10:22, 10:22] = 220
    assert visualization.pre_post_windows(images, [0, 3, 6, 9, 12], gaps=[3]) == []


def test_decoder_uses_linear_inverse_zscore_not_sigmoid():
    predicted = torch.tensor([[[[2.0]], [[-3.0]], [[0.5]]]])
    restored = visualization.denormalize_tactile(predicted, [0.3, 0.4, 0.5], [0.1, 0.1, 0.2])
    torch.testing.assert_close(restored.flatten(), torch.tensor([0.5, 0.1, 0.6]))


def test_good_static_background_is_not_enough_for_contact_reconstruction():
    gt = torch.full((5, 3, 32, 32), 128, dtype=torch.uint8)
    for index in range(5):
        gt[index, :, 10:22, 10:22] += index * 12
    correct = visualization.reconstruction_metrics(gt, gt.float() / 255)
    constant = visualization.reconstruction_metrics(gt, gt[:1].expand_as(gt).float() / 255)
    assert correct["nonconstant_contact_reconstruction"]
    assert correct["contact_delta_cosine"] == pytest.approx(1)
    assert constant["mean_psnr_db"] > 20
    assert not constant["nonconstant_contact_reconstruction"]


def test_video_reader_uses_each_camera_file_offset_and_preserves_episode_times(tmp_path, monkeypatch):
    import av

    class Frame:
        def __init__(self, pts):
            self.pts = pts

        def to_ndarray(self, format):
            assert format == "rgb24"
            return np.full((16, 16, 3), self.pts, dtype=np.uint8)

    class Container:
        streams = SimpleNamespace(video=[SimpleNamespace(time_base=Fraction(1, 10), thread_count=0)])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def seek(self, offset, stream):
            pass

        def decode(self, stream):
            return iter(Frame(index) for index in range(30))

    monkeypatch.setattr(av, "open", lambda path: Container())
    info = {"fps": 10, "video_path": "videos/{video_key}/chunk-{chunk_index}/file-{file_index}.mp4"}
    episode = {"length": 9}
    for key, offset in (("tactile", 0.6), ("rgb", 1.1)):
        episode.update(
            {
                f"videos/{key}/chunk_index": 2,
                f"videos/{key}/file_index": 7,
                f"videos/{key}/from_timestamp": offset,
            }
        )
        images, synchronization = visualization.read_frames(tmp_path, info, episode, key, [0, 2, 4, 6, 8])
        assert images[:, 0, 0, 0].tolist() == [round(offset * 10) + index for index in [0, 2, 4, 6, 8]]
        np.testing.assert_allclose(
            synchronization["actual_episode_times_s"], [0, 0.2, 0.4, 0.6, 0.8], atol=1e-8
        )
        assert synchronization["video"].endswith(f"videos/{key}/chunk-2/file-7.mp4")
    monkeypatch.setattr(Container, "decode", lambda self, stream: iter(Frame(index) for index in range(11)))
    images, _ = visualization.read_frames(
        tmp_path, info, episode, "tactile", [0, 2, 4], episode_times=[0, 0.2, 0.400002]
    )
    assert images[:, 0, 0, 0].tolist() == [6, 8, 10]
    with pytest.raises(ValueError, match="episode boundary"):
        visualization.read_frames(tmp_path, info, episode, "rgb", [0, 9])
    with pytest.raises(ValueError, match="increasing"):
        visualization.read_frames(tmp_path, info, episode, "rgb", [3, 2])


def test_codec_rejects_checkpoint_without_trained_decoder(tmp_path, monkeypatch):
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: {"module": {"physical_encoder.tactile_cnn.norm.weight": torch.ones(512)}},
    )
    with pytest.raises(ValueError, match="both trained"):
        visualization.TactileCodec(tmp_path / "checkpoint.pt")


def test_episode_frame_times_check_real_parquet_rows_and_timestamps(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "data.parquet"
    pq.write_table(
        pa.table(
            {
                "episode_index": [7, 7, 7, 8],
                "frame_index": [0, 3, 6, 0],
                "timestamp": [0, 0.10001, 0.2, 0],
            }
        ),
        path,
    )
    info = {"data_path": "data.parquet", "fps": 30}
    episode = {"data/chunk_index": 0, "data/file_index": 0, "episode_index": 7}
    result = visualization.episode_frame_times(tmp_path, info, episode, [0, 3, 6])
    assert result["timestamp_s"] == pytest.approx([0, 0.10001, 0.2])
    with pytest.raises(ValueError, match="exactly"):
        visualization.episode_frame_times(tmp_path, info, episode, [0, 1, 6])
    with pytest.raises(ValueError, match="nominal"):
        visualization.episode_frame_times(tmp_path, {**info, "fps": 10}, episode, [0, 3, 6])


def test_rgb_presentation_crop_is_fixed_across_time_and_preserves_originals():
    images = np.arange(5 * 10 * 20 * 3).reshape(5, 10, 20, 3)
    original = images.copy()
    cropped = visualization.crop_rgb(images, [0.25, 0.3, 0.75, 1])
    np.testing.assert_array_equal(cropped, images[:, 3:10, 5:15])
    np.testing.assert_array_equal(images, original)
    with pytest.raises(ValueError, match="normalized"):
        visualization.crop_rgb(images, [0, 0.9, 1, 0.5])
    with pytest.raises(ValueError, match="empty"):
        visualization.crop_rgb(images, [0, 0, 0.001, 1])


@pytest.mark.parametrize("override", [False, True])
def test_export_respects_per_case_rgb_view_and_crop_with_explicit_overrides(tmp_path, monkeypatch, override):
    from PIL import Image

    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    tactile, wrist, ego = (
        "observation.images.tactile_right_0",
        "observation.images.right_wrist_camera",
        "observation.images.ego",
    )
    (root / "meta/info.json").write_text(
        json.dumps({"features": {key: {"dtype": "video"} for key in (tactile, wrist, ego)}})
    )
    pattern = np.broadcast_to(np.arange(32, dtype=np.uint8)[None, None, :, None], (5, 32, 32, 3))
    frames = {key: pattern + offset for key, offset in ((tactile, 20), (wrist, 60), (ego, 100))}
    monkeypatch.setattr(
        visualization, "read_frames", lambda root, info, episode, key, *a, **k: (frames[key], {})
    )
    monkeypatch.setattr(
        visualization,
        "episode_frame_times",
        lambda *args: {"timestamp_s": [index / 30 for index in range(5)]},
    )
    record = {
        "dataset_root": str(root),
        "dataset": "test",
        "task": "grasp",
        "episode": {"episode_index": 0},
        "frame_indices": list(range(5)),
        "fps": 30,
        "tactile_key": tactile,
        "pixel_mean_01": [0] * 3,
        "pixel_std_01": [1] * 3,
        "figure_rgb_key": ego,
        "figure_rgb_crop": [0, 0, 0.5, 1],
    }
    codec = SimpleNamespace(
        provenance={"image_size": 32},
        reconstruct=lambda images, mean, std: images.float() / 255,
    )
    options = {"figure_rgb_key": wrist, "rgb_crop": [0.5, 0, 1, 1]} if override else {}
    directory = visualization.export_case(record, codec, tmp_path / "output", **options)
    chosen = wrist if override else ego
    expected_crop = frames[chosen][0, :, 16:] if override else frames[chosen][0, :, :16]
    np.testing.assert_array_equal(
        np.array(Image.open(next((directory / "rgb").glob("00_*.png")))), frames[chosen][0]
    )
    np.testing.assert_array_equal(
        np.array(Image.open(next((directory / "rgb_figure").glob("00_*.png")))), expected_crop
    )
    np.testing.assert_array_equal(
        np.array(Image.open(next((directory / "gt_tactile").glob("00_*.png")))), frames[tactile][0]
    )
