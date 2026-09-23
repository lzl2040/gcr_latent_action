from types import SimpleNamespace

import pytest
import torch

from lerobot.common.datasets_v30 import video_utils


def test_torchcodec_timestamp_error_retries_with_pyav(monkeypatch) -> None:
    expected = torch.zeros(2, 3, 4, 4, dtype=torch.uint8)
    calls = []

    def fail_torchcodec(*args, **kwargs):
        raise video_utils.FrameTimestampError("one-frame timestamp offset")

    def decode_pyav(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(video_utils, "decode_video_frames_torchcodec", fail_torchcodec)
    monkeypatch.setattr(video_utils, "decode_video_frames_torchvision", decode_pyav)

    actual = video_utils.decode_video_frames(
        "video.mp4",
        [1.0, 2.0],
        5e-4,
        backend="torchcodec",
        return_type="uint8",
    )

    assert actual is expected
    assert len(calls) == 1
    assert calls[0][1]["backend"] == "pyav"
    assert calls[0][1]["return_type"] == "uint8"


def test_non_timestamp_torchcodec_errors_are_not_hidden(monkeypatch) -> None:
    def fail_torchcodec(*args, **kwargs):
        raise RuntimeError("decoder initialization failed")

    monkeypatch.setattr(video_utils, "decode_video_frames_torchcodec", fail_torchcodec)

    with pytest.raises(RuntimeError, match="decoder initialization failed"):
        video_utils.decode_video_frames(
            "video.mp4",
            [1.0],
            5e-4,
            backend="torchcodec",
        )


def test_torchcodec_timestamp_check_keeps_float64_precision() -> None:
    class FakeDecoder:
        metadata = SimpleNamespace(average_fps=30.0)

        def get_frames_at(self, indices):
            assert indices == [264964]
            return SimpleNamespace(
                data=torch.zeros(1, 3, 2, 2, dtype=torch.uint8),
                pts_seconds=torch.tensor([8832.133268229167], dtype=torch.float64),
            )

    class FakeCache:
        def get_decoder(self, video_path):
            return FakeDecoder()

    frames = video_utils.decode_video_frames_torchcodec(
        "video.mp4",
        [8832.133333333333],
        2e-4,
        decoder_cache=FakeCache(),
        return_type="uint8",
    )

    assert frames.dtype == torch.uint8
    assert frames.shape == (1, 3, 2, 2)
