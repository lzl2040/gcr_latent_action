import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.common.policies.qwen3vl_mot.generation_eval import (
    GenerationEvalConfig,
    GenerationEvaluator,
    _save_action_artifacts,
    _save_tactile_artifacts,
    _save_video_artifacts,
    persist_generation_manifest,
)


def test_generation_artifacts_are_directly_viewable(tmp_path):
    video_metrics = _save_video_artifacts(
        tmp_path,
        torch.zeros(3, 3, 8, 8, dtype=torch.uint8),
        torch.ones(3, 3, 8, 8),
        fps=4,
    )
    assert video_metrics["video_pixel_mae"] == 1.0
    assert (tmp_path / "video_comparison.mp4").stat().st_size > 0
    assert (tmp_path / "video_contact_sheet.png").stat().st_size > 0

    stats = {
        "mean": torch.cat([torch.tensor([10.0, 20.0]), torch.zeros(38)]),
        "std": torch.cat([torch.tensor([2.0, 4.0]), torch.ones(38)]),
        "min": torch.cat([torch.tensor([8.0, 16.0]), torch.full((38,), -torch.inf)]),
        "max": torch.cat([torch.tensor([12.0, 24.0]), torch.full((38,), torch.inf)]),
    }
    valid_mask = torch.cat([torch.ones(2), torch.zeros(38)])
    action_metrics = _save_action_artifacts(
        tmp_path,
        torch.zeros(4, 40),
        torch.full((4, 40), 100.0),
        stats,
        valid_mask,
    )
    assert action_metrics["action_valid_dimensions"] == 2
    assert action_metrics["action_clip_fraction"] == 1.0
    assert (tmp_path / "action_comparison.png").stat().st_size > 0
    assert "Denormalized valid action" in (tmp_path / "action_summary.html").read_text(encoding="utf-8")

    tactile_metrics = _save_tactile_artifacts(
        tmp_path,
        torch.zeros(2, 3, 8, 8, dtype=torch.uint8),
        torch.zeros(2, 3, 8, 8),
        torch.tensor([1, 0], dtype=torch.bool),
        torch.full((2, 3), -1.0),
        torch.ones(2, 3),
    )
    assert tactile_metrics["tactile_valid_pads"] == 1
    assert tactile_metrics["tactile_pixel_mae"] == 0.0
    assert (tmp_path / "tactile_comparison.png").stat().st_size > 0


def test_generation_manifest_warns_and_preserves_previous_split(tmp_path, caplog):
    first = {"format_version": 1, "fingerprint": "abc", "samples": []}
    path = persist_generation_manifest(tmp_path, first)

    assert json.loads(path.read_text(encoding="utf-8")) == first
    original_mtime = path.stat().st_mtime_ns
    persist_generation_manifest(tmp_path, first)
    assert path.stat().st_mtime_ns == original_mtime
    assert not caplog.records
    assert not (tmp_path / "split_manifest_history").exists()

    second = {"format_version": 1, "fingerprint": "different", "samples": []}
    assert persist_generation_manifest(tmp_path, second) == path
    assert json.loads(path.read_text(encoding="utf-8")) == second
    history = list((tmp_path / "split_manifest_history").glob("*.json"))
    assert len(history) == 1
    assert json.loads(history[0].read_text(encoding="utf-8")) == first
    assert "Generation validation split changed" in caplog.text
    assert "not directly comparable" in caplog.text

    caplog.clear()
    persist_generation_manifest(tmp_path, second)
    assert not caplog.records
    third = {**second, "samples": [{"episode_index": 1, "frame_index": 10}]}
    persist_generation_manifest(tmp_path, third)
    assert json.loads(path.read_text(encoding="utf-8")) == third
    history = list((tmp_path / "split_manifest_history").glob("*.json"))
    assert len(history) == 2
    assert {json.loads(item.read_text())["fingerprint"] for item in history} == {"abc", "different"}


@pytest.mark.parametrize("fail_during", ["archive", "update"])
def test_generation_manifest_keeps_current_split_when_persistence_fails(
    tmp_path, monkeypatch, fail_during
):
    import lerobot.common.policies.qwen3vl_mot.generation_eval as generation_eval

    first = {"format_version": 1, "fingerprint": "abc", "samples": []}
    path = persist_generation_manifest(tmp_path, first)
    write = generation_eval._atomic_write_text

    def fail_write(target, text):
        is_archive = target.parent.name == "split_manifest_history"
        if is_archive == (fail_during == "archive"):
            raise OSError("disk unavailable")
        write(target, text)

    monkeypatch.setattr(generation_eval, "_atomic_write_text", fail_write)
    with pytest.raises(OSError, match="disk unavailable"):
        persist_generation_manifest(tmp_path, {**first, "fingerprint": "different"})
    assert json.loads(path.read_text(encoding="utf-8")) == first


@pytest.mark.parametrize("multiview", [False, True])
@pytest.mark.parametrize("validation_split", ["heldout", "training"])
@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("changed_split", [False, True])
def test_generation_evaluator_publishes_complete_step_directory(
    tmp_path, caplog, multiview, validation_split, joint, changed_split
):
    class FakeDataset:
        heldout_eval_fingerprint = "split-1"
        norm_stats = [
            {
                "action": {
                    "mean": torch.zeros(40),
                    "std": torch.ones(40),
                    "min": torch.full((40,), -1.0),
                    "max": torch.full((40,), 1.0),
                    "mask": torch.cat([torch.ones(2), torch.zeros(38)]),
                }
            }
        ]

        def heldout_validation_samples(self):
            return [
                {
                    "dataset": "heldout",
                    "dataset_index": 0,
                    "training_source": False,
                    "episode_index": 3,
                    "episode_start": 0,
                    "episode_end": 40,
                    "frame_index": 4,
                    "horizon": 31,
                    "fps": 30.0,
                    "anchor_stride": 6,
                }
            ]

        def __init__(self):
            if joint:
                self.norm_stats = [{"action": self.norm_stats[0]["action"], "state": {
                    **self.norm_stats[0]["action"], "mean": torch.full((40,), 10.0),
                    "min": torch.full((40,), -torch.inf), "max": torch.full((40,), torch.inf),
                }}]

        def heldout_validation_manifest(self):
            return {
                "format_version": 1,
                "fingerprint": self.heldout_eval_fingerprint,
                "samples": self.heldout_validation_samples(),
            }

        def get_heldout_item(self, dataset_idx, frame_idx):
            assert (dataset_idx, frame_idx) == (0, 4)
            item = {
                "video": torch.zeros(3, 3, 8, 8, dtype=torch.uint8),
                "action": torch.zeros(2, 40),
                "action_mask": torch.cat([torch.ones(2), torch.zeros(38)]),
                "tactile_image": torch.zeros(
                    1,
                    2,
                    3,
                    8,
                    8,
                    dtype=torch.uint8,
                ),
                "tactile_image_mask": torch.ones(1),
                "tactile_img_mean": torch.full((1, 3), -1.0),
                "tactile_img_std": torch.ones(1, 3),
                "task": "move",
            }
            if multiview:
                item["video_views"] = item["video"][None].expand(3, -1, -1, -1, -1).clone()
                item["video_views"][2] = 255
                item["video_view_mask"] = torch.tensor([1, 0, 1], dtype=torch.bool)
            if joint:
                item["observation.state"] = torch.ones(4, 40)
                item["observation.state"][0] = 1000
                item["state_mask"] = torch.cat([torch.ones(2), torch.zeros(38)])
            return item

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, batch, task_type, step):
            assert task_type == "generate_validation"
            assert step == 11
            self.calls += 1
            batch_size = batch["action"].shape[0]
            prediction = {
                "video_latents": torch.zeros(batch_size, 4, 1, 1, 1),
                "action_normalized": torch.zeros(batch_size, 2, 40),
                "tactile_decoded_z": torch.zeros(
                    batch_size,
                    1,
                    3,
                    8,
                    8,
                ),
                "tactile_mask": torch.ones(
                    batch_size,
                    1,
                    dtype=torch.bool,
                ),
            }
            if multiview:
                prediction["video_latents_views"] = prediction["video_latents"][:, None].expand(
                    -1, 3, -1, -1, -1, -1
                )
                prediction["video_view_mask"] = torch.ones(batch_size, 3, dtype=torch.bool)
            if joint:
                prediction["state_normalized"] = torch.zeros(batch_size, 4, 40)
            return prediction

    dataset = FakeDataset()
    dataset.validation_split = validation_split
    model = FakeModel().train()
    if changed_split:
        previous_manifest = {
            "format_version": 1,
            "fingerprint": "previous-split",
            "samples": [{"dataset": "heldout", "episode_index": 1, "frame_index": 0}],
        }
        persist_generation_manifest(tmp_path / "generation_eval", previous_manifest)
    evaluator = GenerationEvaluator(
        config=GenerationEvalConfig(
            enabled=True,
            datasets="heldout",
            episodes_per_dataset=1,
            seed=11,
            batch_size=1,
        ),
        dataset=dataset,
        output_dir=tmp_path,
        policy_config=SimpleNamespace(
            cosmos3_dir="/unused", n_action_steps=2, rgb_views=("primary", "secondary", "wrist")
        ),
        device=torch.device("cpu"),
        rank=0,
        local_rank=0,
    )
    assert json.loads((evaluator.root / "split_manifest.json").read_text()) == (
        dataset.heldout_validation_manifest()
    )
    if changed_split:
        assert "Generation validation split changed" in caplog.text
        history = list((evaluator.root / "split_manifest_history").glob("*.json"))
        assert len(history) == 1
        assert json.loads(history[0].read_text()) == previous_manifest
    evaluator._prepare_video_decoder = lambda: None
    evaluator._decode_video = lambda latents: torch.zeros(
        latents.shape[0],
        3,
        3,
        8,
        8,
    )

    evaluator.run(model, step=12)

    step_dir = tmp_path / "generation_eval" / "step_00000012"
    assert model.training
    assert model.calls == 1
    assert (step_dir / "index.html").is_file()
    assert (step_dir / "metrics.json").is_file()
    payload = json.loads((step_dir / "metrics.json").read_text())
    assert payload["split_fingerprint"] == dataset.heldout_eval_fingerprint
    if joint:
        sample_dir = step_dir / "heldout" / "episode_000003_frame_000000004"
        assert (sample_dir / "state_comparison.png").is_file()
        assert (sample_dir / "state_summary.html").is_file()
        assert "Future state prediction" in (sample_dir / "index.html").read_text()
        assert payload["samples"][0]["metrics"]["state_mae"] == 1.0
        assert payload["samples"][0]["metrics"]["state_valid_dimensions"] == 2
    assert payload["validation_split"] == validation_split
    assert ("not held-out generalization" in (step_dir / "index.html").read_text()) == (
        validation_split == "training"
    )
    if multiview:
        sample_dir = step_dir / "heldout" / "episode_000003_frame_000000004"
        assert (sample_dir / "video_comparison.mp4").is_file()
        assert (sample_dir / "wrist" / "video_comparison.mp4").is_file()
        assert not (sample_dir / "secondary").exists()
        document = (sample_dir / "index.html").read_text(encoding="utf-8")
        assert 'src="wrist/video_comparison.mp4"' in document
        assert "secondary camera" not in document
        metrics = json.loads((step_dir / "metrics.json").read_text())["samples"][0]["metrics"]
        assert metrics["video_valid_views"] == ["primary", "wrist"]
        assert metrics["video_primary_pixel_mae"] == 0.0
        assert metrics["video_wrist_pixel_mae"] == 1.0
        assert metrics["video_pixel_mae"] == 0.5
    assert (tmp_path / "generation_eval" / "latest_generation_eval").read_text(
        encoding="utf-8"
    ).strip() == step_dir.name

    evaluator.run(model, step=12)
    assert model.calls == 1


def test_multiview_video_export_skips_samples_without_valid_video(tmp_path):
    evaluator = object.__new__(GenerationEvaluator)
    evaluator.policy_config = SimpleNamespace(rgb_views=("primary", "secondary", "wrist"))
    evaluator.config = GenerationEvalConfig()
    evaluator._decode_video = lambda _: pytest.fail("Missing cameras must not be decoded")
    batch = {
        "video_view_mask": torch.zeros(1, 3, dtype=torch.bool),
        "video_views": torch.zeros(1, 3, 3, 3, 8, 8, dtype=torch.uint8),
    }
    prediction = {
        "video_view_mask": torch.ones(1, 3, dtype=torch.bool),
        "video_latents_views": torch.zeros(1, 3, 4, 1, 1, 1),
    }
    assert evaluator._save_sample_videos(tmp_path, batch, prediction, 0) == {"video_valid_views": []}
    assert not list(tmp_path.iterdir())
