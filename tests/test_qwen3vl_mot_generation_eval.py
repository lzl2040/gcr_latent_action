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


def test_generation_manifest_refuses_split_drift(tmp_path):
    first = {"format_version": 1, "fingerprint": "abc", "samples": []}
    path = persist_generation_manifest(tmp_path, first)

    assert json.loads(path.read_text(encoding="utf-8")) == first
    persist_generation_manifest(tmp_path, first)
    with pytest.raises(RuntimeError, match="split changed"):
        persist_generation_manifest(
            tmp_path,
            {"format_version": 1, "fingerprint": "different", "samples": []},
        )


def test_generation_evaluator_publishes_complete_step_directory(tmp_path):
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

        def heldout_validation_manifest(self):
            return {
                "format_version": 1,
                "fingerprint": self.heldout_eval_fingerprint,
                "samples": self.heldout_validation_samples(),
            }

        def get_heldout_item(self, dataset_idx, frame_idx):
            assert (dataset_idx, frame_idx) == (0, 4)
            return {
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

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, batch, task_type, step):
            assert task_type == "generate_validation"
            assert step == 11
            self.calls += 1
            batch_size = batch["action"].shape[0]
            return {
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

    model = FakeModel().train()
    evaluator = GenerationEvaluator(
        config=GenerationEvalConfig(
            enabled=True,
            datasets="heldout",
            episodes_per_dataset=1,
            seed=11,
            batch_size=1,
        ),
        dataset=FakeDataset(),
        output_dir=tmp_path,
        policy_config=SimpleNamespace(cosmos3_dir="/unused", n_action_steps=2),
        device=torch.device("cpu"),
        rank=0,
        local_rank=0,
    )
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
    assert (tmp_path / "generation_eval" / "latest_generation_eval").read_text(
        encoding="utf-8"
    ).strip() == step_dir.name

    evaluator.run(model, step=12)
    assert model.calls == 1
