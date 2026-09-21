import math
from types import SimpleNamespace

import draccus
import pytest
import torch

from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.qwen3vl_mot.configuration_qwen3vl_mot import Qwen3VLMoTConfig
from lerobot.scripts.dps_train_contrast import (
    _compute_epoch_schedule,
    _data_read_batch_counts,
    _load_stage2_resume_policy_config,
)
from lerobot.scripts.fsdp_train_contrast import (
    FSDPTrainPipelineConfig,
    _normalize_resume_position,
)


def test_epoch_schedule_covers_source_then_adds_extra_epochs() -> None:
    total_source_frames = 10_000_000_000
    steps_per_epoch = 2_441
    global_samples_per_step = 4_096
    schedule = _compute_epoch_schedule(
        total_source_frames=total_source_frames,
        steps_per_epoch=steps_per_epoch,
        global_samples_per_step=global_samples_per_step,
        extra_epochs=100,
    )

    expected_source_epochs = math.ceil(
        total_source_frames / (steps_per_epoch * global_samples_per_step)
    )
    assert schedule.samples_per_epoch == steps_per_epoch * global_samples_per_step
    assert schedule.source_equivalent_epochs == expected_source_epochs
    assert schedule.total_epochs == expected_source_epochs + 100
    assert schedule.source_equivalent_steps == expected_source_epochs * steps_per_epoch
    assert schedule.total_steps == schedule.total_epochs * steps_per_epoch
    assert schedule.source_equivalent_epochs == 1_001
    assert schedule.total_epochs == 1_101


def test_epoch_schedule_rejects_invalid_sizes() -> None:
    with pytest.raises(ValueError):
        _compute_epoch_schedule(0, 10, 32)
    with pytest.raises(ValueError):
        _compute_epoch_schedule(100, 0, 32)
    with pytest.raises(ValueError):
        _compute_epoch_schedule(100, 10, 0)
    with pytest.raises(ValueError):
        _compute_epoch_schedule(100, 10, 32, extra_epochs=-1)


def test_data_read_batch_counts_are_grouped_by_dataset() -> None:
    fallback_counts, sample_counts = _data_read_batch_counts(
        {
            "dataset_id": torch.tensor([0, 1, 1, 2, 2]),
            "data_read_fallback": torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0]),
        },
        num_datasets=3,
    )

    torch.testing.assert_close(sample_counts, torch.tensor([1, 2, 2]))
    torch.testing.assert_close(fallback_counts, torch.tensor([0, 1, 2]))


def test_optimizer_factory_uses_computed_schedule_length() -> None:
    recorded = {}

    class OptimizerConfig:
        @staticmethod
        def build(parameters):
            return torch.optim.SGD(parameters, lr=0.1)

    class SchedulerConfig:
        @staticmethod
        def build(_optimizer, num_training_steps):
            recorded["num_training_steps"] = num_training_steps
            return None

    cfg = SimpleNamespace(
        use_policy_training_preset=False,
        optimizer=OptimizerConfig(),
        scheduler=SchedulerConfig(),
        steps=123,
    )
    policy = torch.nn.Linear(2, 2)

    make_optimizer_and_scheduler(cfg, policy, num_training_steps=4_567)

    assert recorded["num_training_steps"] == 4_567


def test_fsdp_cli_decodes_nested_runtime_config() -> None:
    cfg = draccus.parse(
        FSDPTrainPipelineConfig,
        args=[
            "--dataset.repo_id=whatever",
            "--policy.type=qwen3vl_mot",
            "--fsdp.fp8=false",
            "--fsdp.fp8_scope=generation",
            "--batch_size=16",
        ],
    )

    assert cfg.fsdp.fp8 is False
    assert cfg.fsdp.fp8_scope == "generation"
    assert cfg.batch_size == 16


def test_fsdp_resume_position_advances_completed_epoch() -> None:
    assert _normalize_resume_position(3, 100, 100) == (4, 0)
    assert _normalize_resume_position(3, 137, 100) == (4, 37)


def test_stage2_resume_uses_saved_embedded_stage1_config(tmp_path) -> None:
    checkpoint_root = tmp_path / "run"
    checkpoint_root.mkdir()
    saved_policy = Qwen3VLMoTConfig(
        stage1_checkpoint="/path/that/no/longer/exists",
        stage1_policy_config={"vision_backbone": "qwen3vl"},
        initialize_from_stage1=False,
        generation_depth=3,
        optimizer_lr=3e-5,
        scheduler_decay_steps=77_000,
    )
    saved_policy._save_pretrained(checkpoint_root)
    cfg = SimpleNamespace(
        weight_resume=True,
        output_dir=checkpoint_root,
        job_name="run",
        policy=Qwen3VLMoTConfig(stage1_checkpoint=""),
        use_policy_training_preset=True,
        optimizer=None,
        scheduler=None,
    )

    _load_stage2_resume_policy_config(cfg)

    assert cfg.policy.generation_depth == 3
    assert cfg.policy.initialize_from_stage1 is False
    assert cfg.policy.stage1_checkpoint == "/path/that/no/longer/exists"
    assert cfg.optimizer.lr == 3e-5
    assert cfg.scheduler.num_decay_steps == 77_000
