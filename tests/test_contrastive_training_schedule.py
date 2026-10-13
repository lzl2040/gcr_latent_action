import json
import math
import os
import subprocess
import sys
from pathlib import Path
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
    _planned_update_steps,
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


def test_fsdp_stage2_steps_cap_optimizer_updates() -> None:
    assert _planned_update_steps(1_001, 4, 1_000) == 251
    assert _planned_update_steps(1_001, 4, 17) == 17
    with pytest.raises(ValueError, match="steps must be positive"):
        _planned_update_steps(1_001, 4, 0)


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


def test_fsdp_output_dir_is_exact_and_validation_is_idempotent(tmp_path) -> None:
    output_dir = tmp_path / "stage2_run"
    cfg = draccus.parse(
        FSDPTrainPipelineConfig,
        args=[
            "--dataset.repo_id=whatever",
            "--policy.type=qwen3vl_mot",
            "--job_name=stage2_run",
            f"--output_dir={output_dir}",
        ],
    )

    cfg.validate()
    assert cfg.output_dir == output_dir

    cfg.validate()
    assert cfg.output_dir == output_dir


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
        qwen3vl_dir="/Data/saved-qwen",
        cosmos3_dir="/Data/saved-cosmos",
        generation_depth=3,
        optimizer_lr=3e-5,
        scheduler_decay_steps=77_000,
    )
    saved_policy._save_pretrained(checkpoint_root)
    cfg = SimpleNamespace(
        weight_resume=True,
        output_dir=checkpoint_root,
        job_name="run",
        policy=Qwen3VLMoTConfig(
            stage1_checkpoint="",
            qwen3vl_dir="/mnt/runtime-qwen",
            cosmos3_dir="/mnt/runtime-cosmos",
        ),
        use_policy_training_preset=True,
        optimizer=None,
        scheduler=None,
    )

    _load_stage2_resume_policy_config(cfg)

    assert cfg.policy.generation_depth == 3
    assert cfg.policy.initialize_from_stage1 is False
    assert cfg.policy.stage1_checkpoint == "/path/that/no/longer/exists"
    assert cfg.policy.qwen3vl_dir == "/mnt/runtime-qwen"
    assert cfg.policy.cosmos3_dir == "/mnt/runtime-cosmos"
    assert cfg.optimizer.lr == 3e-5
    assert cfg.scheduler.num_decay_steps == 77_000


def test_stage2_resume_allows_task_weight_curriculum(tmp_path) -> None:
    checkpoint_root = tmp_path / "run"
    checkpoint_root.mkdir()
    task_names = ("i2v", "action_prediction", "tactile_prediction")
    saved_policy = Qwen3VLMoTConfig(
        stage1_policy_config={"vision_backbone": "qwen3vl"},
        initialize_from_stage1=False,
        task_names=task_names,
        task_weights=(0.6, 0.2, 0.2),
    )
    saved_policy._save_pretrained(checkpoint_root)
    cfg = SimpleNamespace(
        weight_resume=True,
        output_dir=checkpoint_root,
        policy=Qwen3VLMoTConfig(
            task_names=task_names,
            task_weights=(0.7, 0.0, 0.3),
        ),
        use_policy_training_preset=False,
        optimizer=None,
        scheduler=None,
    )

    _load_stage2_resume_policy_config(
        cfg,
        args=[
            '--policy.task_names=["i2v","action_prediction","tactile_prediction"]',
            "--policy.task_weights=[0.7,0.0,0.3]",
        ],
    )

    assert cfg.policy.task_names == task_names
    assert cfg.policy.task_weights == (0.7, 0.0, 0.3)


def test_stage2_resume_rejects_task_list_change(tmp_path) -> None:
    checkpoint_root = tmp_path / "run"
    checkpoint_root.mkdir()
    saved_policy = Qwen3VLMoTConfig(
        stage1_policy_config={"vision_backbone": "qwen3vl"},
        initialize_from_stage1=False,
        task_names=("i2v", "tactile_prediction"),
        task_weights=(0.7, 0.3),
    )
    saved_policy._save_pretrained(checkpoint_root)
    cfg = SimpleNamespace(
        weight_resume=True,
        output_dir=checkpoint_root,
        policy=Qwen3VLMoTConfig(
            task_names=("i2v",),
            task_weights=(1.0,),
        ),
        use_policy_training_preset=False,
        optimizer=None,
        scheduler=None,
    )

    with pytest.raises(ValueError, match="Cannot change `policy.task_names`"):
        _load_stage2_resume_policy_config(
            cfg,
            args=[
                '--policy.task_names=["i2v"]',
                "--policy.task_weights=[1.0]",
            ],
        )


@pytest.mark.parametrize("saved_version", [None, 1])
@pytest.mark.parametrize("old_name", ["forward_dynamics", "inverse_dynamics", "action_video_prediction"])
def test_stage2_resume_rejects_previous_dynamics_contract(tmp_path, saved_version, old_name):
    Qwen3VLMoTConfig(
        stage1_policy_config={"vision_backbone": "qwen3vl"}, initialize_from_stage1=False,
        task_names=("joint_prediction",), task_weights=(1.0,),
    )._save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    payload = json.loads(config_path.read_text())
    payload["task_names"] = [old_name]
    if saved_version is None:
        del payload["task_contract_version"]
    else:
        payload["task_contract_version"] = saved_version
    config_path.write_text(json.dumps(payload))
    cfg = SimpleNamespace(
        weight_resume=True, output_dir=tmp_path, policy=Qwen3VLMoTConfig(),
        use_policy_training_preset=False,
    )
    with pytest.raises(ValueError, match="previous task contract"):
        _load_stage2_resume_policy_config(cfg, args=[])


def test_stage2_resume_preserves_new_three_task_contract_and_rejects_reordering(tmp_path):
    task_names = ("forward_dynamics", "inverse_dynamics", "joint_prediction")
    Qwen3VLMoTConfig(
        stage1_policy_config={"vision_backbone": "qwen3vl"}, initialize_from_stage1=False,
        task_names=task_names, task_weights=(0.3, 0.3, 0.4),
    )._save_pretrained(tmp_path)
    cfg = SimpleNamespace(
        weight_resume=True, output_dir=tmp_path, policy=Qwen3VLMoTConfig(),
        use_policy_training_preset=False,
    )
    _load_stage2_resume_policy_config(cfg, args=[])
    assert cfg.policy.task_contract_version == 2
    assert cfg.policy.task_names == task_names
    assert cfg.policy.task_weights == (0.3, 0.3, 0.4)
    cfg.policy = Qwen3VLMoTConfig(task_names=tuple(reversed(task_names)), task_weights=(1, 1, 1))
    with pytest.raises(ValueError, match="Cannot change `policy.task_names`"):
        _load_stage2_resume_policy_config(cfg, args=[
            '--policy.task_names=["joint_prediction","inverse_dynamics","forward_dynamics"]',
        ])


@pytest.mark.parametrize("saved_views", [("primary",), ("primary", "secondary", "wrist")])
def test_stage2_resume_preserves_rgb_views_and_rejects_changes(tmp_path, saved_views):
    saved = Qwen3VLMoTConfig(
        stage1_policy_config={"vision_backbone": "qwen3vl"},
        initialize_from_stage1=False, rgb_views=saved_views,
    )
    saved._save_pretrained(tmp_path)
    cfg = SimpleNamespace(
        weight_resume=True, output_dir=tmp_path, policy=Qwen3VLMoTConfig(),
        use_policy_training_preset=False,
    )
    _load_stage2_resume_policy_config(cfg, args=[])
    assert cfg.policy.rgb_views == saved_views
    other = ("primary", "wrist") if len(saved_views) == 1 else ("primary",)
    cfg.policy = Qwen3VLMoTConfig(rgb_views=other)
    with pytest.raises(ValueError, match="Cannot change `policy.rgb_views`"):
        _load_stage2_resume_policy_config(cfg, args=['--policy.rgb_views=["primary","wrist"]'])


@pytest.mark.parametrize("space", ["raw", "embedding"])
def test_stage2_resume_preserves_physical_flow_space_and_rejects_changes(tmp_path, space):
    saved = Qwen3VLMoTConfig(
        stage1_policy_config={"vision_backbone": "qwen3vl"},
        initialize_from_stage1=False, physical_flow_space=space,
    )
    saved._save_pretrained(tmp_path)
    cfg = SimpleNamespace(
        weight_resume=True, output_dir=tmp_path, policy=Qwen3VLMoTConfig(),
        use_policy_training_preset=False,
    )
    _load_stage2_resume_policy_config(cfg, args=[])
    assert cfg.policy.physical_flow_space == space
    other = "embedding" if space == "raw" else "raw"
    cfg.policy = Qwen3VLMoTConfig(physical_flow_space=other)
    with pytest.raises(ValueError, match="Cannot change `policy.physical_flow_space`"):
        _load_stage2_resume_policy_config(cfg, args=[f"--policy.physical_flow_space={other}"])


def test_old_stage2_config_without_physical_flow_field_restores_raw_contract(tmp_path):
    Qwen3VLMoTConfig(
        stage1_policy_config={"vision_backbone": "qwen3vl"}, initialize_from_stage1=False
    )._save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    payload = json.loads(config_path.read_text())
    del payload["physical_flow_space"]
    del payload["physical_reconstruction_loss_weight"]
    config_path.write_text(json.dumps(payload))
    cfg = SimpleNamespace(
        weight_resume=True, output_dir=tmp_path,
        policy=Qwen3VLMoTConfig(physical_flow_space="embedding"),
        use_policy_training_preset=False,
    )
    _load_stage2_resume_policy_config(cfg, args=[])
    assert cfg.policy.physical_flow_space == "raw"


@pytest.mark.parametrize("checkpoint_present,explicit_mode,expected_mode", [
    (False, "", "embedding"), (False, "raw", "raw"),
    (True, "", None), (True, "embedding", "embedding"),
])
def test_fsdp_launcher_preserves_saved_physical_flow_or_defaults_fresh_to_embedding(
    tmp_path, checkpoint_present, explicit_mode, expected_mode
):
    run_dir = tmp_path / "run"
    if checkpoint_present:
        checkpoint = run_dir / "checkpoint_00000001"
        checkpoint.mkdir(parents=True)
        (run_dir / "latest_checkpoint").write_text(checkpoint.name)
    env = {
        **os.environ,
        "CHECK_PATHS": "false", "DRY_RUN": "true", "WEIGHT_RESUME": "true",
        "OUTPUT_DIR": str(run_dir), "STAGE1_CHECKPOINT": str(tmp_path / "stage1"),
        "PHYSICAL_FLOW_SPACE": explicit_mode, "PYTHON_BIN": sys.executable,
        "NPROC_PER_NODE": "1", "NNODES": "1", "MASTER_PORT": "29599",
    }
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["bash", str(root / "train_qwen3vl_mot_fsdp.sh")],
        cwd=root, env=env, capture_output=True, text=True, check=True, timeout=30,
    )
    if expected_mode is None:
        assert "--policy.physical_flow_space=" not in result.stdout
        assert "physical_flow_space=<from checkpoint>" in result.stdout
    else:
        assert f"--policy.physical_flow_space={expected_mode}" in result.stdout


@pytest.mark.parametrize("kind", ["random", "edge", "nano"])
def test_stage2_resume_preserves_generation_backbone_and_rejects_switches(tmp_path, kind):
    Qwen3VLMoTConfig(
        stage1_policy_config={"vision_backbone": "qwen3vl"}, initialize_from_stage1=False,
        generation_backbone=kind, generation_pretrained_dir="/not/needed/on/resume",
    )._save_pretrained(tmp_path)
    cfg = SimpleNamespace(
        weight_resume=True, output_dir=tmp_path, policy=Qwen3VLMoTConfig(generation_backbone="edge"),
        use_policy_training_preset=False,
    )
    _load_stage2_resume_policy_config(cfg, args=[])
    assert cfg.policy.generation_backbone == kind
    assert cfg.policy.generation_pretrained_dir == "/not/needed/on/resume"
    cfg.policy = Qwen3VLMoTConfig(generation_backbone="nano" if kind != "nano" else "edge")
    with pytest.raises(ValueError, match="Cannot change `policy.generation_backbone`"):
        _load_stage2_resume_policy_config(cfg, args=["--policy.generation_backbone=nano"])


def test_stage2_resume_rejects_generation_tuning_mode_switch(tmp_path):
    Qwen3VLMoTConfig(
        stage1_policy_config={"vision_backbone": "qwen3vl"}, initialize_from_stage1=False,
        generation_backbone="edge", generation_tuning_mode="frozen",
    )._save_pretrained(tmp_path)
    cfg = SimpleNamespace(
        weight_resume=True, output_dir=tmp_path, policy=Qwen3VLMoTConfig(generation_backbone="edge"),
        use_policy_training_preset=False,
    )
    with pytest.raises(ValueError, match="Cannot change `policy.generation_tuning_mode`"):
        _load_stage2_resume_policy_config(cfg, args=["--policy.generation_tuning_mode=full"])


@pytest.mark.parametrize("kind", ["random", "edge", "nano"])
@pytest.mark.parametrize("resume", [False, True])
def test_fsdp_launcher_selects_pretrained_generator_only_on_fresh_runs(tmp_path, kind, resume):
    run_dir = tmp_path / "run"
    if resume:
        checkpoint = run_dir / "checkpoint_00000001"
        checkpoint.mkdir(parents=True)
        (run_dir / "latest_checkpoint").write_text(checkpoint.name)
    env = {
        **os.environ, "CHECK_PATHS": "false", "DRY_RUN": "true", "WEIGHT_RESUME": "true",
        "OUTPUT_DIR": str(run_dir), "STAGE1_CHECKPOINT": str(tmp_path / "stage1"),
        "GENERATION_BACKBONE": "" if resume else kind, "GENERATION_PRETRAINED_DIR": "",
        "GENERATION_TUNING_MODE": "", "PYTHON_BIN": sys.executable,
        "NPROC_PER_NODE": "1", "NNODES": "1", "MASTER_PORT": "29599",
    }
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["bash", str(root / "train_qwen3vl_mot_fsdp.sh")],
        cwd=root, env=env, capture_output=True, text=True, check=True, timeout=30,
    )
    if resume:
        assert "--policy.generation_backbone=" not in result.stdout
        assert "--policy.generation_pretrained_dir=" not in result.stdout
    else:
        assert f"--policy.generation_backbone={kind}" in result.stdout
        if kind != "random":
            name = "Cosmos3-Edge" if kind == "edge" else "Cosmos3-Nano"
            assert f"--policy.generation_pretrained_dir=/mnt/wangxiaofa/pt_weights/{name}" in result.stdout


@pytest.mark.parametrize("stage", ["auto", "pretrain", "finetune"])
def test_fsdp_launcher_wires_stage_and_auto_training_validation(tmp_path, stage):
    root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "CHECK_PATHS": "false", "DRY_RUN": "true", "WEIGHT_RESUME": "false",
        "OUTPUT_DIR": str(tmp_path / "run"), "STAGE1_CHECKPOINT": str(tmp_path / "stage1"),
        "PYTHON_BIN": sys.executable, "NNODES": "1", "NPROC_PER_NODE": "1",
        "TRAINING_STAGE": stage, "DATA_MIX": "libero", "GENERATION_EVAL_ENABLED": "true",
        "GENERATION_EVAL_DATASETS": "auto", "MASTER_PORT": "29599",
    }
    result = subprocess.run(
        ["bash", str(root / "train_qwen3vl_mot_fsdp.sh")],
        cwd=root, env=env, capture_output=True, text=True, check=True, timeout=30,
    )
    assert f"--stage={stage}" in result.stdout
    assert "--data_mix=libero" in result.stdout
    assert "--generation_eval.datasets=auto" in result.stdout
    assert "--generation_eval.enabled=true" in result.stdout
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("mode,expected", [("auto", "auto"), ("true", True), ("false", False)])
def test_fsdp_config_parses_auto_and_explicit_fp8_modes(mode, expected):
    cfg = draccus.parse(
        FSDPTrainPipelineConfig,
        args=["--dataset.repo_id=whatever", "--policy.type=qwen3vl_mot", f"--fsdp.fp8={mode}"],
    )
    cfg.fsdp.validate()
    assert cfg.fsdp.fp8 == expected


@pytest.mark.parametrize("mode", [None, "auto", "true", "false"])
def test_fsdp_launcher_preserves_requested_fp8_mode_in_dry_run(tmp_path, mode):
    root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "CHECK_PATHS": "false", "DRY_RUN": "true", "WEIGHT_RESUME": "false",
        "OUTPUT_DIR": str(tmp_path / "run"), "STAGE1_CHECKPOINT": str(tmp_path / "stage1"),
        "PYTHON_BIN": sys.executable, "NNODES": "1", "NPROC_PER_NODE": "1",
        "FP8_EMULATE": "false", "MASTER_PORT": "29599",
    }
    if mode is None:
        env.pop("FP8_ENABLED", None)
    else:
        env["FP8_ENABLED"] = mode
    result = subprocess.run(
        ["bash", str(root / "train_qwen3vl_mot_fsdp.sh")],
        cwd=root, env=env, capture_output=True, text=True, check=True, timeout=30,
    )
    assert f"--fsdp.fp8={mode or 'auto'}" in result.stdout
    assert "--fsdp.fp8_emulate=false" in result.stdout
