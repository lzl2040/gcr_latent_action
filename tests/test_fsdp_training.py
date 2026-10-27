from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch import nn

from lerobot.common.utils.deepspeed_checkpoint import optimizer_group_signature
from lerobot.common.utils.fsdp_training import (
    FSDP_CHECKPOINT_IO_COMPAT_VERSION,
    FSDPCheckpointIOConfig,
    FSDPTrainingConfig,
    _create_fsdp_device_mesh,
    _local_model_state_statistics,
    _make_checkpoint_writer,
    _restore_replicated_frozen_parameters,
    _run_checkpoint_phase,
    _verify_model_checkpoint,
    convert_policy_to_fp8,
    create_checkpoint_process_group,
    find_fsdp_wrap_modules,
    load_fsdp_checkpoint,
    make_distributed_timeout,
    resolve_fp8_mode,
    resolve_latest_fsdp_checkpoint,
)


class _Generation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.eligible = nn.Linear(128, 256, bias=False)
        self.output = nn.Linear(128, 40, bias=False)
        self.lora_A = nn.Linear(128, 128, bias=False)


class _LanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(128, 128, bias=False)])


class _UnderstandingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _LanguageModel()


class _Understanding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _UnderstandingModel()


class _Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.generation = _Generation()
        self.understanding = _Understanding()


class _LocalShard:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor


class _ShardedTensorLike(torch.Tensor):
    def __new__(cls):
        return torch.Tensor._make_subclass(
            cls,
            torch.empty(0),
            False,
        )

    def local_shards(self):
        return [_LocalShard(torch.ones(2, 3))]


def test_fp8_conversion_respects_scope_alignment_and_lora_exclusion() -> None:
    float8_linear_cls = pytest.importorskip(
        "torchao.float8.float8_linear"
    ).Float8Linear
    policy = _Policy()
    converted = convert_policy_to_fp8(
        policy,
        FSDPTrainingConfig(
            fp8_scope="generation",
            fp8_emulate=True,
            fp8_min_features=128,
        ),
        torch.device("cpu"),
    )

    assert converted == ("generation.eligible",)
    assert isinstance(policy.generation.eligible, float8_linear_cls)
    assert type(policy.generation.output) is nn.Linear
    assert type(policy.generation.lora_A) is nn.Linear
    assert type(policy.understanding.model.language_model.layers[0]) is nn.Linear


def test_fp8_conversion_can_include_vlm_transformer() -> None:
    float8_linear_cls = pytest.importorskip(
        "torchao.float8.float8_linear"
    ).Float8Linear
    policy = _Policy()
    converted = convert_policy_to_fp8(
        policy,
        FSDPTrainingConfig(
            fp8_scope="generation_vlm",
            fp8_emulate=True,
            fp8_min_features=128,
        ),
        torch.device("cpu"),
    )

    assert converted == (
        "generation.eligible",
        "understanding.model.language_model.layers.0",
    )
    assert isinstance(
        policy.understanding.model.language_model.layers[0],
        float8_linear_cls,
    )


@pytest.mark.parametrize(
    "capability,expected",
    [((8, 0), False), ((8, 6), False), ((9, 0), True), ((10, 0), True)],
)
def test_auto_fp8_selects_only_supported_native_hardware(monkeypatch, capability, expected, caplog):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: capability)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    config = FSDPTrainingConfig()
    assert resolve_fp8_mode(config, torch.device("cuda", 0)) is expected
    assert config.fp8_emulate is False
    if not expected:
        assert "BF16" in caplog.text


def test_a100_auto_keeps_bf16_linears_and_backward_without_emulation(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (8, 0))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    policy = _Policy().to(dtype=torch.bfloat16)
    inputs = torch.ones(2, 128, dtype=torch.bfloat16)
    expected = policy.generation.eligible(inputs).detach()
    config = FSDPTrainingConfig()
    assert convert_policy_to_fp8(policy, config, torch.device("cuda", 0)) == ()
    assert type(policy.generation.eligible) is nn.Linear
    actual = policy.generation.eligible(inputs)
    torch.testing.assert_close(actual, expected)
    actual.float().square().mean().backward()
    assert torch.isfinite(policy.generation.eligible.weight.grad).all()
    assert policy.generation.eligible.weight.dtype == torch.bfloat16
    assert config.fp8_emulate is False


def test_explicit_native_fp8_on_a100_fails_with_bf16_guidance(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (8, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "NVIDIA A100-SXM4-80GB")
    with pytest.raises(RuntimeError, match="FP8_ENABLED=false"):
        resolve_fp8_mode(FSDPTrainingConfig(fp8=True), torch.device("cuda", 0))


def test_explicit_fp8_disabled_does_not_probe_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: pytest.fail("CUDA must not be probed"))
    assert resolve_fp8_mode(FSDPTrainingConfig(fp8=False), torch.device("cpu")) is False


def test_auto_fp8_uses_same_bf16_mode_when_another_rank_lacks_support(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    original_tensor = torch.tensor

    def cpu_support_flag(value, *, dtype, device):
        assert device == torch.device("cuda", 0)
        return original_tensor(value, dtype=dtype)

    def other_rank_is_a100(flag, *, op):
        assert op == torch.distributed.ReduceOp.MIN
        assert flag.item() == 1
        flag.zero_()

    monkeypatch.setattr(torch, "tensor", cpu_support_flag)
    monkeypatch.setattr(torch.distributed, "all_reduce", other_rank_is_a100)
    assert resolve_fp8_mode(FSDPTrainingConfig(), torch.device("cuda", 0)) is False


def test_find_fsdp_wrap_modules_selects_repeated_trainable_blocks() -> None:
    model = nn.Module()
    model.blocks = nn.ModuleList(
        [
            nn.Linear(128, 128, bias=False),
            nn.Linear(128, 128, bias=False),
            nn.Linear(4, 4, bias=False),
        ]
    )
    model.frozen_blocks = nn.ModuleList([nn.Linear(128, 128, bias=False)])
    model.frozen_blocks[0].requires_grad_(False)

    selected = find_fsdp_wrap_modules(model, min_wrap_params=1_000)

    assert tuple(name for name, _ in selected) == ("blocks.0", "blocks.1")


def test_fsdp_training_config_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError, match="fp8 must be"):
        FSDPTrainingConfig(fp8="invalid").validate()
    with pytest.raises(ValueError, match="Unsupported FP8 recipe"):
        FSDPTrainingConfig(fp8_recipe="invalid").validate()
    with pytest.raises(ValueError, match="Unsupported FP8 scope"):
        FSDPTrainingConfig(fp8_scope="invalid").validate()


def test_resolve_latest_checkpoint_requires_pointer(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="latest_checkpoint"):
        resolve_latest_fsdp_checkpoint(tmp_path)


def test_restore_replicated_frozen_parameters_uses_canonical_fsdp_names() -> None:
    inner = nn.Module()
    inner.weight = nn.Parameter(torch.zeros(2, 3), requires_grad=False)
    root = nn.Module()
    root.block = nn.Module()
    root.block._fsdp_wrapped_module = inner
    wrapper = type("Wrapper", (), {"module": root})()

    _restore_replicated_frozen_parameters(
        wrapper,
        {"block.weight": torch.full((2, 3), 7.0)},
    )

    torch.testing.assert_close(inner.weight, torch.full((2, 3), 7.0))


def test_checkpoint_phase_reports_rank_local_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, op, group=None: None,
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 3)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 1)

    def gather_error(output, error, group=None) -> None:
        output[0] = error

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather_error)

    def fail() -> None:
        raise OSError("rank-local write failed")

    with pytest.raises(
        RuntimeError,
        match=(
            "Distributed checkpoint phase 'save optimizer' failed: "
            "rank 3: OSError: rank-local write failed"
        ),
    ):
        _run_checkpoint_phase("save optimizer", fail, torch.device("cpu"))


def test_checkpoint_phase_logs_start_and_completion(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, op, group=None: None,
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    caplog.set_level(
        logging.INFO,
        logger="lerobot.common.utils.fsdp_training",
    )

    result = _run_checkpoint_phase(
        "save model shards",
        lambda: "saved",
        torch.device("cpu"),
    )

    assert result == "saved"
    assert "Distributed checkpoint phase started: save model shards" in caplog.text
    assert "Distributed checkpoint phase completed: save model shards" in caplog.text


def test_checkpoint_phase_uses_cpu_for_gloo_control_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_group = object()
    observed = {}

    def all_reduce(tensor, op, group=None) -> None:
        observed["device"] = tensor.device.type
        observed["group"] = group

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group=None: "gloo")

    result = _run_checkpoint_phase(
        "save model shards",
        lambda: "saved",
        torch.device("cuda", 0),
        checkpoint_group,
    )

    assert result == "saved"
    assert observed == {"device": "cpu", "group": checkpoint_group}


def test_checkpoint_process_group_uses_gloo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_group = object()
    observed = {}

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group=None: "nccl")

    def new_group(*, backend, timeout):
        observed["backend"] = backend
        observed["timeout"] = timeout
        return checkpoint_group

    monkeypatch.setattr(torch.distributed, "new_group", new_group)
    timeout = make_distributed_timeout("90")

    assert create_checkpoint_process_group(timeout) is checkpoint_group
    assert observed == {"backend": "gloo", "timeout": timeout}


def test_fsdp_device_mesh_reuses_default_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default_group = torch.distributed.group.WORLD
    expected_mesh = object()
    observed = {}

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def from_group(group, device_type, mesh_dim_names):
        observed["group"] = group
        observed["device_type"] = device_type
        observed["mesh_dim_names"] = mesh_dim_names
        return expected_mesh

    monkeypatch.setattr(
        "lerobot.common.utils.fsdp_training.DeviceMesh.from_group",
        from_group,
    )

    assert _create_fsdp_device_mesh(torch.device("cuda", 3)) is expected_mesh
    assert observed == {
        "group": default_group,
        "device_type": "cuda",
        "mesh_dim_names": ("fsdp",),
    }


def test_checkpoint_io_defaults_disable_fsync(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.delenv("FSDP_CHECKPOINT_SYNC_FILES", raising=False)
    monkeypatch.delenv("FSDP_CHECKPOINT_THREADS", raising=False)
    monkeypatch.delenv("FSDP_CHECKPOINT_HEARTBEAT_SECONDS", raising=False)

    config = FSDPCheckpointIOConfig.from_environment()
    writer = _make_checkpoint_writer(tmp_path / "model", config)

    assert config == FSDPCheckpointIOConfig(
        sync_files=False,
        thread_count=1,
        heartbeat_seconds=60,
    )
    assert writer.sync_files is False
    assert writer.thread_count == 1
    assert writer.overwrite is False


def test_model_state_statistics_use_local_shards_before_tensor_dispatch() -> None:
    tensor_count, total_bytes = _local_model_state_statistics(
        {"weight": _ShardedTensorLike()}
    )

    assert tensor_count == 1
    assert total_bytes == 2 * 3 * torch.ones(1).element_size()


def test_checkpoint_writer_round_trip_is_visible(tmp_path) -> None:
    checkpoint_dir = tmp_path / "model"
    writer = _make_checkpoint_writer(
        checkpoint_dir,
        FSDPCheckpointIOConfig(sync_files=False),
    )
    dcp.save(
        {"model": {"weight": torch.arange(8)}},
        storage_writer=writer,
        no_dist=True,
    )

    file_count, total_bytes = _verify_model_checkpoint(checkpoint_dir)

    assert file_count == 1
    assert total_bytes > 0


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("FSDP_CHECKPOINT_SYNC_FILES", "yes"),
        ("FSDP_CHECKPOINT_THREADS", "0"),
        ("FSDP_CHECKPOINT_THREADS", "1.5"),
        ("FSDP_CHECKPOINT_HEARTBEAT_SECONDS", "-1"),
    ],
)
def test_checkpoint_io_rejects_invalid_environment(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        FSDPCheckpointIOConfig.from_environment()


def test_distributed_timeout_defaults_to_one_hour() -> None:
    assert FSDP_CHECKPOINT_IO_COMPAT_VERSION == 2
    assert make_distributed_timeout().total_seconds() == 3_600
    assert make_distributed_timeout("90").total_seconds() == 5_400


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "invalid"])
def test_distributed_timeout_rejects_invalid_values(value: str) -> None:
    with pytest.raises(
        ValueError,
        match="DISTRIBUTED_TIMEOUT_MINUTES must be a positive integer",
    ):
        make_distributed_timeout(value)


@pytest.fixture
def resume_checkpoint(tmp_path, monkeypatch):
    import lerobot.common.utils.fsdp_training as training

    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    for _ in range(27):
        optimizer.zero_grad()
        model(torch.ones(1, 2)).square().sum().backward()
        optimizer.step()
        scheduler.step()
    config = FSDPTrainingConfig(fp8=False, replicate_frozen_params=False)
    geometry = {
        "steps_per_epoch": 10,
        "total_source_frames": 100,
        "dataset_names": ["original"],
        "full_training_sampler_fingerprint": "old-anchors",
    }
    metadata = {
        "world_size": 1,
        "fsdp_config": asdict(config),
        "training_geometry": geometry,
        "micro_step": 27,
        "update_step": 27,
        "epoch": 2,
        "batch_in_epoch": 7,
    }
    checkpoint_dir = tmp_path / "checkpoint_00000027"
    checkpoint_dir.mkdir()
    (tmp_path / "latest_checkpoint").write_text(checkpoint_dir.name)
    (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata))
    dcp.save(
        {"model": model.state_dict()},
        storage_writer=_make_checkpoint_writer(
            checkpoint_dir / "model", FSDPCheckpointIOConfig(heartbeat_seconds=0)
        ),
        no_dist=True,
    )
    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()
    local_state = {
        "optimizer": optimizer_state,
        "optimizer_signature": optimizer_group_signature(model, optimizer),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.tensor([1, 2, 3], dtype=torch.uint8),
    }
    torch.save(local_state, checkpoint_dir / "optimizer_rank_00000.pt")
    torch.save({"scheduler": scheduler_state}, checkpoint_dir / "trainer_state.pt")
    monkeypatch.setattr(training, "_run_checkpoint_phase", lambda name, fn, *args: fn())
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    cuda_states = []
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state, device: cuda_states.append(state))
    return SimpleNamespace(
        output_dir=tmp_path,
        checkpoint_dir=checkpoint_dir,
        config=config,
        geometry=geometry,
        metadata=metadata,
        model_state=model.state_dict(),
        optimizer_state=optimizer_state,
        scheduler_state=scheduler_state,
        local_state=local_state,
        cuda_states=cuda_states,
    )


@pytest.mark.parametrize(
    "changes,expected_position",
    [
        ({}, (2, 7)),
        ({"total_source_frames": 120}, (2, 7)),
        ({"total_epochs": 50, "total_micro_steps": 500}, (2, 7)),
        ({"steps_per_epoch": 20, "total_source_frames": 200}, (1, 7)),
        ({"steps_per_epoch": 3, "total_source_frames": 30}, (9, 0)),
        ({"steps_per_epoch": 9, "total_source_frames": 90}, (3, 0)),
        ({"full_training_sampler_fingerprint": "padded-tail-anchors"}, (2, 7)),
        ({"sample_pool_fingerprint": "new-pool"}, (2, 7)),
        ({"dataset_names": ["original", "added"]}, (2, 7)),
    ],
)
def test_fsdp_resume_allows_data_changes_and_preserves_training_state(
    resume_checkpoint, caplog, changes, expected_position
):
    checkpoint = resume_checkpoint
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    current_geometry = {**checkpoint.geometry, **changes}

    resume = load_fsdp_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=checkpoint.output_dir,
        fsdp_config=checkpoint.config,
        training_geometry=current_geometry,
        device=torch.device("cpu"),
        checkpoint_io=FSDPCheckpointIOConfig(heartbeat_seconds=0),
    )

    assert resume.micro_step == resume.update_step == 27
    assert (resume.epoch, resume.batch_in_epoch) == expected_position
    assert resume.epoch * current_geometry["steps_per_epoch"] + resume.batch_in_epoch == resume.micro_step
    assert 0 <= resume.batch_in_epoch < current_geometry["steps_per_epoch"]
    torch.testing.assert_close(model.state_dict(), checkpoint.model_state, rtol=0, atol=0)
    torch.testing.assert_close(optimizer.state_dict(), checkpoint.optimizer_state, rtol=0, atol=0)
    assert scheduler.state_dict() == checkpoint.scheduler_state
    assert random.getstate() == checkpoint.local_state["python_rng_state"]
    np.testing.assert_equal(np.random.get_state(), checkpoint.local_state["numpy_rng_state"])
    torch.testing.assert_close(torch.get_rng_state(), checkpoint.local_state["torch_rng_state"])
    torch.testing.assert_close(checkpoint.cuda_states[0], checkpoint.local_state["cuda_rng_state"])
    if changes:
        assert "geometry changed across resume" in caplog.text
        assert "sample order will not be an exact continuation" in caplog.text
        assert "Remapped checkpoint data position" in caplog.text
    else:
        assert not caplog.records

    optimizer.zero_grad()
    model(torch.ones(1, 2)).square().sum().backward()
    optimizer.step()
    scheduler.step()
    assert scheduler.last_epoch == 28
    assert all(state["step"].item() == 28 for state in optimizer.state.values())


@pytest.mark.parametrize(
    "change,error",
    [
        ("world_size", "require the same world size"),
        ("fsdp_config", "FSDP/FP8 settings changed"),
        ("optimizer_signature", "Optimizer parameter groups changed"),
    ],
)
def test_fsdp_resume_retains_non_data_compatibility_checks(resume_checkpoint, change, error):
    checkpoint = resume_checkpoint
    if change == "optimizer_signature":
        state = {**checkpoint.local_state, "optimizer_signature": []}
        torch.save(state, checkpoint.checkpoint_dir / "optimizer_rank_00000.pt")
    else:
        metadata = checkpoint.metadata.copy()
        if change == "world_size":
            metadata["world_size"] = 2
        else:
            metadata["fsdp_config"] = {**metadata["fsdp_config"], "fp8": True}
        (checkpoint.checkpoint_dir / "metadata.json").write_text(json.dumps(metadata))
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    with pytest.raises(ValueError, match=error):
        load_fsdp_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            output_dir=checkpoint.output_dir,
            fsdp_config=checkpoint.config,
            training_geometry={**checkpoint.geometry, "total_source_frames": 200},
            device=torch.device("cpu"),
            checkpoint_io=FSDPCheckpointIOConfig(heartbeat_seconds=0),
        )
