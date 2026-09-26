from __future__ import annotations

import logging

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch import nn

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
    make_distributed_timeout,
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
