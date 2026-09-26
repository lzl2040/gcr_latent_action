from __future__ import annotations

import logging

import pytest
import torch
from torch import nn

from lerobot.common.utils.fsdp_training import (
    FSDPTrainingConfig,
    _restore_replicated_frozen_parameters,
    _run_checkpoint_phase,
    convert_policy_to_fp8,
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
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, op: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)

    def gather_error(output, error) -> None:
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
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, op: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
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


def test_distributed_timeout_defaults_to_one_hour() -> None:
    assert make_distributed_timeout().total_seconds() == 3_600
    assert make_distributed_timeout("90").total_seconds() == 5_400


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "invalid"])
def test_distributed_timeout_rejects_invalid_values(value: str) -> None:
    with pytest.raises(
        ValueError,
        match="DISTRIBUTED_TIMEOUT_MINUTES must be a positive integer",
    ):
        make_distributed_timeout(value)
