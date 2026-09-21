from __future__ import annotations

import pytest
import torch
from torch import nn

from lerobot.common.utils.fsdp_training import (
    FSDPTrainingConfig,
    convert_policy_to_fp8,
    find_fsdp_wrap_modules,
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
