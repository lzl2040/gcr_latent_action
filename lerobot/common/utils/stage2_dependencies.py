#!/usr/bin/env python

from __future__ import annotations

import inspect
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from packaging.version import InvalidVersion, Version

STAGE2_INSTALL_COMMAND = (
    "python -m pip install --upgrade 'peft>=0.18,<0.19' 'torchao>=0.15,<0.16' 'bitsandbytes>=0.48,<0.51'"
)


@dataclass(frozen=True)
class VersionRange:
    minimum: Version
    maximum: Version

    def accepts(self, installed: Version) -> bool:
        return self.minimum <= installed < self.maximum

    def describe(self) -> str:
        return f">={self.minimum},<{self.maximum}"


@dataclass(frozen=True)
class TorchProfile:
    name: str
    versions: VersionRange
    requires_python_only_torchao: bool


_PACKAGE_RANGES = {
    "peft": VersionRange(Version("0.18.0"), Version("0.19.0")),
    "torchao": VersionRange(Version("0.15.0"), Version("0.16.0")),
    "bitsandbytes": VersionRange(Version("0.48.0"), Version("0.51.0")),
}
_TORCH_PROFILES = (
    TorchProfile(
        name="torch-2.7-python-only",
        versions=VersionRange(Version("2.7.0"), Version("2.8.0")),
        requires_python_only_torchao=True,
    ),
    TorchProfile(
        name="torch-2.9.1",
        versions=VersionRange(Version("2.9.1"), Version("2.9.2")),
        requires_python_only_torchao=False,
    ),
)
_REQUIRED_PACKAGES = ("torch", *_PACKAGE_RANGES)


def validate_stage2_versions(
    package_versions: Mapping[str, str | None],
) -> tuple[TorchProfile | None, tuple[str, ...]]:
    parsed: dict[str, Version] = {}
    errors: list[str] = []
    for package in _REQUIRED_PACKAGES:
        raw_version = package_versions.get(package)
        if raw_version is None:
            errors.append(f"{package} is not installed")
            continue
        try:
            parsed[package] = Version(raw_version)
        except InvalidVersion:
            errors.append(f"{package} has an invalid version: {raw_version!r}")

    profile = None
    torch_version = parsed.get("torch")
    if torch_version is not None:
        profile = next(
            (candidate for candidate in _TORCH_PROFILES if candidate.versions.accepts(torch_version)),
            None,
        )
        if profile is None:
            supported = " or ".join(candidate.versions.describe() for candidate in _TORCH_PROFILES)
            errors.append(f"torch=={torch_version} is unsupported; expected {supported}")

    for package, supported_range in _PACKAGE_RANGES.items():
        installed = parsed.get(package)
        if installed is not None and not supported_range.accepts(installed):
            errors.append(f"{package}=={installed} is unsupported; expected {supported_range.describe()}")
    return profile, tuple(errors)


def _installed_versions() -> dict[str, str | None]:
    installed: dict[str, str | None] = {}
    for package in _REQUIRED_PACKAGES:
        try:
            installed[package] = version(package)
        except PackageNotFoundError:
            installed[package] = None
    return installed


def _probe_fsdp_api(torch_module: Any) -> None:
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        set_model_state_dict,
    )
    from torch.distributed.fsdp import FullyShardedDataParallel
    from torch.distributed.fsdp.wrap import CustomPolicy

    fsdp_parameters = inspect.signature(FullyShardedDataParallel).parameters
    missing_fsdp_parameters = {"ignored_states", "use_orig_params"} - set(fsdp_parameters)
    if missing_fsdp_parameters:
        missing = ", ".join(sorted(missing_fsdp_parameters))
        raise RuntimeError(f"FSDP is missing required parameters: {missing}")

    state_dict_parameters = inspect.signature(StateDictOptions).parameters
    missing_state_dict_parameters = {"full_state_dict", "cpu_offload", "strict"} - set(state_dict_parameters)
    if missing_state_dict_parameters:
        missing = ", ".join(sorted(missing_state_dict_parameters))
        raise RuntimeError(f"StateDictOptions is missing required parameters: {missing}")
    if not all(callable(item) for item in (get_model_state_dict, set_model_state_dict, CustomPolicy)):
        raise RuntimeError("required distributed checkpoint or FSDP policy APIs are unavailable")

    scaled_mm_schema = str(torch_module.ops.aten._scaled_mm.default._schema)
    missing_schema_items = {
        "scale_a",
        "scale_b",
        "out_dtype",
        "use_fast_accum",
    } - {item for item in ("scale_a", "scale_b", "out_dtype", "use_fast_accum") if item in scaled_mm_schema}
    if missing_schema_items:
        missing = ", ".join(sorted(missing_schema_items))
        raise RuntimeError(f"aten::_scaled_mm schema is missing: {missing}")


def _run_fp8_forward_backward(
    torch_module: Any,
    *,
    device: Any,
    dtype: Any,
    emulate: bool,
) -> None:
    from torch import nn
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training

    model = nn.Sequential(nn.Linear(128, 128, bias=False)).to(device=device, dtype=dtype)
    fp8_config = replace(
        Float8LinearConfig.from_recipe_name("rowwise_with_gw_hp"),
        emulate=emulate,
        enable_fsdp_float8_all_gather=False,
    )
    convert_to_float8_training(model, config=fp8_config)
    batch_size = 2 if emulate else 16
    inputs = torch_module.randn(
        batch_size,
        128,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    model(inputs).float().square().mean().backward()
    gradients = (inputs.grad, *(parameter.grad for parameter in model.parameters()))
    if any(gradient is None or not torch_module.isfinite(gradient).all() for gradient in gradients):
        mode = "emulated" if emulate else "native"
        raise RuntimeError(f"{mode} FP8 forward/backward produced invalid gradients")


def _probe_torchao_fp8(torch_module: Any) -> None:
    _run_fp8_forward_backward(
        torch_module,
        device=torch_module.device("cpu"),
        dtype=torch_module.float32,
        emulate=True,
    )
    if not torch_module.cuda.is_available():
        return
    device = torch_module.device("cuda", 0)
    if torch_module.cuda.get_device_capability(device)[0] < 9:
        return
    try:
        _run_fp8_forward_backward(
            torch_module,
            device=device,
            dtype=torch_module.bfloat16,
            emulate=False,
        )
        torch_module.cuda.synchronize(device)
    except Exception as exc:
        raise RuntimeError(f"native CUDA FP8 forward/backward failed: {exc}") from exc


def probe_stage2_runtime(
    profile: TorchProfile,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    errors: list[str] = []
    details: dict[str, Any] = {}
    if profile.requires_python_only_torchao and os.getenv("TORCHAO_FORCE_SKIP_LOADING_SO_FILES") != "1":
        errors.append(
            "torch 2.7.x requires TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1 "
            "because torchao 0.15 C++ extensions target torch 2.9.1"
        )

    try:
        import torch

        _probe_fsdp_api(torch)
    except Exception as exc:
        errors.append(f"required PyTorch FSDP/FP8 APIs are unavailable: {exc}")
    else:
        if not errors:
            try:
                _probe_torchao_fp8(torch)
            except Exception as exc:
                errors.append(f"TorchAO FP8 runtime probe failed: {exc}")

    try:
        from bitsandbytes.optim import AdamW8bit

        inspect.signature(AdamW8bit)
    except Exception as exc:
        errors.append(f"bitsandbytes AdamW8bit is unavailable: {exc}")

    try:
        import lerobot.common.optim.optimizers as optimizer_module
    except Exception as exc:
        errors.append(f"repository optimizer module cannot be imported: {exc}")
    else:
        compat_version = getattr(
            optimizer_module,
            "ADAMW8BIT_SIGNATURE_COMPAT_VERSION",
            None,
        )
        details["optimizer_path"] = optimizer_module.__file__
        details["optimizer_compat_version"] = compat_version
        if compat_version != 1:
            errors.append(
                "the uploaded source predates AdamW8bit signature compatibility "
                f"(marker={compat_version!r}, expected=1)"
            )
    return tuple(errors), details


def run_stage2_dependency_check() -> None:
    installed = _installed_versions()
    profile, errors = validate_stage2_versions(installed)
    runtime_errors: tuple[str, ...] = ()
    details: dict[str, Any] = {}
    if not errors and profile is not None:
        runtime_errors, details = probe_stage2_runtime(profile)
    all_errors = (*errors, *runtime_errors)
    if all_errors:
        formatted_errors = "\n  - ".join(all_errors)
        raise SystemExit(
            "Stage 2 dependency check failed:\n"
            f"  - {formatted_errors}\n"
            "Install the supported packages without replacing the image's "
            "PyTorch/CUDA wheel:\n"
            f"  {STAGE2_INSTALL_COMMAND}"
        )

    assert profile is not None
    torchao_mode = (
        "python-only" if os.getenv("TORCHAO_FORCE_SKIP_LOADING_SO_FILES") == "1" else "native-extensions"
    )
    versions = " ".join(f"{package}={installed[package]}" for package in _REQUIRED_PACKAGES)
    print(f"Stage 2 dependency check: profile={profile.name} torchao_mode={torchao_mode} {versions}")
    print(
        "Stage 2 source check: "
        f"optimizer={details['optimizer_path']} "
        f"adamw8bit_compat={details['optimizer_compat_version']}"
    )


if __name__ == "__main__":
    run_stage2_dependency_check()
