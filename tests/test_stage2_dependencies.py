from __future__ import annotations

import pytest

from lerobot.common.utils.stage2_dependencies import (
    STAGE2_INSTALL_COMMAND,
    validate_stage2_versions,
)


def _versions(**overrides: str) -> dict[str, str]:
    versions = {
        "torch": "2.9.1+cu130",
        "peft": "0.18.1",
        "torchao": "0.15.0",
        "bitsandbytes": "0.50.2",
    }
    versions.update(overrides)
    return versions


def test_stage2_versions_support_torch_27_python_only_profile() -> None:
    profile, errors = validate_stage2_versions(_versions(torch="2.7.0+cu126"))

    assert errors == ()
    assert profile is not None
    assert profile.name == "torch-2.7-python-only"
    assert profile.requires_python_only_torchao


def test_stage2_versions_support_local_torch_291_profile() -> None:
    profile, errors = validate_stage2_versions(_versions())

    assert errors == ()
    assert profile is not None
    assert profile.name == "torch-2.9.1"
    assert not profile.requires_python_only_torchao


@pytest.mark.parametrize("torch_version", ["2.6.0", "2.8.0", "2.9.0", "2.10.0"])
def test_stage2_versions_reject_unverified_torch_profiles(
    torch_version: str,
) -> None:
    profile, errors = validate_stage2_versions(_versions(torch=torch_version))

    assert profile is None
    assert any(f"torch=={torch_version} is unsupported" in error for error in errors)


@pytest.mark.parametrize(
    ("package", "installed"),
    [
        ("peft", "0.19.0"),
        ("torchao", "0.16.0"),
        ("bitsandbytes", "0.51.0"),
    ],
)
def test_stage2_versions_reject_incompatible_packages(
    package: str,
    installed: str,
) -> None:
    profile, errors = validate_stage2_versions(_versions(**{package: installed}))

    assert profile is not None
    assert any(f"{package}=={installed} is unsupported" in error for error in errors)


def test_stage2_install_command_preserves_existing_torch_build() -> None:
    assert "'torchao>=0.15,<0.16'" in STAGE2_INSTALL_COMMAND
    assert "'torch>=" not in STAGE2_INSTALL_COMMAND
    assert "'torch==" not in STAGE2_INSTALL_COMMAND
