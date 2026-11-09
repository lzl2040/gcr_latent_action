"""Compatibility with the flat-batch policy API of this LeRobot checkout."""

import importlib.metadata
import logging
from typing import NamedTuple

import torch
from packaging.version import Version

ACTION = "action"
OBS_STATE = "observation.state"


def _available(name: str, minimum: str) -> bool:
    try:
        return Version(importlib.metadata.version(name)) >= Version(minimum)
    except importlib.metadata.PackageNotFoundError:
        return False


_transformers_available = _available("transformers", "5.22.0")
_diffusers_available = _available("diffusers", "0.38.0")


def require_package(name: str, *, extra: str) -> None:
    minimum = {"transformers": "5.22.0", "diffusers": "0.38.0"}[name]
    if not _available(name, minimum):
        raise ImportError(
            f"Cosmos3 requires {name}>={minimum}. Use a separate environment with "
            f'pip install -e ".[{extra}]"; do not upgrade the Transformers 4 MoT environment.'
        )


def pad_vector(vector: torch.Tensor, size: int) -> torch.Tensor:
    if vector.shape[-1] > size:
        raise ValueError(f"Action/state width {vector.shape[-1]} exceeds max_action_dim={size}.")
    return torch.nn.functional.pad(vector, (0, size - vector.shape[-1]))


def resolve_safetensors_device(device) -> str:
    return str(device)


def log_model_loading_keys(missing, unexpected) -> None:
    if missing or unexpected:
        logging.warning("Cosmos3 checkpoint: missing=%s unexpected=%s", missing, unexpected)


class ChunkPolicySpec(NamedTuple):
    prediction_steps: int
    execution_steps: int


class TransitionKey:
    OBSERVATION = "observation"
    ACTION = ACTION


class FlatBatchStep:
    """Run an upstream complementary-data step without the newer pipeline framework."""

    def __call__(self, batch: dict) -> dict:
        self.transition = {
            TransitionKey.OBSERVATION: {k: v for k, v in batch.items() if k.startswith("observation.")},
            TransitionKey.ACTION: batch.get(ACTION),
        }
        try:
            return self.complementary_data(dict(batch))
        finally:
            self.transition = None
