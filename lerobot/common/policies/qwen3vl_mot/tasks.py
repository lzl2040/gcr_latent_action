"""Task definitions for the stage-two multimodal flow model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ModalityRole(str, Enum):
    """How a modality participates in one generation task."""

    ABSENT = "absent"
    CLEAN = "clean"
    NOISY = "noisy"
    FUTURE_NOISY = "future_noisy"
    CURRENT_ONLY = "current_only"

    @property
    def is_present(self) -> bool:
        return self is not ModalityRole.ABSENT

    @property
    def is_target(self) -> bool:
        return self in (ModalityRole.NOISY, ModalityRole.FUTURE_NOISY)


@dataclass(frozen=True)
class TaskSpec:
    """One fixed routing pattern through understanding and generation."""

    name: str
    understanding_image: bool
    require_text: bool = False
    video: ModalityRole = ModalityRole.ABSENT
    state: ModalityRole = ModalityRole.ABSENT
    action: ModalityRole = ModalityRole.ABSENT
    tactile: ModalityRole = ModalityRole.ABSENT

    def role(self, modality: str) -> ModalityRole:
        try:
            return getattr(self, modality)
        except AttributeError as exc:
            raise KeyError(f"Unknown stage-two modality {modality!r}.") from exc


TASK_SPECS: dict[str, TaskSpec] = {
    "t2v": TaskSpec(
        name="t2v",
        understanding_image=False,
        require_text=True,
        video=ModalityRole.NOISY,
    ),
    "i2v": TaskSpec(
        name="i2v",
        understanding_image=True,
        video=ModalityRole.FUTURE_NOISY,
    ),
    "forward_dynamics": TaskSpec(
        name="forward_dynamics",
        understanding_image=True,
        video=ModalityRole.FUTURE_NOISY,
        state=ModalityRole.FUTURE_NOISY,
        action=ModalityRole.CLEAN,
    ),
    "inverse_dynamics": TaskSpec(
        name="inverse_dynamics",
        understanding_image=True,
        video=ModalityRole.CLEAN,
        state=ModalityRole.CLEAN,
        action=ModalityRole.NOISY,
    ),
    "action_prediction": TaskSpec(
        name="action_prediction",
        understanding_image=True,
        state=ModalityRole.CURRENT_ONLY,
        action=ModalityRole.NOISY,
    ),
    "state_prediction": TaskSpec(
        name="state_prediction",
        understanding_image=True,
        video=ModalityRole.CLEAN,
        state=ModalityRole.FUTURE_NOISY,
        action=ModalityRole.CLEAN,
    ),
    "tactile_prediction": TaskSpec(
        name="tactile_prediction",
        understanding_image=True,
        video=ModalityRole.CLEAN,
        state=ModalityRole.CLEAN,
        action=ModalityRole.CLEAN,
        tactile=ModalityRole.NOISY,
    ),
}

DEFAULT_TASK_NAMES = tuple(TASK_SPECS)
DEFAULT_TASK_WEIGHTS = (0.10, 0.20, 0.20, 0.15, 0.15, 0.10, 0.10)


def resolve_task_specs(
    names: tuple[str, ...],
    weights: tuple[float, ...],
) -> tuple[tuple[TaskSpec, ...], tuple[float, ...]]:
    if len(names) != len(weights):
        raise ValueError(
            f"`task_names` and `task_weights` must have equal length, got "
            f"{len(names)} and {len(weights)}."
        )
    if not names:
        raise ValueError("At least one stage-two task must be enabled.")
    if len(set(names)) != len(names):
        raise ValueError(f"`task_names` contains duplicates: {names}.")

    unknown = [name for name in names if name not in TASK_SPECS]
    if unknown:
        raise ValueError(
            f"Unknown stage-two tasks {unknown}; available tasks are {tuple(TASK_SPECS)}."
        )
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("`task_weights` must be non-negative and sum to a positive value.")

    total = float(sum(weights))
    return tuple(TASK_SPECS[name] for name in names), tuple(float(weight) / total for weight in weights)
