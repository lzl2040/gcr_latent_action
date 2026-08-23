"""Telling a real instruction apart from filler.

Converted LeRobot datasets very often carry a ``task`` column that is syntactically a string
but semantically absent: ``""``, ``"none"``, ``"n/a"``, or the dataset's own name. Treating
those as language is not harmless. The text tower embeds them into a perfectly ordinary,
*constant* vector, so a model that conditions on language learns "instruction = none" as a
task, and a contrastive model can read that constant as a dataset fingerprint and use it to
solve the matching problem without looking at the images.

This lives in its own module because both the contrastive loader and the vision-only
perception loader need the identical rule, and the latter imports the former -- putting the
rule in either one would create an import cycle or, worse, two copies that drift.
"""

from __future__ import annotations

# Strings that are syntactically language but carry no instruction.
_PLACEHOLDER_TASKS = {
    "", "none", "null", "nil", "n/a", "na", "nan", "-", "--", "unknown", "unknown task",
    "no task", "notask", "task", "tasks", "undefined", "empty", "todo", "placeholder",
}


def is_real_instruction(task: str | None, dataset_name: str = "") -> bool:
    """Whether ``task`` is an actual instruction rather than filler."""
    if task is None:
        return False
    text = str(task).strip()
    if len(text) < 3:
        # Shorter than "go" plus a letter: nothing survives tokenization that is worth
        # conditioning on, and this also catches "?" / "." style filler.
        return False
    lowered = text.lower().rstrip(".!")
    if lowered in _PLACEHOLDER_TASKS:
        return False
    # Several converters fill the task column with the dataset (or repo) name.
    if dataset_name and lowered in (dataset_name.lower(), dataset_name.lower().replace("_", " ")):
        return False
    return True


def dataset_task_strings(ds_meta) -> list[str]:
    """Every distinct instruction a dataset declares, across both metadata layouts.

    v2.1 stores ``{task_index: str}``; v3.0 stores a DataFrame *indexed by the task string*
    with ``task_index`` as its only column, which is easy to mistake for "there are no task
    strings here" when read column-wise.
    """
    tasks = getattr(ds_meta, "tasks", None)
    if tasks is None:
        return []
    if isinstance(tasks, dict):
        return [str(v) for v in tasks.values()]
    index = getattr(tasks, "index", None)
    if index is not None:
        return [str(v) for v in index]
    try:
        return [str(v) for v in tasks]
    except TypeError:
        return []
