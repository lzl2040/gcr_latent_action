"""Explicit stage-one module transfer for the Qwen3-VL MoT policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TransferReport:
    copied_tensors: int
    copied_parameters: int
    destination_parameters: int

    @property
    def coverage(self) -> float:
        return self.copied_parameters / max(1, self.destination_parameters)


def merge_peft_module(module: nn.Module) -> nn.Module:
    """Merge a PEFT adapter into its base module when the checkpoint used LoRA."""
    merge = getattr(module, "merge_and_unload", None)
    return merge() if callable(merge) else module


def _normalise_source_name(name: str, prefixes: tuple[str, ...]) -> str:
    while name.startswith("base_model.model."):
        name = name[len("base_model.model.") :]
    for prefix in prefixes:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def transfer_matching_module(
    source: nn.Module,
    destination: nn.Module,
    *,
    source_prefixes: tuple[str, ...],
    ignored_destination_prefixes: tuple[str, ...] = (),
) -> TransferReport:
    """Copy same-shaped parameters after removing known wrapper prefixes."""
    destination_state = destination.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    for source_name, tensor in source.state_dict().items():
        normalised = _normalise_source_name(source_name, source_prefixes)
        candidates = [normalised]
        if ".base_layer." in normalised:
            candidates.append(normalised.replace(".base_layer.", "."))
        for candidate in candidates:
            target = destination_state.get(candidate)
            if target is not None and target.shape == tensor.shape:
                mapped[candidate] = tensor
                break

    destination.load_state_dict(mapped, strict=False)
    eligible = {
        name: tensor
        for name, tensor in destination_state.items()
        if not any(name.startswith(prefix) for prefix in ignored_destination_prefixes)
    }
    return TransferReport(
        copied_tensors=len(mapped),
        copied_parameters=sum(destination_state[name].numel() for name in mapped),
        destination_parameters=sum(tensor.numel() for tensor in eligible.values()),
    )


def transfer_stage1_perception(
    stage1_perception: nn.Module,
    qwen_model: nn.Module,
) -> tuple[TransferReport, TransferReport]:
    """Transfer Qwen-compatible vision and text tensors from the stage-one perception side."""
    vision_source = merge_peft_module(stage1_perception.vision_backbone)
    stage1_perception.vision_backbone = vision_source
    vision_report = transfer_matching_module(
        vision_source,
        qwen_model.visual,
        source_prefixes=("vision_model.", "model.visual.", "visual."),
        ignored_destination_prefixes=("merger.linear_fc", "deepstack_merger_list."),
    )

    text_source = merge_peft_module(stage1_perception.text_backbone)
    stage1_perception.text_backbone = text_source
    text_report = transfer_matching_module(
        text_source,
        qwen_model.language_model,
        source_prefixes=(
            "language_model.",
            "model.language_model.",
            "text_model.",
            "model.text_model.",
        ),
    )
    return vision_report, text_report
