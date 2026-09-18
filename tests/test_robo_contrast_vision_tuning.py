from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers.models.dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel

from lerobot.common.policies.ace.configuration_robo_contrast import RoboContrastConfig
from lerobot.common.policies.ace.modeling_robo_contrast import (
    RoboContrast,
    _configure_vision_tuning,
    _vision_lora_target_modules,
)
from lerobot.common.utils.deepspeed_checkpoint import (
    DATA_PARALLEL_WORLD_SIZE_KEY,
    OPTIMIZER_GROUP_SIGNATURE_KEY,
    align_fresh_scheduler_to_step,
    load_checkpoint_with_optimizer_fallback,
    optimizer_group_signature,
)


def _tiny_dinov3() -> DINOv3ViTModel:
    return DINOv3ViTModel(
        DINOv3ViTConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            image_size=32,
            patch_size=16,
            num_register_tokens=0,
        )
    )


def _policy_shell(backbone: nn.Module, mode: str) -> RoboContrast:
    policy = RoboContrast.__new__(RoboContrast)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(vision_tuning_mode=mode)
    policy.perception_encoder = nn.Module()
    policy.perception_encoder.vision_backbone = backbone
    return policy


def test_legacy_freeze_flag_maps_to_lora_or_full() -> None:
    lora = RoboContrastConfig(freeze_vision_encoder=True)
    full = RoboContrastConfig(freeze_vision_encoder=False)
    frozen = RoboContrastConfig(
        freeze_vision_encoder=False,
        vision_tuning_mode="frozen",
    )
    assert lora.vision_tuning_mode == "lora" and lora.freeze_vision_encoder
    assert full.vision_tuning_mode == "full" and not full.freeze_vision_encoder
    assert frozen.vision_tuning_mode == "frozen" and frozen.freeze_vision_encoder


def test_lora_targets_only_last_attention_blocks() -> None:
    targets, layers = _vision_lora_target_modules(
        _tiny_dinov3(),
        backbone_kind="dinov3",
        num_layers=2,
    )

    assert layers == [1, 2]
    assert len(targets) == 8
    assert all(name.startswith(("layer.1.", "layer.2.")) for name in targets)
    assert all(".attention." in name for name in targets)


def test_vision_lora_preserves_initial_output_and_receives_gradients() -> None:
    torch.manual_seed(0)
    backbone = _tiny_dinov3()
    backbone.eval()
    pixels = torch.randn(2, 3, 32, 32)
    expected = backbone(pixel_values=pixels).last_hidden_state.detach()
    config = RoboContrastConfig(
        vision_backbone="dinov3",
        vision_tuning_mode="lora",
        vision_lora_rank=4,
        vision_lora_alpha=4,
        vision_lora_layers=2,
    )

    tuned = _configure_vision_tuning(backbone, config)
    tuned.eval()
    actual = tuned(pixel_values=pixels).last_hidden_state
    torch.testing.assert_close(actual, expected)

    trainable = [
        (name, parameter)
        for name, parameter in tuned.named_parameters()
        if parameter.requires_grad
    ]
    assert trainable
    assert all("lora_" in name for name, _ in trainable)

    actual.float().square().mean().backward()
    assert all(parameter.grad is not None for _, parameter in trainable)


def test_full_and_frozen_vision_modes_set_base_trainability() -> None:
    full = _configure_vision_tuning(
        _tiny_dinov3(),
        RoboContrastConfig(vision_tuning_mode="full"),
    )
    frozen = _configure_vision_tuning(
        _tiny_dinov3(),
        RoboContrastConfig(vision_tuning_mode="frozen"),
    )

    assert all(parameter.requires_grad for parameter in full.parameters())
    assert not any(parameter.requires_grad for parameter in frozen.parameters())


def test_optimizer_uses_reduced_lr_for_trainable_vision_parameters() -> None:
    model = RoboContrast.__new__(RoboContrast)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        optimizer_lr=1e-4,
        vision_lr_scale=0.1,
        tactile_lr_scale=0.2,
    )
    model.perception_encoder = nn.Module()
    model.perception_encoder.vision_backbone = nn.Linear(4, 4)
    model.perception_encoder.projection = nn.Linear(4, 4)
    model.physical_encoder = nn.Module()
    model.physical_encoder.tactile_cnn = nn.Linear(4, 4)

    groups = model.get_optim_params()

    assert len(groups) == 3
    assert "lr" not in groups[0]
    assert groups[0]["group_name"] == "main"
    assert groups[1]["group_name"] == "vision"
    assert groups[2]["group_name"] == "tactile"
    assert groups[1]["lr"] == pytest.approx(1e-5)
    assert groups[2]["lr"] == pytest.approx(2e-5)


def test_frozen_checkpoint_loads_into_lora_without_changing_output() -> None:
    torch.manual_seed(11)
    source = _policy_shell(_tiny_dinov3(), "frozen").eval()
    pixels = torch.randn(2, 3, 32, 32)
    expected = source.perception_encoder.vision_backbone(
        pixel_values=pixels
    ).last_hidden_state

    config = RoboContrastConfig(
        vision_tuning_mode="lora",
        vision_lora_rank=4,
        vision_lora_alpha=4,
        vision_lora_layers=1,
    )
    target = _policy_shell(
        _configure_vision_tuning(_tiny_dinov3(), config),
        "lora",
    ).eval()
    incompatible = target.load_state_dict(source.state_dict(), strict=True)
    actual = target.perception_encoder.vision_backbone(
        pixel_values=pixels
    ).last_hidden_state

    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_lora_checkpoint_cannot_silently_load_into_non_lora_mode() -> None:
    config = RoboContrastConfig(
        vision_tuning_mode="lora",
        vision_lora_rank=4,
        vision_lora_alpha=4,
        vision_lora_layers=1,
    )
    source = _policy_shell(_configure_vision_tuning(_tiny_dinov3(), config), "lora")
    target = _policy_shell(_tiny_dinov3(), "full")

    with pytest.raises(RuntimeError, match="silently discard"):
        target.load_state_dict(source.state_dict(), strict=False)


def test_lora_checkpoint_requires_matching_adapter_config() -> None:
    source_config = RoboContrastConfig(
        vision_tuning_mode="lora",
        vision_lora_rank=4,
        vision_lora_alpha=4,
        vision_lora_layers=1,
    )
    target_config = RoboContrastConfig(
        vision_tuning_mode="lora",
        vision_lora_rank=4,
        vision_lora_alpha=8,
        vision_lora_layers=1,
    )
    source = _policy_shell(
        _configure_vision_tuning(_tiny_dinov3(), source_config),
        "lora",
    )
    target = _policy_shell(
        _configure_vision_tuning(_tiny_dinov3(), target_config),
        "lora",
    )

    with pytest.raises(RuntimeError, match="rank/alpha"):
        target.load_state_dict(source.state_dict(), strict=False)


def test_optimizer_group_mismatch_falls_back_to_model_only_resume() -> None:
    class FakeScheduler:
        def __init__(self):
            self.position = "fresh"
            self.stepped_to = None

        def state_dict(self):
            return {"position": self.position}

        def load_state_dict(self, state):
            self.position = state["position"]

        def step(self, step):
            self.stepped_to = step

    class FakeEngine:
        def __init__(self, scheduler, saved_signature):
            self.scheduler = scheduler
            self.saved_signature = saved_signature
            self.calls = []

        def load_checkpoint(self, checkpoint_dir, **kwargs):
            self.calls.append((checkpoint_dir, kwargs))
            if kwargs["load_optimizer_states"]:
                self.scheduler.position = "loaded-before-optimizer-error"
                raise ValueError("loaded state dict has a different number of parameter groups")
            return "/checkpoint/global_step10", {
                "step": 17,
                OPTIMIZER_GROUP_SIGNATURE_KEY: self.saved_signature,
                DATA_PARALLEL_WORLD_SIZE_KEY: 8,
            }

    scheduler = FakeScheduler()
    signature = "sha256:matching"
    engine = FakeEngine(scheduler, signature)
    load_path, state, optimizer_restored = load_checkpoint_with_optimizer_fallback(
        engine,
        "/checkpoint",
        signature,
        8,
        scheduler,
    )
    align_fresh_scheduler_to_step(scheduler, state["step"], gradient_accumulation_steps=4)

    assert load_path == "/checkpoint/global_step10"
    assert not optimizer_restored
    assert scheduler.position == "fresh"
    assert scheduler.stepped_to == 4
    assert not engine.calls[0][1]["load_optimizer_states"]
    assert engine.calls[1][1]["load_optimizer_states"]


def test_optimizer_signature_detects_same_group_count_with_different_shapes() -> None:
    first = nn.Linear(10, 1, bias=False)
    second = nn.Linear(2, 1, bias=False)
    first_optimizer = torch.optim.AdamW(first.parameters(), lr=1e-4)
    second_optimizer = torch.optim.AdamW(second.parameters(), lr=1e-4)

    assert optimizer_group_signature(first, first_optimizer) != optimizer_group_signature(
        second,
        second_optimizer,
    )


def test_optimizer_signature_mismatch_skips_optimizer_checkpoint() -> None:
    class FakeEngine:
        def __init__(self):
            self.calls = []

        def load_checkpoint(self, checkpoint_dir, **kwargs):
            self.calls.append(kwargs)
            assert not kwargs["load_optimizer_states"]
            return "/checkpoint/global_step10", {
                "step": 17,
                OPTIMIZER_GROUP_SIGNATURE_KEY: "sha256:old",
                DATA_PARALLEL_WORLD_SIZE_KEY: 8,
            }

    engine = FakeEngine()
    _, _, optimizer_restored = load_checkpoint_with_optimizer_fallback(
        engine,
        "/checkpoint",
        "sha256:new",
        8,
    )

    assert not optimizer_restored
    assert len(engine.calls) == 1


def test_world_size_change_skips_zero_optimizer_checkpoint() -> None:
    class FakeEngine:
        def __init__(self):
            self.calls = []

        def load_checkpoint(self, checkpoint_dir, **kwargs):
            self.calls.append(kwargs)
            assert not kwargs["load_optimizer_states"]
            return "/checkpoint/global_step10", {
                "step": 17,
                OPTIMIZER_GROUP_SIGNATURE_KEY: "sha256:same",
                DATA_PARALLEL_WORLD_SIZE_KEY: 4,
            }

    engine = FakeEngine()
    _, _, optimizer_restored = load_checkpoint_with_optimizer_fallback(
        engine,
        "/checkpoint",
        "sha256:same",
        8,
    )

    assert not optimizer_restored
    assert len(engine.calls) == 1
