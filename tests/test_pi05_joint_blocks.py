import copy

import pytest
import torch
from torch import nn
from transformers import GemmaConfig, PaliGemmaConfig
from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration

from lerobot.common.policies.pi05.modeling_gemma import GemmaModel
from lerobot.common.policies.pi05.paligemma_with_expert import PaliGemmaWithExpertModel
from lerobot.scripts.train_pi05_robodojo_fsdp import canonical_pi05_state_dict


def tiny_joint_model():
    text = GemmaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        use_adarms=False,
    )
    expert = GemmaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        use_adarms=True,
        adarms_cond_dim=8,
    )
    config = PaliGemmaConfig(
        text_config=text.to_dict(),
        vision_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 28,
            "patch_size": 14,
        },
        projection_dim=16,
        image_token_index=31,
    )
    model = PaliGemmaWithExpertModel.__new__(PaliGemmaWithExpertModel)
    nn.Module.__init__(model)
    model.paligemma = PaliGemmaForConditionalGeneration(config)
    model.paligemma.model.language_model = GemmaModel(text)
    model.paligemma.lm_head = nn.Identity()
    model.gemma_expert = GemmaForCausalLM(expert)
    model.gemma_expert.model = GemmaModel(expert)
    model.gemma_expert.model.embed_tokens = None
    model.joint_blocks = None
    return model


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_joint_block_forward_and_gradients_match_legacy(checkpointing, dtype):
    torch.manual_seed(7)
    legacy = tiny_joint_model().to(dtype).train()
    legacy.gemma_expert.model.gradient_checkpointing = checkpointing
    joint = copy.deepcopy(legacy)
    original_parameters = {id(parameter) for parameter in joint.parameters()}
    joint.enable_joint_blocks()
    joint.enable_joint_blocks()
    assert {id(parameter) for parameter in joint.parameters()} == original_parameters
    assert len(joint.paligemma.language_model.layers) == 0
    assert len(joint.gemma_expert.model.layers) == 0
    calls = []
    for index, block in enumerate(joint.joint_blocks):
        block.gradient_checkpointing = checkpointing
        block.register_forward_pre_hook(lambda module, args, index=index: calls.append(index))
    prefix = torch.randn(2, 3, 16, dtype=dtype)
    suffix = torch.randn(2, 2, 8, dtype=dtype)
    condition = torch.randn(2, 8, dtype=dtype)
    mask = torch.zeros(2, 1, 5, 5, dtype=dtype)
    mask[:, :, :3, 3:] = torch.finfo(dtype).min
    positions = torch.arange(5).expand(2, -1)

    def run(model):
        inputs = [prefix.clone().requires_grad_(), suffix.clone().requires_grad_()]
        cond = condition.clone().requires_grad_()
        outputs, cache = model(
            inputs_embeds=inputs,
            attention_mask=mask,
            position_ids=positions,
            adarms_cond=[None, cond],
        )
        assert cache is None
        sum(value.float().square().sum() for value in outputs).backward()
        gradients = {
            f"model.{name}": parameter.grad
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        return outputs, [value.grad for value in inputs] + [cond.grad], gradients

    expected_outputs, expected_inputs, expected_gradients = run(legacy)
    actual_outputs, actual_inputs, actual_gradients = run(joint)
    assert calls == [0, 1]
    for expected, actual in zip(
        expected_outputs + expected_inputs, actual_outputs + actual_inputs, strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_gradients = canonical_pi05_state_dict(actual_gradients)
    assert actual_gradients.keys() == expected_gradients.keys()
    for name in expected_gradients:
        torch.testing.assert_close(actual_gradients[name], expected_gradients[name], rtol=0, atol=0)
    canonical = canonical_pi05_state_dict({f"model.{k}": v for k, v in joint.state_dict().items()})
    assert canonical.keys() == {f"model.{key}" for key in legacy.state_dict()}
    legacy.load_state_dict(
        {name.removeprefix("model."): value for name, value in canonical.items()}, strict=True
    )


def test_joint_training_layout_rejects_cached_inference():
    model = tiny_joint_model()
    model.enable_joint_blocks()
    with pytest.raises(ValueError, match="export for inference"):
        model(inputs_embeds=[torch.randn(1, 3, 16), None])


def test_export_rejects_duplicate_canonical_names():
    with pytest.raises(ValueError, match="Duplicate"):
        canonical_pi05_state_dict(
            {
                "model.joint_blocks.0.language_layer.weight": torch.ones(1),
                "model.paligemma.model.language_model.layers.0.weight": torch.ones(1),
            }
        )
