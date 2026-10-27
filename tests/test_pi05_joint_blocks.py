import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import GemmaConfig, PaliGemmaConfig
from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration

from lerobot.common.policies.pi05.configuration_pi05 import PI05Config
from lerobot.common.policies.pi05.modeling_gemma import GemmaModel
from lerobot.common.policies.pi05.modeling_pi05 import PI05FlowMatching
from lerobot.common.policies.pi05.paligemma_with_expert import (
    PaliGemmaWithExpertModel,
    joint_attention,
)
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


@pytest.mark.parametrize("attention_backend", ["eager", "sdpa"])
def test_flow_matching_joint_layout_matches_legacy_with_three_cameras(attention_backend):
    torch.manual_seed(8)
    legacy = PI05FlowMatching.__new__(PI05FlowMatching)
    nn.Module.__init__(legacy)
    legacy.config = PI05Config(
        dtype="float32", chunk_size=2, n_action_steps=2, max_action_dim=4, device="cpu"
    )
    legacy.paligemma_with_expert = tiny_joint_model()
    legacy.action_in_proj = nn.Linear(4, 8)
    legacy.action_out_proj = nn.Linear(8, 4)
    legacy.time_mlp_in = nn.Linear(8, 8)
    legacy.time_mlp_out = nn.Linear(8, 8)
    legacy.gradient_checkpointing_enable()
    joint = copy.deepcopy(legacy)
    joint.paligemma_with_expert.enable_joint_blocks()
    for block in joint.paligemma_with_expert.joint_blocks:
        block.gradient_checkpointing = True
        block.attention_backend = attention_backend
    inputs = {
        "images": [torch.randn(2, 3, 28, 28) for _ in range(3)],
        "img_masks": [torch.ones(2, dtype=torch.bool) for _ in range(3)],
        "tokens": torch.randint(0, 30, (2, 3)),
        "masks": torch.ones(2, 3, dtype=torch.bool),
        "actions": torch.randn(2, 2, 4),
        "noise": torch.randn(2, 2, 4),
        "time": torch.tensor([0.3, 0.8]),
    }
    expected = legacy(**inputs)
    actual = joint(**inputs)
    tolerance = {"rtol": 0, "atol": 0} if attention_backend == "eager" else {"rtol": 2e-5, "atol": 1e-7}
    torch.testing.assert_close(actual, expected, **tolerance)
    expected.mean().backward()
    actual.mean().backward()
    gradients = canonical_pi05_state_dict(
        {name: p.grad for name, p in joint.named_parameters() if p.grad is not None}
    )
    expected_gradients = {name: p.grad for name, p in legacy.named_parameters() if p.grad is not None}
    assert gradients.keys() == expected_gradients.keys()
    for name in expected_gradients:
        torch.testing.assert_close(gradients[name], expected_gradients[name], **tolerance)


def test_export_rejects_duplicate_canonical_names():
    with pytest.raises(ValueError, match="Duplicate"):
        canonical_pi05_state_dict(
            {
                "model.joint_blocks.0.language_layer.weight": torch.ones(1),
                "model.paligemma.model.language_model.layers.0.weight": torch.ones(1),
            }
        )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is required to validate the fused kernel"
            ),
        ),
    ],
)
def test_sdpa_joint_mask_outputs_and_gradients_match_eager(device):
    torch.manual_seed(19)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    head_dim, heads, length = (256, 8, 1000) if device == "cuda" else (8, 2, 7)
    module = SimpleNamespace(num_key_value_groups=heads, scaling=head_dim**-0.5, training=True)
    tensors = [torch.randn(1, count, length, head_dim, device=device, dtype=dtype) for count in (heads, 1, 1)]
    mask = torch.zeros(1, 1, length, length, device=device, dtype=torch.float32)
    mask[:, :, : length - 2, length - 2 :] = -2.3819763e38
    mask[:, :, :, 1] = -2.3819763e38
    # Real padding masks can also produce completely masked query rows.
    mask[:, :, 1, :] = -2.3819763e38

    def run(backend):
        inputs = [value.clone().requires_grad_() for value in tensors]
        result = joint_attention(module, *inputs, mask, backend)
        # Padding queries never contribute to the real action loss or subsequent keys.
        valid_queries = torch.ones(length, device=device)
        valid_queries[1] = 0
        (result.float().square() * valid_queries[None, :, None, None]).mean().backward()
        return [result] + [value.grad for value in inputs]

    expected, actual = run("eager"), run("sdpa")
    for left, right in zip(expected, actual, strict=True):
        assert torch.isfinite(right).all()
        if device == "cpu":
            torch.testing.assert_close(left, right, rtol=2e-5, atol=1e-7)
        else:
            # BF16 eager rounds attention scores/probabilities; fused SDPA does not.
            relative_error = (left.float() - right.float()).norm() / left.float().norm()
            assert relative_error < 0.03
    if device == "cuda":
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            run("sdpa")
        assert any("_scaled_dot_product_efficient_attention" in item.key for item in profile.key_averages())
