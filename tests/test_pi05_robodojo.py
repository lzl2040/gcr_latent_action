from unittest.mock import Mock

import pytest
import torch
from torch import nn
from transformers import PaliGemmaConfig

from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pi05.configuration_pi05 import PI05Config
from lerobot.common.policies.pi05.modeling_pi05 import PI05FlowMatching, PI05Policy
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature


def quantile_stats():
    return {
        "action": {
            "mean": torch.zeros(2),
            "q01": torch.tensor([2.0, -4.0]),
            "q99": torch.tensor([6.0, 4.0]),
        }
    }


def test_quantile_normalization_round_trip():
    features = {"action": PolicyFeature(FeatureType.ACTION, (2,))}
    modes = {FeatureType.ACTION: NormalizationMode.QUANTILES}
    normalize = Normalize(features, modes, quantile_stats())
    unnormalize = Unnormalize(features, modes, quantile_stats())
    values = torch.tensor([[[2.0, -4.0], [4.0, 0.0], [6.0, 4.0]]])
    normalized = normalize({"action": values})
    torch.testing.assert_close(normalized["action"], torch.tensor([[[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]]))
    torch.testing.assert_close(unnormalize(normalized)["action"], values)


@pytest.mark.parametrize("kind", ["missing", "reversed", "nan"])
def test_quantile_invalid_stats_fail(kind):
    features = {"action": PolicyFeature(FeatureType.ACTION, (2,))}
    stats = quantile_stats()
    if kind == "reversed":
        stats["action"]["q99"][0] = 0
    elif kind == "nan":
        stats["action"]["q01"][0] = float("nan")
    normalizer = Normalize(
        features, {FeatureType.ACTION: NormalizationMode.QUANTILES}, None if kind == "missing" else stats
    )
    with pytest.raises(ValueError, match="QUANTILES"):
        normalizer({"action": torch.ones(1, 2)})


def test_constant_quantile_is_finite():
    stats = quantile_stats()
    stats["action"]["q99"] = stats["action"]["q01"].clone()
    features = {"action": PolicyFeature(FeatureType.ACTION, (2,))}
    modes = {FeatureType.ACTION: NormalizationMode.QUANTILES}
    normalize = Normalize(features, modes, stats)
    inverse = Unnormalize(features, modes, stats)
    batch = {"action": stats["action"]["q01"][None]}
    assert torch.isfinite(normalize(batch)["action"]).all()
    torch.testing.assert_close(inverse(normalize(batch))["action"], batch["action"])


def bare_policy():
    policy = PI05Policy.__new__(PI05Policy)
    nn.Module.__init__(policy)
    policy.config = PI05Config(image_resolution=(28, 28), device="cpu")
    policy.model = nn.Linear(1, 1)
    policy.config.input_features = {
        f"observation.images.{name}": PolicyFeature(FeatureType.VISUAL, (3, 28, 28))
        for name in ("primary", "secondary", "wrist")
    }
    return policy


def test_three_cameras_and_missing_camera():
    policy = bare_policy()
    batch = {key: torch.full((1, 3, 28, 28), i / 2) for i, key in enumerate(policy.config.image_features)}
    images, masks = policy._preprocess_images(batch)
    assert len(images) == 3 and all(mask.all() for mask in masks)
    for image, value in zip(images, (-1, 0, 1), strict=True):
        torch.testing.assert_close(image, torch.full_like(image, value))
    batch.pop("observation.images.wrist")
    with pytest.raises(ValueError, match="Missing configured camera"):
        policy._preprocess_images(batch)


def test_state_prompt_padding_and_token_limit():
    policy = bare_policy()
    batch = {"observation.state": torch.zeros(1, 20), "task": ["Stack_the bowls.\n"]}
    prompts = policy._prepare_state_prompt(batch)
    assert batch["task"] == ["Stack_the bowls.\n"]
    assert len(prompts["task"][0].split("State: ")[1].split(";")[0].split()) == 32
    policy.language_tokenizer = Mock(
        return_value={
            "input_ids": torch.zeros(1, 200, dtype=torch.long),
            "attention_mask": torch.ones(1, 200, dtype=torch.long),
        }
    )
    policy.prepare_language(prompts)
    assert policy.language_tokenizer.call_args.kwargs["max_length"] == 200


@pytest.fixture
def tiny_flow(monkeypatch):
    import lerobot.common.policies.pi05.modeling_pi05 as modeling
    import lerobot.common.policies.pi05.paligemma_with_expert as backbone

    def tiny_paligemma():
        return PaliGemmaConfig(
            projection_dim=32,
            vision_config={
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "image_size": 28,
                "patch_size": 14,
            },
        )

    monkeypatch.setattr(
        backbone,
        "CONFIG_MAPPING",
        {
            "paligemma": tiny_paligemma,
            "gemma": backbone.CONFIG_MAPPING["gemma"],
        },
    )
    monkeypatch.setattr(modeling, "get_gemma_config", lambda _: backbone.GemmaConfig(32, 1, 64, 2, 1, 16))
    config = PI05Config(chunk_size=4, n_action_steps=2, max_action_dim=20, image_resolution=(28, 28))
    return PI05FlowMatching(config)


def test_actual_flow_forward_backward_and_sampling(tiny_flow):
    torch.manual_seed(0)
    images = [torch.rand(1, 3, 28, 28) for _ in range(3)]
    masks = [torch.ones(1, dtype=torch.bool) for _ in images]
    tokens = torch.tensor([[2, 3, 4]])
    token_mask = torch.ones_like(tokens, dtype=torch.bool)
    actions = torch.rand(1, 4, 20)
    loss = tiny_flow(images, masks, tokens, token_mask, actions).mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(tiny_flow.action_out_proj.weight.grad).all()
    predicted = tiny_flow.sample_actions(images, masks, tokens, token_mask, num_steps=2)
    assert predicted.shape == actions.shape and torch.isfinite(predicted).all()


def test_official_tied_embedding_loading_and_missing_weight_rejection(tiny_flow):
    policy = bare_policy()
    policy.model = tiny_flow
    policy.model.paligemma_with_expert.paligemma.lm_head = nn.Identity()
    weights = dict(policy.model.state_dict())
    embedding = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    weights["paligemma_with_expert.paligemma.lm_head.weight"] = weights.pop(embedding)
    policy.load_base_state_dict(weights)
    weights.pop("action_in_proj.weight")
    with pytest.raises(ValueError, match="missing=.*action_in_proj.weight"):
        policy.load_base_state_dict(weights)
