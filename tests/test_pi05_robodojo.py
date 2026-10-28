import json
import random
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn
from transformers import PaliGemmaConfig

from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pi05.configuration_pi05 import PI05Config
from lerobot.common.policies.pi05.modeling_pi05 import PI05FlowMatching, PI05Policy
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature


@pytest.fixture
def wandb_args(tmp_path):
    return SimpleNamespace(
        wandb=True,
        wandb_project="pi05-tests",
        wandb_entity="team",
        wandb_run_name="batch16",
        wandb_mode="offline",
        output_dir=tmp_path,
        resume=False,
    )


@pytest.mark.parametrize(("enabled", "rank"), [(False, 0), (True, 1)])
def test_wandb_disabled_or_nonzero_rank_has_no_sdk_or_files(monkeypatch, wandb_args, enabled, rank):
    from lerobot.scripts.train_pi05_robodojo import wandb_logging

    wandb_args.wandb = enabled
    monkeypatch.setitem(sys.modules, "wandb", None)
    with wandb_logging(wandb_args, {}, rank=rank) as run:
        assert run is None
    assert not list(wandb_args.output_dir.iterdir())


def test_wandb_persists_identity_and_preserves_rng(monkeypatch, wandb_args):
    from lerobot.scripts.train_pi05_robodojo import get_rng_state, set_rng_state, wandb_logging

    run = Mock(id="run123", project="pi05-tests", entity="team")

    def initialize(**kwargs):
        random.random()
        np.random.rand()
        torch.rand(1)
        return run

    sdk = SimpleNamespace(init=Mock(side_effect=initialize), Settings=Mock())
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    rng = get_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(1))
    set_rng_state(rng)
    with wandb_logging(wandb_args, {"effective_batch_size": 128}) as received:
        assert received is run
        run.define_metric.assert_called_once_with("train/*", step_metric="train/step")
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(1), expected[2], rtol=0, atol=0)
    identity = json.loads((wandb_args.output_dir / "wandb_run.json").read_text())
    assert identity == {"id": "run123", "project": "pi05-tests", "entity": "team"}
    assert sdk.init.call_args.kwargs["config"] == {"effective_batch_size": 128}
    assert sdk.init.call_args.kwargs["save_code"] is False
    sdk.Settings.assert_called_once_with(disable_git=True, disable_code=True)
    run.finish.assert_called_once_with()


def test_wandb_online_resume_reuses_identity_and_rejects_another_project(monkeypatch, wandb_args):
    from lerobot.scripts.train_pi05_robodojo import wandb_logging

    (wandb_args.output_dir / "wandb_run.json").write_text(
        json.dumps({"id": "previous", "project": "original", "entity": "team"})
    )
    wandb_args.resume = True
    wandb_args.wandb_project = None
    wandb_args.wandb_entity = None
    wandb_args.wandb_mode = "online"
    run = Mock(id="previous", project="original", entity="team")
    sdk = SimpleNamespace(init=Mock(return_value=run), Settings=Mock())
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    with wandb_logging(wandb_args, {}):
        pass
    assert sdk.init.call_args.kwargs["id"] == "previous"
    assert sdk.init.call_args.kwargs["resume"] == "allow"
    assert sdk.init.call_args.kwargs["project"] == "original"
    assert sdk.init.call_args.kwargs["entity"] == "team"
    wandb_args.wandb_project = "different"
    with pytest.raises(ValueError, match="original project"), wandb_logging(wandb_args, {}):
        pytest.fail("Changed project must fail before entering training.")
    assert sdk.init.call_count == 1


def test_wandb_marks_failed_training(monkeypatch, wandb_args):
    from lerobot.scripts.train_pi05_robodojo import wandb_logging

    run = Mock(id="failed", project="pi05-tests", entity="team")
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=Mock(return_value=run), Settings=Mock()))
    with pytest.raises(RuntimeError, match="training failed"), wandb_logging(wandb_args, {}):
        raise RuntimeError("training failed")
    run.finish.assert_called_once_with(exit_code=1)


def quantile_stats():
    return {
        "action": {
            "mean": torch.zeros(2),
            "q01": torch.tensor([2.0, -4.0]),
            "q99": torch.tensor([6.0, 4.0]),
        }
    }


def test_pi05_uses_standard_adamw():
    parameter = nn.Parameter(torch.ones(1))
    optimizer = PI05Config().get_optimizer_preset().build([parameter])
    assert type(optimizer) is torch.optim.AdamW


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


@pytest.mark.parametrize("mode", [NormalizationMode.MEAN_STD, NormalizationMode.MIN_MAX])
def test_other_normalization_modes_are_unchanged(mode):
    features = {"action": PolicyFeature(FeatureType.ACTION, (2,))}
    stats = {
        "action": {
            "mean": torch.tensor([2.0, 4.0]),
            "std": torch.tensor([2.0, 4.0]),
            "min": torch.zeros(2),
            "max": torch.tensor([4.0, 8.0]),
        }
    }
    values = {"action": torch.tensor([[2.0, 4.0], [4.0, 8.0]])}
    normalize = Normalize(features, {FeatureType.ACTION: mode}, stats)
    inverse = Unnormalize(features, {FeatureType.ACTION: mode}, stats)
    normalized = normalize(values)
    torch.testing.assert_close(normalized["action"], torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
    torch.testing.assert_close(inverse(normalized)["action"], values["action"])


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


def test_non_square_camera_padding_remains_in_siglip_range():
    policy = bare_policy()
    batch = {key: torch.ones(1, 3, 48, 64) for key in policy.config.image_features}
    images, _ = policy._preprocess_images(batch)
    for image in images:
        assert image.shape == (1, 3, 28, 28)
        assert image.min() == -1 and image.max() == 1


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
    batch["observation.state"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite robot observations"):
        policy._prepare_state_prompt(batch)


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


def test_robodojo_delta_feature_and_padding_mapping(tmp_path, monkeypatch):
    from lerobot.scripts import train_pi05_robodojo as training

    meta = tmp_path / "meta"
    meta.mkdir()
    features = {key: {"shape": [20]} for key in (training.STATE_KEY, training.ACTION_KEY)}
    features.update({key: {"dtype": "video"} for key in training.CAMERAS.values()})
    (meta / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "fps": 25,
                "features": features,
            }
        )
    )
    stats = {"mean": [0.0] * 20, "q01": [-1.0] * 20, "q99": [1.0] * 20}
    (meta / "stats.json").write_text(
        json.dumps(
            {
                training.STATE_KEY: stats,
                training.ACTION_KEY: stats,
            }
        )
    )
    sample = {
        training.STATE_KEY: torch.arange(20).float(),
        training.ACTION_KEY: torch.arange(80).reshape(4, 20).float(),
        f"{training.ACTION_KEY}_is_pad": torch.tensor([False, True, True, True]),
        "action": torch.full((4, 16), -99.0),
        "task": "Stack the three bowls together.",
    }
    sample.update({key: torch.zeros(3, 48, 64) for key in training.CAMERAS.values()})
    source = Mock()
    source.__getitem__ = Mock(return_value=sample)
    factory = Mock(return_value=source)
    monkeypatch.setattr(training, "LeRobotDataset", factory)
    dataset = training.RoboDojoDeltaDataset(tmp_path, chunk_size=4)
    result = dataset[0]
    torch.testing.assert_close(result["action"], sample[training.ACTION_KEY])
    torch.testing.assert_close(result["observation.state"], sample[training.STATE_KEY])
    assert result["action_is_pad"].tolist() == [False, True, True, True]
    assert len([key for key in result if key.startswith("observation.images.")]) == 3
    assert factory.call_args.kwargs["delta_timestamps"] == {training.ACTION_KEY: [0.0, 0.04, 0.08, 0.12]}


def test_full_policy_loss_and_raw_observation_prediction(tiny_flow):
    policy = bare_policy()
    policy.model = tiny_flow
    policy.config.chunk_size = 4
    policy.config.max_action_dim = 20
    policy.config.output_features = {"action": PolicyFeature(FeatureType.ACTION, (20,))}
    policy.config.input_features["observation.state"] = PolicyFeature(FeatureType.STATE, (20,))
    stats = {
        key: {"mean": torch.zeros(20), "q01": torch.full((20,), -2.0), "q99": torch.full((20,), 2.0)}
        for key in ("observation.state", "action")
    }
    modes = {FeatureType.STATE: NormalizationMode.QUANTILES, FeatureType.ACTION: NormalizationMode.QUANTILES}
    policy.normalize_inputs = Normalize(policy.config.input_features, modes, stats)
    policy.normalize_targets = Normalize(policy.config.output_features, modes, stats)
    policy.unnormalize_outputs = Unnormalize(policy.config.output_features, modes, stats)
    policy.language_tokenizer = Mock(
        return_value={
            "input_ids": torch.tensor([[2, 3, 4]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }
    )
    batch = {key: torch.rand(1, 3, 28, 28) for key in policy.config.image_features}
    batch.update(
        {
            "observation.state": torch.zeros(1, 20),
            "action": torch.ones(1, 4, 20),
            "action_is_pad": torch.tensor([[False, False, True, True]]),
            "task": ["Stack bowls."],
        }
    )
    loss, _ = policy(batch)
    assert torch.isfinite(loss)
    loss.backward()
    prediction = policy.predict_action_chunk(batch)
    assert prediction.shape == (1, 4, 20) and torch.isfinite(prediction).all()
    assert batch["task"] == ["Stack bowls."]
    batch["action_is_pad"][:] = True
    with pytest.raises(ValueError, match="no valid action targets"):
        policy(batch)
