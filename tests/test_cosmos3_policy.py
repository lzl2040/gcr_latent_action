import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from scipy.spatial.transform import Rotation
from torch import nn

pytest.importorskip("transformers", minversion="5.22.0")
pytest.importorskip("diffusers", minversion="0.38.0")

from lerobot.common.policies.cosmos3 import modeling_cosmos3 as modeling
from lerobot.common.policies.cosmos3.configuration_cosmos3 import Cosmos3NanoConfig
from lerobot.common.policies.cosmos3.processor_cosmos3 import Cosmos3Processor, cosmos3_libero_action
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature


class Tokenizer:
    eos_token = "<end>"
    unk_token_id = -1

    def convert_tokens_to_ids(self, token):
        return 2

    def apply_chat_template(self, messages, **kwargs):
        return "user:" + messages[0]["content"][0]["text"] + ":assistant"

    def __call__(self, texts, **kwargs):
        lengths = [len(text) for text in texts]
        mask = torch.arange(max(lengths))[None] < torch.tensor(lengths)[:, None]
        return {"input_ids": mask.long(), "attention_mask": mask.long()}

    def save_pretrained(self, path):
        Path(path).mkdir(exist_ok=True)


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setattr(
        "lerobot.common.policies.cosmos3.processor_cosmos3.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: Tokenizer(),
    )
    return Cosmos3NanoConfig(
        dtype=torch.bfloat16,
        fps=20,
        view_size=(256, 256),
        cfg_dropout_rate=0,
        input_features={
            "observation.images.image": PolicyFeature(FeatureType.VISUAL, (3, 256, 256)),
            "observation.images.image2": PolicyFeature(FeatureType.VISUAL, (3, 256, 256)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (10,))},
    )


@pytest.fixture
def stats():
    return {"action": {"q01": torch.full((10,), -2.0), "q99": torch.full((10,), 4.0)}}


def raw_batch(frames=None):
    shape = (2, 3, 256, 256) if frames is None else (2, frames, 3, 256, 256)
    return {
        "observation.images.image": torch.zeros(shape, dtype=torch.uint8),
        "observation.images.image2": torch.full(shape, 255, dtype=torch.uint8),
        "task": ["move the bowl", "move the plate"],
    }


def test_native_canvas_prompt_normalization_and_roundtrip(config, stats, tmp_path):
    processor = Cosmos3Processor(config, stats)
    batch = raw_batch(17)
    batch["action"] = torch.zeros(2, 16, 10)
    prepared = processor(batch)
    assert prepared["pixel_values_videos_gen"].shape == (2, 3, 17, 192, 320)
    assert prepared["video_grid_thw_gen"].tolist() == [[5, 10, 20]] * 2
    torch.testing.assert_close(prepared["action"], torch.full((2, 16, 10), -1 / 3))
    torch.testing.assert_close(processor.unnormalize_action(prepared["action"]), batch["action"])
    caption = json.loads(processor.prompt.caption("move the bowl", 192, 320, 0, 20))
    assert caption["actions"][0]["description"] == "move the bowl."
    assert caption["actions"][0]["idle_frame"] == "0 out of 16."
    assert caption["resolution"] == {"H": 192, "W": 320}
    processor.save_pretrained(tmp_path)
    restored = Cosmos3Processor(config, pretrained_path=tmp_path)
    torch.testing.assert_close(
        restored(raw_batch())["pixel_values_videos_gen"], processor(raw_batch())["pixel_values_videos_gen"]
    )
    assert "negative_input_ids" in restored(raw_batch())
    with pytest.raises(ValueError, match="statistic"):
        Cosmos3Processor(config, {"action": {"q01": torch.zeros(40), "q99": torch.ones(40)}})


def test_libero_native_axis_angle_and_continuous_gripper():
    controls = torch.tensor([[0.2, -0.1, 0.3, 0.2, -0.4, 0.5, 0.25]], dtype=torch.float64)
    matrix = torch.from_numpy(Rotation.from_rotvec(controls[0, 3:6].numpy()).as_matrix())
    action = torch.cat((controls[:, :3], matrix[:, 0][None], matrix[:, 1][None], controls[:, 6:]), -1)
    native = cosmos3_libero_action(action)
    torch.testing.assert_close(native[:, :6], controls[:, :6], atol=1e-10, rtol=1e-10)
    assert native[0, -1].item() == 0.5
    with pytest.raises(ValueError, match="10D"):
        cosmos3_libero_action(torch.zeros(1, 40))


class TinyNetwork(nn.Module):
    _action_modules = modeling.Cosmos3VFMNetwork._action_modules
    _keep_in_fp32_modules = modeling.Cosmos3VFMNetwork._keep_in_fp32_modules
    _set_precision = modeling.Cosmos3VFMNetwork._set_precision

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.action2llm = modeling.DomainAwareLinear(config.max_action_dim, 8, 2)
        self.llm2action = modeling.DomainAwareLinear(8, config.max_action_dim, 2)
        self.action_modality_embed = modeling.ModalityEmbedding(8)
        self.time_embedder = nn.Linear(1, 8)
        self.register_buffer("freq", torch.arange(8).float(), persistent=False)
        self._set_precision()


class TinyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.network = TinyNetwork(config)
        self.config = config

    def forward(self, **batch):
        loss = self.network.action2llm.fc.weight.float().mean() * batch["actions"].float().square().mean()
        return loss, {"video_loss": loss.detach(), "action_loss": loss.detach()}

    def generate(self, **batch):
        size = batch["input_ids"].shape[0]
        return torch.zeros(size, self.config.chunk_size, self.config.max_action_dim), None


def test_strict_meta_loading_action_reinit_and_saved_resume(config, stats, tmp_path, monkeypatch):
    monkeypatch.setattr(modeling, "OmniMoTModel", TinyModel)
    source = modeling.Cosmos3NanoPolicy(config, dataset_stats=stats)
    source._save_pretrained(tmp_path)
    loaded_config = PreTrainedConfig.from_pretrained(tmp_path)
    loaded_config.load_action_modules = False
    loaded_config.max_action_dim = 20
    policy = modeling.Cosmos3NanoPolicy.from_pretrained(tmp_path, config=loaded_config)
    assert policy.config.load_action_modules
    assert not any(t.is_meta for t in (*policy.parameters(), *policy.buffers()))
    assert policy.model.network.time_embedder.weight.dtype == torch.float32
    batch = raw_batch(17)
    batch["action"] = torch.zeros(2, 16, 10)
    policy.train()
    loss, metrics = policy(batch, task_type="train_stage2", step=0)
    loss.backward()
    assert torch.isfinite(policy.model.network.action2llm.fc.weight.grad).all()
    assert policy.model.network.action2llm.fc.weight.grad.abs().sum() > 0
    assert metrics["action_flow_loss"] == metrics["action_loss"]
    torch.testing.assert_close(policy.predict_action_chunk(raw_batch()), torch.ones(2, 16, 10))
    saved = tmp_path / "trained"
    saved.mkdir()
    policy._save_pretrained(saved)
    restored = modeling.Cosmos3NanoPolicy.from_pretrained(saved)
    torch.testing.assert_close(
        restored.model.network.action2llm.fc.weight, policy.model.network.action2llm.fc.weight
    )
    state = load_file(saved / "model.safetensors")
    state["unexpected"] = torch.ones(1)
    save_file(state, saved / "model.safetensors")
    with pytest.raises(RuntimeError, match="strict load"):
        modeling.Cosmos3NanoPolicy.from_pretrained(saved)


def test_fsdp_resume_restores_saved_shape_and_rejects_contract_changes(config, tmp_path):
    from lerobot.scripts.fsdp_train_contrast import _load_cosmos3_resume_policy_config

    saved = deepcopy(config)
    saved.max_action_dim = 20
    saved._save_pretrained(tmp_path)
    cfg = SimpleNamespace(
        weight_resume=True, policy=deepcopy(config), output_dir=tmp_path, use_policy_training_preset=False,
    )
    _load_cosmos3_resume_policy_config(cfg, args=[])
    assert cfg.policy.max_action_dim == 20
    assert cfg.policy.load_action_modules
    cfg.policy = deepcopy(config)
    with pytest.raises(ValueError, match="max_action_dim"):
        _load_cosmos3_resume_policy_config(cfg, args=["--policy.max_action_dim=64"])


def test_dcp_eval_restores_changed_action_width_and_training_processor(config, stats, tmp_path, monkeypatch):
    import torch.distributed.checkpoint as dcp

    from lerobot.evaluation.eval_cosmos3_libero import load_policy

    monkeypatch.setattr(modeling, "OmniMoTModel", TinyModel)
    source = tmp_path / "source"
    source.mkdir()
    modeling.Cosmos3NanoPolicy(config, dataset_stats=stats)._save_pretrained(source)
    adapted = deepcopy(config)
    adapted.max_action_dim = 20
    adapted.load_action_modules = False
    trained = modeling.Cosmos3NanoPolicy.from_pretrained(source, config=adapted)
    with torch.no_grad():
        trained.model.network.action2llm.fc.weight.fill_(0.25)
    trained.get_processor().stats["action"]["q01"] = torch.full((10,), -5.0)
    trained.get_processor().stats["action"]["q99"] = torch.full((10,), 5.0)
    run = tmp_path / "run"
    run.mkdir()
    trained.config._save_pretrained(run)
    trained.save_processor(run)
    checkpoint = run / "checkpoint_00000001"
    dcp.save({"model": trained.state_dict()}, checkpoint_id=checkpoint / "model", no_dist=True)
    restored = load_policy(run, torch.device("cpu"), checkpoint)
    torch.testing.assert_close(
        restored.model.network.action2llm.fc.weight, trained.model.network.action2llm.fc.weight,
    )
    torch.testing.assert_close(restored.predict_action_chunk(raw_batch()), torch.zeros(2, 16, 10))
