import copy
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from lerobot.common.datasets.contrastive_dataset import (
    MultiModalContrastiveDataset,
    _evenly_spaced_frame_timestamps,
)
from lerobot.common.optim.optimizers import AdamW8bitConfig
from lerobot.common.policies.ace.configuration_robo_contrast import RoboContrastConfig
from lerobot.common.policies.qwen3vl_mot.configuration_qwen3vl_mot import Qwen3VLMoTConfig
from lerobot.common.policies.qwen3vl_mot.modeling_generation import (
    AsymmetricAttention,
    GenerationExpert,
    GenerationStream,
)
from lerobot.common.policies.qwen3vl_mot.modeling_qwen3vl_mot import (
    Qwen3VLMoTPolicy,
    _load_stage1_config,
)
from lerobot.common.policies.qwen3vl_mot.stage1_transfer import TransferReport, transfer_matching_module
from lerobot.common.policies.qwen3vl_mot.tasks import (
    TASK_SPECS,
    ModalityRole,
    resolve_task_specs,
)


def test_stage1_config_loader_accepts_saved_policy_config(tmp_path):
    expected = RoboContrastConfig(
        vision_backbone="qwen3vl",
        vision_tuning_mode="lora",
        perception_recon_target="vae",
    )
    expected._save_pretrained(tmp_path)

    loaded = _load_stage1_config(tmp_path)

    assert loaded == expected


def test_stage2_model_path_defaults_use_cluster_mount():
    config = Qwen3VLMoTConfig()

    assert config.qwen3vl_dir == "/mnt/wangxiaofa/pt_weights/Qwen3-VL-4B-Instruct"
    assert config.cosmos3_dir == "/mnt/wangxiaofa/pt_weights/Cosmos3-Edge"


def test_task_family_has_requested_directional_roles():
    assert TASK_SPECS["t2v"].video is ModalityRole.NOISY
    assert not TASK_SPECS["t2v"].understanding_image
    assert TASK_SPECS["i2v"].understanding_image
    assert TASK_SPECS["i2v"].video is ModalityRole.FUTURE_NOISY
    assert TASK_SPECS["joint_prediction"].video is ModalityRole.FUTURE_NOISY
    assert TASK_SPECS["joint_prediction"].state is ModalityRole.FUTURE_NOISY
    assert TASK_SPECS["joint_prediction"].action is ModalityRole.NOISY
    assert TASK_SPECS["forward_dynamics"].action is ModalityRole.CLEAN
    assert TASK_SPECS["forward_dynamics"].state is ModalityRole.ABSENT
    assert TASK_SPECS["inverse_dynamics"].video is ModalityRole.CLEAN
    assert TASK_SPECS["inverse_dynamics"].action is ModalityRole.NOISY
    assert TASK_SPECS["inverse_dynamics"].state is ModalityRole.ABSENT
    assert TASK_SPECS["action_prediction"].state is ModalityRole.CURRENT_ONLY
    assert TASK_SPECS["tactile_prediction"].tactile is ModalityRole.NOISY


def test_task_weights_are_normalized():
    tasks, weights = resolve_task_specs(
        ("i2v", "tactile_prediction"),
        (7.0, 3.0),
    )

    assert tuple(task.name for task in tasks) == ("i2v", "tactile_prediction")
    assert weights == (0.7, 0.3)


def test_replaced_joint_task_name_requires_fresh_contract():
    with pytest.raises(ValueError, match="replaced by `joint_prediction`"):
        Qwen3VLMoTConfig(task_names=("action_video_prediction",), task_weights=(1.0,))
    with pytest.raises(ValueError, match="task contract version"):
        Qwen3VLMoTConfig(task_contract_version=1)


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("explicit_config", [False, True])
def test_direct_model_loading_rejects_old_task_contract_before_construction(
    tmp_path, monkeypatch, remote, explicit_config
):
    import json

    import lerobot.common.policies.qwen3vl_mot.modeling_qwen3vl_mot as stage2

    config = Qwen3VLMoTConfig(task_names=("inverse_dynamics",), task_weights=(1.0,))
    config._save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    payload = json.loads(config_path.read_text())
    del payload["task_contract_version"]
    config_path.write_text(json.dumps(payload))
    requests = []

    def fetch_config(**kwargs):
        requests.append(kwargs)
        return str(config_path)

    monkeypatch.setattr(stage2, "hf_hub_download", fetch_config)
    monkeypatch.setattr(Qwen3VLMoTPolicy, "__init__", lambda *args, **kwargs: pytest.fail(
        "Old task semantics must be rejected before loading the model."
    ))
    options = {"config": config} if explicit_config else {}
    with pytest.raises(ValueError, match="previous task contract"):
        Qwen3VLMoTPolicy.from_pretrained(
            "test/stage2-model" if remote else tmp_path,
            revision="old-revision", local_files_only=True, **options,
        )
    if remote:
        assert requests == [{
            "repo_id": "test/stage2-model", "filename": "config.json",
            "revision": "old-revision", "local_files_only": True,
        }]
    else:
        assert not requests


def test_current_hub_task_contract_reuses_downloaded_config(tmp_path, monkeypatch):
    import lerobot.common.policies.qwen3vl_mot.modeling_qwen3vl_mot as stage2

    config = Qwen3VLMoTConfig(
        task_names=("forward_dynamics", "inverse_dynamics", "joint_prediction"),
        task_weights=(0.3, 0.3, 0.4),
    )
    config._save_pretrained(tmp_path)
    monkeypatch.setattr(stage2, "hf_hub_download", lambda **kwargs: str(tmp_path / "config.json"))
    result = object()

    def restore(cls, model_id, **kwargs):
        assert cls is Qwen3VLMoTPolicy
        assert model_id == "test/stage2-model"
        assert kwargs["config"] == config
        assert kwargs["revision"] == "new-revision"
        return result

    monkeypatch.setattr(stage2.PreTrainedPolicy, "from_pretrained", classmethod(restore))
    assert Qwen3VLMoTPolicy.from_pretrained(
        "test/stage2-model", revision="new-revision"
    ) is result


def test_direct_model_loading_rejects_overridden_task_order(tmp_path, monkeypatch):
    names = ("forward_dynamics", "inverse_dynamics", "joint_prediction")
    Qwen3VLMoTConfig(task_names=names, task_weights=(1, 1, 1))._save_pretrained(tmp_path)
    config = Qwen3VLMoTConfig(task_names=tuple(reversed(names)), task_weights=(1, 1, 1))
    monkeypatch.setattr(Qwen3VLMoTPolicy, "__init__", lambda *args, **kwargs: pytest.fail(
        "Task order must be checked before model construction."
    ))
    with pytest.raises(ValueError, match="Cannot change `policy.task_names`"):
        Qwen3VLMoTPolicy.from_pretrained(tmp_path, config=config)


def test_task_selection_redistributes_unavailable_tactile_weight():
    policy = Qwen3VLMoTPolicy.__new__(Qwen3VLMoTPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(tactile_generation_target="spatial_patches")
    policy.task_specs = (
        TASK_SPECS["i2v"],
        TASK_SPECS["tactile_prediction"],
    )
    policy.task_weights = (0.01, 0.99)
    batch = {
        "image_t0": torch.zeros(2, 3, 8, 8, dtype=torch.uint8),
        "pair_is_valid": torch.ones(2),
        "state_mask": torch.ones(2, 4),
        "action_mask": torch.ones(2, 4),
        "tactile_signal_mask": torch.zeros(2),
        "tactile_image_mask": torch.zeros(2, 2),
    }

    task, valid_rows = policy._select_task(batch, forced_task=None)

    assert task.name == "i2v"
    assert valid_rows.tolist() == [True, True]


def test_asymmetric_attention_reads_only_unmasked_understanding_kv():
    torch.manual_seed(0)
    attention = AsymmetricAttention(
        hidden_dim=16,
        num_heads=4,
        num_kv_heads=2,
        dropout=0.0,
    ).eval()
    generation = torch.randn(2, 3, 16)
    understanding = torch.randn(2, 4, 16)
    understanding_keep = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.bool)
    generation_keep = torch.ones(2, 3, dtype=torch.bool)

    expected = attention(generation, understanding, understanding_keep, generation_keep)
    changed_masked = understanding.clone()
    changed_masked[~understanding_keep] = 10_000
    actual = attention(generation, changed_masked, understanding_keep, generation_keep)
    torch.testing.assert_close(actual, expected)

    changed_visible = understanding.clone()
    changed_visible[understanding_keep] += 10
    visible_result = attention(generation, changed_visible, understanding_keep, generation_keep)
    assert not torch.allclose(visible_result, expected)

    native_key = attention._heads(attention.k_proj(understanding)).detach().requires_grad_(True)
    native_value = attention._heads(attention.v_proj(understanding)).detach().requires_grad_(True)
    native = attention(
        generation,
        None,
        understanding_keep,
        generation_keep,
        native_key,
        native_value,
    )
    torch.testing.assert_close(native, expected)
    native.square().mean().backward()
    assert native_key.grad is not None
    assert native_value.grad is not None


def test_generation_expert_preserves_stream_shapes():
    model = GenerationExpert(
        understanding_dim=12,
        hidden_dim=16,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        input_dims={"video": 4, "action": 8},
        output_dims={"video": 4, "action": 8},
        task_names=("i2v",),
        gradient_checkpointing=False,
    )
    streams = [
        GenerationStream(
            "video",
            torch.randn(2, 5, 4),
            torch.rand(2),
            torch.ones(2, 5, dtype=torch.bool),
        ),
        GenerationStream(
            "action",
            torch.randn(2, 3, 8),
            torch.rand(2),
            torch.ones(2, 3, dtype=torch.bool),
        ),
    ]
    output = model(
        streams,
        [torch.randn(2, 7, 12), torch.randn(2, 7, 12)],
        torch.ones(2, 7, dtype=torch.bool),
        "i2v",
    )
    assert output["video"].shape == (2, 5, 4)
    assert output["action"].shape == (2, 3, 8)


@pytest.mark.parametrize("roles", [("primary",), ("primary", "secondary", "wrist")])
@pytest.mark.parametrize("flow_space", ["raw", "embedding"])
def test_default_generation_expert_is_between_one_and_two_billion_parameters(roles, flow_space):
    config = Qwen3VLMoTConfig(rgb_views=roles, physical_flow_space=flow_space)
    with torch.device("meta"):
        model = GenerationExpert(
            understanding_dim=2560,
            hidden_dim=config.generation_hidden_dim,
            depth=config.generation_depth,
            num_heads=config.generation_num_heads,
            num_kv_heads=config.generation_num_kv_heads,
            intermediate_dim=config.generation_intermediate_dim,
            hidden_act=config.generation_hidden_act,
            dropout=config.generation_dropout,
            input_dims={
                "video": config.video_latent_dim * config.video_latent_patch_size**2,
                "state": config.physical_hidden_dim,
                "action": config.physical_hidden_dim,
                "tactile": config.physical_hidden_dim,
            },
            output_dims={
                "video": config.video_latent_dim * config.video_latent_patch_size**2,
                "state": config.physical_hidden_dim if flow_space == "embedding" else config.group_size * config.max_state_dim,
                "action": config.physical_hidden_dim if flow_space == "embedding" else config.group_size * config.max_action_dim,
                "tactile": config.physical_hidden_dim,
            },
            task_names=config.task_names,
            gradient_checkpointing=True,
            num_video_views=len(config.rgb_views),
        )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert 1_000_000_000 <= parameters <= 2_000_000_000
    assert parameters == 1_429_471_744 + (6144 if len(roles) > 1 else 0) + (
        2 * 864 * 2049 if flow_space == "embedding" else 0
    )
    assert (model.video_view_embedding is None) == (len(roles) == 1)


def test_native_kv_checkpointing_matches_non_checkpointed_gradients():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

    torch.manual_seed(7)
    rotary_config = Qwen3VLTextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=128,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 0]},
    )
    plain = GenerationExpert(
        understanding_dim=12,
        hidden_dim=16,
        depth=2,
        num_heads=4,
        num_kv_heads=2,
        mlp_ratio=2.0,
        dropout=0.0,
        input_dims={"video": 4},
        output_dims={"video": 4},
        task_names=("i2v",),
        gradient_checkpointing=False,
        rotary_config=rotary_config,
    )
    with torch.no_grad():
        for parameter in plain.parameters():
            parameter.normal_(std=0.02)
    checkpointed = copy.deepcopy(plain)
    checkpointed.gradient_checkpointing = True
    plain.train()
    checkpointed.train()

    values = torch.randn(2, 5, 4, requires_grad=True)
    values_checkpointed = values.detach().clone().requires_grad_(True)
    key_values = [
        (
            torch.randn(2, 2, 7, 4, requires_grad=True),
            torch.randn(2, 2, 7, 4, requires_grad=True),
        )
        for _ in range(2)
    ]
    key_values_checkpointed = [
        (
            key.detach().clone().requires_grad_(True),
            value.detach().clone().requires_grad_(True),
        )
        for key, value in key_values
    ]
    keep = torch.ones(2, 5, dtype=torch.bool)
    context_keep = torch.ones(2, 7, dtype=torch.bool)
    sigma = torch.rand(2)

    plain_output = plain(
        [GenerationStream("video", values, sigma, keep)],
        [],
        context_keep,
        "i2v",
        understanding_key_values=key_values,
    )["video"]
    checkpointed_output = checkpointed(
        [GenerationStream("video", values_checkpointed, sigma, keep)],
        [],
        context_keep,
        "i2v",
        understanding_key_values=key_values_checkpointed,
    )["video"]
    torch.testing.assert_close(checkpointed_output, plain_output)
    plain_output.square().mean().backward()
    checkpointed_output.square().mean().backward()
    torch.testing.assert_close(values_checkpointed.grad, values.grad)
    for (plain_key, plain_value), (checkpointed_key, checkpointed_value) in zip(
        key_values,
        key_values_checkpointed,
        strict=True,
    ):
        torch.testing.assert_close(checkpointed_key.grad, plain_key.grad)
        torch.testing.assert_close(checkpointed_value.grad, plain_value.grad)
    for (_, plain_parameter), (_, checkpointed_parameter) in zip(
        plain.named_parameters(),
        checkpointed.named_parameters(),
        strict=True,
    ):
        if plain_parameter.grad is None:
            assert checkpointed_parameter.grad is None
        else:
            torch.testing.assert_close(checkpointed_parameter.grad, plain_parameter.grad)


@pytest.mark.parametrize("multiview", [False, True])
def test_tiny_qwen_understanding_emits_native_kv_and_query_gradients(multiview):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    from lerobot.common.policies.qwen3vl_mot.modeling_understanding import (
        Qwen3VLUnderstandingExpert,
        _base_qwen,
        _selected_layers,
    )

    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 100,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 1024,
            "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]},
            "pad_token_id": 0,
        },
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 256,
            "deepstack_visual_indexes": [0],
        },
        image_token_id=90,
        video_token_id=91,
        vision_start_token_id=92,
        vision_end_token_id=93,
    )

    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, texts, **kwargs):
            length = kwargs["max_length"]
            ids = torch.zeros(len(texts), length, dtype=torch.long)
            keep = torch.zeros_like(ids)
            ids[:, :2] = torch.tensor([2, 3])
            keep[:, :2] = 1
            return {"input_ids": ids, "attention_mask": keep}

    expert = object.__new__(Qwen3VLUnderstandingExpert)
    nn.Module.__init__(expert)
    expert.model = Qwen3VLModel(config)
    for parameter in expert.model.parameters():
        parameter.requires_grad_(False)
    expert.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    expert.tokenizer = _Tokenizer()
    expert.num_queries = 2
    expert.max_text_tokens = 4
    expert.query_tokens = nn.Parameter(torch.randn(2, 32) * 0.02)
    expert.selected_layer_indices = _selected_layers(2, 2)
    expert.gradient_checkpointing = True
    expert.tuning_mode = "frozen"
    expert.eval()
    output = expert(torch.full((1, 3, 32, 32), 255, dtype=torch.uint8), ["move"])
    float_output = expert(torch.ones(1, 3, 32, 32), ["move"])
    torch.testing.assert_close(float_output.latent_queries, output.latent_queries)
    assert output.latent_queries.shape == (1, 2, 32)
    assert output.hidden_states == []
    assert len(output.key_values) == 2
    assert output.key_values[0][0].shape == (1, 2, 72, 8)
    if multiview:
        images = torch.randint(0, 256, (3, 3, 3, 32, 32), dtype=torch.uint8)
        mask = torch.tensor([[1, 1, 0], [1, 0, 1], [0, 0, 0]], dtype=torch.bool)
        names = ("primary", "secondary", "wrist")
        kwargs = {"image_mask": mask, "view_names": names}
        output = expert(images, ["move"] * 3, **kwargs)
        assert output.key_values[0][0].shape == (3, 2, 228, 8)
        assert output.attention_mask.sum(dim=1).tolist() == [140, 140, 4]
        modified = images.clone()
        modified[~mask] = 255 - modified[~mask]
        torch.testing.assert_close(
            expert(modified, ["move"] * 3, **kwargs).latent_queries, output.latent_queries
        )
        # Exercise the dummy visual forward used by a rank with no valid camera.
        empty = expert(images, ["move"] * 3, image_mask=torch.zeros_like(mask), view_names=names)
        assert empty.attention_mask.sum(dim=1).tolist() == [4, 4, 4]
        assert torch.isfinite(empty.latent_queries).all()
    expert.train()
    vision_parameter = next(expert.model.visual.parameters())
    vision_parameter.requires_grad_(True)
    output = (
        expert(images, ["move"] * 3, **kwargs)
        if multiview else expert(torch.zeros(1, 3, 32, 32, dtype=torch.uint8), ["move"])
    )
    output.latent_queries.square().mean().backward()
    assert expert.query_tokens.grad is not None
    assert vision_parameter.grad is not None
    if multiview:
        expert.zero_grad(set_to_none=True)
        empty = expert(images, ["move"] * 3, image_mask=torch.zeros_like(mask), view_names=names)
        empty.latent_queries.square().mean().backward()
        assert vision_parameter.grad is not None
        assert torch.count_nonzero(vision_parameter.grad) == 0

    vision_lora_expert = object.__new__(Qwen3VLUnderstandingExpert)
    nn.Module.__init__(vision_lora_expert)
    vision_lora_expert.model = Qwen3VLModel(config)
    vision_lora_expert._configure_tuning(
        tuning_mode="lora",
        rank=2,
        alpha=2,
        dropout=0.0,
        text_layers=0,
        vision_layers=1,
    )
    trainable_names = [
        name
        for name, parameter in vision_lora_expert.model.named_parameters()
        if parameter.requires_grad
    ]
    assert any("visual.blocks" in name and "lora_" in name for name in trainable_names)
    assert not any("language_model.layers" in name for name in trainable_names)

    full_expert = object.__new__(Qwen3VLUnderstandingExpert)
    nn.Module.__init__(full_expert)
    full_expert.model = Qwen3VLModel(config)
    full_expert._configure_tuning(
        tuning_mode="full",
        rank=2,
        alpha=2,
        dropout=0.0,
        text_layers=0,
        vision_layers=1,
    )
    full_base = _base_qwen(full_expert.model)
    assert all(
        parameter.requires_grad
        for parameter in full_base.language_model.layers.parameters()
    )
    assert not full_base.language_model.embed_tokens.weight.requires_grad
    assert not full_base.language_model.norm.weight.requires_grad
    full_trainable_names = [
        name
        for name, parameter in full_expert.model.named_parameters()
        if parameter.requires_grad
    ]
    assert any("visual.blocks" in name and "lora_" in name for name in full_trainable_names)
    assert not any(
        "visual.blocks" in name and "base_layer" in name
        for name in full_trainable_names
    )


def test_stage1_transfer_strips_wrapper_prefixes():
    source = nn.Module()
    source.vision_model = nn.Linear(3, 5)
    destination = nn.Linear(3, 5)
    with torch.no_grad():
        source.vision_model.weight.fill_(2.0)
        source.vision_model.bias.fill_(3.0)
    report = transfer_matching_module(
        source,
        destination,
        source_prefixes=("vision_model.",),
    )
    assert report.coverage == 1.0
    torch.testing.assert_close(destination.weight, source.vision_model.weight)
    torch.testing.assert_close(destination.bias, source.vision_model.bias)


def test_incompatible_stage1_text_transfer_preserves_pretrained_qwen_weights():
    source = nn.Linear(3, 5)
    destination = nn.Linear(4, 6)
    original = copy.deepcopy(destination.state_dict())
    report = transfer_matching_module(source, destination, source_prefixes=())
    assert report.coverage == 0
    assert report.copied_tensors == 0
    for name, value in destination.state_dict().items():
        torch.testing.assert_close(value, original[name])


def test_world_video_extraction_resizes_and_pads():
    dataset = object.__new__(MultiModalContrastiveDataset)
    dataset.world_video_frames = 4
    dataset.cfg = SimpleNamespace(
        dataset=SimpleNamespace(image_transforms=SimpleNamespace(img_size=8))
    )
    frames = torch.rand(2, 3, 4, 6)
    video = dataset._extract_video({"camera": frames}, "camera")
    assert video.shape == (4, 3, 8, 8)
    assert video.dtype == torch.uint8
    torch.testing.assert_close(video[-1], video[-2])


def test_world_video_timestamps_stay_on_source_frame_boundaries():
    timestamps = _evenly_spaced_frame_timestamps(horizon=31, count=9, index_fps=30)
    assert len(timestamps) == 9
    assert timestamps[0] == 0.0
    assert timestamps[-1] == 31 / 30
    assert all(abs(timestamp * 30 - round(timestamp * 30)) < 1e-8 for timestamp in timestamps)


def test_config_rejects_non_divisible_grouping():
    try:
        Qwen3VLMoTConfig(chunk_size=10, group_size=4)
    except ValueError as exc:
        assert "must be divisible" in str(exc)
    else:
        raise AssertionError("Expected an invalid grouping configuration to fail.")


def test_default_stage2_lora_tunes_vision_but_not_text():
    config = Qwen3VLMoTConfig()
    assert config.understanding_tuning_mode == "lora"
    assert config.understanding_text_lora_layers == 0
    assert config.understanding_vision_lora_layers > 0
    assert isinstance(config.get_optimizer_preset(), AdamW8bitConfig)


def test_stage2_scheduler_uses_plateau_argument():
    config = Qwen3VLMoTConfig()
    scheduler = config.get_scheduler_preset()
    assert scheduler.num_platform_steps == config.scheduler_plateau_steps


def test_stage2_defaults_use_consecutive_32_step_actions():
    config = Qwen3VLMoTConfig()
    assert config.window_mode == "frames"
    assert config.chunk_size == 32
    assert config.n_action_steps == 32

    dataset = object.__new__(MultiModalContrastiveDataset)
    dataset.window_mode = config.window_mode
    dataset.chunk_size = config.chunk_size
    dataset.chunk_seconds = config.chunk_seconds
    dataset.chunk_frames_min = config.chunk_frames_min
    dataset.chunk_frames_max = config.chunk_frames_max
    dataset.frame_horizon_override = config.frame_horizon
    offsets, horizon = dataset._window_offsets(fps=30)

    assert offsets == list(range(32))
    assert horizon == 31


def test_stage2_rejects_resampled_or_misaligned_action_windows():
    for kwargs in (
        {"window_mode": "duration"},
        {"frame_horizon": 48},
    ):
        try:
            Qwen3VLMoTConfig(**kwargs)
        except ValueError as exc:
            assert "action" in str(exc) or "frame_horizon" in str(exc)
        else:
            raise AssertionError(f"Expected invalid temporal config to fail: {kwargs}")


class _FakePerception(nn.Module):
    def __init__(self, queries: int, width: int):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.vision_backbone = nn.Linear(1, 1)
        self.text_backbone = nn.Linear(1, 1)
        self.predictor = nn.Identity()
        self.vae = nn.Identity()
        self.queries = queries
        self.width = width

    def forward(
        self,
        image_t0,
        image_t1,
        texts,
        has_text=None,
        probe=False,
        return_latent_action=False,
    ):
        self.last_images = (image_t0.clone(), image_t1.clone())
        del texts, has_text, probe
        latent = torch.ones(
            image_t0.shape[0],
            self.queries,
            self.width,
            device=image_t0.device,
        )
        return latent.mean(dim=1), None, {"latent_action": latent}


class _FakePhysical(nn.Module):
    def __init__(self, groups: int, hidden: int, tactile_tokens: int):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.num_cls_tokens = 1
        self.groups = groups
        self.hidden = hidden
        self.tactile_tokens = tactile_tokens
        self.last_state = None
        self.include_tactile_history = []
        self.input_history = []
        self.use_ftp1_tactile = False
        self.use_anytouch_tactile = False
        self.tactile_feat_dim = 6
        self.tactile_cnn = nn.Identity()
        self.tactile_recon = nn.Identity()

    def forward(self, batch, **kwargs):
        self.include_tactile_history.append(kwargs["include_tactile"])
        self.last_state = batch["observation.state"].detach().clone()
        self.input_history.append({
            key: batch[key].detach().clone()
            for key in ("observation.state", "state_mask", "action", "action_mask")
        })
        batch_size = batch["action"].shape[0]
        length = 1 + 2 * self.groups + self.tactile_tokens
        tokens = self.anchor + torch.zeros(
            batch_size,
            length,
            self.hidden,
            device=self.anchor.device,
        )
        keep = torch.ones(batch_size, length, device=self.anchor.device, dtype=torch.bool)
        keep[:, 1 + 2 * self.groups :] = (
            batch["tactile_signal_mask"].to(self.anchor.device).reshape(-1, 1) > 0
        )
        return tokens, keep, None

    def encode_tactile_patches(self, images):
        values = images.float().mean(dim=2) / 255.0
        values = torch.nn.functional.adaptive_avg_pool2d(
            values.reshape(-1, 1, *values.shape[-2:]),
            (2, 2),
        )
        values = values.expand(-1, self.tactile_feat_dim, -1, -1)
        return values.view(
            images.shape[0],
            images.shape[1],
            self.tactile_feat_dim,
            2,
            2,
        )

    def decode_tactile_patches(self, patches):
        decoded = patches[..., :3, :, :]
        return torch.nn.functional.interpolate(
            decoded.reshape(-1, 3, *decoded.shape[-2:]),
            size=(8, 8),
            mode="bilinear",
            align_corners=False,
        ).view(*patches.shape[:-3], 3, 8, 8)


class _FakeUnderstanding(nn.Module):
    def __init__(self, _model_dir, *, num_queries, **kwargs):
        super().__init__()
        del kwargs
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

        self.model = nn.Linear(1, 1)
        self.model.config = SimpleNamespace(
            text_config=Qwen3VLTextConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=4,
                head_dim=4,
                max_position_embeddings=128,
                rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 0]},
            )
        )
        self.query_tokens = nn.Parameter(torch.randn(num_queries, 12))
        self.num_queries = num_queries
        self.tuning_mode = "frozen"

    @property
    def hidden_dim(self):
        return 12

    def forward(self, images, texts, **kwargs):
        self.last_images = images
        self.last_view_arguments = kwargs
        from lerobot.common.policies.qwen3vl_mot.modeling_understanding import UnderstandingOutput

        batch_size = len(texts)
        hidden = self.query_tokens.new_zeros(batch_size, 5, 12)
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        hidden = hidden + queries.mean() * 0
        key = hidden.new_zeros(batch_size, 4, 5, 4)
        return UnderstandingOutput(
            hidden_states=[],
            key_values=[(key, key.clone())],
            attention_mask=torch.ones(batch_size, 5, dtype=torch.bool, device=hidden.device),
            latent_queries=queries.to(torch.bfloat16),
        )


class _FakeVAE(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.channels = channels
        self.decoder = None

    def encode(self, video):
        batch_size = video.shape[0]
        latent = self.anchor + torch.zeros(
            batch_size,
            self.channels,
            3,
            2,
            2,
            device=video.device,
        )
        return SimpleNamespace(latent_dist=SimpleNamespace(mean=latent))


def test_spatial_tactile_target_and_generation_are_decodable():
    policy = object.__new__(Qwen3VLMoTPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        tactile_generation_target="spatial_patches",
        chunk_size=4,
        group_size=2,
        num_groups=2,
        n_action_steps=2,
        max_action_dim=4,
        max_state_dim=4,
        max_tactile_signal_dim=3,
        max_tactile_views=2,
        tactile_frames=2,
        tactile_img_size=8,
        physical_hidden_dim=8,
        physical_tuning_mode="frozen",
        use_tactile_conditioning=False,
        world_video_frames=9,
        video_image_size=16,
        video_latent_dim=4,
        video_latent_patch_size=2,
        inference_steps=2,
        sigma_min=1e-3,
        sigma_max=0.999,
        time_sampling_distribution="uniform",
        time_sampling_beta_probability=0.5,
        task_weights=(1 / 3, 1 / 3, 1 / 3),
    )
    policy.physical_encoder = _FakePhysical(
        groups=2,
        hidden=8,
        tactile_tokens=4,
    )
    policy.understanding = _FakeUnderstanding("", num_queries=2)
    policy.video_vae = _FakeVAE(4)
    policy.temporal_compression = 4
    policy.register_buffer(
        "video_latent_mean",
        torch.zeros(1, 4, 1, 1, 1),
        persistent=False,
    )
    policy.register_buffer(
        "video_latent_std",
        torch.ones(1, 4, 1, 1, 1),
        persistent=False,
    )
    policy.generation = GenerationExpert(
        understanding_dim=12,
        hidden_dim=16,
        depth=1,
        num_heads=4,
        num_kv_heads=4,
        intermediate_dim=32,
        dropout=0.0,
        input_dims={
            "video": 16,
            "state": 8,
            "action": 8,
            "tactile": 6,
        },
        output_dims={
            "video": 16,
            "state": 8,
            "action": 8,
            "tactile": 6,
        },
        task_names=("i2v", "action_prediction", "tactile_prediction"),
        gradient_checkpointing=False,
    )
    batch = {
        "image_t0": torch.zeros(2, 3, 8, 8, dtype=torch.uint8),
        "video": torch.zeros(2, 9, 3, 8, 8, dtype=torch.uint8),
        "task": ["move", "touch"],
        "pair_is_valid": torch.ones(2),
        "observation.state": torch.randn(2, 4, 4),
        "state_mask": torch.ones(2, 4),
        "action": torch.randn(2, 4, 4),
        "action_mask": torch.ones(2, 4),
        "tactile_signal": torch.zeros(2, 4, 3),
        "tactile_signal_mask": torch.zeros(2),
        "tactile_image": torch.randint(
            0,
            256,
            (2, 2, 2, 3, 8, 8),
            dtype=torch.uint8,
        ),
        "tactile_image_mask": torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
        "sample_rate": torch.full((2,), 10, dtype=torch.long),
    }

    stream, target, mask, _ = policy._prepare_tactile_stream(
        TASK_SPECS["tactile_prediction"],
        batch,
        torch.ones(2, dtype=torch.bool),
    )
    assert stream.values.shape == (2, 8, 6)
    assert stream.position_ids.shape == (8, 3)
    assert target.shape == stream.values.shape
    assert mask.sum().item() == 3 * 2 * 2 * 6

    prediction = policy.generate_validation(batch, seed=123)
    assert prediction["action_normalized"].shape == (2, 2, 4)
    assert prediction["video_latents"].shape == (2, 4, 3, 2, 2)
    assert prediction["tactile_decoded_z"].shape == (2, 2, 3, 8, 8)
    torch.testing.assert_close(
        prediction["tactile_mask"],
        batch["tactile_image_mask"].bool(),
    )


@pytest.mark.parametrize("multiview", [False, True])
@pytest.mark.parametrize("flow_space", ["raw", "embedding"])
@pytest.mark.parametrize("generation_variant", ["random", "edge", "nano", "edge_frozen"])
@pytest.mark.parametrize("time_distribution", ["uniform", "beta", "mixed", "shifted_uniform"])
def test_policy_runs_all_task_routes_with_lightweight_backends(
    tmp_path, monkeypatch, multiview, flow_space, generation_variant, time_distribution
):
    import lerobot.common.policies.ace.cosmos3_encoders as cosmos3_encoders
    import lerobot.common.policies.qwen3vl_mot.modeling_qwen3vl_mot as stage2
    from lerobot.common.policies.qwen3vl_mot.cosmos3_pretrained import (
        COSMOS_GENERATION_CONFIGS,
        generation_checkpoint_mapping,
    )

    generation_kind = generation_variant.split("_")[0]
    generator_dir = tmp_path / "cosmos"
    if generation_kind != "random":
        native = {
            **COSMOS_GENERATION_CONFIGS[generation_kind],
            "hidden_size": 16, "num_hidden_layers": 1, "num_attention_heads": 4,
            "num_key_value_heads": 4, "head_dim": 4, "intermediate_size": 32,
            "rope_axes_dim": [1, 1, 0], "latent_channel": 4, "patch_latent_dim": 16,
        }
        monkeypatch.setitem(COSMOS_GENERATION_CONFIGS, generation_kind, native)
        (generator_dir / "transformer").mkdir(parents=True)
        (generator_dir / "vae").mkdir()
        import json

        (generator_dir / "transformer/config.json").write_text(json.dumps(native))
        (generator_dir / "vae/config.json").write_text(json.dumps({"z_dim": 4}))
        source = GenerationExpert(
            understanding_dim=8, hidden_dim=16, depth=1, num_heads=4, num_kv_heads=4,
            intermediate_dim=32, hidden_act=native["hidden_act"], dropout=0,
            input_dims={"video": 16}, output_dims={"video": 16}, task_names=("i2v",),
            gradient_checkpointing=False, cosmos_config=native,
        )
        with torch.no_grad():
            source.output_heads["video"].weight.normal_(std=0.02)
        source_parameters = dict(source.named_parameters())
        save_file({
            original: source_parameters[target].detach().contiguous()
            for target, original in generation_checkpoint_mapping(source).items()
        }, generator_dir / "transformer/diffusion_pytorch_model.safetensors")

    stage1_dir = tmp_path / "stage1"
    stage1_dir.mkdir()
    save_file(
        {
            "perception_encoder.vision_backbone.weight": torch.zeros(1),
            "physical_encoder.state_proj.weight": torch.zeros(1),
            "physical_encoder.action_proj.weight": torch.zeros(1),
            "physical_encoder.blocks.0.weight": torch.zeros(1),
        },
        stage1_dir / "model.safetensors",
    )
    stage1_config = RoboContrastConfig(
        vision_backbone="qwen3vl",
        qwen3vl_dir="/Data/old-qwen",
        cosmos3_dir="/Data/old-cosmos",
        chunk_size=4,
        group_size=2,
        n_action_steps=2,
        hidden_dim=8,
        num_attention_heads=4,
        num_physical_layers=1,
        max_action_dim=4,
        max_state_dim=4,
        max_tactile_signal_dim=3,
        max_tactile_views=2,
        tactile_tokens_per_pad=1,
        tactile_frames=2,
        tactile_img_size=8,
        num_change_queries=2,
    )
    stage1_config._save_pretrained(stage1_dir)
    teacher = SimpleNamespace(
        config=stage1_config,
        perception_encoder=_FakePerception(queries=2, width=8),
        physical_encoder=_FakePhysical(groups=2, hidden=8, tactile_tokens=4),
    )

    def load_fake_teacher(*args, **kwargs):
        teacher.config = kwargs["config"]
        return teacher

    monkeypatch.setattr(stage2.RoboContrast, "from_pretrained", load_fake_teacher)
    monkeypatch.setattr(stage2, "Qwen3VLUnderstandingExpert", _FakeUnderstanding)
    monkeypatch.setattr(
        stage2,
        "transfer_stage1_perception",
        lambda *args, **kwargs: (
            TransferReport(1, 1, 1),
            TransferReport(0, 0, 1),
        ),
    )
    monkeypatch.setattr(
        cosmos3_encoders,
        "build_cosmos3_vae",
        lambda *args, **kwargs: (
            _FakeVAE(4),
            4,
            4,
            torch.zeros(4),
            torch.ones(4),
        ),
    )
    config = Qwen3VLMoTConfig(
        stage1_checkpoint=str(stage1_dir),
        qwen3vl_dir="/mnt/runtime-qwen",
        cosmos3_dir=str(generator_dir) if generation_kind != "random" else "/mnt/runtime-cosmos",
        generation_backbone=generation_kind,
        generation_pretrained_dir=str(generator_dir),
        generation_tuning_mode="frozen" if generation_variant.endswith("_frozen") else "full",
        understanding_tuning_mode="frozen",
        num_latent_queries=2,
        latent_action_dim=8,
        generation_hidden_dim=16,
        generation_depth=1,
        generation_num_heads=4,
        generation_num_kv_heads=4,
        generation_intermediate_dim=32,
        understanding_kv_layers=1,
        video_latent_dim=4,
        world_video_frames=9,
        video_image_size=16,
        chunk_size=4,
        group_size=2,
        n_action_steps=2,
        max_action_dim=4,
        max_state_dim=4,
        max_tactile_signal_dim=3,
        physical_hidden_dim=8,
        max_tactile_views=2,
        tactile_tokens_per_pad=1,
        tactile_frames=2,
        tactile_img_size=8,
        use_tactile_conditioning=True,
        generation_gradient_checkpointing=multiview,
        rgb_views=("primary", "secondary", "wrist") if multiview else ("primary",),
        physical_flow_space=flow_space,
        action_prediction_reconstruction_loss_weight=0.1 if flow_space == "embedding" else 0.0,
        state_prediction_reconstruction_loss_weight=0.1 if flow_space == "embedding" else 0.0,
        time_sampling_distribution=time_distribution,
        time_sampling_shift=3.0,
        inference_steps=3,
    )
    policy = stage2.Qwen3VLMoTPolicy(config)
    if generation_kind != "random":
        assert policy.generation_load_report.copied_tensors > 0
        assert policy.generation.output_heads["video"].weight.count_nonzero() > 0
    assert teacher.config.qwen3vl_dir == config.qwen3vl_dir
    assert teacher.config.cosmos3_dir == config.cosmos3_dir
    batch = {
        "image_t0": torch.zeros(2, 3, 8, 8, dtype=torch.uint8),
        "image_t1": torch.ones(2, 3, 8, 8, dtype=torch.uint8),
        "video": torch.zeros(2, 9, 3, 8, 8, dtype=torch.uint8),
        "task": ["move", "touch"],
        "has_text": torch.ones(2),
        "pair_is_valid": torch.ones(2),
        "observation.state": torch.randn(2, 4, 4),
        "state_mask": torch.ones(2, 4),
        "action": torch.randn(2, 4, 4),
        "action_mask": torch.ones(2, 4),
        "tactile_signal": torch.randn(2, 4, 3),
        "tactile_signal_mask": torch.ones(2),
        "tactile_image": torch.zeros(2, 2, 2, 3, 8, 8, dtype=torch.uint8),
        "tactile_image_mask": torch.ones(2, 2),
        "sample_rate": torch.full((2,), 10, dtype=torch.long),
    }
    if multiview:
        batch.update({
            "images_t0": batch["image_t0"][:, None].expand(-1, 3, -1, -1, -1).clone(),
            "image_view_mask": torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool),
            "video_views": batch["video"][:, None].expand(-1, 3, -1, -1, -1, -1).clone(),
            "video_view_mask": torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool),
        })
        batch["images_t0"][:, 1:] = 125
        batch["video_views"][:, 1:] = 125
    generation_inputs = {}
    generation_outputs = {}
    decoder_inputs = {"action": [], "state": []}
    sigma_calls = []
    original_sample_sigma = policy._sample_sigma

    def record_sigma(batch_size, device):
        sigma_calls.append(batch_size)
        return original_sample_sigma(batch_size, device)

    policy._sample_sigma = record_sigma

    def record_generation_inputs(module, args):
        generation_inputs.clear()
        generation_inputs.update({
            stream.name: (stream.values.detach().clone(), stream.sigma.detach().clone())
            for stream in args[0]
        })

    def record_generation_outputs(module, args, output):
        generation_outputs.clear()
        generation_outputs.update({name: value.detach().clone() for name, value in output.items()})

    policy.generation.register_forward_pre_hook(record_generation_inputs)
    policy.generation.register_forward_hook(record_generation_outputs)
    if flow_space == "embedding":
        for name in decoder_inputs:
            policy.physical_decoders[name].register_forward_pre_hook(
                lambda module, args, name=name: decoder_inputs[name].append(args[0].detach().clone())
            )

    for task_name in TASK_SPECS:
        policy.physical_encoder.include_tactile_history.clear()
        sigma_calls.clear()
        for recorded in decoder_inputs.values():
            recorded.clear()
        rng = torch.get_rng_state()
        loss, metrics = policy(batch, task_type=task_name)
        assert torch.isfinite(loss)
        assert metrics[f"task_{task_name}"] == 1.0
        assert policy.understanding.last_images is (
            None if task_name == "t2v" else batch["images_t0" if multiview else "image_t0"]
        )
        if multiview and task_name != "t2v":
            assert policy.understanding.last_view_arguments["image_mask"] is batch["image_view_mask"]
            assert policy.understanding.last_view_arguments["view_names"] == config.rgb_views
        if task_name == "t2v":
            assert metrics["video_target_elements"] == (144 if multiview else 96)
        elif task_name in ("i2v", "joint_prediction", "forward_dynamics"):
            assert metrics["video_target_elements"] == (96 if multiview else 64)
        if task_name in ("joint_prediction", "state_prediction"):
            assert metrics["state_target_elements"] == (32 if flow_space == "embedding" else 24)
        if task_name in (
            "joint_prediction",
            "inverse_dynamics",
            "action_prediction",
        ):
            assert metrics["action_target_elements"] == 32
        assert "action_reconstruction_loss" not in metrics
        assert "state_reconstruction_loss" not in metrics
        if task_name == "joint_prediction":
            assert sigma_calls == [2]
            for name in ("action", "state"):
                torch.testing.assert_close(generation_inputs[name][1], generation_inputs["video"][1])
        for name in ("action", "state"):
            if flow_space == "embedding" and TASK_SPECS[task_name].role(name).is_target:
                assert len(decoder_inputs[name]) == 1
                noisy, sigma = generation_inputs[name]
                expected = (
                    noisy[:, :config.num_groups].float()
                    - sigma.float()[:, None, None] * generation_outputs[name][:, :config.num_groups].float()
                )
                torch.testing.assert_close(decoder_inputs[name][0], expected)
                with torch.no_grad():
                    predicted = policy._decode_physical_embedding(name, expected).reshape(2, 4, 4)
                    target = batch["action" if name == "action" else "observation.state"]
                    if name == "state":
                        predicted, target = predicted[:, 1:], target[:, 1:]
                    expected_loss = (predicted - target).square().mean()
                assert metrics[f"{name}_prediction_reconstruction_loss"] == pytest.approx(expected_loss.item())
                assert metrics[f"{name}_prediction_reconstruction_elements"] == (32 if name == "action" else 24)
            else:
                assert not decoder_inputs[name]
                assert metrics[f"{name}_prediction_reconstruction_loss"] == 0
                assert metrics[f"{name}_prediction_reconstruction_elements"] == 0
        if task_name in ("forward_dynamics", "inverse_dynamics"):
            assert "state_flow_loss" not in metrics
            assert "state_reconstruction_loss" not in metrics
            assert not policy.physical_encoder.last_state.any()
            torch.set_rng_state(rng)
            changed_loss, changed_metrics = policy({
                **batch, "observation.state": batch["observation.state"] + 1000,
                "state_mask": torch.zeros_like(batch["state_mask"]),
            }, task_type=task_name)
            torch.testing.assert_close(changed_loss, loss)
            assert changed_metrics["valid_rows"] == metrics["valid_rows"]
            policy.zero_grad(set_to_none=True)
            changed_loss.backward()
            assert policy.generation.output_heads["state"].weight.grad is None
            if flow_space == "embedding":
                assert policy.physical_decoders["state"][-1].weight.grad is None
        if task_name == "joint_prediction" and flow_space == "raw":
            torch.testing.assert_close(policy.physical_encoder.last_state[:, 0], batch["observation.state"][:, 0])
        if task_name == "action_prediction" or (task_name == "joint_prediction" and flow_space == "embedding"):
            torch.testing.assert_close(
                policy.physical_encoder.input_history[-2]["observation.state"]
                if flow_space == "embedding" else policy.physical_encoder.last_state,
                batch["observation.state"][:, :1].expand(-1, 4, -1),
            )
        if task_name == "tactile_prediction":
            assert metrics["tactile_target_elements"] == 64
            assert policy.physical_encoder.include_tactile_history == (
                [False, False, True] if flow_space == "embedding" else [False, True]
            )
        if flow_space == "embedding" and task_name == "joint_prediction":
            policy.zero_grad(set_to_none=True)
            loss.backward()
            assert policy.physical_decoders["action"][-1].weight.grad is not None
            assert policy.physical_decoders["state"][-1].weight.grad is not None
            assert policy.generation.output_heads["action"].weight.grad is not None
            assert policy.generation.output_heads["state"].weight.grad is not None
            assert policy.physical_encoder.anchor.grad is None
            assert not policy.physical_encoder.training
        if multiview and task_name == "joint_prediction":
            if flow_space == "raw":
                loss.backward()
            assert policy.generation.video_view_embedding.weight.grad is not None
            assert policy.teacher_perception.anchor.grad is None
            torch.testing.assert_close(policy.teacher_perception.last_images[0], batch["image_t0"])
            torch.testing.assert_close(policy.teacher_perception.last_images[1], batch["image_t1"])
    if flow_space == "embedding":
        old_action_weight = policy.config.action_loss_weight
        old_latent_weight = policy.config.latent_action_loss_weight
        policy.config.action_loss_weight = 0.0
        policy.config.latent_action_loss_weight = 0.0
        masked = copy.deepcopy(batch)
        masked["action_mask"][0, 1] = 0
        masked["pair_is_valid"][1] = 0
        if multiview:
            masked["video_view_mask"][1] = False
        torch.manual_seed(83)
        active_loss, active_metrics = policy(masked, task_type="inverse_dynamics")
        assert active_metrics["action_prediction_reconstruction_elements"] == 12
        policy.config.action_prediction_reconstruction_loss_weight = 0.0
        torch.manual_seed(83)
        baseline_loss, baseline_metrics = policy(masked, task_type="inverse_dynamics")
        assert baseline_metrics["action_prediction_reconstruction_loss"] == 0
        auxiliary = active_loss - baseline_loss
        assert auxiliary.item() == pytest.approx(
            0.1 * active_metrics["action_prediction_reconstruction_loss"], abs=1e-6
        )
        policy.zero_grad(set_to_none=True)
        auxiliary.backward()
        assert policy.generation.output_heads["action"].weight.grad.abs().sum() > 0
        assert policy.physical_decoders["action"][-1].weight.grad.abs().sum() > 0
        assert policy.generation.output_heads["state"].weight.grad is None
        assert policy.physical_decoders["state"][-1].weight.grad is None
        assert policy.physical_encoder.anchor.grad is None
        policy.config.action_loss_weight = old_action_weight
        policy.config.latent_action_loss_weight = old_latent_weight
        policy.config.action_prediction_reconstruction_loss_weight = 0.1
        old_state_weight = policy.config.state_loss_weight
        policy.config.state_loss_weight = 0.0
        policy.config.latent_action_loss_weight = 0.0
        masked = copy.deepcopy(batch)
        masked["state_mask"][0, 1] = 0
        masked["pair_is_valid"][1] = 0
        if multiview:
            masked["video_view_mask"][1] = False
        torch.manual_seed(91)
        state_loss, state_metrics = policy(masked, task_type="state_prediction")
        assert state_metrics["state_prediction_reconstruction_elements"] == 9
        policy.config.state_prediction_reconstruction_loss_weight = 0.0
        torch.manual_seed(91)
        state_baseline, baseline_metrics = policy(masked, task_type="state_prediction")
        assert baseline_metrics["state_prediction_reconstruction_loss"] == 0
        state_auxiliary = state_loss - state_baseline
        assert state_auxiliary.item() == pytest.approx(
            0.1 * state_metrics["state_prediction_reconstruction_loss"], abs=1e-6
        )
        policy.zero_grad(set_to_none=True)
        state_auxiliary.backward()
        assert policy.generation.output_heads["state"].weight.grad.abs().sum() > 0
        assert policy.physical_decoders["state"][-1].weight.grad.abs().sum() > 0
        assert policy.generation.output_heads["action"].weight.grad is None
        assert policy.physical_decoders["action"][-1].weight.grad is None
        assert policy.physical_encoder.anchor.grad is None
        policy.config.state_loss_weight = old_state_weight
        policy.config.latent_action_loss_weight = old_latent_weight
        policy.config.state_prediction_reconstruction_loss_weight = 0.1
        oracle_clean, _, _, _ = policy._clean_physical_embedding(
            batch, "action", torch.ones(2, dtype=torch.bool)
        )

        def perfect_velocity(module, args, outputs):
            stream = next(stream for stream in args[0] if stream.name == "action")
            velocity = (stream.values.float() - oracle_clean) / stream.sigma[:, None, None]
            return {**outputs, "action": velocity}

        oracle_hook = policy.generation.register_forward_hook(perfect_velocity)
        _, oracle_metrics = policy(batch, task_type="inverse_dynamics")
        oracle_hook.remove()
        with torch.no_grad():
            oracle_action = policy._decode_physical_embedding("action", oracle_clean)
            oracle_error = (oracle_action - batch["action"].reshape_as(oracle_action)).square().mean()
        assert oracle_metrics["action_prediction_reconstruction_loss"] == pytest.approx(
            oracle_error.item(), abs=1e-5
        )
        policy.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            mixed_loss, mixed_metrics = policy(batch, task_type="inverse_dynamics")
        assert torch.isfinite(mixed_loss)
        assert mixed_metrics["action_prediction_reconstruction_elements"] == 32
        mixed_loss.backward()
        assert torch.isfinite(policy.generation.output_heads["action"].weight.grad).all()
    canonical_action = policy.sample_canonical_action(batch)
    assert canonical_action.shape == (2, 2, 4)
    if multiview:
        generated = policy.generate_validation(batch, seed=17)
        assert generated["video_latents_views"].shape == (2, 3, 4, 3, 2, 2)
        torch.testing.assert_close(generated["video_latents"], generated["video_latents_views"][:, 0])
        torch.testing.assert_close(generated["video_view_mask"], batch["image_view_mask"])
    torch.testing.assert_close(
        policy.physical_encoder.last_state,
        batch["observation.state"][:, :1].expand(-1, 4, -1),
    )
    policy.config.task_weights = tuple(1.0 if name == "joint_prediction" else 0.0 for name in TASK_SPECS)
    joint = policy.generate_validation(batch, seed=37)
    assert joint["action_normalized"].shape == (2, 2, 4)
    assert joint["state_normalized"].shape == (2, 4, 4)
    assert joint["video_latents"].shape == (2, 4, 3, 2, 2)
    assert not joint["tactile_mask"].any()
    torch.testing.assert_close(joint["state_normalized"][:, 0], batch["observation.state"][:, 0])
    if multiview:
        assert joint["video_latents_views"].shape == (2, 3, 4, 3, 2, 2)
        assert not joint["video_latents_views"][~batch["image_view_mask"]].any()
    changed = copy.deepcopy(batch)
    changed["observation.state"][:, 1:] += 1000
    changed["action"] += 1000
    changed["video"][:, 1:] = 255
    if multiview:
        changed["video_views"][:, :, 1:] = 255
    repeated = policy.generate_validation(changed, seed=37)
    for name in joint:
        torch.testing.assert_close(joint[name], repeated[name])

    stage2_dir = tmp_path / "stage2"
    stage2_dir.mkdir()
    policy.config.stage1_policy_config["qwen3vl_dir"] = "/Data/embedded-old-qwen"
    policy.config.stage1_policy_config["cosmos3_dir"] = "/Data/embedded-old-cosmos"
    policy._save_pretrained(stage2_dir)
    (stage1_dir / "model.safetensors").unlink()
    (stage1_dir / "config.json").unlink()
    stage1_dir.rmdir()
    if generation_kind != "random":
        (generator_dir / "transformer/diffusion_pytorch_model.safetensors").unlink()
        (generator_dir / "transformer/config.json").unlink()

    restored_stage1_configs = []

    class _FakeRestoredStage1:
        def __init__(self, restored_config):
            restored_stage1_configs.append(restored_config)
            self.config = restored_config
            self.perception_encoder = _FakePerception(queries=2, width=8)
            self.physical_encoder = _FakePhysical(groups=2, hidden=8, tactile_tokens=4)

    monkeypatch.setattr(stage2, "RoboContrast", _FakeRestoredStage1)
    restored = stage2.Qwen3VLMoTPolicy.from_pretrained(stage2_dir)
    assert restored.config.stage1_policy_config is not None
    assert restored.config.stage1_checkpoint == str(stage1_dir)
    assert restored_stage1_configs[0].qwen3vl_dir == config.qwen3vl_dir
    assert restored_stage1_configs[0].cosmos3_dir == config.cosmos3_dir
    assert restored.config.rgb_views == config.rgb_views
    assert restored.config.physical_flow_space == flow_space
    assert restored.config.generation_backbone == generation_kind
    if generation_kind != "random":
        assert not hasattr(restored, "generation_load_report")
        for name, parameter in policy.generation.named_parameters():
            torch.testing.assert_close(parameter, dict(restored.generation.named_parameters())[name])
    if flow_space == "embedding":
        for name, parameter in policy.physical_decoders.named_parameters():
            torch.testing.assert_close(parameter, dict(restored.physical_decoders.named_parameters())[name])
        torch.manual_seed(31)
        expected_action = policy.sample_canonical_action(batch)
        torch.manual_seed(31)
        torch.testing.assert_close(restored.sample_canonical_action(batch), expected_action)
    else:
        assert not hasattr(restored, "physical_decoders")


def test_multiview_dataset_keeps_padded_tails_and_masks_missing_duplicate_or_invalid_views(monkeypatch):
    import lerobot.common.datasets.contrastive_dataset as dataset_module

    dataset = object.__new__(MultiModalContrastiveDataset)
    dataset.cfg = SimpleNamespace(
        dataset=SimpleNamespace(image_transforms=SimpleNamespace(img_size=8))
    )
    dataset.rgb_views = ("primary", "secondary", "wrist")
    dataset.world_video_frames = 3
    monkeypatch.setitem(dataset_module.OXE_DATASET_CONFIGS, "test_cameras", {
        "image_obs_keys": {"primary": "front", "secondary": "images.rgb.side", "wrist": "hand"}
    })
    metadata = SimpleNamespace(
        video_keys=["observation.images.front", "images.rgb.side", "observations.images.hand"],
        image_keys=[],
    )
    keys = dataset._resolve_image_keys("test_cameras", None, ds_meta=metadata)
    assert keys == {
        "primary": "observation.images.front",
        "secondary": "images.rgb.side",
        "wrist": "observations.images.hand",
    }
    item = {key: torch.full((3, 3, 8, 8), (i + 1) * 50, dtype=torch.uint8)
            for i, key in enumerate(keys.values())}
    item["observations.images.hand_is_pad"] = torch.tensor([0, 0, 1], dtype=torch.bool)
    result = dataset._extract_rgb_views(item, keys)
    assert result["images_t0"].shape == (3, 3, 8, 8)
    assert result["video_views"].shape == (3, 3, 3, 8, 8)
    assert result["image_view_mask"].tolist() == [True, True, True]
    assert result["video_view_mask"].tolist() == [True, True, True]
    assert result["video_views"][:, 0, 0, 0, 0].tolist() == [50, 100, 150]
    keys["secondary"] = None
    keys["wrist"] = keys["primary"]
    result = dataset._extract_rgb_views(item, keys)
    assert result["image_view_mask"].tolist() == [True, False, False]
    assert result["video_view_mask"].tolist() == [True, False, False]
    assert not result["video_views"][1:].any()
    item[f"{keys['primary']}_is_pad"] = torch.ones(3, dtype=torch.bool)
    result = dataset._extract_rgb_views(item, keys)
    assert not result["image_view_mask"].any()
    assert not result["video_view_mask"].any()


@pytest.mark.parametrize("version", ["v2.1", "v3.0"])
def test_multiview_dataset_decodes_only_enabled_cameras_on_the_same_timestamp_grid(monkeypatch, version):
    import lerobot.common.datasets.contrastive_dataset as dataset_module

    roles = ("primary", "secondary", "wrist")
    keys = {role: f"observation.images.{role}" for role in roles}
    keys["wrist"] = keys["secondary"]
    metadata = SimpleNamespace(
        fps=30, video_keys=list(set(keys.values())) + ["observation.images.unused"],
        features={key: {} for key in set(keys.values())},
    )
    for name in ("LeRobotDatasetMetadata", "LeRobotDatasetMetadataV30"):
        monkeypatch.setattr(dataset_module, name, lambda *args, **kwargs: metadata)
    captured = {}

    def load_dataset(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    for name in ("LeRobotDataset", "LeRobotDatasetV30"):
        monkeypatch.setattr(dataset_module, name, load_dataset)
    monkeypatch.setattr(dataset_module, "resolve_delta_timestamps", lambda *args: {})
    monkeypatch.setattr(dataset_module, "resolve_true_fps", lambda *args: 30.0)
    dataset = object.__new__(MultiModalContrastiveDataset)
    dataset.rgb_views = roles
    dataset.use_wrist_image = True
    dataset.world_video_frames = 9
    dataset.tactile_frames = 2
    dataset.max_tactile_views = 2
    dataset._video_backend = "torchcodec"
    dataset._resolve_image_keys = lambda *args, **kwargs: keys
    dataset._window_offsets = lambda fps: (list(range(32)), 31)
    loaded, _, _, horizon, _ = dataset._build_dataset(
        SimpleNamespace(policy=Qwen3VLMoTConfig(rgb_views=roles)),
        "test_multiview", "/unused", version, {},
    )
    assert loaded.video_keys_to_decode == [keys["primary"], keys["secondary"]]
    assert set(captured["delta_timestamps"]) == set(loaded.video_keys_to_decode)
    assert horizon == 31
    expected = _evenly_spaced_frame_timestamps(31, 9, 30)
    assert all(stamps == expected for stamps in captured["delta_timestamps"].values())


def test_multiview_video_stream_and_sampling_keep_each_current_frame_without_future_leakage():
    class TemporalVAE(_FakeVAE):
        def encode(self, video):
            self.last_encoded_batch = video.shape[0]
            values = video.mean(dim=1, keepdim=True)[:, :, ::4]
            values = torch.nn.functional.adaptive_avg_pool3d(values, (values.shape[2], 2, 2))
            return SimpleNamespace(latent_dist=SimpleNamespace(
                mean=values.expand(-1, self.channels, -1, -1, -1)
            ))

    policy = object.__new__(Qwen3VLMoTPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        rgb_views=("primary", "secondary", "wrist"), world_video_frames=9,
        video_image_size=16, video_latent_dim=4, video_latent_patch_size=2,
        inference_steps=2, sigma_min=1e-3, sigma_max=0.999,
        time_sampling_distribution="uniform", time_sampling_beta_probability=0.5,
    )
    policy.video_vae = TemporalVAE(4)
    policy.temporal_compression = 4
    policy.register_buffer("video_latent_mean", torch.zeros(1, 4, 1, 1, 1))
    policy.register_buffer("video_latent_std", torch.ones(1, 4, 1, 1, 1))
    policy.generation = GenerationExpert(
        understanding_dim=12, hidden_dim=16, depth=1, num_heads=4, dropout=0.0,
        input_dims={"video": 16}, output_dims={"video": 16},
        task_names=("i2v", "t2v"), gradient_checkpointing=False, num_video_views=3,
    )
    video = torch.stack([
        torch.full((9, 3, 8, 8), value, dtype=torch.uint8) for value in (0, 127, 255)
    ]).unsqueeze(0).expand(2, -1, -1, -1, -1, -1).clone()
    mask = torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.bool)
    batch = {"video": video[:, 0], "video_views": video,
             "image_view_mask": mask, "video_view_mask": mask}
    clean, positions, keep, views = policy._video_stream_tokens(batch)
    assert policy.video_vae.last_encoded_batch == 4
    assert clean.shape == (2, 9, 16)
    assert views.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert positions[:, 0].tolist() == [0, 1, 2] * 3
    for name, expected_elements in (("i2v", 128), ("t2v", 192)):
        stream, _, target_mask, _ = policy._prepare_video_stream(
            TASK_SPECS[name].video, batch, torch.ones(2, dtype=torch.bool)
        )
        assert target_mask.sum() == expected_elements
        torch.testing.assert_close(stream.keep, keep)
        assert not stream.values[~keep].any()
        if name == "i2v":
            torch.testing.assert_close(stream.values[:, positions[:, 0] == 0],
                                       clean[:, positions[:, 0] == 0])
    understanding = _FakeUnderstanding("", num_queries=2)(None, ["move"] * 2)

    def sample(values):
        return policy._sample_video_latents(
            values, understanding, generator=torch.Generator().manual_seed(123)
        )

    generated = sample(batch)
    assert generated.shape == (2, 3, 4, 3, 2, 2)
    assert not generated[~mask].any()
    expected_current = clean.view(2, 3, 3, 16)[:, :, 0].view(2, 3, 4, 2, 2)
    torch.testing.assert_close(generated[:, :, :, 0], expected_current)
    changed = video.clone()
    changed[:, :, 1:] = 255 - changed[:, :, 1:]
    changed[~mask] = 255 - changed[~mask]
    torch.testing.assert_close(sample({**batch, "video_views": changed}), generated)
    no_views = {**batch, "image_view_mask": torch.zeros_like(mask)}
    assert not sample(no_views).any()


@pytest.mark.parametrize("roles", [(), ("wrist",), ("primary", "primary"), ("primary", "unknown")])
def test_rgb_views_reject_invalid_camera_contracts(roles):
    with pytest.raises(ValueError, match="rgb_views"):
        Qwen3VLMoTConfig(rgb_views=roles)


def _embedding_physical_policy():
    class CoupledPhysical(_FakePhysical):
        def forward(self, batch, **kwargs):
            tokens, keep, loss = super().forward(batch, **kwargs)
            state = batch["observation.state"].reshape(-1, self.groups, self.hidden)
            action = batch["action"].reshape_as(state)
            context = (state.mean(dim=(1, 2)) + action.mean(dim=(1, 2)))[:, None, None]
            tokens[:, 1 : 1 + self.groups] = state + context
            tokens[:, 1 + self.groups : 1 + 2 * self.groups] = action + context
            return tokens, keep, loss

    policy = object.__new__(Qwen3VLMoTPolicy)
    nn.Module.__init__(policy)
    policy.config = Qwen3VLMoTConfig(
        physical_flow_space="embedding", chunk_size=4, group_size=2, n_action_steps=4,
        physical_hidden_dim=8, max_action_dim=4, max_state_dim=4,
        inference_steps=3, num_latent_queries=2, latent_action_dim=8,
        task_names=("action_prediction",), task_weights=(1.0,),
    )
    policy.physical_encoder = CoupledPhysical(groups=2, hidden=8, tactile_tokens=4).eval()
    policy.physical_encoder.requires_grad_(False)
    policy.physical_decoders = nn.ModuleDict({
        name: nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 8))
        for name in ("state", "action")
    })
    policy.generation = GenerationExpert(
        understanding_dim=12, hidden_dim=16, depth=1, num_heads=4, dropout=0.0,
        input_dims={"state": 8, "action": 8}, output_dims={"state": 8, "action": 8},
        task_names=("action_prediction",), gradient_checkpointing=False,
    )
    batch = {
        "image_t0": torch.zeros(2, 3, 8, 8, dtype=torch.uint8),
        "observation.state": torch.arange(32).reshape(2, 4, 4).float() / 32,
        "state_mask": torch.tensor([[1, 1, 0, 1], [1, 1, 1, 1]]).float(),
        "action": torch.arange(32).flip(0).reshape(2, 4, 4).float() / 32,
        "action_mask": torch.tensor([[1, 0, 1, 1], [1, 1, 1, 1]]).float(),
        "tactile_signal_mask": torch.zeros(2),
    }
    return policy, batch


def test_physical_flow_corrupts_clean_isolated_embeddings_not_encoder_inputs():
    policy, batch = _embedding_physical_policy()
    valid = torch.tensor([1, 0], dtype=torch.bool)
    policy._sample_sigma = lambda batch_size, device: torch.full((batch_size,), 0.25, device=device)
    clean, raw, keep, raw_mask = policy._clean_physical_embedding(batch, "action", valid)
    torch.manual_seed(17)
    expected_noise = torch.randn_like(clean)
    torch.manual_seed(17)
    streams, targets, _, reconstructions = policy._prepare_physical_streams(
        TASK_SPECS["action_prediction"], batch, valid
    )
    action = next(stream for stream in streams if stream.name == "action")
    expected_values = 0.75 * clean + 0.25 * expected_noise
    expected_values[~keep] = 0
    torch.testing.assert_close(action.values, expected_values)
    torch.testing.assert_close(targets["action"][0][keep], (expected_noise - clean)[keep])
    assert targets["action"][1].sum() == 16
    assert raw_mask.sum() == 12
    assert "state" not in reconstructions
    torch.testing.assert_close(reconstructions["action"][0], raw)
    torch.testing.assert_close(reconstructions["action"][1], raw_mask)
    assert not action.values[1].any()
    assert not action.keep[1].any()
    for recorded in policy.physical_encoder.input_history:
        assert not (recorded["state_mask"].any() and recorded["action_mask"].any())
        for name, key in (("state", "observation.state"), ("action", "action")):
            if not recorded[f"{name}_mask"].any():
                assert not recorded[key].any()
                continue
            source = batch[key]
            if name == "state":
                source = source[:, :1].expand_as(source)
            expected = torch.where(recorded[f"{name}_mask"].bool()[:, None],
                                   source, torch.zeros_like(source))
            torch.testing.assert_close(recorded[key], expected)
    decoded = policy._decode_physical_embedding("action", clean)
    loss, _, count = policy._distributed_mse(decoded, raw, raw_mask)
    assert count == 12
    loss.backward()
    assert policy.physical_decoders["action"][-1].weight.grad is not None
    assert policy.physical_encoder.anchor.grad is None


@pytest.mark.parametrize("task_name", [
    "action_prediction", "joint_prediction", "inverse_dynamics",
    "forward_dynamics", "state_prediction",
])
def test_clean_physical_conditions_do_not_leak_target_trajectories(task_name):
    policy, batch = _embedding_physical_policy()
    task = TASK_SPECS[task_name]
    valid = torch.ones(2, dtype=torch.bool)

    def prepare(values):
        torch.manual_seed(19)
        streams, targets, _, _ = policy._prepare_physical_streams(task, values, valid)
        return {stream.name: stream for stream in streams}, targets

    before, targets = prepare(batch)
    changed = {key: value.clone() for key, value in batch.items()}
    if task.state is ModalityRole.ABSENT:
        assert "state" not in before
        assert "state" not in targets
        changed["observation.state"] += 100
        after, _ = prepare(changed)
        torch.testing.assert_close(before["action"].values, after["action"].values)
        return
    if task_name == "joint_prediction":
        changed["action"] += 100
        changed["observation.state"][:, 1:] += 100
        after, _ = prepare(changed)
        torch.testing.assert_close(before["state"].values[:, -1], after["state"].values[:, -1])
        assert not torch.equal(before["state"].values[:, :-1], after["state"].values[:, :-1])
        assert not torch.equal(before["action"].values, after["action"].values)
        assert not targets["state"][1][:, -1].any()
        return
    if task.action.is_target:
        changed["action"] += 100
        if task.state is ModalityRole.CURRENT_ONLY:
            changed["observation.state"][:, 1:] += 100
        after, _ = prepare(changed)
        torch.testing.assert_close(before["state"].values, after["state"].values)
        assert not torch.equal(before["action"].values, after["action"].values)
    else:
        changed["observation.state"][:, 1:] += 100
        after, _ = prepare(changed)
        torch.testing.assert_close(before["action"].values, after["action"].values)
        # All contextual state embeddings are noised, plus one clean current-only token.
        assert before["state"].values.shape == (2, 3, 8)
        torch.testing.assert_close(before["state"].values[:, -1], after["state"].values[:, -1])
        assert not torch.equal(before["state"].values[:, :2], after["state"].values[:, :2])
        assert targets["state"][1][:, :2].all()
        assert not targets["state"][1][:, -1].any()
        assert targets["state"][1].sum() == 32


def test_embedding_action_sampling_never_encodes_noisy_actions_or_future_state():
    policy, batch = _embedding_physical_policy()
    understanding = _FakeUnderstanding("", num_queries=2)(None, ["move"] * 2)

    def sample(values):
        policy.physical_encoder.input_history.clear()
        prediction = policy._sample_canonical_action(
            values, understanding, generator=torch.Generator().manual_seed(23)
        )
        assert len(policy.physical_encoder.input_history) == 1
        encoded = policy.physical_encoder.input_history[0]
        assert not encoded["action"].any()
        assert not encoded["action_mask"].any()
        expected_state = values["observation.state"][:, :1].expand(-1, 4, -1)
        torch.testing.assert_close(
            encoded["observation.state"],
            expected_state * values["state_mask"][:, None],
        )
        return prediction

    prediction = sample(batch)
    assert prediction.shape == (2, 4, 4)
    assert not prediction[0, :, 1].any()
    changed = {**batch, "action": torch.full_like(batch["action"], 999),
               "observation.state": batch["observation.state"].clone()}
    changed["observation.state"][:, 1:] = -999
    torch.testing.assert_close(sample(changed), prediction)
    missing = {**batch, "action_mask": torch.zeros_like(batch["action_mask"])}
    assert not sample(missing).any()


def test_embedding_euler_updates_latents_and_decodes_only_after_integration():
    class ConstantVelocity(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1))
            self.inputs = []
            self.task_to_id = {"action_prediction": 0}

        def forward(self, streams, *args, **kwargs):
            self.inputs.append({stream.name: stream.values.clone() for stream in streams})
            return {"action": torch.full_like(streams[1].values, 2.0)}

    policy, batch = _embedding_physical_policy()
    policy.generation = ConstantVelocity()
    policy.physical_decoders["action"] = nn.Sequential(nn.Identity(), nn.Linear(8, 8, bias=False))
    with torch.no_grad():
        policy.physical_decoders["action"][1].weight.copy_(torch.eye(8))
    understanding = _FakeUnderstanding("", num_queries=2)(None, ["move"] * 2)
    initial = torch.randn(2, 2, 8, generator=torch.Generator().manual_seed(29))
    prediction = policy._sample_canonical_action(
        batch, understanding, generator=torch.Generator().manual_seed(29)
    )
    expected = (initial - 2).reshape(2, 4, 4) * batch["action_mask"][:, None]
    torch.testing.assert_close(prediction, expected)
    assert len(policy.generation.inputs) == policy.config.inference_steps
    assert len(policy.physical_encoder.input_history) == 1
    torch.testing.assert_close(policy.generation.inputs[0]["action"], initial)
    for recorded in policy.generation.inputs[1:]:
        torch.testing.assert_close(recorded["state"], policy.generation.inputs[0]["state"])


@pytest.mark.parametrize("flow_space", ["raw", "embedding"])
@pytest.mark.parametrize("multiview", [False, True])
@pytest.mark.parametrize("zero_weight_action", [False, True])
def test_three_task_joint_sampling_uses_only_current_conditions(flow_space, multiview, zero_weight_action):
    class CurrentVideoVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)
            self.inputs = []

        def encode(self, video):
            self.inputs.append(video.clone())
            latent = torch.nn.functional.adaptive_avg_pool3d(video.mean(1, keepdim=True), (1, 2, 2))
            return SimpleNamespace(latent_dist=SimpleNamespace(mean=latent.expand(-1, 4, -1, -1, -1)))

    policy, batch = _embedding_physical_policy()
    policy.config.physical_flow_space = flow_space
    if flow_space == "raw":
        policy.config.physical_hidden_dim = 12
        policy.physical_encoder = _FakePhysical(groups=2, hidden=12, tactile_tokens=4)
    policy.config.task_names = ("forward_dynamics", "inverse_dynamics", "joint_prediction")
    policy.config.task_weights = (1 / 3, 1 / 3, 1 / 3)
    if zero_weight_action:
        policy.config.task_names += ("action_prediction",)
        policy.config.task_weights += (0.0,)
    policy.config.video_latent_dim = 4
    policy.config.video_image_size = 16
    policy.config.rgb_views = ("primary", "secondary", "wrist") if multiview else ("primary",)
    policy.config.tactile_generation_target = "spatial_patches"
    policy.video_vae = CurrentVideoVAE()
    policy.temporal_compression = 4
    policy.register_buffer("video_latent_mean", torch.zeros(1, 4, 1, 1, 1))
    policy.register_buffer("video_latent_std", torch.ones(1, 4, 1, 1, 1))
    policy.understanding = _FakeUnderstanding("", num_queries=2)
    policy.generation = GenerationExpert(
        understanding_dim=12, hidden_dim=16, depth=1, num_heads=4, dropout=0.0,
        input_dims={
            "video": 16, "state": policy.config.physical_hidden_dim, "action": policy.config.physical_hidden_dim,
        },
        output_dims={"video": 16, "state": 8, "action": 8},
        task_names=policy.config.task_names, gradient_checkpointing=False,
        num_video_views=len(policy.rgb_views),
    )
    with torch.no_grad():
        for head in policy.generation.output_heads.values():
            head.weight.normal_(std=0.02)
    batch["video"] = torch.zeros(2, 9, 3, 8, 8, dtype=torch.uint8)
    batch["tactile_image_mask"] = torch.zeros(2, 2)
    batch["task"] = ["move", "touch"]
    if multiview:
        batch["images_t0"] = batch["image_t0"][:, None].expand(-1, 3, -1, -1, -1).clone()
        batch["image_view_mask"] = torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.bool)
        batch["video_views"] = batch["video"][:, None].expand(-1, 3, -1, -1, -1, -1).clone()

    recorded = []

    def record_streams(module, args):
        recorded.append({stream.name: stream.values.clone() for stream in args[0]})

    policy.generation.register_forward_pre_hook(record_streams)
    prediction = policy.generate_validation(batch, seed=47)
    assert len(recorded) == policy.config.inference_steps
    assert all(set(streams) == {"video", "state", "action"} for streams in recorded)
    assert prediction["action_normalized"].shape == (2, 4, 4)
    assert prediction["state_normalized"].shape == (2, 4, 4)
    assert prediction["video_latents"].shape == (2, 4, 3, 2, 2)
    torch.testing.assert_close(
        prediction["state_normalized"][:, 0], batch["observation.state"][:, 0] * batch["state_mask"]
    )
    assert not prediction["state_normalized"][0, :, 2].any()
    assert not prediction["action_normalized"][0, :, 1].any()
    assert prediction["tactile_decoded_z"].numel() == 0
    assert not prediction["tactile_mask"].any()
    assert all(encoded.shape[2] == 1 for encoded in policy.video_vae.inputs)
    torch.testing.assert_close(prediction["video_latents"][:, :, 0], -torch.ones(2, 4, 2, 2))
    if multiview:
        assert prediction["video_latents_views"].shape == (2, 3, 4, 3, 2, 2)
        assert not prediction["video_latents_views"][~batch["image_view_mask"]].any()
    if flow_space == "embedding":
        assert len(policy.physical_encoder.input_history) == 1
        encoded = policy.physical_encoder.input_history[0]
        assert not encoded["action"].any()
        assert not encoded["action_mask"].any()
        torch.testing.assert_close(
            encoded["observation.state"],
            batch["observation.state"][:, :1].expand(-1, 4, -1) * batch["state_mask"][:, None],
        )
        for streams in recorded[1:]:
            torch.testing.assert_close(streams["state"][:, -1], recorded[0]["state"][:, -1])

    changed = copy.deepcopy(batch)
    changed["observation.state"][:, 1:] = -999
    changed["action"].fill_(999)
    changed["video"][:, 1:] = 255
    if multiview:
        changed["video_views"][:, :, 1:] = 255
    repeated = policy.generate_validation(changed, seed=47)
    for name in prediction:
        torch.testing.assert_close(repeated[name], prediction[name])

    policy.config.time_sampling_distribution = "shifted_uniform"
    policy.config.time_sampling_shift = 3.0
    shifted = policy.generate_validation(batch, seed=47)
    for name in prediction:
        torch.testing.assert_close(shifted[name], prediction[name], rtol=0, atol=0)
    current_only = {
        name: batch[name] for name in ("image_t0", "state_mask", "action_mask", "task")
    }
    current_only["observation.state"] = batch["observation.state"][:, 0]
    if multiview:
        current_only.update({name: batch[name] for name in ("images_t0", "image_view_mask")})
    minimal = policy.generate_validation(current_only, seed=47)
    for name in prediction:
        torch.testing.assert_close(minimal[name], prediction[name])
    torch.manual_seed(47)
    torch.testing.assert_close(policy.sample_canonical_action(batch), prediction["action_normalized"])
    torch.manual_seed(47)
    ambient = policy.generate_validation(batch, seed=None)
    ambient_rng_state = torch.get_rng_state()
    torch.testing.assert_close(ambient["action_normalized"], prediction["action_normalized"])
    torch.manual_seed(47)
    torch.testing.assert_close(policy.sample_canonical_action(batch), ambient["action_normalized"])
    torch.testing.assert_close(torch.get_rng_state(), ambient_rng_state)
    with pytest.raises(ValueError, match="valid current state"):
        policy.generate_validation({**batch, "state_mask": torch.zeros_like(batch["state_mask"])}, seed=47)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_embedding_physical_path_with_real_stage1_transformer(dtype):
    from lerobot.common.policies.ace.modeling_robo_contrast import PhysicalEncoder

    policy, batch = _embedding_physical_policy()
    stage1_config = RoboContrastConfig(
        chunk_size=4, group_size=2, n_action_steps=4, hidden_dim=8, projection_dim=8,
        num_attention_heads=2, num_physical_layers=1, max_action_dim=4, max_state_dim=4,
        max_tactile_signal_dim=3, max_tactile_views=1, tactile_tokens_per_pad=1,
        tactile_frames=2, tactile_img_size=32, tactile_feat_dim=8, tactile_pretrained=False,
    )
    policy.physical_encoder = PhysicalEncoder(stage1_config).to(dtype).eval().requires_grad_(False)
    batch.update({
        "tactile_signal": torch.zeros(2, 4, 3),
        "tactile_image": torch.zeros(2, 1, 2, 3, 32, 32, dtype=torch.uint8),
        "tactile_image_mask": torch.zeros(2, 1),
        "sample_rate": torch.full((2,), 10, dtype=torch.long),
    })
    valid = torch.ones(2, dtype=torch.bool)
    before, _, _, _ = policy._clean_physical_embedding(batch, "state", valid, current_only=True)
    changed = {**batch, "action": torch.full_like(batch["action"], 100),
               "observation.state": batch["observation.state"].clone()}
    changed["observation.state"][:, 1:] = -100
    after, _, _, _ = policy._clean_physical_embedding(changed, "state", valid, current_only=True)
    torch.testing.assert_close(before, after)
    assert not before.requires_grad
    streams, targets, sigmas, reconstructions = policy._prepare_physical_streams(
        TASK_SPECS["joint_prediction"], batch, valid
    )
    assert streams[0].values.shape == (2, 3, 8)
    assert targets["state"][1].sum() == 32
    raw, mask = reconstructions["state"]
    velocity = torch.zeros_like(streams[0].values[:, :2], requires_grad=True)
    predicted_clean = streams[0].values[:, :2] - sigmas["state"][:, None, None] * velocity
    mask = mask.clone()
    mask[:, 0, :4] = False
    loss, _, _ = policy._distributed_mse(policy._decode_physical_embedding("state", predicted_clean), raw, mask)
    loss.backward()
    assert velocity.grad.abs().sum() > 0
    assert policy.physical_decoders["state"][-1].weight.grad is not None
    assert all(parameter.grad is None for parameter in policy.physical_encoder.parameters())


@pytest.mark.parametrize("kwargs", [
    {"physical_flow_space": "unknown"},
    {"physical_flow_space": "embedding", "physical_tuning_mode": "full"},
])
def test_embedding_flow_rejects_invalid_codec_contracts(kwargs):
    with pytest.raises(ValueError):
        Qwen3VLMoTConfig(**kwargs)


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("name", ["action", "state"])
def test_predicted_action_reconstruction_rejects_invalid_weights(weight, name):
    with pytest.raises(ValueError, match="finite and non-negative"):
        Qwen3VLMoTConfig(**{f"{name}_prediction_reconstruction_loss_weight": weight})


@pytest.mark.parametrize("name", ["action", "state"])
def test_predicted_action_reconstruction_requires_embedding_at_model_construction(name):
    config = Qwen3VLMoTConfig(**{f"{name}_prediction_reconstruction_loss_weight": 0.1})
    with pytest.raises(ValueError, match="requires embedding-space"):
        config.validate_features()
    Qwen3VLMoTConfig(
        physical_flow_space="embedding", action_prediction_reconstruction_loss_weight=0.1
    ).validate_features()


def test_legacy_clean_codec_weight_is_retired_with_explicit_warning(caplog):
    config = Qwen3VLMoTConfig(physical_flow_space="embedding", physical_reconstruction_loss_weight=1.0)
    assert config.physical_reconstruction_loss_weight is None
    assert config.action_prediction_reconstruction_loss_weight == 0.1
    assert config.state_prediction_reconstruction_loss_weight == 0.1
    assert "Clean physical codec reconstruction has been removed" in caplog.text


@pytest.mark.parametrize("distribution,mean,low_probability,high_probability", [
    ("uniform", 0.5, 0.1, 0.2),
    ("beta", 0.6, 0.1 ** 1.5, 1 - 0.8 ** 1.5),
    ("mixed", 0.55, 0.5 * (0.1 + 0.1 ** 1.5), 0.5 * (0.2 + 1 - 0.8 ** 1.5)),
    ("shifted_uniform", 0.6760407835, 0.1 / 2.8, 3 / 7),
])
@pytest.mark.parametrize("device", [
    "cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")),
])
def test_time_sampling_distribution_statistics_and_bounds(
    distribution, mean, low_probability, high_probability, device
):
    policy, _ = _embedding_physical_policy()
    policy.config.time_sampling_distribution = distribution
    policy.config.time_sampling_shift = 3.0
    torch.manual_seed(204)
    with torch.autocast(device, dtype=torch.bfloat16):
        sigma = policy._sample_sigma(50000, torch.device(device))
    assert sigma.shape == (50000,)
    assert sigma.device.type == device
    assert sigma.dtype == torch.float32
    assert torch.isfinite(sigma).all()
    assert sigma.min() >= policy.config.sigma_min
    assert sigma.max() <= policy.config.sigma_max
    u = (sigma - policy.config.sigma_min) / (policy.config.sigma_max - policy.config.sigma_min)
    assert u.mean().item() == pytest.approx(mean, abs=0.005)
    assert (u < 0.1).float().mean().item() == pytest.approx(low_probability, abs=0.004)
    assert (u > 0.8).float().mean().item() == pytest.approx(high_probability, abs=0.006)


@pytest.mark.parametrize("shift", [0.25, 1.0, 3.0, 10.0])
def test_shifted_time_sampling_matches_exact_mapping_and_custom_bounds(monkeypatch, shift):
    policy, _ = _embedding_physical_policy()
    policy.config.time_sampling_distribution = "shifted_uniform"
    policy.config.time_sampling_shift = shift
    policy.config.sigma_min, policy.config.sigma_max = 0.15, 0.8
    uniform = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0])

    def fixed_uniform(size, *, device, dtype):
        assert size == uniform.numel()
        return uniform.to(device=device, dtype=dtype)

    monkeypatch.setattr(torch, "rand", fixed_uniform)
    actual = policy._sample_sigma(uniform.numel(), torch.device("cpu"))
    shifted = shift * uniform / (1 + (shift - 1) * uniform)
    expected = 0.15 + 0.65 * shifted
    torch.testing.assert_close(actual, expected)
    assert actual[0].item() == pytest.approx(0.15)
    assert actual[-1].item() == pytest.approx(0.8)


def test_shifted_time_sampling_is_monotone_and_identity_at_one():
    policy, _ = _embedding_physical_policy()
    torch.manual_seed(73)
    uniform = policy._sample_sigma(1000, torch.device("cpu"))
    policy.config.time_sampling_distribution = "shifted_uniform"
    previous = None
    for shift in (0.25, 1.0, 3.0, 10.0):
        policy.config.time_sampling_shift = shift
        torch.manual_seed(73)
        actual = policy._sample_sigma(1000, torch.device("cpu"))
        if shift == 1.0:
            torch.testing.assert_close(actual, uniform)
        if previous is not None:
            assert (actual >= previous).all()
        previous = actual


@pytest.mark.parametrize("shift", [1e-300, 1e300])
def test_shifted_time_sampling_remains_finite_for_extreme_positive_shifts(shift):
    policy, _ = _embedding_physical_policy()
    policy.config.time_sampling_distribution = "shifted_uniform"
    policy.config.time_sampling_shift = shift
    sigma = policy._sample_sigma(1000, torch.device("cpu"))
    assert torch.isfinite(sigma).all()
    assert sigma.min() >= policy.config.sigma_min
    assert sigma.max() <= policy.config.sigma_max


@pytest.mark.parametrize("distribution", ["uniform", "beta", "mixed"])
def test_time_sampling_shift_does_not_change_existing_distributions(distribution):
    policy, _ = _embedding_physical_policy()
    policy.config.time_sampling_distribution = distribution
    samples = []
    for shift in (0.5, 3.0):
        policy.config.time_sampling_shift = shift
        torch.manual_seed(73)
        samples.append(policy._sample_sigma(1000, torch.device("cpu")))
    torch.testing.assert_close(*samples, rtol=0, atol=0)


@pytest.mark.parametrize("probability,reference", [(0.0, "uniform"), (1.0, "beta")])
def test_mixed_time_sampling_endpoints_match_pure_distribution(probability, reference):
    policy, _ = _embedding_physical_policy()
    policy.config.time_sampling_distribution = "mixed"
    policy.config.time_sampling_beta_probability = probability
    torch.manual_seed(13)
    mixed = policy._sample_sigma(100, torch.device("cpu"))
    policy.config.time_sampling_distribution = reference
    torch.manual_seed(13)
    torch.testing.assert_close(mixed, policy._sample_sigma(100, torch.device("cpu")))


@pytest.mark.parametrize("kwargs", [
    {"time_sampling_distribution": "invalid"}, {"time_sampling_beta_alpha": 0},
    {"time_sampling_beta_alpha": float("nan")}, {"time_sampling_beta_beta": -1},
    {"time_sampling_beta_beta": float("inf")}, {"time_sampling_beta_probability": -0.1},
    {"time_sampling_beta_probability": 1.1}, {"time_sampling_beta_probability": float("nan")},
    {"time_sampling_shift": 0}, {"time_sampling_shift": -1},
    {"time_sampling_shift": float("nan")}, {"time_sampling_shift": float("inf")},
    {"time_sampling_shift": float("-inf")},
])
def test_time_sampling_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError, match="time_sampling"):
        Qwen3VLMoTConfig(**kwargs)


def test_physical_decoder_outputs_eight_groups_of_160_then_32_by_40():
    policy, _ = _embedding_physical_policy()
    policy.config = Qwen3VLMoTConfig(physical_flow_space="embedding")
    for name in ("state", "action"):
        policy.physical_decoders[name] = nn.Sequential(
            nn.LayerNorm(1024), nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, 160)
        )
        decoded = policy._decode_physical_embedding(name, torch.randn(2, 8, 1024))
        assert decoded.shape == (2, 8, 160)
        assert decoded.reshape(2, 32, 40).shape == (2, 32, 40)
    assert sum(p.numel() for p in policy.physical_decoders.parameters()) == 2_431_296
