import copy
from types import SimpleNamespace

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
from lerobot.common.policies.qwen3vl_mot.modeling_qwen3vl_mot import _load_stage1_config
from lerobot.common.policies.qwen3vl_mot.stage1_transfer import TransferReport, transfer_matching_module
from lerobot.common.policies.qwen3vl_mot.tasks import TASK_SPECS, ModalityRole


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
    assert TASK_SPECS["forward_dynamics"].action is ModalityRole.CLEAN
    assert TASK_SPECS["forward_dynamics"].state is ModalityRole.FUTURE_NOISY
    assert TASK_SPECS["inverse_dynamics"].video is ModalityRole.CLEAN
    assert TASK_SPECS["inverse_dynamics"].action is ModalityRole.NOISY
    assert TASK_SPECS["action_prediction"].state is ModalityRole.CURRENT_ONLY
    assert TASK_SPECS["tactile_prediction"].tactile is ModalityRole.NOISY


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


def test_default_generation_expert_is_between_one_and_two_billion_parameters():
    config = Qwen3VLMoTConfig()
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
                "state": config.group_size * config.max_state_dim,
                "action": config.group_size * config.max_action_dim,
                "tactile": config.physical_hidden_dim,
            },
            task_names=config.task_names,
            gradient_checkpointing=True,
        )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert 1_000_000_000 <= parameters <= 2_000_000_000
    assert parameters == 1_429_469_696


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


def test_tiny_qwen_understanding_emits_native_kv_and_query_gradients():
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
    expert.train()
    vision_parameter = next(expert.model.visual.parameters())
    vision_parameter.requires_grad_(True)
    output = expert(torch.zeros(1, 3, 32, 32, dtype=torch.uint8), ["move"])
    output.latent_queries.square().mean().backward()
    assert expert.query_tokens.grad is not None
    assert vision_parameter.grad is not None

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
        del image_t1, texts, has_text, probe
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

    def forward(self, batch, **kwargs):
        self.include_tactile_history.append(kwargs["include_tactile"])
        self.last_state = batch["observation.state"].detach().clone()
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

    def forward(self, images, texts):
        del images
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


def test_policy_runs_all_task_routes_with_lightweight_backends(tmp_path, monkeypatch):
    import lerobot.common.policies.ace.cosmos3_encoders as cosmos3_encoders
    import lerobot.common.policies.qwen3vl_mot.modeling_qwen3vl_mot as stage2

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
        cosmos3_dir="/mnt/runtime-cosmos",
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
        generation_gradient_checkpointing=False,
    )
    policy = stage2.Qwen3VLMoTPolicy(config)
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
    for task_name in TASK_SPECS:
        policy.physical_encoder.include_tactile_history.clear()
        loss, metrics = policy(batch, task_type=task_name)
        assert torch.isfinite(loss)
        assert metrics[f"task_{task_name}"] == 1.0
        if task_name == "t2v":
            assert metrics["video_target_elements"] == 96
        elif task_name in ("i2v", "forward_dynamics"):
            assert metrics["video_target_elements"] == 64
        if task_name in ("forward_dynamics", "state_prediction"):
            assert metrics["state_target_elements"] == 24
        if task_name in ("inverse_dynamics", "action_prediction"):
            assert metrics["action_target_elements"] == 32
        if task_name == "action_prediction":
            torch.testing.assert_close(
                policy.physical_encoder.last_state,
                batch["observation.state"][:, :1].expand(-1, 4, -1),
            )
        if task_name == "tactile_prediction":
            assert metrics["tactile_target_elements"] == 64
            assert policy.physical_encoder.include_tactile_history == [False, True]
    canonical_action = policy.sample_canonical_action(batch)
    assert canonical_action.shape == (2, 2, 4)
    torch.testing.assert_close(
        policy.physical_encoder.last_state,
        batch["observation.state"][:, :1].expand(-1, 4, -1),
    )

    stage2_dir = tmp_path / "stage2"
    stage2_dir.mkdir()
    policy.config.stage1_policy_config["qwen3vl_dir"] = "/Data/embedded-old-qwen"
    policy.config.stage1_policy_config["cosmos3_dir"] = "/Data/embedded-old-cosmos"
    policy._save_pretrained(stage2_dir)
    (stage1_dir / "model.safetensors").unlink()
    (stage1_dir / "config.json").unlink()
    stage1_dir.rmdir()

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
