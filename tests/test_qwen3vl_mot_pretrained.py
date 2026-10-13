import copy
import json

import pytest
import torch
from diffusers.models.embeddings import Timesteps
from diffusers.models.normalization import RMSNorm
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F

from lerobot.common.policies.qwen3vl_mot.configuration_qwen3vl_mot import Qwen3VLMoTConfig
from lerobot.common.policies.qwen3vl_mot.cosmos3_pretrained import (
    COSMOS_GENERATION_CONFIGS,
    CosmosRMSNorm,
    CosmosRotaryEmbedding,
    generation_checkpoint_mapping,
    inspect_generation_checkpoint,
    load_generation_checkpoint,
    rotate_half,
    validate_generation_vae,
)
from lerobot.common.policies.qwen3vl_mot.modeling_generation import (
    GenerationBlock,
    GenerationExpert,
    GenerationStream,
    TimestepEmbedding,
)
from lerobot.common.policies.qwen3vl_mot.modeling_qwen3vl_mot import Qwen3VLMoTPolicy


def _tiny_native(kind="edge", *, checkpointing=False, dtype=torch.float32):
    native = {
        **COSMOS_GENERATION_CONFIGS[kind],
        "hidden_size": 16, "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 4, "intermediate_size": 32,
        "rope_axes_dim": [1, 1, 0], "latent_channel": 1, "patch_latent_dim": 4,
    }
    model = GenerationExpert(
        understanding_dim=12, hidden_dim=16, depth=2, num_heads=4, num_kv_heads=2,
        intermediate_dim=32, hidden_act=native["hidden_act"], dropout=0,
        input_dims={"video": 4, "state": 8, "action": 8, "tactile": 8},
        output_dims={"video": 4, "state": 8, "action": 8, "tactile": 8},
        task_names=("i2v", "joint_prediction"), gradient_checkpointing=checkpointing,
        cosmos_config=native, num_video_views=3,
    ).to(dtype=dtype)
    model.initialize_cosmos_adapters()
    with torch.no_grad():
        model.output_heads["video"].weight.normal_(std=0.1)
    return model


def _write_checkpoint(path, model, *, sharded=True):
    root = path / "transformer"
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps(model.cosmos_config))
    parameters = dict(model.named_parameters())
    state = {
        source: parameters[target].detach().contiguous().clone()
        for target, source in generation_checkpoint_mapping(model).items()
    }
    state["embed_tokens.weight"] = torch.zeros(3, 16)
    state["action_proj_in.fc.weight"] = torch.zeros(2, 16)
    if not sharded:
        save_file(state, root / "diffusion_pytorch_model.safetensors")
        return state
    filenames = ("part1.safetensors", "part2.safetensors")
    weight_map = {name: filenames[index % 2] for index, name in enumerate(state)}
    for filename in filenames:
        save_file({name: value for name, value in state.items() if weight_map[name] == filename},
                  root / filename)
    (root / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    return state


@pytest.mark.parametrize("kind", ["edge", "nano"])
@pytest.mark.parametrize("sharded", [False, True])
def test_pretrained_load_covers_all_native_weights_and_nonzero_video_head(tmp_path, kind, sharded):
    expected = _tiny_native(kind)
    state = _write_checkpoint(tmp_path, expected, sharded=sharded)
    actual = _tiny_native(kind)
    original_adapter = actual.blocks[0].attention.key_adapter.weight.clone()
    report = load_generation_checkpoint(actual, tmp_path)
    parameters = dict(actual.named_parameters())
    mapping = generation_checkpoint_mapping(actual)
    for target, source in mapping.items():
        torch.testing.assert_close(parameters[target], state[source])
    assert report.copied_tensors == len(mapping)
    assert report.copied_parameters == sum(parameters[name].numel() for name in mapping)
    assert report.ignored_tensors == 2
    assert actual.output_heads["video"].weight.count_nonzero() > 0
    torch.testing.assert_close(actual.blocks[0].attention.key_adapter.weight, original_adapter)


@pytest.mark.parametrize("failure", ["missing", "extra", "shape", "config", "shard"])
def test_pretrained_loader_rejects_incomplete_or_incompatible_checkpoints(tmp_path, failure):
    model = _tiny_native()
    _write_checkpoint(tmp_path, model)
    root = tmp_path / "transformer"
    index_path = root / "diffusion_pytorch_model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    key = "proj_out.weight"
    if failure == "missing":
        del index["weight_map"][key]
    elif failure == "extra":
        index["weight_map"]["layers.0.mlp_moe_gen.unmapped.weight"] = "part1.safetensors"
    elif failure == "shape":
        filename = index["weight_map"][key]
        from safetensors.torch import load_file

        tensors = load_file(root / filename)
        tensors[key] = torch.zeros(1, 1)
        save_file(tensors, root / filename)
    elif failure == "config":
        native = {**model.cosmos_config, "rope_theta": 42}
        (root / "config.json").write_text(json.dumps(native))
    else:
        index["weight_map"][key] = "missing.safetensors"
    index_path.write_text(json.dumps(index))
    with pytest.raises((RuntimeError, ValueError, FileNotFoundError)):
        inspect_generation_checkpoint(model, tmp_path)


@pytest.mark.parametrize("kind", ["edge", "nano"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_rmsnorm_matches_reference_math(kind, dtype):
    module = CosmosRMSNorm(16, 1e-5, nemotron=kind == "edge").to(dtype=dtype)
    values = torch.randn(2, 3, 16).to(dtype)
    with torch.no_grad():
        module.weight.uniform_(0.5, 1.5)
    if kind == "nano":
        reference = RMSNorm(16, eps=1e-5).to(dtype=dtype)
        reference.load_state_dict(module.state_dict())
        expected = reference(values)
    else:
        expected = (values.float() * torch.rsqrt(values.float().square().mean(-1, keepdim=True) + 1e-5)
                    * module.weight.float()).to(dtype)
    torch.testing.assert_close(module(values), expected, rtol=0, atol=0)


def test_native_timestep_uses_sigma_not_sigma_times_1000():
    module = TimestepEmbedding(16, native_cosmos=True)
    sigma = torch.tensor([0.0, 0.25, 0.9])
    projection = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
    expected = module.mlp(projection(sigma))
    torch.testing.assert_close(module(sigma), expected)
    assert not torch.allclose(module(sigma), module.mlp(projection(sigma * 1000)))


def test_native_rope_matches_interleaved_reference_at_large_fractional_positions():
    module = CosmosRotaryEmbedding(128, 5_000_000, [24, 20, 20])
    positions = torch.randn(3, 2, 5) * 3 + 15000
    values = torch.zeros(2, 5, 16)
    inverse = 1 / (5_000_000 ** (torch.arange(0, 128, 2).float() / 128))
    frequencies = positions.unsqueeze(-1) * inverse
    combined = frequencies[0].clone()
    combined[..., 1:60:3] = frequencies[1, ..., 1:60:3]
    combined[..., 2:60:3] = frequencies[2, ..., 2:60:3]
    phases = torch.cat([combined, combined], dim=-1)
    cos, sin = module(values, positions)
    torch.testing.assert_close(cos, phases.cos())
    torch.testing.assert_close(sin, phases.sin())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cos_autocast, sin_autocast = module(values, positions)
    torch.testing.assert_close(cos_autocast, cos, rtol=0, atol=0)
    torch.testing.assert_close(sin_autocast, sin, rtol=0, atol=0)
    module.to(dtype=torch.bfloat16)
    assert module._frequencies().dtype == torch.float32
    assert not tuple(module.buffers())
    cos_after_cast, sin_after_cast = module(values, positions)
    torch.testing.assert_close(cos_after_cast, cos, rtol=0, atol=0)
    torch.testing.assert_close(sin_after_cast, sin, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["edge", "nano"])
def test_native_generation_block_matches_full_reference_generation_path(kind):
    model = _tiny_native(kind).eval()
    block = model.blocks[0]
    generation = torch.randn(2, 3, 16)
    key, value = torch.randn(2, 2, 4, 4), torch.randn(2, 2, 4, 4)
    keep_g = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    keep_u = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 1]], dtype=torch.bool)
    cos_g, sin_g = model.rotary_embedding(generation, torch.arange(3).view(1, 1, -1).expand(3, 2, -1))
    cos_u, sin_u = model.rotary_embedding(generation, torch.arange(4).view(1, 1, -1).expand(3, 2, -1))
    actual = block(generation, None, keep_u, keep_g, key, value, cos_g, sin_g, cos_u, sin_u)
    attention = block.attention
    normalized = block.generation_norm(generation)
    q = attention.q_norm(attention._heads(attention.q_proj(normalized), 4))
    kg = attention.k_norm(attention._heads(attention.k_proj(normalized), 2))
    vg = attention._heads(attention.v_proj(normalized), 2)
    q = q * cos_g[:, None] + rotate_half(q) * sin_g[:, None]
    kg = kg * cos_g[:, None] + rotate_half(kg) * sin_g[:, None]
    ku = attention.understanding_k_norm(key)
    ku = ku * cos_u[:, None] + rotate_half(ku) * sin_u[:, None]
    all_k, all_v = torch.cat([ku, kg], 2), torch.cat([value, vg], 2)
    mask = torch.cat([keep_u, keep_g], 1)[:, None, None]
    bias = torch.zeros_like(mask, dtype=q.dtype).masked_fill(~mask, torch.finfo(q.dtype).min)
    out = F.scaled_dot_product_attention(
        q, all_k.repeat_interleave(2, 1), all_v.repeat_interleave(2, 1), attn_mask=bias
    ).transpose(1, 2).reshape_as(generation)
    residual = generation + attention.out_proj(out) * keep_g[..., None]
    expected = (residual + block.mlp(block.mlp_norm(residual))) * keep_g[..., None]
    torch.testing.assert_close(actual, expected)


def test_native_time_is_not_added_to_clean_i2v_frames_or_physical_conditions():
    model = _tiny_native()
    stream = GenerationStream(
        "video", torch.randn(2, 4, 4), torch.ones(2), torch.ones(2, 4, dtype=torch.bool),
        noise_mask=torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.bool),
        fps=torch.tensor([8.0, 12.0]),
        position_ids=torch.tensor([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1]]),
    )
    hidden, positions = model._embed_stream(stream, torch.zeros(2, 16))
    projected = model.input_projections["video"](stream.values)
    torch.testing.assert_close(hidden[:, :2], projected[:, :2])
    torch.testing.assert_close(hidden[:, 2:], projected[:, 2:] + model.timestep_embedding(stream.sigma)[:, None])
    torch.testing.assert_close(positions[:, 2:, 0], torch.tensor([[3.0, 3.0], [2.0, 2.0]]))


@pytest.mark.parametrize("kind", ["edge", "nano"])
def test_native_condition_adapter_checkpointing_gradients_and_qwen_derotation(kind):
    plain = _tiny_native(kind).train()
    checkpointed = copy.deepcopy(plain)
    checkpointed.gradient_checkpointing = True
    positions = torch.arange(5).view(1, 1, -1).expand(3, 2, -1)
    phase = torch.rand(2, 5, 2)
    cos, sin = torch.cat([phase.cos()] * 2, -1), torch.cat([phase.sin()] * 2, -1)
    keep = torch.ones(2, 5, dtype=torch.bool)
    key = torch.randn(2, 2, 5, 4, requires_grad=True)
    value = torch.randn(2, 2, 5, 4, requires_grad=True)
    values = torch.randn(2, 4, 4, requires_grad=True)
    stream = GenerationStream("video", values, torch.rand(2), torch.ones(2, 4, dtype=torch.bool))
    rotated = key * cos[:, None] + rotate_half(key) * sin[:, None]
    expected = plain([stream], [], keep, "i2v", [(key, value)], understanding_position_ids=positions)["video"]
    actual = plain(
        [stream], [], keep, "i2v", [(rotated, value)],
        understanding_position_ids=positions, understanding_rotary=(cos, sin),
    )["video"]
    torch.testing.assert_close(actual, expected)
    other = checkpointed(
        [stream], [], keep, "i2v", [(rotated, value)],
        understanding_position_ids=positions, understanding_rotary=(cos, sin),
    )["video"]
    torch.testing.assert_close(other, actual)
    actual.square().mean().backward(retain_graph=True)
    other.square().mean().backward()
    assert key.grad is not None and value.grad is not None and values.grad is not None
    for (_, parameter), (_, other_parameter) in zip(
        plain.named_parameters(), checkpointed.named_parameters(), strict=True
    ):
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other_parameter.grad)
    assert plain.blocks[0].attention.key_adapter.weight.grad.abs().sum() > 0


def test_native_freezing_leaves_conditions_and_new_heads_trainable():
    model = _tiny_native()
    model.configure_cosmos_tuning("frozen")
    mapping = generation_checkpoint_mapping(model)
    assert all(not parameter.requires_grad for name, parameter in model.named_parameters() if name in mapping)
    assert model.blocks[0].attention.key_adapter.weight.requires_grad
    assert model.output_heads["action"].weight.requires_grad
    assert model.video_view_embedding.weight.requires_grad
    model.configure_cosmos_tuning("full")
    assert all(parameter.requires_grad for parameter in model.parameters())


@pytest.mark.parametrize("kind,hidden,depth,heads", [("edge", 2048, 28, 16), ("nano", 4096, 36, 32)])
def test_native_policy_presets_keep_qwen_unchanged(kind, hidden, depth, heads):
    config = Qwen3VLMoTConfig(generation_backbone=kind)
    assert config.qwen3vl_dir.endswith("/Qwen3-VL-4B-Instruct")
    assert (config.generation_hidden_dim, config.generation_depth, config.generation_num_heads) == (hidden, depth, heads)
    assert config.generation_num_kv_heads == 8


def test_native_vae_compatibility_checks_statistics(tmp_path):
    edge, nano = tmp_path / "edge", tmp_path / "nano"
    for root in (edge, nano):
        (root / "vae").mkdir(parents=True)
        (root / "vae/config.json").write_text(json.dumps({"z_dim": 48, "latents_mean": [0.0]}))
    validate_generation_vae(edge, nano)
    (nano / "vae/config.json").write_text(json.dumps({"z_dim": 48, "latents_mean": [1.0]}))
    with pytest.raises(ValueError, match="latents_mean"):
        validate_generation_vae(edge, nano)


def test_native_video_fps_matches_sparse_window_wall_clock():
    policy = Qwen3VLMoTPolicy.__new__(Qwen3VLMoTPolicy)
    nn.Module.__init__(policy)
    policy.config = Qwen3VLMoTConfig(generation_backbone="edge")
    fps = policy._generation_video_fps({"sample_rate": torch.tensor([10, 30])})
    torch.testing.assert_close(fps, torch.tensor([80 / 31, 240 / 31]))
    with pytest.raises(ValueError, match="positive"):
        policy._generation_video_fps({"sample_rate": torch.tensor([0])})


@pytest.mark.parametrize("kind,expected", [("edge", 1_486_261_952), ("nano", 7_066_219_200)])
def test_native_default_parameter_counts(kind, expected):
    config = Qwen3VLMoTConfig(generation_backbone=kind, physical_flow_space="embedding")
    native = COSMOS_GENERATION_CONFIGS[kind]
    with torch.device("meta"):
        model = GenerationExpert(
            understanding_dim=2560, hidden_dim=config.generation_hidden_dim,
            depth=config.generation_depth, num_heads=config.generation_num_heads,
            num_kv_heads=config.generation_num_kv_heads, intermediate_dim=config.generation_intermediate_dim,
            hidden_act=config.generation_hidden_act, dropout=0,
            input_dims={"video": 192, "state": 1024, "action": 1024, "tactile": 1024},
            output_dims={"video": 192, "state": 1024, "action": 1024, "tactile": 1024},
            task_names=config.task_names, gradient_checkpointing=True, cosmos_config=native,
        )
    assert sum(parameter.numel() for parameter in model.parameters()) == expected


@pytest.mark.parametrize("kind", ["edge", "nano"])
def test_native_bf16_forward_and_backward_are_finite(kind):
    model = _tiny_native(kind, checkpointing=True, dtype=torch.bfloat16).train()
    stream = GenerationStream(
        "video", torch.randn(2, 3, 4), torch.rand(2), torch.ones(2, 3, dtype=torch.bool)
    )
    key = torch.randn(2, 2, 4, 4, requires_grad=True)
    value = torch.randn(2, 2, 4, 4, requires_grad=True)
    output = model([stream], [], torch.ones(2, 4, dtype=torch.bool), "i2v", [(key, value)])["video"]
    output.float().square().mean().backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(key.grad).all()
    assert torch.isfinite(model.blocks[0].attention.key_adapter.weight.grad).all()


def test_native_timestep_stays_fp32_under_autocast_and_fp8_filter():
    from lerobot.common.utils.fsdp_training import FSDPTrainingConfig, _is_fp8_linear

    module = TimestepEmbedding(128, native_cosmos=True).to(dtype=torch.bfloat16)
    sigma = torch.tensor([0.25, 0.9])
    reference = copy.deepcopy(module).float()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = module(sigma)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, reference(sigma), rtol=0, atol=0)
    assert not _is_fp8_linear(module.mlp[0], "generation.timestep_embedding.mlp.0", FSDPTrainingConfig())


def test_legacy_block_preserves_optimizer_parameter_order():
    block = GenerationBlock(16, 4, 2, 32, "relu2", 0)
    assert list(dict(block.named_parameters())) == [
        "generation_norm.weight", "generation_norm.bias",
        "understanding_norm.weight", "understanding_norm.bias",
        "attention.q_proj.weight", "attention.k_proj.weight", "attention.v_proj.weight",
        "attention.out_proj.weight", "mlp_norm.weight", "mlp_norm.bias",
        "mlp.up.weight", "mlp.down.weight",
    ]


@pytest.mark.parametrize("kind", ["edge", "nano"])
@pytest.mark.parametrize("mode", ["full", "frozen"])
def test_native_fp8_emulation_supports_trainable_conditioning(kind, mode):
    from lerobot.common.utils.fsdp_training import FSDPTrainingConfig, convert_policy_to_fp8

    native = {
        **COSMOS_GENERATION_CONFIGS[kind],
        "hidden_size": 128, "num_hidden_layers": 1, "num_attention_heads": 1,
        "num_key_value_heads": 1, "intermediate_size": 256,
    }
    policy = nn.Module()
    policy.generation = GenerationExpert(
        understanding_dim=128, hidden_dim=128, depth=1, num_heads=1, num_kv_heads=1,
        intermediate_dim=256, hidden_act=native["hidden_act"], dropout=0,
        input_dims={"video": 192}, output_dims={"video": 192}, task_names=("i2v",),
        gradient_checkpointing=True, cosmos_config=native,
    ).bfloat16()
    policy.generation.initialize_cosmos_adapters()
    with torch.no_grad():
        policy.generation.output_heads["video"].weight.normal_(std=0.02)
    policy.generation.configure_cosmos_tuning(mode)
    converted = convert_policy_to_fp8(
        policy, FSDPTrainingConfig(fp8_emulate=True, fp8_scope="generation"), torch.device("cpu")
    )
    assert converted and not any("timestep_embedding" in name for name in converted)
    stream = GenerationStream(
        "video", torch.randn(2, 3, 192), torch.rand(2), torch.ones(2, 3, dtype=torch.bool)
    )
    output = policy.generation(
        [stream], [], torch.ones(2, 4, dtype=torch.bool), "i2v",
        [(torch.randn(2, 1, 4, 128), torch.randn(2, 1, 4, 128))],
    )["video"]
    output.float().square().mean().backward()
    assert torch.isfinite(output).all()
    gradient = policy.generation.blocks[0].attention.key_adapter.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
