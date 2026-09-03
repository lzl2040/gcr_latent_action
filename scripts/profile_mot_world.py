"""Stage-level timing for the MoT world model, to find where a step actually goes.

Prints wall time for the vision tower, the und stack, and the gen stack separately, plus a
per-layer breakdown of the gen expert.  Times are forward-only unless stated, so they can be
compared against the FLOP ratio between the two experts.

Run:  CUDA_VISIBLE_DEVICES=2 python -u scripts/profile_mot_world.py --batch 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot.common.policies.mot.modeling_mot import MoTConfig  # noqa: E402
from lerobot.common.policies.mot.world_model import MoTWorldModel, WorldModelConfig  # noqa: E402


def timeit(fn, n=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi_dir", default="/Data/lzl/huggingface/Phi-4-multimodal-instruct")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--latent_frames", type=int, default=2)
    ap.add_argument("--text_len", type=int, default=32)
    ap.add_argument("--action_len", type=int, default=32)
    args = ap.parse_args()

    device, dtype = "cuda", torch.bfloat16
    torch.manual_seed(0)
    mot = MoTConfig.from_phi_dir(args.phi_dir)
    model = MoTWorldModel(WorldModelConfig(mot=mot))
    model = model.to(device=device, dtype=dtype).eval()

    b = args.batch
    images = torch.rand(b, 3, model.vision_image_size, model.vision_image_size, device=device, dtype=dtype)
    text_ids = torch.randint(0, 30000, (b, args.text_len), device=device)

    with torch.no_grad():
        t_vision = timeit(lambda: model.vision(images, encoder_grad=False))
        image_embeds = model.vision(images, encoder_grad=False)
        embeds = torch.cat([image_embeds, model.mot.embed_tokens(text_ids)], dim=1)
        pos = torch.arange(embeds.shape[1], device=device)
        pos = pos.view(1, 1, -1).expand(3, b, -1)
        t_und = timeit(lambda: model.mot.forward_und_kv(embeds, pos, True))

        kv, rope_und = model.mot.forward_und_kv(embeds, pos, True)
        n_gen = args.latent_frames * model.latent_side**2 + args.action_len
        gen = torch.randn(b, n_gen, mot.gen_hidden_size, device=device, dtype=dtype)
        gpos = model.gen_positions(args.latent_frames, args.action_len, device).unsqueeze(1).expand(3, b, -1)
        sigma = torch.rand(b, device=device, dtype=dtype)
        t_gen = timeit(lambda: model.mot.forward_gen(gen, gpos, kv, rope_und, sigma * 1000.0))

        layer = model.mot.layers[0]
        rope_gen = tuple(r.to(dtype) for r in model.mot.rotary_emb(gpos))
        k0, v0 = kv[0]
        t_layer = timeit(lambda: layer.gen_forward(gen, rope_gen, rope_und, k0, v0), n=20, warmup=5)

        def gen_attn_only():
            x = layer.input_layernorm_moe_gen(gen)
            q = layer.norm_added_q(layer.add_q_proj(x).view(b, n_gen, layer.n_gen, 128)).transpose(1, 2)
            k = layer.norm_added_k(layer.add_k_proj(x).view(b, n_gen, layer.n_kv, 128)).transpose(1, 2)
            v = layer.add_v_proj(x).view(b, n_gen, layer.n_kv, 128).transpose(1, 2)
            return q, k, v

        t_qkv = timeit(gen_attn_only, n=20, warmup=5)

        def gen_cat_only():
            kn = layer.k_norm_und_for_gen(k0.transpose(1, 2)).transpose(1, 2)
            return torch.cat([kn, kn[:, :, : n_gen // 2]], dim=2)

        t_cat = timeit(gen_cat_only, n=20, warmup=5)

    # Derive both counts from the model rather than hardcoding them: a stale constant here
    # silently rescales every TFLOP/s figure when the gen expert is resized.
    rep = model.mot.param_report()
    und_matmul = 0
    for index, layer in enumerate(model.mot.layers):
        modules = [layer.self_attn.qkv_proj]
        if index < len(model.mot.layers) - 1:
            modules += [
                layer.self_attn.o_proj,
                layer.mlp.gate_up_proj,
                layer.mlp.down_proj,
            ]
        und_matmul += sum(p.numel() for module in modules for p in module.parameters())
    und_flops = 2 * und_matmul * embeds.shape[1] * b
    gen_flops = 2 * rep["gen"] * n_gen * b
    print(f"[shape] und tokens {embeds.shape[1]}, gen tokens {n_gen}, batch {b}")
    print(f"[param] und matmul {und_matmul / 1e9:.3f}B, gen {rep['gen'] / 1e9:.3f}B")
    print(f"[time]  vision+projector  {t_vision * 1000:8.1f} ms")
    print(f"[time]  und stack (32L)   {t_und * 1000:8.1f} ms   {und_flops / t_und / 1e12:6.1f} TFLOP/s")
    print(f"[time]  gen stack (32L)   {t_gen * 1000:8.1f} ms   {gen_flops / t_gen / 1e12:6.1f} TFLOP/s")
    print(f"[time]  gen one layer     {t_layer * 1000:8.1f} ms  (x32 = {t_layer * 32 * 1000:.1f} ms)")
    print(f"[time]    of which qkv    {t_qkv * 1000:8.1f} ms")
    print(f"[time]    k_und norm+cat  {t_cat * 1000:8.1f} ms")
    print(f"[ratio] und/gen FLOPs {und_flops / gen_flops:.2f}x, und/gen time {t_und / t_gen:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
