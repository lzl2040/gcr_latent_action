"""Smoke-test the Phi-4-Multimodal MoT world model.

Checks that matter here:
  * every trainable parameter actually receives a gradient (a silently disconnected gen
    expert would still train to a plausible-looking loss on the video head alone);
  * the frozen und expert receives none;
  * patchify/unpatchify round-trip exactly, since a wrong latent layout is invisible in the
    loss but destroys spatial structure;
  * peak memory and step time at the batch size we intend to train at.

Run:  CUDA_VISIBLE_DEVICES=2 python -u scripts/smoke_mot_world.py --batch 1
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


def build(args) -> MoTWorldModel:
    overrides = {
        k: v
        for k, v in (
            ("gen_hidden_size", args.gen_hidden),
            ("gen_num_attention_heads", args.gen_heads),
            ("gen_intermediate_size", args.gen_intermediate),
        )
        if v is not None
    }
    mot = MoTConfig.from_phi_dir(args.phi_dir, **overrides)
    microbatch = args.microbatch if args.microbatch is not None else (
        32 if args.scope == "gen_only" else 16
    )
    cfg = WorldModelConfig(
        mot=mot,
        trainable_scope=args.scope,
        freeze_vision_projector=args.freeze_projector,
        training_execution=args.execution,
        und_microbatch_size=microbatch,
        mot_checkpoint_segment_size=args.checkpoint_segment,
    )
    return MoTWorldModel(cfg)


def check_patchify(model: MoTWorldModel, device, dtype) -> bool:
    c = model.config.mot.latent_channels
    t, g = 2, model.config.latent_grid
    x = torch.randn(2, c, t, g, g, device=device, dtype=dtype)
    back = model.unpatchify(model.patchify(x), t)
    err = (back - x).abs().max().item()
    ok = err == 0.0
    print(f"[check] patchify round-trip max|diff| {err:.3e}   {'OK' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi_dir", default="/Data/lzl/huggingface/Phi-4-multimodal-instruct")
    # Default to whatever MoTConfig says, so this script cannot silently keep testing an
    # old gen size after the config moves.
    ap.add_argument("--gen_hidden", type=int, default=None)
    ap.add_argument("--gen_heads", type=int, default=None)
    ap.add_argument("--gen_intermediate", type=int, default=None)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--latent_frames", type=int, default=2)
    ap.add_argument("--text_len", type=int, default=32)
    ap.add_argument("--action_len", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--random_init", action="store_true")
    ap.add_argument("--gen_checkpointing", action="store_true")
    ap.add_argument("--freeze_projector", action="store_true")
    ap.add_argument("--scope", default="gen_only")
    ap.add_argument("--execution", choices=("interleaved", "cached"), default="interleaved")
    ap.add_argument("--microbatch", type=int, default=None)
    ap.add_argument("--checkpoint_segment", type=int, default=4)
    ap.add_argument("--skip_opt", action="store_true")
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    print("[build] loading ...")
    model = build(args).to(device=device, dtype=dtype)
    if not args.random_init:
        model.load_pretrained()
    model.mot.gradient_checkpointing = args.gen_checkpointing
    model.train()

    rep = model.param_report()
    print(
        f"[params] vision {rep['vision_frozen'] / 1e6:.1f}M | "
        f"projector {rep['vision_projector'] / 1e6:.1f}M | "
        f"und(frozen) {rep['und_frozen'] / 1e9:.3f}B | gen {rep['gen_trainable'] / 1e6:.1f}M"
    )
    print(
        f"[params] total {rep['total'] / 1e9:.3f}B, trainable {rep['trainable'] / 1e9:.3f}B "
        f"({100 * rep['trainable'] / rep['total']:.1f}%)"
    )

    ok = check_patchify(model, device, dtype)

    b, cfg = args.batch, model.config
    latents = torch.randn(b, cfg.mot.latent_channels, args.latent_frames, cfg.latent_grid, cfg.latent_grid,
                          device=device, dtype=dtype)
    images = torch.rand(b, 3, model.vision_image_size, model.vision_image_size, device=device, dtype=dtype)
    text_ids = torch.randint(0, 30000, (b, args.text_len), device=device)
    actions = torch.randn(b, args.action_len, cfg.mot.action_dim, device=device, dtype=dtype)
    domain = torch.randint(0, cfg.mot.num_embodiment_domains, (b,), device=device)

    torch.cuda.reset_peak_memory_stats()
    # "joint_action" is the only task that exercises every head, so the gradient-coverage check
    # below is meaningful; the other four are covered by scripts/check_mot_tasks.py.
    out = model(latents, images, text_ids, actions, domain, task="joint_action")
    print(f"[forward] loss={out['loss'].item():.4f} video={out['loss_video'].item():.4f} "
          f"action={out['loss_action'].item():.4f}")
    out["loss"].backward()

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    missing = [n for n, p in trainable if p.grad is None]
    frozen_with_grad = [n for n, p in model.named_parameters() if not p.requires_grad and p.grad is not None]
    print(f"[grad] trainable tensors {len(trainable)}, without grad {len(missing)}   "
          f"{'OK' if not missing else 'FAIL'}")
    if missing:
        print("       " + ", ".join(missing[:6]))
    print(f"[grad] frozen tensors carrying grad {len(frozen_with_grad)}   "
          f"{'OK' if not frozen_with_grad else 'FAIL'}")
    ok &= not missing and not frozen_with_grad

    gen_named = [n for n, _ in trainable if n.startswith("mot.layers")]
    print(f"[grad] gen-expert tensors receiving grad: {len(gen_named)}")

    model.zero_grad(set_to_none=True)
    opt = (
        None
        if args.skip_opt
        else torch.optim.AdamW([p for _, p in trainable], lr=1e-4)
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        if opt is None:
            model.zero_grad(set_to_none=True)
        else:
            opt.zero_grad(set_to_none=True)
        loss = model(latents, images, text_ids, actions, domain, task="joint_action")["loss"]
        loss.backward()
        if opt is not None:
            opt.step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / args.steps
    peak = torch.cuda.max_memory_allocated() / 2**30
    mode = "fwd+bwd" if opt is None else "optimizer step"
    print(f"[perf] batch {b}: {dt:.3f}s/{mode}, peak {peak:.1f} GiB "
          f"({b / dt:.1f} samples/s)")

    print("\nALL CHECKS PASSED" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
