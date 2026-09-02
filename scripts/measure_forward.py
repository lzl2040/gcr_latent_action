"""Split the step into forward / backward / optimizer for each trainable scope.

"How long is one forward" has three different answers depending on what is included, and they
differ by more than 3x here, so the question is worth answering with a decomposition rather
than a single number:

* forward under no_grad -- what inference costs, no activations stored;
* forward under grad -- what training's forward costs, which is *more* than the no_grad one
  because activations must be kept and gradient checkpointing re-runs work later;
* forward + backward -- the part that scales with the GPU;
* + optimizer -- the full step, which is what wall-clock projections must use.

Synthetic tensors on purpose: this isolates compute from the loader, which on this machine is
a spinning disk and would otherwise dominate (see scripts/measure_io.py).

Run:  python -u scripts/measure_forward.py --batch 32
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot.common.datasets.canonical_space import CANON_DIM  # noqa: E402
from lerobot.common.policies.mot.modeling_mot import MoTConfig  # noqa: E402
from lerobot.common.policies.mot.world_model import (  # noqa: E402
    TRAINABLE_SCOPES,
    MoTWorldModel,
    WorldModelConfig,
)


def timed(fn, reps: int, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi_dir", default="/Data/lzl/huggingface/Phi-4-mini-instruct")
    ap.add_argument("--qwen3vl_dir", default="/Data/lzl/huggingface/Qwen3-VL-4B-Instruct")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--latent_frames", type=int, default=3)
    ap.add_argument("--text_len", type=int, default=32)
    ap.add_argument("--action_len", type=int, default=32)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--scopes", default="gen_only,freeze_vision")
    ap.add_argument("--task", default="action")
    ap.add_argument("--skip_opt", action="store_true",
                    help="skip the optimizer stage; its state does not fit when the card is shared")
    args = ap.parse_args()

    device, dtype = torch.device("cuda"), torch.bfloat16
    b = args.batch

    print(f"task={args.task}  batch={b}  latent_frames={args.latent_frames}  "
          f"({args.reps} reps after 2 warmups)\n")
    print(f"{'scope':>14s} {'fwd(no_grad)':>13s} {'fwd(grad)':>11s} {'fwd+bwd':>9s} "
          f"{'+opt=step':>10s} {'peak':>9s}")

    for scope in args.scopes.split(","):
        if scope not in TRAINABLE_SCOPES:
            raise SystemExit(f"unknown scope {scope!r}")
        torch.cuda.reset_peak_memory_stats()
        mot = MoTConfig.from_phi_dir(args.phi_dir, action_dim=CANON_DIM)
        cfg = WorldModelConfig(mot=mot, qwen3vl_dir=args.qwen3vl_dir, trainable_scope=scope)
        model = MoTWorldModel(cfg).to(device=device, dtype=dtype)
        model.mot.gradient_checkpointing = True
        model.train()

        c, g = mot.latent_channels, cfg.latent_grid
        size = model.vision_image_size
        latents = torch.randn(b, c, args.latent_frames, g, g, device=device, dtype=dtype)
        images = torch.rand(b, 3, size, size, device=device, dtype=dtype)
        text_ids = torch.randint(0, 1000, (b, args.text_len), device=device)
        actions = torch.randn(b, args.action_len, mot.action_dim, device=device, dtype=dtype)
        domain_id = torch.zeros(b, dtype=torch.long, device=device)

        def fwd():
            return model(latents=latents, pixel_values=images, text_ids=text_ids,
                         actions=actions, domain_id=domain_id, task=args.task)

        def fwd_nograd():
            with torch.no_grad():
                fwd()

        def fwd_bwd():
            out = fwd()
            out["loss"].backward()
            for p in model.parameters():
                p.grad = None

        t_nograd = timed(fwd_nograd, args.reps)
        t_grad = timed(lambda: fwd(), args.reps)
        t_fb = timed(fwd_bwd, args.reps)

        opt = None
        t_step = float("nan")
        if not args.skip_opt:
            opt = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=1e-4, fused=True
            )

            def step():
                out = fwd()
                out["loss"].backward()
                opt.step()
                opt.zero_grad(set_to_none=True)

            t_step = timed(step, args.reps)
        peak = torch.cuda.max_memory_allocated() / 2**30

        step_str = "     n/a" if args.skip_opt else f"{t_step:8.0f}"
        print(f"{scope:>14s} {t_nograd:11.0f}ms {t_grad:9.0f}ms {t_fb:7.0f}ms "
              f"{step_str}ms {peak:7.1f}G")

        del model, opt, latents, images, actions
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
