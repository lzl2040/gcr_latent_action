"""Check the Cosmos stage-2/stage-3 task family: T2I, T2V, I2V, V2V and action.

All five tasks share one rectified-flow code path and differ only in how many latent frames
start clean and whether the understanding stream gets an image. Two things about that are
worth testing rather than assuming.

**Context frames must be excluded from the loss.** They are handed to the model unchanged, so
predicting them is free; if they leaked into the average, the loss would fall for a reason that
has nothing to do with learning. The test makes the leak impossible to miss by scaling the
context latents by 1000: the masked loss stays O(1), while a broken mask lands around 1e6.

**T2I/T2V must not run the vision tower.** They have no input frame, and quietly feeding a
blank one would spend a full ViT pass teaching the model what "no image" looks like. A counting
hook proves the tower is skipped instead of fed zeros.

Run:  python -u scripts/check_mot_tasks.py --batch 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot.common.policies.mot.modeling_mot import MoTConfig  # noqa: E402
from lerobot.common.policies.mot.world_model import (  # noqa: E402
    TASK_SPECS,
    MoTWorldModel,
    WorldModelConfig,
)

CONTEXT_SCALE = 1000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi_dir", default="/Data/lzl/huggingface/Phi-4-mini-instruct")
    ap.add_argument("--qwen3vl_dir", default="/Data/lzl/huggingface/Qwen3-VL-4B-Instruct")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--latent_frames", type=int, default=3)
    ap.add_argument("--text_len", type=int, default=32)
    ap.add_argument("--action_len", type=int, default=32)
    ap.add_argument("--random_init", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    mot = MoTConfig.from_phi_dir(args.phi_dir)
    cfg = WorldModelConfig(mot=mot, qwen3vl_dir=args.qwen3vl_dir)
    model = MoTWorldModel(cfg)
    if args.random_init:
        model.mot.freeze_und()
    else:
        model.load_pretrained()
    model = model.to(device=device, dtype=dtype)
    model.mot.gradient_checkpointing = True
    model.train()

    rep = model.param_report()
    print(
        f"params: total {rep['total'] / 1e9:.3f}B  trainable {rep['trainable'] / 1e9:.3f}B  "
        f"gen {rep['gen_trainable'] / 1e9:.3f}B"
    )

    # Count vision-tower invocations so "no image" is proved, not assumed.
    vision_calls = {"n": 0}
    model.vision.register_forward_hook(lambda *_: vision_calls.__setitem__("n", vision_calls["n"] + 1))

    b = args.batch
    c, g = mot.latent_channels, cfg.latent_grid
    size = model.vision_image_size
    ok = True

    for task, spec in TASK_SPECS.items():
        t_lat = spec.latent_frames or args.latent_frames
        latents = torch.randn(b, c, t_lat, g, g, device=device, dtype=dtype)
        # Make any leak of the clean context frames into the loss enormous.
        if spec.context:
            latents[:, :, : spec.context] *= CONTEXT_SCALE
        images = torch.rand(b, 3, size, size, device=device, dtype=dtype)
        text_ids = torch.randint(0, 1000, (b, args.text_len), device=device)
        actions = torch.randn(b, args.action_len, mot.action_dim, device=device, dtype=dtype)
        domain_id = torch.zeros(b, dtype=torch.long, device=device)

        before = vision_calls["n"]
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        out = model(
            latents=latents,
            pixel_values=images,
            text_ids=text_ids,
            actions=actions,
            domain_id=domain_id,
            task=task,
        )
        out["loss"].backward()
        torch.cuda.synchronize() if device.type == "cuda" else None
        dt = time.perf_counter() - t0
        used_vision = vision_calls["n"] > before

        loss = out["loss_video"].item()
        # Random init predicts ~0, so the target's own scale sets the floor:
        # E||noise - latents||^2 ~ 2 per element on the *target* frames only.
        mask_ok = loss < 100.0
        vision_ok = used_vision == spec.image
        grads = sum(
            1
            for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
        )
        good = mask_ok and vision_ok and torch.isfinite(out["loss"]).item()
        ok &= good
        print(
            f"[{task:6s}] ctx={spec.context} T={t_lat} img={int(spec.image)} act={int(spec.action)}  "
            f"loss_video={loss:9.3f}"
            + (f" loss_action={out['loss_action'].item():.3f}" if "loss_action" in out else "")
            + f"  vision_ran={int(used_vision)}(want {int(spec.image)})"
            f"  grad_tensors={grads}  {dt * 1e3:6.0f} ms  {'OK' if good else 'FAIL'}"
        )
        if not mask_ok:
            print(
                f"         FAIL: loss {loss:.1f} suggests the {spec.context} clean context "
                f"frames (scaled x{CONTEXT_SCALE:.0f}) leaked into the average"
            )
        if not vision_ok:
            print(f"         FAIL: vision tower ran={used_vision}, expected {spec.image}")
        model.zero_grad(set_to_none=True)

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
