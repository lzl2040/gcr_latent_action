"""Correctness checks for the Qwen3-VL vision trunk wrapper.

Qwen3-VL's ViT is wrapped by hand (manual patchification, token reordering, merger
bypass), so the usual "it runs and the shape is right" is not evidence of correctness.
Each check below targets a specific way the wrapper could be silently wrong.
"""

import argparse
import contextlib
import time

import numpy as np
import torch

from lerobot.common.policies.ace.qwen3vl_encoder import build_qwen3vl_vision


def check_patch_order(model_dir: str, size: int) -> None:
    """Our patchification must equal Qwen's own image processor, byte for byte.

    If the patch order were wrong the tower would still run, but every patch would receive
    the position embedding of a different location.
    """
    from transformers import AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained(model_dir)
    torch.manual_seed(0)
    img = torch.randint(0, 255, (1, 3, size, size), dtype=torch.uint8)
    ref = proc(images=[img[0].permute(1, 2, 0).numpy()], do_resize=False, return_tensors="pt")

    mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
    std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
    x = (img.float() * proc.rescale_factor - mean) / std
    p, m, tp, c = 16, 2, 2, 3
    b, _, h, w = x.shape
    gh, gw = h // p, w // p
    bh, bw = gh // m, gw // m
    q = x.view(b, c, bh, m, p, bw, m, p).permute(0, 2, 5, 3, 6, 1, 4, 7)
    q = q.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, tp, -1, -1)
    mine = q.reshape(b * gh * gw, c * tp * p * p)

    diff = (mine - ref["pixel_values"]).abs().max().item()
    print(f"  patch order vs official processor : max abs diff {diff:.3e}  {'OK' if diff < 1e-5 else 'FAIL'}")

    row_major = x.view(b, c, gh, p, gw, p).permute(0, 2, 4, 1, 3, 5).reshape(b, gh * gw, c * p * p)
    tok = mine.view(b, bh, bw, m, m, -1).permute(0, 1, 3, 2, 4, 5).reshape(b, gh * gw, -1)
    first = tok.view(b, gh * gw, c, tp, p, p)[:, :, :, 0].reshape(b, gh * gw, c * p * p)
    d2 = (first - row_major).abs().max().item()
    print(f"  reorder -> row-major              : max abs diff {d2:.3e}  {'OK' if d2 == 0 else 'FAIL'}")


def check_independence(model, size: int, device: str) -> None:
    """Images in a batch must not attend to each other.

    The tower uses variable-length attention driven by `cu_seqlens`. If that ever stopped
    segmenting the batch, features would depend on whatever else was sampled alongside
    them -- a leak that no shape check and no loss curve would reveal.
    """
    torch.manual_seed(1)
    x = torch.randn(3, 3, size, size, device=device)
    with torch.no_grad():
        full = model(pixel_values=x).last_hidden_state
        solo = model(pixel_values=x[1:2]).last_hidden_state
        flipped = model(pixel_values=x.flip(0)).last_hidden_state.flip(0)
    d1 = (solo - full[1:2]).abs().max().item()
    d2 = (flipped - full).abs().max().item()
    print(f"  batch independence ({device:4s})         : max abs diff {d1:.3e}  {'OK' if d1 < 1e-4 else 'FAIL'}")
    print(f"  order independence ({device:4s})         : max abs diff {d2:.3e}  {'OK' if d2 < 1e-4 else 'FAIL'}")


def check_spatial_grid(model, size: int, device: str) -> None:
    """Token i must correspond to image cell i after the reorder.

    Perturb one 16x16 cell and confirm the token that moves most is the one indexed by that
    cell in row-major order. Attention mixes information, so other tokens move too -- the
    claim being tested is only that the *argmax* lands on the right index.
    """
    grid = size // 16
    torch.manual_seed(2)
    base = torch.randn(1, 3, size, size, device=device)
    for cell in (0, grid + 1, 2 * grid + 3, grid * grid - 1):
        r, c = divmod(cell, grid)
        bumped = base.clone()
        bumped[:, :, r * 16 : (r + 1) * 16, c * 16 : (c + 1) * 16] += 4.0
        with torch.no_grad():
            a = model(pixel_values=base).last_hidden_state
            b = model(pixel_values=bumped).last_hidden_state
        moved = (b - a).norm(dim=-1)[0].argmax().item()
        print(
            f"  perturb cell {cell:3d} (row {r}, col {c}) -> most-changed token {moved:3d}  "
            f"{'OK' if moved == cell else 'FAIL'}"
        )


def check_batched_attention(model_dir: str, size: int, device: str) -> None:
    """The optimized path must reproduce the stock implementation exactly.

    Batched attention and the grid caches both replace hot code inside a frozen tower, so an
    error would surface only as slightly worse features -- nothing would crash.
    """
    fast, _, _ = build_qwen3vl_vision(model_dir, optimized=True)
    slow, _, _ = build_qwen3vl_vision(model_dir, optimized=False)
    fast, slow = fast.to(device).eval(), slow.to(device).eval()
    torch.manual_seed(3)
    x = torch.randn(8, 3, size, size, device=device)

    for tag, ctx in (
        ("fp32", contextlib.nullcontext()),
        ("bf16", torch.autocast(device, dtype=torch.bfloat16)),
    ):
        with torch.no_grad(), ctx:
            a = fast(pixel_values=x).last_hidden_state.float()
            a2 = fast(pixel_values=x).last_hidden_state.float()  # second call: caches warm
            b = slow(pixel_values=x).last_hidden_state.float()
        rel = ((a - b).norm() / b.norm()).item()
        cache_rel = ((a - a2).norm() / a2.norm()).item()
        print(f"  optimized vs stock ({tag})        : rel diff {rel:.3e}  {'OK' if rel < 1e-5 else 'FAIL'}")
        print(f"  cached call is stable ({tag})     : rel diff {cache_rel:.3e}  {'OK' if cache_rel == 0 else 'FAIL'}")

    # With a trainable pos_embed the cache must switch itself off, or unfreezing the tower
    # would silently pin the position embedding at its initial value.
    for p in fast.parameters():
        p.requires_grad_(True)
    fast.vision_model.pos_embed.weight.grad = None
    out = fast(pixel_values=x[:2]).last_hidden_state.sum()
    out.backward()
    g = fast.vision_model.pos_embed.weight.grad
    ok = g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    print(f"  pos_embed gets grad when unfrozen : {'OK' if ok else 'FAIL'}")

    for tag, mod in (("stock    ", slow), ("optimized", fast)):
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
            for _ in range(2):
                mod(pixel_values=x)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                mod(pixel_values=x)
            if device == "cuda":
                torch.cuda.synchronize()
            print(f"  {tag} tower: {(time.perf_counter() - t0) / 5 * 1000:7.1f} ms / {x.shape[0]} images")


def check_normalization(model_dir: str) -> None:
    """The model's pixel normalisation must match this checkpoint's preprocessor config.

    Worth testing rather than trusting: `Qwen2VLImageProcessorFast` hard-codes
    OPENAI_CLIP mean/std as the *class* default (what Qwen2-VL used), and only
    `preprocessor_config.json` overrides it to 0.5 for Qwen3-VL. Reading the defaults off
    the transformers source, or copying constants from Qwen2-VL code, silently feeds a
    frozen tower out-of-distribution input.
    """
    from transformers import AutoImageProcessor

    from lerobot.common.policies.ace.modeling_robo_contrast import SIGLIP_MEAN, SIGLIP_STD

    proc = AutoImageProcessor.from_pretrained(model_dir)
    want_mean = [round(float(v), 6) for v in proc.image_mean]
    want_std = [round(float(v), 6) for v in proc.image_std]
    got_mean = [round(float(v), 6) for v in SIGLIP_MEAN.flatten()]
    got_std = [round(float(v), 6) for v in SIGLIP_STD.flatten()]
    ok = want_mean == got_mean and want_std == got_std and abs(proc.rescale_factor - 1 / 255) < 1e-9
    print(
        f"  normalisation matches checkpoint  : model {got_mean}/{got_std} vs "
        f"processor {want_mean}/{want_std}  {'OK' if ok else 'FAIL'}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="/Data/lzl/huggingface/Qwen3-VL-4B-Instruct")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model, dim, size = build_qwen3vl_vision(args.model_dir)
    model = model.to(args.device).eval()
    print(f"\nhidden={dim} image={size} tokens={(size // 16) ** 2}\n")
    check_normalization(args.model_dir)
    check_patch_order(args.model_dir, size)
    check_independence(model, size, args.device)
    check_spatial_grid(model, size, args.device)
    check_batched_attention(args.model_dir, size, args.device)


if __name__ == "__main__":
    main()
