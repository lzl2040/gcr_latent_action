"""Verify the hand-written und expert reproduces HuggingFace's Phi-4-mini exactly.

The und expert re-implements Phi's forward pass so that we control RoPE and can export
per-layer key/value tensors.  That is only safe if it is numerically identical to the
reference implementation -- otherwise the frozen pretrained weights are worthless.

Run:  python -u scripts/check_mot_und.py
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot.common.policies.mot.modeling_mot import MoTConfig, MoTModel  # noqa: E402

PHI_DIR = "/Data/lzl/huggingface/Phi-4-mini-instruct"


def reference_hidden(phi_dir: str, input_ids: torch.Tensor, device: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(phi_dir, dtype=dtype, attn_implementation="sdpa")
    model = model.to(device).eval()
    rotary = model.model.rotary_emb
    meta = {
        "attention_scaling": float(rotary.attention_scaling),
        "inv_freq": rotary.inv_freq.detach().float().cpu().clone(),
    }
    with torch.no_grad():
        out = model.model(input_ids=input_ids.to(device), use_cache=False)
    hidden = out.last_hidden_state.float().cpu()
    del model, out
    gc.collect()
    torch.cuda.empty_cache()
    return hidden, meta


def ours_hidden(phi_dir: str, input_ids: torch.Tensor, device: str, dtype: torch.dtype, scaling: float):
    cfg = MoTConfig.from_phi_dir(phi_dir)
    model = MoTModel(cfg, attention_scaling=scaling)
    model.load_phi_weights()
    model = model.to(device=device, dtype=dtype).eval()

    b, length = input_ids.shape
    pos = torch.arange(length, device=device).view(1, 1, length).expand(3, b, length)
    with torch.no_grad():
        hidden, kv, rope = model.forward_und(model.embed_tokens(input_ids.to(device)), pos)
    report = model.param_report()
    inv_freq = model.rotary_emb.inv_freq.detach().float().cpu().clone()
    out = hidden.float().cpu()
    n_kv = len(kv)
    del model, kv
    gc.collect()
    torch.cuda.empty_cache()
    return out, report, inv_freq, n_kv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi_dir", default=PHI_DIR)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seq", type=int, default=48)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--fp32", action="store_true", default=True)
    args = ap.parse_args()

    dtype = torch.float32 if args.fp32 else torch.bfloat16
    torch.manual_seed(0)
    input_ids = torch.randint(0, 32000, (args.batch, args.seq))

    print(f"[setup] dtype={dtype} batch={args.batch} seq={args.seq}")
    ref, meta = reference_hidden(args.phi_dir, input_ids, args.device, dtype)
    print(f"[hf]    last_hidden_state {tuple(ref.shape)}  attention_scaling={meta['attention_scaling']:.6f}")

    ours, report, inv_freq, n_kv = ours_hidden(
        args.phi_dir, input_ids, args.device, dtype, meta["attention_scaling"]
    )
    print(f"[ours]  last_hidden_state {tuple(ours.shape)}  per-layer kv exported: {n_kv}")

    ok = True

    freq_err = (inv_freq - meta["inv_freq"]).abs().max().item()
    status = "OK" if freq_err < 1e-6 else "FAIL"
    ok &= freq_err < 1e-6
    print(f"[check] inv_freq max|diff|        {freq_err:.3e}   {status}")
    print(f"        (len {inv_freq.numel()}, must equal rotary_dim//2)")

    denom = ref.abs().max().item()
    abs_err = (ours - ref).abs().max().item()
    rel_err = abs_err / max(denom, 1e-12)
    tol = 2e-4 if dtype is torch.float32 else 5e-2
    status = "OK" if rel_err < tol else "FAIL"
    ok &= rel_err < tol
    print(f"[check] hidden max|diff|          {abs_err:.3e}")
    print(f"[check] hidden relative           {rel_err:.3e} (tol {tol})   {status}")

    cos = torch.nn.functional.cosine_similarity(ours.flatten(), ref.flatten(), dim=0).item()
    status = "OK" if cos > 0.9999 else "FAIL"
    ok &= cos > 0.9999
    print(f"[check] cosine similarity         {cos:.8f}   {status}")

    print(
        f"[params] und={report['und'] / 1e9:.3f}B  gen={report['gen'] / 1e6:.1f}M  "
        f"total={report['total'] / 1e9:.3f}B"
    )

    print("\nALL CHECKS PASSED" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
