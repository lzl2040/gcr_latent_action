"""Prove that cached and Cosmos3-style interleaved MoT execution are equivalent.

The two schedules must produce the same GEN output and parameter gradients:

* cached: run all UND layers, retain their K/V, then run all GEN layers;
* interleaved: run UND layer N and immediately consume its K/V in GEN layer N.

Run:
    python -u scripts/check_mot_execution.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot.common.policies.mot.modeling_mot import MoTConfig, MoTModel  # noqa: E402


def collect_grads(model: MoTModel) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().float().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    torch.manual_seed(0)

    config = MoTConfig(
        und_hidden_size=256,
        num_hidden_layers=3,
        und_num_attention_heads=8,
        und_intermediate_size=512,
        vocab_size=1000,
        head_dim=32,
        num_key_value_heads=4,
        partial_rotary_factor=0.75,
        mrope_section=(4, 4, 4),
        vision_lora_rank=16,
        vision_lora_alpha=32.0,
        gen_hidden_size=192,
        gen_num_attention_heads=8,
        gen_intermediate_size=512,
    )
    model = MoTModel(config).to(device=device, dtype=dtype).train()
    model.gradient_checkpointing = False

    batch, und_len, gen_len = 2, 7, 11
    und_hidden = torch.randn(
        batch,
        und_len,
        config.und_hidden_size,
        device=device,
        dtype=dtype,
    )
    gen_hidden = torch.randn(
        batch,
        gen_len,
        config.gen_hidden_size,
        device=device,
        dtype=dtype,
    )
    und_pos = torch.arange(und_len, device=device).view(1, 1, -1).expand(3, batch, -1)
    gen_pos = torch.arange(gen_len, device=device).view(1, 1, -1).expand(3, batch, -1)
    timestep = torch.rand(batch, gen_len, device=device, dtype=dtype) * 1000.0

    kv, rope_und = model.forward_und_kv(und_hidden, und_pos)
    cached = model.forward_gen(gen_hidden, gen_pos, kv, rope_und, timestep)
    cached.float().square().mean().backward()
    cached_grads = collect_grads(model)

    model.zero_grad(set_to_none=True)
    interleaved = model.forward_interleaved(
        und_hidden,
        und_pos,
        gen_hidden,
        gen_pos,
        timestep,
        checkpoint_layers=False,
    )
    interleaved.float().square().mean().backward()
    interleaved_grads = collect_grads(model)

    output_diff = (cached - interleaved).abs().max().item()
    grad_names_match = cached_grads.keys() == interleaved_grads.keys()
    grad_diff = max(
        (
            (cached_grads[name] - interleaved_grads[name]).abs().max().item()
            for name in cached_grads.keys() & interleaved_grads.keys()
        ),
        default=0.0,
    )
    tolerance = 1e-5 if dtype == torch.float32 else 2e-2
    equivalent = output_diff <= tolerance and grad_names_match and grad_diff <= tolerance
    print(
        f"[{'PASS' if equivalent else 'FAIL'}] cached == interleaved  "
        f"output_max_diff={output_diff:.3e} grad_max_diff={grad_diff:.3e} "
        f"grad_keys={len(cached_grads)}"
    )

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        microbatched = torch.cat(
            [
                model.forward_interleaved(
                    und_hidden[start : start + 1],
                    und_pos[:, start : start + 1],
                    gen_hidden[start : start + 1],
                    gen_pos[:, start : start + 1],
                    timestep[start : start + 1],
                    checkpoint_layers=False,
                )
                for start in range(batch)
            ],
            dim=0,
        )
    microbatch_diff = (cached - microbatched).abs().max().item()
    microbatch_cosine = torch.nn.functional.cosine_similarity(
        cached.float().flatten(),
        microbatched.float().flatten(),
        dim=0,
    ).item()
    microbatch_ok = microbatch_diff <= max(tolerance, 5e-2) and microbatch_cosine >= 0.999
    print(
        f"[{'PASS' if microbatch_ok else 'FAIL'}] batch slicing  "
        f"output_max_diff={microbatch_diff:.3e} cosine={microbatch_cosine:.8f}"
    )

    model.gradient_checkpointing = True
    checkpointed = model.forward_interleaved(
        und_hidden,
        und_pos,
        gen_hidden,
        gen_pos,
        timestep,
        checkpoint_layers=True,
    )
    checkpointed.float().square().mean().backward()
    checkpoint_grads = collect_grads(model)
    checkpoint_diff = (cached - checkpointed).abs().max().item()
    checkpoint_grad_names_match = cached_grads.keys() == checkpoint_grads.keys()
    checkpoint_grad_diff = max(
        (
            (cached_grads[name] - checkpoint_grads[name]).abs().max().item()
            for name in cached_grads.keys() & checkpoint_grads.keys()
        ),
        default=0.0,
    )
    checkpoint_ok = (
        checkpoint_diff <= tolerance
        and checkpoint_grad_names_match
        and checkpoint_grad_diff <= tolerance
    )
    print(
        f"[{'PASS' if checkpoint_ok else 'FAIL'}] interleaved checkpoint  "
        f"output_max_diff={checkpoint_diff:.3e} "
        f"grad_max_diff={checkpoint_grad_diff:.3e}"
    )

    model.zero_grad(set_to_none=True)
    model.set_und_trainable(False)
    model.set_gen_trainable(True)
    frozen = model.forward_interleaved(
        und_hidden,
        und_pos,
        gen_hidden,
        gen_pos,
        timestep,
        und_requires_grad=False,
        checkpoint_layers=True,
    )
    frozen.float().square().mean().backward()
    frozen_ok = all(
        parameter.grad is None
        for parameter in model.parameters()
        if not parameter.requires_grad
    )
    gen_grad_count = sum(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    frozen_ok = frozen_ok and gen_grad_count > 0
    print(
        f"[{'PASS' if frozen_ok else 'FAIL'}] frozen UND checkpoint  "
        f"trainable_grad_tensors={gen_grad_count}"
    )

    ok = equivalent and microbatch_ok and checkpoint_ok and frozen_ok
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
