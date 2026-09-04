#!/usr/bin/env python
"""Validate AnyTouch checkpoint loading, window ordering and output shape."""

import argparse
import time

import torch

from lerobot.common.policies.ace.anytouch_tactile import (
    ANYTOUCH_OUT_DIM,
    AnyTouchTactileTower,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="/Data/lzl/huggingface/anytouch_encoder.pth",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tokens", type=int, default=2, choices=(1, 2))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--forward_batch_size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tower = AnyTouchTactileTower(
        args.checkpoint,
        num_tokens=args.tokens,
        forward_batch_size=args.forward_batch_size,
    ).to(
        device=device, dtype=dtype
    )
    images = torch.randint(
        0,
        256,
        (args.batch_size, 4, 3, 224, 224),
        dtype=torch.uint8,
        device=device,
    )
    changed = images.clone()
    changed[:, -1] = 255 - changed[:, -1]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = tower(images)
        repeated = tower(images)
        changed_output = tower(changed)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    expected = (args.batch_size, args.tokens, ANYTOUCH_OUT_DIM)
    if output.shape != expected:
        raise AssertionError(f"Expected {expected}, got {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise AssertionError("AnyTouch output contains non-finite values")
    repeat_error = (output - repeated).abs().max().item()
    if repeat_error != 0.0:
        raise AssertionError(f"Frozen AnyTouch output is not deterministic: {repeat_error}")
    if args.tokens == 2:
        first_delta = (output[:, 0] - changed_output[:, 0]).abs().max().item()
        second_delta = (output[:, 1] - changed_output[:, 1]).abs().max().item()
        if first_delta != 0.0 or second_delta <= 0.0:
            raise AssertionError(
                "Changing frame 3 must leave window [0,1,2] unchanged and alter [1,2,3]: "
                f"{first_delta=}, {second_delta=}"
            )

    total = sum(parameter.numel() for parameter in tower.parameters())
    trainable = sum(
        parameter.numel() for parameter in tower.parameters() if parameter.requires_grad
    )
    peak = (
        torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
    )
    print(
        f"AnyTouch OK: shape={tuple(output.shape)} params={total / 1e6:.1f}M "
        f"trainable={trainable} elapsed={elapsed:.3f}s peak={peak:.2f}GiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
