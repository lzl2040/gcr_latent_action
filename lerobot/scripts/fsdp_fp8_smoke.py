#!/usr/bin/env python
"""Native-H100 smoke for FSDP, TorchAO FP8 Linear, and AdamW8bit."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from bitsandbytes.optim import AdamW8bit
from torch import nn

from lerobot.common.utils.fsdp_training import (
    FSDPTrainingConfig,
    convert_policy_to_fp8,
    load_fsdp_checkpoint,
    save_fsdp_checkpoint,
    wrap_policy_with_fsdp,
)


class SmokeBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.up = nn.Linear(hidden_dim, hidden_dim * 4, bias=False)
        self.down = nn.Linear(hidden_dim * 4, hidden_dim, bias=False)
        self.norm = nn.RMSNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.down(torch.nn.functional.silu(self.up(self.norm(x))))


class SmokeGeneration(nn.Module):
    def __init__(self, hidden_dim: int, depth: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [SmokeBlock(hidden_dim) for _ in range(depth)]
        )
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.output(x)


class SmokePolicy(nn.Module):
    def __init__(self, hidden_dim: int, depth: int) -> None:
        super().__init__()
        self.generation = SmokeGeneration(hidden_dim, depth)
        self.frozen_conditioner = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.frozen_conditioner.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            conditioning = self.frozen_conditioner(x)
        return self.generation(x + conditioning)

    def get_optim_params(self):
        return [
            {
                "params": self.generation.parameters(),
                "group_name": "generation",
            }
        ]


def _build(
    *,
    hidden_dim: int,
    depth: int,
    device: torch.device,
    config: FSDPTrainingConfig,
):
    policy = SmokePolicy(hidden_dim, depth).to(device=device, dtype=torch.bfloat16)
    converted = convert_policy_to_fp8(policy, config, device)
    model, summary = wrap_policy_with_fsdp(policy, config, device)
    optimizer = AdamW8bit(
        model.get_optim_params(),
        lr=1e-4,
        min_8bit_size=4096,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return model, optimizer, scheduler, converted, summary


def _step(
    model,
    optimizer: torch.optim.Optimizer,
    scheduler,
    inputs: torch.Tensor,
) -> float:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(inputs).float().square().mean()
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite smoke loss: {loss.item()}.")
    loss.backward()
    model.clip_grad_norm_(1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--recipe", default="rowwise_with_gw_hp")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = torch.device("cuda", local_rank)
    config = FSDPTrainingConfig(
        fp8=True,
        fp8_recipe=args.recipe,
        fp8_scope="generation",
        fp8_emulate=False,
        fp8_min_features=128,
        min_wrap_params=1_000_000,
        replicate_frozen_params=True,
    )
    training_geometry = {
        "batch_size_per_rank": args.batch_size,
        "gradient_accumulation_steps": 1,
        "sequence_length": args.sequence_length,
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
    }

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    model, optimizer, scheduler, converted, summary = _build(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        device=device,
        config=config,
    )
    torch.manual_seed(5678 + rank)
    inputs = torch.randn(
        args.batch_size,
        args.sequence_length,
        args.hidden_dim,
        device=device,
        dtype=torch.bfloat16,
    )

    torch.cuda.synchronize(device)
    start = time.perf_counter()
    first_loss = _step(model, optimizer, scheduler, inputs)
    torch.cuda.synchronize(device)
    first_step_s = time.perf_counter() - start
    save_fsdp_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=args.checkpoint_dir,
        fsdp_config=config,
        training_geometry=training_geometry,
        micro_step=1,
        update_step=1,
        epoch=0,
        batch_in_epoch=1,
        device=device,
    )

    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    restored_model, restored_optimizer, restored_scheduler, _, _ = _build(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        device=device,
        config=config,
    )
    resume = load_fsdp_checkpoint(
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        output_dir=args.checkpoint_dir,
        fsdp_config=config,
        training_geometry=training_geometry,
        device=device,
    )
    restored_loss = _step(
        restored_model,
        restored_optimizer,
        restored_scheduler,
        inputs,
    )
    optimizer_state_dtypes = sorted(
        {
            str(value.dtype)
            for state in restored_optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor)
        }
    )
    if "torch.uint8" not in optimizer_state_dtypes:
        raise AssertionError(
            f"AdamW8bit did not create uint8 moments: {optimizer_state_dtypes}."
        )

    local_summary = {
        "rank": rank,
        "gpu": torch.cuda.get_device_name(device),
        "capability": ".".join(
            str(part) for part in torch.cuda.get_device_capability(device)
        ),
        "converted_fp8_linears": len(converted),
        "wrapped_blocks": len(summary.wrapped_module_names),
        "replicated_frozen_parameters": summary.replicated_frozen_parameters,
        "first_loss": first_loss,
        "restored_loss": restored_loss,
        "first_step_s": first_step_s,
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "optimizer_state_dtypes": optimizer_state_dtypes,
        "resume_step": resume.update_step,
    }
    gathered = [None] * dist.get_world_size() if rank == 0 else None
    dist.gather_object(local_summary, gathered, dst=0)
    if rank == 0:
        print("FP8_FSDP_SMOKE_OK", flush=True)
        print(json.dumps(gathered, indent=2, sort_keys=True), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
