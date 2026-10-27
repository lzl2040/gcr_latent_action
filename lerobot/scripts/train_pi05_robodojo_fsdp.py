"""Multi-node FSDP fine-tuning of PI05 on RoboDojo delta actions."""

import json
import logging
import math
import os
import subprocess
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision, ShardingStrategy
from torch.utils.data import DataLoader, DistributedSampler

from lerobot.common.policies.pi05.configuration_pi05 import PI05Config
from lerobot.common.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.common.utils.fsdp_training import (
    FSDPTrainingConfig,
    create_checkpoint_process_group,
    load_fsdp_checkpoint,
    make_distributed_timeout,
    resolve_latest_fsdp_checkpoint,
    save_fsdp_checkpoint,
)
from lerobot.common.utils.random_utils import set_seed
from lerobot.scripts.train_pi05_robodojo import (
    ACTION_KEY,
    CAMERAS,
    STATE_KEY,
    RoboDojoDeltaDataset,
    build_parser,
    file_hash,
    parse_args,
)


def distributed_args():
    parser = build_parser()
    parser.description = __doc__
    parser.set_defaults(device=None)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--export-final", action="store_true")
    parser.add_argument("--max-updates-this-run", type=int)
    args = parse_args(parser)
    if args.device is not None or args.validate_data_only:
        parser.error("Use the single-device entry for --device or --validate-data-only.")
    if args.grad_accumulation_steps <= 0:
        parser.error("--grad-accumulation-steps must be positive.")
    if not math.isfinite(args.lr) or args.seed < 0:
        parser.error("--lr must be finite and --seed must be nonnegative.")
    if args.max_updates_this_run is not None and args.max_updates_this_run <= 0:
        parser.error("--max-updates-this-run must be positive.")
    if not all(name in os.environ for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
        parser.error("Launch this entry with torchrun, not python alone.")
    return args


def make_loader(dataset, args, rank, world_size):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    generator = torch.Generator()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        generator=generator,
        pin_memory=True,
    )
    if len(loader) == 0:
        raise ValueError("The distributed dataset must produce at least one batch per rank.")
    return loader, sampler, generator


def epoch_iterator(loader, sampler, generator, seed, epoch, batch_in_epoch):
    if not 0 <= batch_in_epoch <= len(loader):
        raise ValueError("Checkpoint batch position is outside the current epoch.")
    sampler.set_epoch(epoch)
    generator.manual_seed(seed + epoch)
    iterator = iter(loader)
    for _ in range(batch_in_epoch):
        next(iterator)
    return iterator


def wrap_pi05(policy, device):
    # Joint prefix/expert attention accesses child weights without calling layer.forward.
    # A root unit keeps those weights gathered; wrapping Gemma layers would bypass FSDP hooks.
    policy.float()
    normalizers = (policy.normalize_inputs, policy.normalize_targets, policy.unnormalize_outputs)
    normalizer_parameters = []
    for normalizer in normalizers:
        # Normalize stores statistics as frozen Parameters, not registered buffers.
        normalizer.to(device)
        normalizer_parameters.extend(normalizer.parameters())
    return FullyShardedDataParallel(
        policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=None,
        ),
        use_orig_params=True,
        device_id=device,
        limit_all_gathers=True,
        ignored_states=normalizer_parameters,
    )


def check_resume_geometry(output_dir, geometry):
    checkpoint = resolve_latest_fsdp_checkpoint(output_dir)
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    saved = metadata["training_geometry"]
    previous, current = dict(saved), dict(geometry)
    if current.get("tokenizer_sha256") is not None and (
        current["tokenizer_sha256"] == previous.get("tokenizer_sha256")
    ):
        previous.pop("tokenizer_path", None)
        current.pop("tokenizer_path", None)
    if previous != current:
        raise ValueError(
            "PI05 resume requires unchanged data, batching, optimizer, schedule and seed settings."
        )
    return saved


def export_policy(model, policy, output_dir, step):
    destination = output_dir / f"step{step}"
    temporary = output_dir / f".step{step}.tmp"
    if destination.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite export {destination} or {temporary}.")
    with FullyShardedDataParallel.summon_full_params(
        model, offload_to_cpu=True, rank0_only=True, writeback=False
    ):
        if dist.get_rank() == 0:
            policy.save_pretrained(temporary)
            temporary.rename(destination)
    dist.barrier()


def train(args, device, checkpoint_group):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    set_seed(args.seed)
    dataset = RoboDojoDeltaDataset(args.dataset_root, args.chunk_size, args.video_backend)
    loader, sampler, generator = make_loader(dataset, args, rank, world_size)
    base_file = args.pretrained_path
    if base_file.is_dir():
        base_file = base_file / "model.safetensors"
    geometry = {
        "wrap_strategy": "pi05_root_full_shard_fp32_master_bf16_compute_fp32_normalizers",
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_stats_sha256": file_hash(args.dataset_root / "meta/stats.json"),
        "base_sha256": file_hash(base_file),
        "tokenizer_path": None if Path(args.tokenizer_path).is_file() else args.tokenizer_path,
        "tokenizer_sha256": file_hash(args.tokenizer_path) if Path(args.tokenizer_path).is_file() else None,
        "dataset_frames": len(dataset),
        "steps_per_epoch": len(loader),
        "batch_size_per_rank": args.batch_size,
        "grad_accumulation_steps": args.grad_accumulation_steps,
        "effective_batch_size": args.batch_size * world_size * args.grad_accumulation_steps,
        "chunk_size": args.chunk_size,
        "steps": args.steps,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "seed": args.seed,
        "video_backend": args.video_backend,
    }
    if args.resume:
        geometry = check_resume_geometry(args.output_dir, geometry)
    elif rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    config = PI05Config(
        tokenizer_path=args.tokenizer_path,
        device="cpu",
        dtype="bfloat16",
        chunk_size=args.chunk_size,
        n_action_steps=min(10, args.chunk_size),
        input_features=dataset.input_features,
        output_features=dataset.output_features,
        gradient_checkpointing=True,
        train_expert_only=False,
        optimizer_lr=args.lr,
        scheduler_warmup_steps=min(args.warmup_steps, max(0, args.steps - 1)),
        scheduler_decay_steps=args.steps,
    )
    policy = PI05Policy(config, dataset_stats=dataset.stats)
    if not args.resume:
        policy.load_base_weights(base_file)
    policy.config.device = str(device)
    model = wrap_pi05(policy, device)
    model.train()
    optimizer = config.get_optimizer_preset().build(model.parameters())
    scheduler_config = config.get_scheduler_preset()
    scheduler_config.num_platform_steps = 0
    scheduler_config.decay_from_platform_end = True
    scheduler_config.num_decay_steps = max(1, args.steps - config.scheduler_warmup_steps)
    scheduler = scheduler_config.build(optimizer, args.steps)
    fsdp_config = FSDPTrainingConfig(fp8=False, replicate_frozen_params=False)
    epoch = batch_in_epoch = update_step = micro_step = 0
    if args.resume:
        state = load_fsdp_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            output_dir=args.output_dir,
            fsdp_config=fsdp_config,
            training_geometry=geometry,
            device=device,
            checkpoint_process_group=checkpoint_group,
        )
        epoch, batch_in_epoch = state.epoch, state.batch_in_epoch
        update_step, micro_step = state.update_step, state.micro_step
        if update_step >= args.steps:
            raise ValueError("The checkpoint already reached --steps; no updates remain.")
    else:
        set_seed(args.seed + rank)
    iterator = epoch_iterator(loader, sampler, generator, args.seed + rank, epoch, batch_in_epoch)
    if rank == 0:
        config.save_pretrained(args.output_dir)
        contract = {
            **geometry,
            "world_size": world_size,
            "state_feature": STATE_KEY,
            "action_feature": ACTION_KEY,
            "camera_mapping": CAMERAS,
            "normalization": "QUANTILES",
            "action_semantics": "World dp, left-multiplied dR, incremental gripper; reanchor each replan.",
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "args": {
                key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
            },
        }
        name = f"launch_step{update_step}.json"
        with (args.output_dir / name).open("x") as stream:
            json.dump(contract, stream, indent=2)
    stop_step = min(args.steps, update_step + (args.max_updates_this_run or args.steps))
    torch.cuda.reset_peak_memory_stats(device)
    log = (args.output_dir / "training.jsonl").open("a" if args.resume else "x") if rank == 0 else None
    try:
        while update_step < stop_step:
            optimizer.zero_grad(set_to_none=True)
            total_loss = torch.zeros((), device=device)
            for accumulation in range(args.grad_accumulation_steps):
                if batch_in_epoch == len(loader):
                    epoch, batch_in_epoch = epoch + 1, 0
                    iterator = epoch_iterator(loader, sampler, generator, args.seed + rank, epoch, 0)
                batch = next(iterator)
                batch_in_epoch += 1
                micro_step += 1
                batch = {
                    key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                synchronize = accumulation == args.grad_accumulation_steps - 1
                with (
                    nullcontext() if synchronize else model.no_sync(),
                    torch.autocast("cuda", dtype=torch.bfloat16),
                ):
                    loss, _ = model(batch)
                    finite = torch.isfinite(loss).to(torch.int32)
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                    if not finite.item():
                        raise FloatingPointError(f"Nonfinite loss at micro-step {micro_step}.")
                    (loss / args.grad_accumulation_steps).backward()
                total_loss += loss.detach() / args.grad_accumulation_steps
            grad_norm = model.clip_grad_norm_(config.optimizer_grad_clip_norm)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Nonfinite gradient norm at update {update_step + 1}.")
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            update_step += 1
            dist.all_reduce(total_loss)
            row = {
                "step": update_step,
                "micro_step": micro_step,
                "loss": total_loss.item() / world_size,
                "grad_norm": grad_norm.item(),
                "lr": lr,
                "epoch": epoch,
                "batch_in_epoch": batch_in_epoch,
            }
            if log is not None:
                log.write(json.dumps(row) + "\n")
                log.flush()
                logging.info("%s", row)
            if update_step % args.save_freq == 0 or update_step == stop_step:
                save_fsdp_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    output_dir=args.output_dir,
                    fsdp_config=fsdp_config,
                    training_geometry=geometry,
                    micro_step=micro_step,
                    update_step=update_step,
                    epoch=epoch,
                    batch_in_epoch=batch_in_epoch,
                    device=device,
                    checkpoint_process_group=checkpoint_group,
                )
        if args.export_final:
            export_policy(model, policy, args.output_dir, update_step)
        peak = torch.tensor(torch.cuda.max_memory_allocated(device), device=device)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        if rank == 0:
            summary = {
                "completed": update_step == args.steps,
                "stopped_at_update": update_step,
                "target_steps": args.steps,
                "final_loss": row["loss"],
                "world_size": world_size,
                "peak_cuda_allocated_bytes_per_rank_max": peak.item(),
            }
            (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    finally:
        if log is not None:
            log.close()


def main():
    args = distributed_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PI05 FSDP training requires CUDA.")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    timeout = make_distributed_timeout(os.environ.get("DISTRIBUTED_TIMEOUT_MINUTES", "60"))
    dist.init_process_group("nccl", timeout=timeout, device_id=device)
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [rank {dist.get_rank()}] %(levelname)s %(message)s",
        force=True,
    )
    try:
        checkpoint_group = create_checkpoint_process_group(timeout)
        train(args, device, checkpoint_group)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
