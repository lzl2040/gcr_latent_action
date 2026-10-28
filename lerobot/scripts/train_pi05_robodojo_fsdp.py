"""Multi-node FSDP fine-tuning of PI05 on RoboDojo delta actions."""

import json
import logging
import math
import os
import subprocess
import tempfile
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullStateDictConfig,
    FullyShardedDataParallel,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.utils.data import DataLoader, DistributedSampler

from lerobot.common.optim.optimizers import AdamW8bitConfig
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
    wandb_logging,
)


def distributed_args():
    parser = build_parser()
    parser.description = __doc__
    parser.set_defaults(device=None)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--resume", action="store_true", help="Resume if latest_checkpoint exists; otherwise start fresh."
    )
    parser.add_argument("--export-final", action="store_true")
    parser.add_argument("--max-updates-this-run", type=int)
    parser.add_argument("--fsdp-wrap-strategy", choices=("layer", "root"), default="layer")
    parser.add_argument("--master-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--optimizer", choices=("adamw", "adamw8bit"), default="adamw")
    parser.add_argument("--attention-backend", choices=("eager", "sdpa"), default="eager")
    args = parse_args(parser)
    if args.device is not None or args.validate_data_only:
        parser.error("Use the single-device entry for --device or --validate-data-only.")
    if args.grad_accumulation_steps <= 0:
        parser.error("--grad-accumulation-steps must be positive.")
    if not math.isfinite(args.lr) or args.seed < 0:
        parser.error("--lr must be finite and --seed must be nonnegative.")
    if args.max_updates_this_run is not None and args.max_updates_this_run <= 0:
        parser.error("--max-updates-this-run must be positive.")
    if args.fsdp_wrap_strategy != "layer" and (
        args.master_dtype != "float32" or args.optimizer != "adamw" or args.attention_backend != "eager"
    ):
        parser.error("Memory-optimized modes require --fsdp-wrap-strategy layer.")
    if not all(name in os.environ for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
        parser.error("Launch this entry with torchrun, not python alone.")
    return args


def prepare_output_directory(output_dir, resume, rank):
    status = [resume, None]
    if rank == 0:
        try:
            marker = output_dir / "latest_checkpoint"
            if resume and not (marker.exists() or marker.is_symlink()):
                status[0] = False
                logging.warning("No latest_checkpoint in %s; starting fresh from base weights.", output_dir)
                if output_dir.exists():
                    if not output_dir.is_dir() or output_dir.is_symlink():
                        raise OSError(f"Refusing to move non-directory or symlink output: {output_dir}")
                    backup = Path(
                        tempfile.mkdtemp(prefix=f"{output_dir.name}.no-checkpoint-", dir=output_dir.parent)
                    )
                    output_dir.rename(backup / "previous_run")
                    logging.warning("Previous output preserved in %s", backup / "previous_run")
            if not status[0]:
                output_dir.mkdir(parents=True, exist_ok=False)
        except OSError as error:
            status[1] = str(error)
    dist.broadcast_object_list(status, src=0)
    if status[1] is not None:
        raise RuntimeError(f"Cannot prepare output directory: {status[1]}")
    return status[0]


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


def wrap_pi05(policy, device, strategy="layer", master_dtype="float32", attention_backend="eager"):
    policy.float()
    if master_dtype == "bfloat16":
        policy.model.bfloat16()
    normalizers = (policy.normalize_inputs, policy.normalize_targets, policy.unnormalize_outputs)
    normalizer_parameters = []
    for normalizer in normalizers:
        # Normalize stores statistics as frozen Parameters, not registered buffers.
        normalizer.to(device=device, dtype=torch.float32)
        normalizer_parameters.extend(normalizer.parameters())
    wrap_policy = None
    if strategy == "layer":
        from lerobot.common.policies.pi05.paligemma_with_expert import JointAttentionBlock

        joint = policy.model.paligemma_with_expert
        joint.enable_joint_blocks()
        for block in joint.joint_blocks:
            block.gradient_checkpointing = policy.config.gradient_checkpointing
            block.attention_backend = attention_backend
        # Prefix weights are called through these modules' forwards, not the decoder layers.
        wrap_policy = ModuleWrapPolicy(
            {JointAttentionBlock, type(joint.paligemma.vision_tower), torch.nn.Embedding}
        )
    elif strategy != "root":
        raise ValueError(f"Unknown PI05 FSDP strategy: {strategy}")
    return FullyShardedDataParallel(
        policy,
        auto_wrap_policy=wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=getattr(torch, master_dtype),
            buffer_dtype=None,
        ),
        use_orig_params=True,
        device_id=device,
        limit_all_gathers=True,
        ignored_states=normalizer_parameters,
        backward_prefetch=None if strategy == "layer" else BackwardPrefetch.BACKWARD_PRE,
    )


def make_optimizer(config, model, name):
    if name == "adamw":
        return config.get_optimizer_preset().build(model.parameters())
    if name != "adamw8bit":
        raise ValueError(f"Unknown PI05 optimizer: {name}")
    return AdamW8bitConfig(
        lr=config.optimizer_lr,
        betas=config.optimizer_betas,
        eps=config.optimizer_eps,
        weight_decay=config.optimizer_weight_decay,
        grad_clip_norm=config.optimizer_grad_clip_norm,
    ).build(model.parameters())


def check_resume_geometry(output_dir, geometry):
    checkpoint = resolve_latest_fsdp_checkpoint(output_dir)
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    saved = metadata["training_geometry"]
    previous, current = dict(saved), dict(geometry)
    for key, default in (
        ("master_dtype", "float32"),
        ("optimizer", "adamw"),
        ("attention_backend", "eager"),
    ):
        previous.setdefault(key, default)
        current.setdefault(key, default)
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


def canonical_pi05_state_dict(state):
    canonical = {}
    for name, value in state.items():
        name = name.replace("_fsdp_wrapped_module.", "")
        prefix, separator, suffix = name.partition(".joint_blocks.")
        if separator:
            layer, branch, parameter = suffix.split(".", 2)
            if branch == "language_layer":
                name = f"{prefix}.paligemma.model.language_model.layers.{layer}.{parameter}"
            elif branch == "expert_layer":
                name = f"{prefix}.gemma_expert.model.layers.{layer}.{parameter}"
            else:
                raise ValueError(f"Unknown joint block branch: {branch}")
        if name in canonical:
            raise ValueError(f"Duplicate exported PI05 parameter: {name}")
        canonical[name] = value
    return canonical


def export_policy(model, policy, output_dir, step):
    destination = output_dir / f"step{step}"
    temporary = output_dir / f".step{step}.tmp"
    if destination.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite export {destination} or {temporary}.")
    # Gather one FSDP unit at a time instead of simultaneously unsharding the whole model.
    with FullyShardedDataParallel.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    ):
        state = model.state_dict()
    if dist.get_rank() == 0:
        temporary.mkdir()
        policy.config.save_pretrained(temporary)
        save_file(canonical_pi05_state_dict(state), temporary / "model.safetensors")
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
        "wrap_strategy": (
            "pi05_joint_blocks_full_shard_fp32_master_bf16_compute_fp32_normalizers_sync_each_microbatch_v1"
            if args.fsdp_wrap_strategy == "layer"
            else "pi05_root_full_shard_fp32_master_bf16_compute_fp32_normalizers"
        ),
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
        "master_dtype": args.master_dtype,
        "optimizer": args.optimizer,
        "attention_backend": args.attention_backend,
    }
    if args.optimizer == "adamw8bit":
        import bitsandbytes

        geometry["bitsandbytes_version"] = bitsandbytes.__version__
    args.resume = prepare_output_directory(args.output_dir, args.resume, rank)
    if args.resume:
        geometry = check_resume_geometry(args.output_dir, geometry)
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
    model = wrap_pi05(policy, device, args.fsdp_wrap_strategy, args.master_dtype, args.attention_backend)
    model.train()
    optimizer = make_optimizer(config, model, args.optimizer)
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
    setup_peak = torch.tensor(torch.cuda.max_memory_allocated(device), device=device)
    dist.all_reduce(setup_peak, op=dist.ReduceOp.MAX)
    torch.cuda.reset_peak_memory_stats(device)
    with (
        wandb_logging(args, contract if rank == 0 else {}, rank=rank) as run,
        (
            (args.output_dir / "training.jsonl").open("a" if args.resume else "x")
            if rank == 0
            else nullcontext()
        ) as log,
    ):
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
                synchronize = (
                    args.fsdp_wrap_strategy == "layer" or accumulation == args.grad_accumulation_steps - 1
                )
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
            memory_peak = torch.tensor(torch.cuda.max_memory_allocated(device), device=device)
            dist.all_reduce(memory_peak, op=dist.ReduceOp.MAX)
            row = {
                "step": update_step,
                "micro_step": micro_step,
                "loss": total_loss.item() / world_size,
                "grad_norm": grad_norm.item(),
                "lr": lr,
                "epoch": epoch,
                "batch_in_epoch": batch_in_epoch,
                "peak_cuda_allocated_bytes_per_rank_max": memory_peak.item(),
            }
            if log is not None:
                log.write(json.dumps(row) + "\n")
                log.flush()
                logging.info("%s", row)
                if run is not None:
                    run.log({f"train/{key}": value for key, value in row.items()})
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
        before_export_peak = torch.tensor(torch.cuda.max_memory_allocated(device), device=device)
        dist.all_reduce(before_export_peak, op=dist.ReduceOp.MAX)
        if args.export_final:
            export_policy(model, policy, args.output_dir, update_step)
        peak = torch.tensor(torch.cuda.max_memory_allocated(device), device=device)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        reserved_peak = torch.tensor(torch.cuda.max_memory_reserved(device), device=device)
        dist.all_reduce(reserved_peak, op=dist.ReduceOp.MAX)
        if rank == 0:
            summary = {
                "completed": update_step == args.steps,
                "stopped_at_update": update_step,
                "target_steps": args.steps,
                "final_loss": row["loss"],
                "world_size": world_size,
                "fsdp_wrap_strategy": args.fsdp_wrap_strategy,
                "master_dtype": args.master_dtype,
                "optimizer": args.optimizer,
                "attention_backend": args.attention_backend,
                "fsdp_units": len(FullyShardedDataParallel.fsdp_modules(model)),
                "peak_cuda_allocated_bytes_before_export_per_rank_max": before_export_peak.item(),
                "peak_cuda_allocated_bytes_per_rank_max": peak.item(),
                "peak_cuda_reserved_bytes_per_rank_max": reserved_peak.item(),
                "setup_peak_cuda_allocated_bytes_per_rank_max": setup_peak.item(),
            }
            (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            if run is not None:
                run.summary.update(summary)


def main():
    args = distributed_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PI05 FSDP training requires CUDA.")
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    visible_devices = torch.cuda.device_count()
    if not 0 <= local_rank < visible_devices or local_world_size > visible_devices:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank}, LOCAL_WORLD_SIZE={local_world_size}, but PyTorch sees "
            f"{visible_devices} CUDA devices (CUDA_VISIBLE_DEVICES="
            f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}). "
            "Set NPROC_PER_NODE to at most the number of visible GPUs per node. "
            "Preserve scheduler-provided GPU visibility; CUDA device indices are local to the visible set."
        )
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
