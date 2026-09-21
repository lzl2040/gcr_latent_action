#!/usr/bin/env python
"""FSDP training entry point for the Qwen3-VL stage-two world model."""

import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat

import torch
from termcolor import colored
from torch import distributed as dist
from torch.utils.data import DataLoader

from lerobot.common.datasets.contrastive_dataset import (
    MultiModalContrastiveDataset,
    contrastive_collate_fn,
)
from lerobot.common.datasets.contrastive_sampler import ContrastiveBatchSampler
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.optim.optimizers import AdamW8bitConfig
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.fsdp_training import (
    FSDPTrainingConfig,
    convert_policy_to_fp8,
    load_fsdp_checkpoint,
    save_fsdp_checkpoint,
    wrap_policy_with_fsdp,
)
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import format_big_number
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.dps_train_contrast import (
    _compute_epoch_schedule,
    _data_read_batch_counts,
    _load_stage2_resume_policy_config,
    _worker_init,
    init_logger,
    move_batch,
)


@dataclass
class FSDPTrainPipelineConfig(TrainPipelineConfig):
    fsdp: FSDPTrainingConfig = field(default_factory=FSDPTrainingConfig)


def _initialize_distributed() -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-two FSDP training requires CUDA.")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def _stage2_metrics(cfg) -> dict[str, AverageMeter]:
    return {
        "loss": AverageMeter("loss", ":.3f"),
        "latent_action_loss": AverageMeter("latent", ":.4f"),
        "video_flow_loss": AverageMeter("video", ":.4f"),
        "action_flow_loss": AverageMeter("action", ":.4f"),
        "state_flow_loss": AverageMeter("state", ":.4f"),
        "tactile_flow_loss": AverageMeter("tactile", ":.4f"),
        "valid_rows": AverageMeter("valid", ":.1f"),
        **{
            f"task_{name}": AverageMeter(name, ":.2f")
            for name in cfg.policy.task_names
        },
        "lr": AverageMeter("lr", ":0.1e"),
        "understanding_lr": AverageMeter("ulr", ":0.1e"),
        "physical_lr": AverageMeter("plr", ":0.1e"),
        "grad_norm": AverageMeter("grad", ":.2f"),
        "memory_gib": AverageMeter("mem", ":.1f"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }


def _group_lr(optimizer: torch.optim.Optimizer, group_name: str) -> float:
    return next(
        (
            group["lr"]
            for group in optimizer.param_groups
            if group.get("group_name") == group_name
        ),
        0.0,
    )


def _normalize_resume_position(
    epoch: int,
    batch_in_epoch: int,
    steps_per_epoch: int,
) -> tuple[int, int]:
    epoch += batch_in_epoch // steps_per_epoch
    batch_in_epoch %= steps_per_epoch
    return epoch, batch_in_epoch


def _train(cfg: FSDPTrainPipelineConfig) -> None:
    cfg.validate()
    _load_stage2_resume_policy_config(cfg)
    cfg.fsdp.validate()
    if cfg.policy.type != "qwen3vl_mot":
        raise ValueError(
            "fsdp_train_contrast.py is stage-two-specific and requires "
            "`policy.type=qwen3vl_mot`."
        )
    if not isinstance(cfg.optimizer, AdamW8bitConfig):
        raise ValueError(
            "Stage-two FSDP currently requires optimizer.type=adamw_8bit because its "
            "checkpoint format stores bitsandbytes states rank-locally."
        )
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if cfg.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    if cfg.save_checkpoint and cfg.save_freq <= 0:
        raise ValueError("save_freq must be positive when checkpoint saving is enabled.")

    os.environ.setdefault("DECORD_LOG_LEVEL", "error")
    rank, local_rank, world_size, device = _initialize_distributed()
    logger = init_logger(cfg, subdir="qwen3vl_mot_fsdp")

    if rank == 0:
        logger.info(pformat(cfg.to_dict()))
        wandb_logger = WandBLogger(cfg) if (cfg.wandb.enable and cfg.wandb.project) else None
        if wandb_logger is None:
            logger.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
    else:
        wandb_logger = None

    if cfg.seed is not None:
        set_seed(cfg.seed + rank)

    dataset = MultiModalContrastiveDataset(
        cfg=cfg,
        data_mix=cfg.data_mix,
        seed=cfg.seed,
        dataset_size_one_epoch=cfg.dataset.dataset_size_one_epoch,
    )
    sampler = ContrastiveBatchSampler(
        episode_ranges=dataset.episode_ranges,
        sample_weights=dataset.sample_weights,
        batch_size=cfg.batch_size,
        num_replicas=world_size,
        rank=rank,
        seed=cfg.seed,
        samples_per_epoch=cfg.dataset.dataset_size_one_epoch,
        horizon=dataset.frame_horizons,
        same_dataset_frac=cfg.policy.same_dataset_frac,
        episode_group_frac=cfg.policy.episode_group_frac,
        episode_group_size=cfg.policy.episode_group_size,
        min_frame_gap=cfg.policy.min_frame_gap,
        sample_costs=dataset.sample_costs,
        dataset_has_physical=dataset.has_physical,
        min_physical_per_batch=0,
        balance_across_ranks=True,
    )
    epoch_schedule = _compute_epoch_schedule(
        total_source_frames=dataset.total_source_frames,
        steps_per_epoch=len(sampler),
        global_samples_per_step=cfg.batch_size * world_size,
    )
    total_update_steps = math.ceil(
        epoch_schedule.total_steps / cfg.gradient_accumulation_steps
    )
    training_geometry = {
        "batch_size_per_rank": cfg.batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "data_mix": cfg.data_mix,
        "dataset_names": list(dataset.dataset_names),
        "dataset_size_one_epoch": cfg.dataset.dataset_size_one_epoch,
        "total_source_frames": dataset.total_source_frames,
        "steps_per_epoch": epoch_schedule.steps_per_epoch,
        "total_epochs": epoch_schedule.total_epochs,
        "total_micro_steps": epoch_schedule.total_steps,
        "seed": cfg.seed,
    }
    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=cfg.num_workers,
        worker_init_fn=_worker_init,
        pin_memory=True,
        collate_fn=contrastive_collate_fn,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=4 if cfg.num_workers > 0 else None,
    )

    if rank == 0:
        logger.info("Dataset: %s", dataset)
        logger.info(
            "Epoch schedule: source_frames=%s configured_samples_per_epoch=%s "
            "actual_samples_per_epoch=%s micro_steps_per_epoch=%s total_epochs=%s "
            "total_micro_steps=%s optimizer_steps=%s",
            format_big_number(dataset.total_source_frames),
            format_big_number(cfg.dataset.dataset_size_one_epoch),
            format_big_number(epoch_schedule.samples_per_epoch),
            format_big_number(epoch_schedule.steps_per_epoch),
            format_big_number(epoch_schedule.total_epochs),
            format_big_number(epoch_schedule.total_steps),
            format_big_number(total_update_steps),
        )

    # All ranks construct identical random Generation weights before FSDP shards them.
    model_seed = cfg.seed if cfg.seed is not None else 0
    set_seed(model_seed)
    logger.info("Creating stage-two policy...")
    policy = make_policy(
        cfg=cfg.policy,
        device="cpu",
        ds_meta=dataset.meta,
        weight_pt_path=cfg.policy.pretrained_path,
    )
    policy.to(device=device, dtype=torch.bfloat16)

    output_path = Path(cfg.output_dir)
    if rank == 0:
        output_path.mkdir(parents=True, exist_ok=True)
        cfg.policy._save_pretrained(output_path)
        cfg._save_pretrained(output_path)
    dist.barrier(device_ids=[local_rank])

    converted_fp8 = convert_policy_to_fp8(policy, cfg.fsdp, device)
    model, wrap_summary = wrap_policy_with_fsdp(policy, cfg.fsdp, device)
    optimizer, lr_scheduler = make_optimizer_and_scheduler(
        cfg,
        model,
        num_training_steps=total_update_steps,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)

    set_seed(model_seed + rank)

    if rank == 0:
        num_total_params = sum(parameter.numel() for parameter in model.parameters())
        logger.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {output_path}")
        logger.info(
            "FSDP full-shard: trainable=%s replicated_frozen=%s wrapped_blocks=%d",
            format_big_number(wrap_summary.trainable_parameters),
            format_big_number(wrap_summary.replicated_frozen_parameters),
            len(wrap_summary.wrapped_module_names),
        )
        logger.info(
            "FP8 Linear: enabled=%s scope=%s recipe=%s converted=%d; master "
            "parameters and optimizer-facing gradients remain BF16",
            cfg.fsdp.fp8,
            cfg.fsdp.fp8_scope,
            cfg.fsdp.fp8_recipe,
            len(converted_fp8),
        )
        logger.info(
            "Parameters visible after FSDP sharding on rank 0: %s",
            format_big_number(num_total_params),
        )
        logger.info(
            "Batch: micro=%d/GPU world=%d accumulation=%d global=%d",
            cfg.batch_size,
            world_size,
            cfg.gradient_accumulation_steps,
            cfg.batch_size * world_size * cfg.gradient_accumulation_steps,
        )
        logger.info(
            "Optimizer learning rates: %s",
            ", ".join(
                f"{group.get('group_name', f'group_{index}')}={group['lr']:.3e}"
                for index, group in enumerate(optimizer.param_groups)
            ),
        )

    micro_step = 0
    update_step = 0
    start_epoch = 0
    start_batch = 0
    if cfg.weight_resume:
        resume = load_fsdp_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            output_dir=output_path,
            fsdp_config=cfg.fsdp,
            training_geometry=training_geometry,
            device=device,
        )
        micro_step = resume.micro_step
        update_step = resume.update_step
        start_epoch, start_batch = _normalize_resume_position(
            resume.epoch,
            resume.batch_in_epoch,
            epoch_schedule.steps_per_epoch,
        )
        expected_micro_step = start_epoch * epoch_schedule.steps_per_epoch + start_batch
        if micro_step != expected_micro_step:
            raise ValueError(
                "Checkpoint data position is inconsistent: "
                f"micro_step={micro_step}, position implies {expected_micro_step}."
            )
        logger.info(
            "Resumed %s at optimizer step %d, micro step %d, epoch %d batch %d",
            resume.checkpoint_dir,
            update_step,
            micro_step,
            start_epoch,
            start_batch,
        )

    train_tracker = MetricsTracker(
        cfg.batch_size * world_size * cfg.gradient_accumulation_steps,
        dataset.num_frames,
        dataset.num_episodes,
        _stage2_metrics(cfg),
        initial_step=update_step,
    )
    interval_read_fallbacks = torch.zeros(len(dataset.dataset_names), dtype=torch.long)
    interval_read_samples = torch.zeros(len(dataset.dataset_names), dtype=torch.long)
    total_read_fallbacks = 0
    total_read_samples = 0
    fwd_bwd_time = 0.0
    dataloading_s = 0.0

    logger.info("Start FSDP training on %d devices", world_size)
    for epoch in range(start_epoch, epoch_schedule.total_epochs):
        epoch_start_batch = start_batch if epoch == start_epoch else 0
        sampler.set_epoch(epoch)
        sampler.set_start_batch(epoch_start_batch)
        dataset.set_epoch(epoch)
        batch_ready = time.perf_counter()

        for batch_offset, batch in enumerate(dataloader):
            batch_idx = epoch_start_batch + batch_offset
            dataloading_s += time.perf_counter() - batch_ready
            batch_fallbacks, batch_samples = _data_read_batch_counts(
                batch,
                len(dataset.dataset_names),
            )
            interval_read_fallbacks += batch_fallbacks
            interval_read_samples += batch_samples

            if micro_step == 0:
                dataset_ids, dataset_counts = torch.unique(
                    batch["dataset_id"],
                    return_counts=True,
                )
                mix = ", ".join(
                    f"{dataset.dataset_names[int(dataset_id)]}:{int(count)}"
                    for dataset_id, count in zip(dataset_ids, dataset_counts, strict=True)
                )
                logger.warning(
                    "First batch rank=%d physical=%d video_only=%d tactile_pads=%d "
                    "datasets={%s}",
                    rank,
                    int(batch["has_physical"].sum().item()),
                    int((batch["has_physical"] < 0.5).sum().item()),
                    int(batch["tactile_image_mask"].sum().item()),
                    mix,
                )

            batch = move_batch(batch, device)
            cycle_position = micro_step % cfg.gradient_accumulation_steps
            cycle_start = micro_step - cycle_position
            accumulation_target = min(
                cfg.gradient_accumulation_steps,
                epoch_schedule.total_steps - cycle_start,
            )
            should_update = cycle_position + 1 == accumulation_target
            sync_context = nullcontext() if should_update else model.no_sync()
            fwd_bwd_start = time.perf_counter()
            task_type = (
                "train_stage2"
                if cfg.task_type == "train_contrastive"
                else cfg.task_type
            )
            with sync_context:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, output_dict = model(
                        batch,
                        task_type=task_type,
                        step=update_step,
                    )
                (loss / accumulation_target).backward()
            micro_step += 1
            fwd_bwd_time += time.perf_counter() - fwd_bwd_start

            if should_update:
                grad_norm = model.clip_grad_norm_(cfg.optimizer.grad_clip_norm)
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1

                reduced_loss = loss.detach().float()
                dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
                reduced_loss /= world_size
                train_tracker.loss = reduced_loss.item()
                train_tracker.lr = optimizer.param_groups[0]["lr"]
                train_tracker.understanding_lr = _group_lr(optimizer, "understanding")
                train_tracker.physical_lr = _group_lr(optimizer, "physical")
                train_tracker.grad_norm = float(grad_norm)
                train_tracker.memory_gib = torch.cuda.max_memory_allocated(device) / 2**30
                train_tracker.update_s = fwd_bwd_time
                train_tracker.dataloading_s = dataloading_s
                for key in train_tracker.metrics:
                    if key in output_dict:
                        setattr(train_tracker, key, output_dict[key])
                train_tracker.step()
                fwd_bwd_time = 0.0
                dataloading_s = 0.0

            is_last_batch = (
                epoch + 1 == epoch_schedule.total_epochs
                and batch_idx + 1 == epoch_schedule.steps_per_epoch
            )
            should_save = should_update and cfg.save_checkpoint and (
                update_step % cfg.save_freq == 0 or is_last_batch
            )
            if should_save:
                logger.info("Saving FSDP checkpoint after optimizer step %d", update_step)
                save_fsdp_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    output_dir=output_path,
                    fsdp_config=cfg.fsdp,
                    training_geometry=training_geometry,
                    micro_step=micro_step,
                    update_step=update_step,
                    epoch=epoch,
                    batch_in_epoch=batch_idx + 1,
                    device=device,
                )

            should_log = (
                should_update
                and cfg.log_freq > 0
                and update_step % cfg.log_freq == 0
            )
            read_stats = None
            if should_log:
                read_stats = torch.cat(
                    [interval_read_fallbacks, interval_read_samples]
                ).to(device)
                dist.all_reduce(read_stats, op=dist.ReduceOp.SUM)
                interval_read_fallbacks.zero_()
                interval_read_samples.zero_()

            if rank == 0 and should_log:
                num_datasets = len(dataset.dataset_names)
                global_fallbacks = read_stats[:num_datasets].cpu()
                global_samples = read_stats[num_datasets:].cpu()
                interval_fallback_count = int(global_fallbacks.sum().item())
                interval_sample_count = int(global_samples.sum().item())
                interval_fallback_rate = interval_fallback_count / max(
                    1,
                    interval_sample_count,
                )
                total_read_fallbacks += interval_fallback_count
                total_read_samples += interval_sample_count
                total_fallback_rate = total_read_fallbacks / max(1, total_read_samples)

                logger.info(
                    "epoch:%d/%d micro_step:%d/%d %s",
                    epoch + 1,
                    epoch_schedule.total_epochs,
                    micro_step,
                    epoch_schedule.total_steps,
                    train_tracker,
                )
                failed_datasets = {
                    dataset.dataset_names[index]: int(global_fallbacks[index].item())
                    for index in range(num_datasets)
                    if global_fallbacks[index].item() > 0
                }
                if failed_datasets:
                    logger.warning(
                        "Data read fallbacks over the last %d samples: %d (%.4f%%), "
                        "total=%d; by_dataset=%s",
                        interval_sample_count,
                        interval_fallback_count,
                        100.0 * interval_fallback_rate,
                        total_read_fallbacks,
                        failed_datasets,
                    )
                if wandb_logger:
                    wandb_log_dict = train_tracker.to_dict()
                    wandb_log_dict["sampler_epoch/current"] = epoch + 1
                    wandb_log_dict["sampler_epoch/total"] = epoch_schedule.total_epochs
                    wandb_log_dict["micro_step"] = micro_step
                    wandb_log_dict["data/read_fallback_count_interval"] = (
                        interval_fallback_count
                    )
                    wandb_log_dict["data/read_fallback_rate_interval"] = (
                        interval_fallback_rate
                    )
                    wandb_log_dict["data/read_fallback_count_total"] = (
                        total_read_fallbacks
                    )
                    wandb_log_dict["data/read_fallback_rate_total"] = total_fallback_rate
                    for index, name in enumerate(dataset.dataset_names):
                        fallback_count = int(global_fallbacks[index].item())
                        if fallback_count == 0:
                            continue
                        sample_count = int(global_samples[index].item())
                        wandb_log_dict[
                            f"data/read_fallback_count_by_dataset/{name}"
                        ] = fallback_count
                        wandb_log_dict[
                            f"data/read_fallback_rate_by_dataset/{name}"
                        ] = fallback_count / max(1, sample_count)
                    for key, value in output_dict.items():
                        wandb_log_dict.setdefault(key, value)
                    wandb_logger.log_dict(wandb_log_dict, update_step)
                train_tracker.reset_averages()
                torch.cuda.reset_peak_memory_stats(device)

            batch_ready = time.perf_counter()

    logger.info("FSDP training finished at optimizer step %d", update_step)


@parser.wrap()
def train(cfg: FSDPTrainPipelineConfig) -> None:
    try:
        _train(cfg)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    train()
