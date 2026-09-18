#!/usr/bin/env python
"""DeepSpeed entry point for perception <-> physical contrastive pre-training.

This is the counterpart of ``dps_train_ace.py`` for the ``robo_contrast`` policy. It is a
separate script because the data path is fundamentally different: batches are built by a
``ContrastiveBatchSampler`` (which decides *which* negatives share a batch) instead of a
plain ``DistributedSampler``, and the batch carries images, language, canonical
state/action and tactile all at once.
"""

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any

import deepspeed
import torch
from termcolor import colored
from torch import distributed as dist
from torch.utils.data import DataLoader

from lerobot.common.datasets.contrastive_dataset import (
    MultiModalContrastiveDataset,
    contrastive_collate_fn,
)
from lerobot.common.datasets.contrastive_eval import build_eval_loaders, evaluate
from lerobot.common.datasets.contrastive_sampler import ContrastiveBatchSampler
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.deepspeed_checkpoint import (
    DATA_PARALLEL_WORLD_SIZE_KEY,
    OPTIMIZER_GROUP_SIGNATURE_KEY,
    align_fresh_scheduler_to_step,
    load_checkpoint_with_optimizer_fallback,
    optimizer_group_signature,
)
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import format_big_number
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig

# Images stay uint8 all the way to the GPU (4x less PCIe traffic than bf16) and are
# normalised inside the model; index-like tensors must stay integral.
_KEEP_DTYPE_KEYS = ("image_t0", "image_t1", "tactile_image")
_EXTRA_TRAINING_EPOCHS = 100


@dataclass(frozen=True)
class EpochSchedule:
    steps_per_epoch: int
    global_samples_per_step: int
    samples_per_epoch: int
    source_equivalent_epochs: int
    extra_epochs: int
    total_epochs: int
    source_equivalent_steps: int
    total_steps: int


def _compute_epoch_schedule(
    total_source_frames: int,
    steps_per_epoch: int,
    global_samples_per_step: int,
    extra_epochs: int = _EXTRA_TRAINING_EPOCHS,
) -> EpochSchedule:
    if total_source_frames <= 0:
        raise ValueError("total_source_frames must be positive.")
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive.")
    if global_samples_per_step <= 0:
        raise ValueError("global_samples_per_step must be positive.")
    if extra_epochs < 0:
        raise ValueError("extra_epochs must be non-negative.")

    samples_per_epoch = steps_per_epoch * global_samples_per_step
    source_equivalent_epochs = math.ceil(total_source_frames / samples_per_epoch)
    total_epochs = source_equivalent_epochs + extra_epochs
    return EpochSchedule(
        steps_per_epoch=steps_per_epoch,
        global_samples_per_step=global_samples_per_step,
        samples_per_epoch=samples_per_epoch,
        source_equivalent_epochs=source_equivalent_epochs,
        extra_epochs=extra_epochs,
        total_epochs=total_epochs,
        source_equivalent_steps=source_equivalent_epochs * steps_per_epoch,
        total_steps=total_epochs * steps_per_epoch,
    )


def init_logger(cfg, subdir: str = "contrast"):
    rank = int(os.environ.get("RANK", 0))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        f"[%(asctime)s] [rank: {rank}] [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handlers: list[logging.Handler] = []
    if rank == 0:
        log_path = Path(cfg.log_dir) / f"{subdir}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
        handlers.append(logging.StreamHandler())
    else:
        # The other ranks say the same things at the same time; keep only what signals
        # trouble, otherwise every message is repeated once per GPU.
        console = logging.StreamHandler()
        console.setLevel(logging.WARNING)
        handlers.append(console)
    for h in handlers:
        h.setFormatter(formatter)

    # The library modules log the things worth auditing -- resolved temporal windows,
    # skipped datasets, canonical-space fallbacks -- but only `__main__` was ever given a
    # handler, so those records fell through to logging's lastResort handler and anything
    # below WARNING was silently dropped. Configure the `lerobot` logger too, rather than
    # the root, which would pull in INFO spam from torch/deepspeed/PIL.
    lib_logger = logging.getLogger("lerobot")
    lib_logger.setLevel(logging.INFO)
    for target in (logger, lib_logger):
        for h in handlers:
            target.addHandler(h)
        # `cfg.validate()` reaches for the module-level `logging.warning`, which quietly runs
        # `basicConfig()` and leaves a handler on the root logger. Records propagating up
        # would then be emitted a second time in logging's default format. Terminate here:
        # these two loggers already have handlers of their own.
        target.propagate = False
    return logger


def _worker_init(_worker_id: int) -> None:
    """One thread per worker.

    Decoding and resizing are already parallelised across workers; letting each of them
    spawn its own intra-op thread pool oversubscribes the CPU and measurably *lowers*
    throughput once several ranks share the machine.
    """
    torch.set_num_threads(1)


def move_batch(batch: dict, device, dtype=torch.bfloat16) -> dict:
    """Move a batch to ``device``, casting *only* floating point tensors to ``dtype``."""
    out = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            out[key] = value
            continue
        value = value.to(device, non_blocking=True)
        if value.is_floating_point() and key not in _KEEP_DTYPE_KEYS:
            value = value.to(dtype)
        out[key] = value
    return out


def update_policy(model_engine, batch: Any, task_type: str, step: int):
    batch = move_batch(batch, model_engine.device)
    loss, output_dict = model_engine(batch, task_type=task_type, step=step)
    model_engine.backward(loss)
    model_engine.step()
    return loss, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()

    os.environ.setdefault("DECORD_LOG_LEVEL", "error")
    deepspeed.init_distributed()
    logger = init_logger(cfg)

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if rank == 0:
        logger.info(pformat(cfg.to_dict()))
        wandb_logger = WandBLogger(cfg) if (cfg.wandb.enable and cfg.wandb.project) else None
        if wandb_logger is None:
            logger.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
    else:
        wandb_logger = None

    if cfg.seed is not None:
        set_seed(cfg.seed + rank)

    with open(cfg.deepspeed) as f:
        deepspeed_configs_in_dict = json.load(f)
    batch_size = deepspeed_configs_in_dict["train_micro_batch_size_per_gpu"]

    # ------------------------------------------------------------------ data
    # The seed is shared across ranks: every process reconstructs the same virtual global
    # batch before deterministically taking its balanced local shard.
    dataset = MultiModalContrastiveDataset(
        cfg=cfg,
        data_mix=cfg.data_mix,
        seed=cfg.seed,
        dataset_size_one_epoch=cfg.dataset.dataset_size_one_epoch,
    )
    logger.info(f"Dataset: {dataset}")
    if not dataset.has_physical.any():
        if (
            cfg.policy.perception_recon_weight <= 0
            or cfg.policy.num_predictor_layers <= 0
        ):
            raise ValueError(
                "The selected mixture contains only video, so train_contrastive needs "
                "perception_recon_weight > 0 and num_predictor_layers > 0."
            )
        cfg.policy.perception_only = True
        logger.warning(
            "The mixture contains no physical supervision. PhysicalEncoder will not be "
            "built; train_contrastive will optimize only the visual-change latent "
            "reconstruction objective."
        )

    sampler = ContrastiveBatchSampler(
        episode_ranges=dataset.episode_ranges,
        sample_weights=dataset.sample_weights,
        batch_size=batch_size,
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
        balance_across_ranks=True,
    )
    epoch_schedule = _compute_epoch_schedule(
        total_source_frames=dataset.total_source_frames,
        steps_per_epoch=len(sampler),
        global_samples_per_step=batch_size * world_size,
    )
    if rank == 0:
        logger.info(
            "Epoch schedule: source_frames=%s configured_samples_per_epoch=%s "
            "actual_samples_per_epoch=%s steps_per_epoch=%s "
            "source_equivalent_epochs=%s extra_epochs=%s total_epochs=%s "
            "source_equivalent_steps=%s total_steps=%s; cfg.steps=%s is ignored",
            format_big_number(dataset.total_source_frames),
            format_big_number(cfg.dataset.dataset_size_one_epoch),
            format_big_number(epoch_schedule.samples_per_epoch),
            format_big_number(epoch_schedule.steps_per_epoch),
            format_big_number(epoch_schedule.source_equivalent_epochs),
            format_big_number(epoch_schedule.extra_epochs),
            format_big_number(epoch_schedule.total_epochs),
            format_big_number(epoch_schedule.source_equivalent_steps),
            format_big_number(epoch_schedule.total_steps),
            format_big_number(cfg.steps),
        )

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

    # A fixed set of frames, identical in every run, scored periodically. In-batch retrieval
    # accuracy on random training batches is too noisy to compare runs with -- see
    # ``contrastive_eval`` for the measurement -- so this is the number to judge a change on.
    eval_loaders = build_eval_loaders(
        dataset=dataset,
        policy_cfg=cfg.policy,
        collate_fn=contrastive_collate_fn,
        num_workers=max(2, cfg.num_workers // 2),
        rank=rank,
        world_size=world_size,
    )

    # ------------------------------------------------------------------ policy
    logger.info("Creating policy...")
    policy = make_policy(
        cfg=cfg.policy,
        device="cpu",
        ds_meta=dataset.meta,
        weight_pt_path=cfg.policy.pretrained_path,
    )

    optimizer, lr_scheduler = make_optimizer_and_scheduler(
        cfg,
        policy,
        num_training_steps=epoch_schedule.total_steps,
    )
    optimizer_signature = optimizer_group_signature(policy, optimizer)

    # Cast weights to bf16 without changing the configured frozen/LoRA/full trainability.
    for params in policy.parameters():
        params.data = params.data.bfloat16()

    if rank == 0:
        num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        num_total_params = sum(p.numel() for p in policy.parameters())
        logger.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        logger.info(
            "Training epochs: %s source-equivalent + %s extra = %s; total steps: %s",
            format_big_number(epoch_schedule.source_equivalent_epochs),
            format_big_number(epoch_schedule.extra_epochs),
            format_big_number(epoch_schedule.total_epochs),
            format_big_number(epoch_schedule.total_steps),
        )
        logger.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logger.info(f"{dataset.num_episodes=}")
        logger.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logger.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=policy,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=cfg.deepspeed,
        model_parameters=[p for p in policy.parameters() if p.requires_grad],
    )
    logger.info(f"Training batch size: {model_engine.train_batch_size()}")
    logger.info(
        "Optimizer learning rates: %s",
        ", ".join(
            f"{group.get('group_name', f'group_{index}')}={group['lr']:.3e}"
            for index, group in enumerate(optimizer.param_groups)
        ),
    )

    step = 0
    cfg.output_dir = os.path.join(cfg.output_dir, cfg.job_name)
    if cfg.weight_resume:
        logger.info(f"Resuming training from {cfg.output_dir}")
        load_path, loaded_state, optimizer_restored = load_checkpoint_with_optimizer_fallback(
            model_engine,
            cfg.output_dir,
            optimizer_signature,
            world_size,
            lr_scheduler,
        )
        if load_path is not None and loaded_state is not None:
            # load_checkpoint returns every non-DeepSpeed-owned key it finds in the
            # checkpoint, which includes internal metadata such as
            # `checkpoint_parallel_dimensions` that newer DeepSpeed writes on save but
            # does not filter out on load. Feeding that dict back into save_checkpoint
            # raises "client_state contains reserved checkpoint key", so only pick out
            # the fields this script actually owns.
            step = loaded_state.get("step", 0)
            if not optimizer_restored:
                align_fresh_scheduler_to_step(
                    lr_scheduler,
                    step,
                    model_engine.gradient_accumulation_steps(),
                )
            logger.info(f"Resumed training from step {step}")

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "contrastive_loss": AverageMeter("contra_loss", ":.3f"),
        "recon_loss": AverageMeter("trecon", ":.4f"),
        "percep_recon_loss": AverageMeter("precon", ":.4f"),
        "retrieval_acc": AverageMeter("acc", ":.3f"),
        "physical_rows": AverageMeter("phys_n", ":.1f"),
        "video_only_rows": AverageMeter("video_n", ":.1f"),
        "tactile_hits": AverageMeter("tac_hit", ":.1f"),
        "tactile_rows": AverageMeter("tac_n", ":.1f"),
        "pos_sim": AverageMeter("pos_sim", ":.3f"),
        "logit_scale": AverageMeter("scale", ":.2f"),
        "tac_sig_gate": AverageMeter("tsig", ":.3f"),
        "tac_img_gate": AverageMeter("timg", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "vision_lr": AverageMeter("vlr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    train_tracker = MetricsTracker(
        model_engine.train_batch_size(),
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=int(step / model_engine.gradient_accumulation_steps()),
    )

    logger.info(f"Start training on {world_size} devices")
    fwd_bwd_time = 0.0
    dataloading_s = 0.0
    dist_step = 50
    start_epoch, start_batch = divmod(step, epoch_schedule.steps_per_epoch)
    if step:
        if start_epoch < epoch_schedule.total_epochs:
            logger.info(
                "Resume position: epoch %d/%d, batch %d/%d",
                start_epoch + 1,
                epoch_schedule.total_epochs,
                start_batch,
                epoch_schedule.steps_per_epoch,
            )
        else:
            logger.info(
                "Checkpoint step %s has already reached planned total step %s.",
                format_big_number(step),
                format_big_number(epoch_schedule.total_steps),
            )

    for epoch in range(start_epoch, epoch_schedule.total_epochs):
        epoch_start_batch = start_batch if epoch == start_epoch else 0
        logger.info(
            "Epoch %d/%d start at batch %d/%d (step %s/%s)",
            epoch + 1,
            epoch_schedule.total_epochs,
            epoch_start_batch,
            epoch_schedule.steps_per_epoch,
            format_big_number(step),
            format_big_number(epoch_schedule.total_steps),
        )
        sampler.set_epoch(epoch)
        sampler.set_start_batch(epoch_start_batch)
        dataset.set_epoch(epoch)
        batch_ready = time.perf_counter()
        for batch_offset, batch in enumerate(dataloader):
            batch_idx = epoch_start_batch + batch_offset
            dataloading_s += time.perf_counter() - batch_ready
            if step == 0:
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

            fwd_bwd_start = time.perf_counter()
            loss, output_dict = update_policy(model_engine, batch, cfg.task_type, step=step)
            step += 1
            fwd_bwd_time += time.perf_counter() - fwd_bwd_start

            if model_engine.is_gradient_accumulation_boundary():
                train_tracker.dataloading_s = dataloading_s
                train_tracker.update_s = fwd_bwd_time
                train_tracker.loss = loss.detach().mean().item()
                train_tracker.recon_loss = output_dict.get("recon_loss", 0.0)
                train_tracker.percep_recon_loss = output_dict.get("percep_recon_loss", 0.0)
                physical_rows = output_dict.get("physical_rows", 0.0)
                if physical_rows > 0:
                    train_tracker.contrastive_loss.update(
                        output_dict.get("contrastive_loss", 0.0),
                        n=physical_rows,
                    )
                    train_tracker.retrieval_acc.update(
                        output_dict.get("retrieval_acc", 0.0),
                        n=physical_rows,
                    )
                    train_tracker.pos_sim.update(
                        output_dict.get("pos_sim", 0.0),
                        n=physical_rows,
                    )
                train_tracker.physical_rows = physical_rows
                train_tracker.video_only_rows = output_dict.get("video_only_rows", 0.0)
                train_tracker.tactile_hits = output_dict.get("tactile_hits", 0.0)
                train_tracker.tactile_rows = output_dict.get("tactile_rows", 0.0)
                train_tracker.logit_scale = output_dict.get("logit_scale", 0.0)
                train_tracker.tac_sig_gate = output_dict.get("tactile_sig_gate", 0.0)
                train_tracker.tac_img_gate = output_dict.get("tactile_img_gate", 0.0)
                train_tracker.lr = optimizer.param_groups[0]["lr"]
                train_tracker.vision_lr = next(
                    (
                        group["lr"]
                        for group in optimizer.param_groups
                        if group.get("group_name") == "vision"
                    ),
                    0.0,
                )
                train_tracker.step()
                fwd_bwd_time = 0.0
                dataloading_s = 0.0

            if eval_loaders and cfg.eval_freq > 0 and step % cfg.eval_freq == 0:
                eval_start = time.perf_counter()
                eval_metrics = evaluate(model_engine, eval_loaders, move_batch)
                if rank == 0:
                    summary = " ".join(
                        f"{k.split('/')[-1]}:{v:.4f}"
                        for k, v in eval_metrics.items()
                        if not k.endswith("_rows")
                    )
                    logger.info(
                        f"eval step:{step} {summary} took:{time.perf_counter() - eval_start:.1f}s"
                    )
                    if wandb_logger:
                        wandb_logger.log_dict(eval_metrics, step)

            is_last_batch = (
                epoch + 1 == epoch_schedule.total_epochs
                and batch_idx + 1 == epoch_schedule.steps_per_epoch
            )
            if cfg.save_checkpoint and (step % cfg.save_freq == 0 or is_last_batch):
                logger.info(f"Checkpoint policy after step {step}")
                os.makedirs(cfg.output_dir, exist_ok=True)
                model_engine.save_checkpoint(
                    save_dir=cfg.output_dir,
                    client_state={
                        "step": step,
                        "epoch": epoch,
                        "batch_in_epoch": batch_idx + 1,
                        OPTIMIZER_GROUP_SIGNATURE_KEY: optimizer_signature,
                        DATA_PARALLEL_WORLD_SIZE_KEY: world_size,
                    },
                )

            if rank == 0 and cfg.log_freq > 0 and step % cfg.log_freq == 0:
                logger.info(
                    "epoch:%d/%d %s",
                    epoch + 1,
                    epoch_schedule.total_epochs,
                    train_tracker,
                )
                if wandb_logger:
                    wandb_log_dict = train_tracker.to_dict()
                    wandb_log_dict["sampler_epoch/current"] = epoch + 1
                    wandb_log_dict["sampler_epoch/total"] = epoch_schedule.total_epochs
                    if output_dict:
                        for key, value in output_dict.items():
                            wandb_log_dict.setdefault(key, value)
                    wandb_logger.log_dict(wandb_log_dict, step)
                train_tracker.reset_averages()

            if step % dist_step == 0 and dist.is_initialized():
                dist.barrier(device_ids=[model_engine.local_rank])

            batch_ready = time.perf_counter()

    logger.info("Training finished")


if __name__ == "__main__":
    train()
