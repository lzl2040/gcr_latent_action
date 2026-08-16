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
import os
import time
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
from lerobot.common.datasets.contrastive_sampler import ContrastiveBatchSampler
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import format_big_number
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig

# Images stay uint8 all the way to the GPU (4x less PCIe traffic than bf16) and are
# normalised inside the model; index-like tensors must stay integral.
_KEEP_DTYPE_KEYS = ("image_t0", "image_t1", "tactile_image")


def init_logger(cfg):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if int(os.environ.get("RANK", 0)) == 0:
        formatter = logging.Formatter(
            "[%(asctime)s] [rank: 0] [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        log_path = Path(cfg.log_dir) / f"contrast/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.addHandler(logging.StreamHandler())
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
    # The seed is shared across ranks: the sampler itself de-correlates ranks by offsetting
    # the batch id, so every rank must agree on the sampling plan.
    dataset = MultiModalContrastiveDataset(
        cfg=cfg,
        data_mix=cfg.data_mix,
        seed=cfg.seed,
        dataset_size_one_epoch=cfg.dataset.dataset_size_one_epoch,
    )
    logger.info(f"Dataset: {dataset}")

    sampler = ContrastiveBatchSampler(
        episode_ranges=dataset.episode_ranges,
        sample_weights=dataset.sample_weights,
        batch_size=batch_size,
        num_replicas=world_size,
        rank=rank,
        seed=cfg.seed,
        samples_per_epoch=cfg.dataset.dataset_size_one_epoch,
        horizon=max(cfg.policy.chunk_size, cfg.policy.frame_horizon),
        same_dataset_frac=cfg.policy.same_dataset_frac,
        episode_group_frac=cfg.policy.episode_group_frac,
        episode_group_size=cfg.policy.episode_group_size,
        min_frame_gap=cfg.policy.min_frame_gap,
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

    # ------------------------------------------------------------------ policy
    logger.info("Creating policy...")
    policy = make_policy(
        cfg=cfg.policy,
        device="cpu",
        ds_meta=dataset.meta,
        weight_pt_path=cfg.policy.pretrained_path,
    )

    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Cast weights to bf16 but never *unfreeze* anything: the SigLIP towers are frozen on
    # purpose, and re-enabling their gradients would blow up both memory and step time.
    for params in policy.parameters():
        params.data = params.data.bfloat16()

    if rank == 0:
        num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        num_total_params = sum(p.numel() for p in policy.parameters())
        logger.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        logger.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
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

    step = 0
    cfg.output_dir = os.path.join(cfg.output_dir, cfg.job_name)
    client_state = {"step": step}
    if cfg.weight_resume:
        logger.info(f"Resuming training from {cfg.output_dir}")
        load_path, loaded_state = model_engine.load_checkpoint(
            cfg.output_dir,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
            load_module_strict=False,
        )
        if load_path is not None and loaded_state is not None:
            client_state = loaded_state
            step = client_state.get("step", 0)
            logger.info(f"Resumed training from step {step}")

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "contra_loss": AverageMeter("contra_loss", ":.3f"),
        "retrieval_acc": AverageMeter("acc", ":.3f"),
        "pos_sim": AverageMeter("pos_sim", ":.3f"),
        "logit_scale": AverageMeter("scale", ":.2f"),
        "tac_sig_gate": AverageMeter("tsig", ":.3f"),
        "tac_img_gate": AverageMeter("timg", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
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
    start_epoch = step // max(1, len(sampler))

    for epoch in range(start_epoch, 100000):
        logger.info(f"Epoch {epoch} start...")
        sampler.set_epoch(epoch)
        dataset.set_epoch(epoch)
        batch_ready = time.perf_counter()
        for batch in dataloader:
            dataloading_s += time.perf_counter() - batch_ready

            fwd_bwd_start = time.perf_counter()
            loss, output_dict = update_policy(model_engine, batch, cfg.task_type, step=step)
            step += 1
            fwd_bwd_time += time.perf_counter() - fwd_bwd_start

            if model_engine.is_gradient_accumulation_boundary():
                train_tracker.dataloading_s = dataloading_s
                train_tracker.update_s = fwd_bwd_time
                train_tracker.loss = loss.detach().mean().item()
                train_tracker.contra_loss = output_dict.get("contrastive_loss", 0.0)
                train_tracker.retrieval_acc = output_dict.get("retrieval_acc", 0.0)
                train_tracker.pos_sim = output_dict.get("pos_sim", 0.0)
                train_tracker.logit_scale = output_dict.get("logit_scale", 0.0)
                train_tracker.tac_sig_gate = output_dict.get("tactile_sig_gate", 0.0)
                train_tracker.tac_img_gate = output_dict.get("tactile_img_gate", 0.0)
                train_tracker.lr = optimizer.param_groups[0]["lr"]
                train_tracker.step()
                fwd_bwd_time = 0.0
                dataloading_s = 0.0

            if cfg.save_checkpoint and (step % cfg.save_freq == 0 or step == cfg.steps):
                logger.info(f"Checkpoint policy after step {step}")
                os.makedirs(cfg.output_dir, exist_ok=True)
                client_state["step"] = step
                client_state["epoch"] = epoch
                model_engine.save_checkpoint(save_dir=cfg.output_dir, client_state=client_state)

            if rank == 0 and cfg.log_freq > 0 and step % cfg.log_freq == 0:
                logger.info(train_tracker)
                if wandb_logger:
                    wandb_log_dict = train_tracker.to_dict()
                    if output_dict:
                        wandb_log_dict.update(output_dict)
                    wandb_logger.log_dict(wandb_log_dict, step)
                train_tracker.reset_averages()

            if step % dist_step == 0 and dist.is_initialized():
                dist.barrier(device_ids=[model_engine.local_rank])

            if step >= cfg.steps:
                break
            batch_ready = time.perf_counter()

        if step >= cfg.steps:
            break

    logger.info("Training finished")


if __name__ == "__main__":
    train()
