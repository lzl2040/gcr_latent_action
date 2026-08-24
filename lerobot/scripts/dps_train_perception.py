#!/usr/bin/env python
"""DeepSpeed entry point for stage-1, vision-only pre-training of the perception branch.

Action-bearing data is a small fraction of the video that exists, and the perception branch
does not need it: its reconstruction objective -- predict the frame-``t+H`` patch features
from frame ``t``, the instruction and the change queries -- is self-supervised on any two
frames. So it is trained here on plain video first, and the physical branch is introduced
afterwards by ``dps_train_contrast.py``, which loads these weights and adds the contrastive
loss.

Differences from the contrastive script, all consequences of there being one branch:

* ``PerceptionVideoDataset`` reads two frames of one camera and nothing else.
* A plain ``DistributedSampler`` replaces ``ContrastiveBatchSampler``: with no contrastive
  loss there are no negatives to choose, so which samples share a batch does not matter.
* ``policy.perception_only`` skips building the physical branch entirely.

The metric to watch is ``qgain`` (``percep_query_gain``), not the loss. See
``PerceptionEncoder._recon_loss``: most of frame ``t+H`` is predictable from frame ``t``
alone, so the loss falls whether or not the change queries -- the only thing being
pre-trained for stage 2 -- carry any information. ``qgain`` is how much worse the
reconstruction gets when each sample is given another sample's queries. Near zero means the
run is not producing anything stage 2 can use, however healthy the loss curve looks.
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
from torch.utils.data import DataLoader, DistributedSampler

from lerobot.common.datasets.contrastive_dataset import contrastive_collate_fn
from lerobot.common.datasets.perception_dataset import PerceptionVideoDataset
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import format_big_number
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.dps_train_contrast import _worker_init, init_logger, move_batch


def update_policy(model_engine, batch: Any, step: int):
    batch = move_batch(batch, model_engine.device)
    loss, output_dict = model_engine(batch, task_type="train_perception", step=step)
    model_engine.backward(loss)
    model_engine.step()
    return loss, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()

    os.environ.setdefault("DECORD_LOG_LEVEL", "error")
    deepspeed.init_distributed()
    # Its own log subdirectory: the two stages are launched with the same `--log_dir`, and
    # timestamp-named files from both would otherwise interleave in one folder.
    logger = init_logger(cfg, subdir="perception")

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
        batch_size = json.load(f)["train_micro_batch_size_per_gpu"]

    # ------------------------------------------------------------------ data
    # The seed is shared across ranks so every rank builds the same sampling plan; the
    # DistributedSampler then gives each rank a disjoint slice of it.
    dataset = PerceptionVideoDataset(
        cfg=cfg,
        data_mix=cfg.data_mix,
        seed=cfg.seed,
        dataset_size_one_epoch=cfg.dataset.dataset_size_one_epoch,
        camera_mode=getattr(cfg.policy, "perception_camera_mode", "primary"),
    )

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.seed, drop_last=True
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        worker_init_fn=_worker_init,
        pin_memory=True,
        collate_fn=contrastive_collate_fn,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=4 if cfg.num_workers > 0 else None,
    )

    # ------------------------------------------------------------------ policy
    logger.info("Creating policy...")
    if not cfg.policy.perception_only:
        # Not fatal, but it silently doubles memory and hands the optimizer a physical branch
        # that receives no gradient on any rank, so it is worth saying out loud.
        logger.warning(
            "policy.perception_only is False: the physical branch will be built and never "
            "trained. Pass --policy.perception_only=true for stage-1 pre-training."
        )
    if cfg.policy.perception_recon_weight <= 0:
        raise ValueError(
            "Stage-1 pre-training optimises the perception reconstruction loss, but "
            "perception_recon_weight is 0, which means the predictor head is not built. "
            "There would be nothing to train."
        )

    policy = make_policy(
        cfg=cfg.policy,
        device="cpu",
        ds_meta=dataset.meta,
        weight_pt_path=cfg.policy.pretrained_path,
    )
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Cast to bf16 without unfreezing anything: the backbones are frozen on purpose.
    for params in policy.parameters():
        params.data = params.data.bfloat16()

    if rank == 0:
        num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        num_total_params = sum(p.numel() for p in policy.parameters())
        logger.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        logger.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logger.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
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
    if cfg.weight_resume:
        logger.info(f"Resuming training from {cfg.output_dir}")
        load_path, loaded_state = model_engine.load_checkpoint(
            cfg.output_dir,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
            load_module_strict=False,
        )
        if load_path is not None and loaded_state is not None:
            # Only the fields this script owns: load_checkpoint also returns DeepSpeed's own
            # metadata (e.g. `checkpoint_parallel_dimensions`), and feeding that back into
            # save_checkpoint raises "client_state contains reserved checkpoint key".
            step = loaded_state.get("step", 0)
            logger.info(f"Resumed training from step {step}")

    train_metrics = {
        "loss": AverageMeter("loss", ":.4f"),
        "query_gain": AverageMeter("qgain", ":.4f"),
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

    logger.info(f"Start perception pre-training on {world_size} devices")
    fwd_bwd_time = 0.0
    dataloading_s = 0.0
    dist_step = 50
    steps_per_epoch = max(1, len(dataloader))
    start_epoch = step // steps_per_epoch
    # `qgain` is probed every `query_probe_freq` steps, so the meter would otherwise average a
    # real value against zeros from every other step and report a number that is not the gain.
    last_query_gain = 0.0

    for epoch in range(start_epoch, 100000):
        logger.info(f"Epoch {epoch} start...")
        sampler.set_epoch(epoch)
        dataset.set_epoch(epoch)
        batch_ready = time.perf_counter()
        for batch in dataloader:
            dataloading_s += time.perf_counter() - batch_ready

            fwd_bwd_start = time.perf_counter()
            loss, output_dict = update_policy(model_engine, batch, step=step)
            step += 1
            fwd_bwd_time += time.perf_counter() - fwd_bwd_start

            if "percep_query_gain" in output_dict:
                last_query_gain = output_dict["percep_query_gain"]

            if model_engine.is_gradient_accumulation_boundary():
                train_tracker.dataloading_s = dataloading_s
                train_tracker.update_s = fwd_bwd_time
                train_tracker.loss = loss.detach().mean().item()
                train_tracker.query_gain = last_query_gain
                train_tracker.lr = optimizer.param_groups[0]["lr"]
                train_tracker.step()
                fwd_bwd_time = 0.0
                dataloading_s = 0.0

            if cfg.save_checkpoint and (step % cfg.save_freq == 0 or step == cfg.steps):
                logger.info(f"Checkpoint policy after step {step}")
                os.makedirs(cfg.output_dir, exist_ok=True)
                # A freshly built dict every time; reusing one that came back from
                # load_checkpoint would smuggle DeepSpeed's reserved keys back in.
                model_engine.save_checkpoint(
                    save_dir=cfg.output_dir, client_state={"step": step, "epoch": epoch}
                )

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

    logger.info("Perception pre-training finished")


if __name__ == "__main__":
    train()
