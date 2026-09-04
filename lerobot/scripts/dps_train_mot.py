#!/usr/bin/env python
"""Distributed DeepSpeed training entry point for the Phi-4-Multimodal MoT world model."""

import json
import math
import os
import time
from itertools import islice
from pathlib import Path

import deepspeed
import torch
from torch import distributed as dist
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler
from transformers import AutoTokenizer

from lerobot.common.datasets.canonical_space import CANON_DIM
from lerobot.common.datasets.contrastive_dataset import (
    MultiModalContrastiveDataset,
    contrastive_collate_fn,
)
from lerobot.common.policies.mot.modeling_mot import MoTConfig
from lerobot.common.policies.mot.vae_latents import WanLatentEncoder
from lerobot.common.policies.mot.world_model import (
    TASK_SPECS,
    MoTWorldModel,
    WorldModelConfig,
    parse_mix,
    sample_task,
)
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.dps_train_contrast import _worker_init, init_logger


def env(name: str, default, cast=int):
    return cast(os.environ.get(name, default))


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value not in {"0", "1", "false", "true"}:
        raise ValueError(f"{name} must be one of 0/1/false/true, got {value!r}")
    return value in {"1", "true"}


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> LambdaLR:
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def task_for_update(mix: dict[str, float], seed: int, update: int) -> str:
    """Choose the same task on every rank, reproducibly across checkpoint resumes."""
    generator = torch.Generator().manual_seed(seed + update)
    return sample_task(mix, generator)


def distributed_mean(value: float, device: torch.device, world_size: int) -> float:
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor / world_size).item()


def validate_paths(phi_dir: str, vae_dir: str, deepspeed_config: str) -> None:
    required = (
        (Path(phi_dir) / "config.json", "Phi-4-Multimodal config"),
        (Path(vae_dir) / "config.json", "Wan VAE config"),
        (Path(deepspeed_config), "DeepSpeed config"),
    )
    missing = [f"{label}: {path}" for path, label in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required training files:\n  " + "\n  ".join(missing))


class ResumableBatchSampler:
    """Skip completed batches in the first resumed epoch without decoding them again."""

    def __init__(self, sampler: DistributedSampler, batch_size: int):
        self.sampler = sampler
        self.batch_size = batch_size
        self.start_batch = 0

    @property
    def full_length(self) -> int:
        return len(self.sampler) // self.batch_size

    def __iter__(self):
        batches = BatchSampler(self.sampler, self.batch_size, drop_last=True)
        return islice(iter(batches), self.start_batch, None)

    def __len__(self) -> int:
        return max(0, self.full_length - self.start_batch)


@parser.wrap()
def train(cfg: TrainPipelineConfig) -> None:
    if cfg.policy is None:
        raise ValueError("--policy.type=robo_contrast is required for the MoT data configuration")
    if cfg.deepspeed is None:
        raise ValueError("--deepspeed must point to a ZeRO configuration")

    phi_dir = os.environ.get("PHI_DIR", "/Data/lzl/huggingface/Phi-4-multimodal-instruct")
    vae_dir = os.environ.get("VAE_DIR", "/Data/lzl/huggingface/Cosmos3-Edge/vae")
    validate_paths(phi_dir, vae_dir, cfg.deepspeed)

    deepspeed.init_distributed()
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16

    cfg.log_dir = cfg.log_dir or "logs"
    run_name = cfg.job_name or "phi4_mot"
    run_dir = Path(cfg.output_dir or "outputs/mot") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = run_dir
    logger = init_logger(cfg, subdir="mot")

    seed = int(cfg.seed or 0)
    set_seed(seed + rank)

    with open(cfg.deepspeed) as f:
        ds_config = json.load(f)
    batch_size = int(ds_config["train_micro_batch_size_per_gpu"])
    grad_accumulation = int(ds_config.get("gradient_accumulation_steps", 1))
    if cfg.steps % grad_accumulation:
        raise ValueError(
            f"steps ({cfg.steps}) must be divisible by gradient accumulation "
            f"({grad_accumulation}) so the final checkpoint is on an optimizer boundary"
        )

    latent_frames = env("LATENT_FRAMES", 3)
    rgb_frames = WanLatentEncoder.required_frames(latent_frames)
    cfg.policy.rgb_frames = rgb_frames
    cfg.policy.use_tactile = False

    dataset = MultiModalContrastiveDataset(
        cfg=cfg,
        data_mix=cfg.data_mix,
        seed=seed,
        dataset_size_one_epoch=cfg.dataset.dataset_size_one_epoch,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=True,
    )
    batch_sampler = ResumableBatchSampler(sampler, batch_size)
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=cfg.num_workers,
        worker_init_fn=_worker_init,
        collate_fn=contrastive_collate_fn,
        pin_memory=True,
        # The dataset rebuilds its index-to-frame plan in set_epoch(). Persistent workers
        # would retain their private epoch-0 copy and silently repeat that plan forever.
        persistent_workers=False,
        prefetch_factor=4 if cfg.num_workers > 0 else None,
    )
    if batch_sampler.full_length == 0:
        raise ValueError(
            f"Per-rank dataset length {len(sampler)} is smaller than batch size {batch_size}"
        )
    steps_per_epoch = batch_sampler.full_length
    training_geometry = {
        "batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": grad_accumulation,
        "world_size": world_size,
        "steps_per_epoch": steps_per_epoch,
        "dataset_size_one_epoch": cfg.dataset.dataset_size_one_epoch,
        "data_mix": cfg.data_mix,
        "seed": seed,
    }

    scope = os.environ.get("SCOPE", "gen_only")
    execution = os.environ.get("EXECUTION", "interleaved")
    checkpoint_segment = env("CKPT_SEGMENT", 4)
    default_microbatch = 32 if scope == "gen_only" else 16
    mot_microbatch = env("MOT_MICROBATCH", default_microbatch)
    grad_checkpointing = env_bool("GRAD_CHECKPOINTING", True)
    freeze_projector = env_bool("FREEZE_VISION_PROJECTOR", False)
    action_loss_weight = env("ACTION_LOSS_WEIGHT", 1.0, float)

    mot_config = MoTConfig.from_phi_dir(phi_dir, action_dim=CANON_DIM)
    model = MoTWorldModel(
        WorldModelConfig(
            mot=mot_config,
            trainable_scope=scope,
            freeze_vision_projector=freeze_projector,
            training_execution=execution,
            mot_checkpoint_segment_size=checkpoint_segment,
            und_microbatch_size=mot_microbatch,
            action_loss_weight=action_loss_weight,
        )
    )
    model.load_pretrained()
    model.mot.gradient_checkpointing = grad_checkpointing
    model.to(dtype=dtype)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    learning_rate = env("LEARNING_RATE", 1e-4, float)
    min_learning_rate = env("MIN_LEARNING_RATE", 1e-5, float)
    weight_decay = env("WEIGHT_DECAY", 0.01, float)
    if learning_rate <= 0:
        raise ValueError(f"LEARNING_RATE must be positive, got {learning_rate}")
    if not 0 <= min_learning_rate <= learning_rate:
        raise ValueError(
            f"MIN_LEARNING_RATE must be in [0, LEARNING_RATE], got {min_learning_rate}"
        )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        weight_decay=weight_decay,
        fused=True,
    )
    total_updates = cfg.steps // grad_accumulation
    warmup_updates = env("WARMUP_STEPS", 500) // grad_accumulation
    scheduler = build_scheduler(
        optimizer,
        total_steps=total_updates,
        warmup_steps=warmup_updates,
        min_lr_ratio=min_learning_rate / learning_rate,
    )

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=cfg.deepspeed,
        model_parameters=trainable,
    )
    model_engine.train()

    tokenizer = AutoTokenizer.from_pretrained(phi_dir)
    vae = WanLatentEncoder(vae_dir, dtype=dtype).to(device)
    if mot_config.latent_channels != vae.latent_channels:
        raise ValueError(
            f"MoT expects {mot_config.latent_channels} latent channels, "
            f"but the VAE produces {vae.latent_channels}"
        )

    mix_name = os.environ.get("MIX", "stage3")
    mix = parse_mix(mix_name)
    text_length = env("TEXT_LEN", 32)

    report = model_engine.module.param_report()
    if rank == 0:
        logger.info(
            "MoT params: total %.3fB, trainable %.3fB; scope=%s, execution=%s",
            report["total"] / 1e9,
            report["trainable"] / 1e9,
            scope,
            execution,
        )
        logger.info(
            "Data: mix=%s, task_mix=%s, pixel_frames=%d, batch/gpu=%d, "
            "gradient_accumulation=%d, global_batch=%d",
            cfg.data_mix,
            ", ".join(f"{name}={weight:.2f}" for name, weight in mix.items()),
            rgb_frames,
            batch_size,
            grad_accumulation,
            model_engine.train_batch_size(),
        )
        logger.info(
            "Memory controls: checkpoint=%s, segment=%d, MoT microbatch=%d",
            grad_checkpointing,
            checkpoint_segment,
            mot_microbatch,
        )

    step = 0
    client_state = None
    resume_dir = Path(os.environ.get("RESUME_DIR") or run_dir)
    if env_bool("RESUME", True):
        load_path, client_state = model_engine.load_checkpoint(
            str(resume_dir),
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
            load_module_strict=False,
        )
        if load_path is not None:
            step = int((client_state or {}).get("step", 0))
            saved_geometry = (client_state or {}).get("training_geometry")
            if saved_geometry is not None:
                mismatches = {
                    key: (saved_geometry.get(key), value)
                    for key, value in training_geometry.items()
                    if saved_geometry.get(key) != value
                }
                if mismatches:
                    details = ", ".join(
                        f"{key}: checkpoint={old}, current={new}"
                        for key, (old, new) in mismatches.items()
                    )
                    raise ValueError(
                        "Cannot resume with different data/accumulation geometry; " + details
                    )
            elif step % grad_accumulation:
                raise ValueError(
                    f"Checkpoint micro-step {step} is not on a boundary for the current "
                    f"gradient accumulation {grad_accumulation}"
                )
            logger.info("Resumed from %s at micro-step %d", load_path, step)
        elif rank == 0:
            logger.info("No DeepSpeed checkpoint found in %s; starting from Phi-4 weights", resume_dir)

    if step >= cfg.steps:
        if rank == 0:
            logger.info(
                "Checkpoint is already at micro-step %d (target %d); nothing to train",
                step,
                cfg.steps,
            )
        if dist.is_initialized():
            dist.barrier(device_ids=[local_rank])
        return

    if rank == 0:
        run_config = {
            "scope": scope,
            "execution": execution,
            "checkpoint_segment": checkpoint_segment,
            "mot_microbatch": mot_microbatch,
            "gradient_checkpointing": grad_checkpointing,
            "latent_frames": latent_frames,
            "pixel_frames": rgb_frames,
            "task_mix": mix,
            "phi_dir": phi_dir,
            "vae_dir": vae_dir,
            "deepspeed": ds_config,
        }
        (run_dir / "mot_run_config.json").write_text(
            json.dumps(run_config, indent=2, ensure_ascii=False) + "\n"
        )
        wandb_logger = WandBLogger(cfg) if (cfg.wandb.enable and cfg.wandb.project) else None
    else:
        wandb_logger = None

    if client_state and "next_batch_in_epoch" in client_state:
        start_epoch = int(client_state.get("epoch", 0))
        start_batch = int(client_state["next_batch_in_epoch"])
        start_epoch += start_batch // steps_per_epoch
        start_batch %= steps_per_epoch
    else:
        # Compatibility with checkpoints written before the sampler offset was recorded.
        start_epoch, start_batch = divmod(step, steps_per_epoch)
    data_started = time.perf_counter()

    for epoch in range(start_epoch, 100000):
        sampler.set_epoch(epoch)
        dataset.set_epoch(epoch)
        batch_sampler.start_batch = start_batch if epoch == start_epoch else 0
        if rank == 0:
            logger.info("Epoch %d start at batch %d", epoch, batch_sampler.start_batch)

        for batch_index, batch in enumerate(dataloader, start=batch_sampler.start_batch):
            data_seconds = time.perf_counter() - data_started
            # Every rank and every micro-step in one optimizer update must solve the same
            # task. Mixing task families inside an accumulation window changes the meaning of
            # the configured task probabilities and produces a hybrid gradient update.
            task = task_for_update(mix, seed, model_engine.global_steps)
            spec = TASK_SPECS[task]

            needed_latents = spec.latent_frames or latent_frames
            if spec.video == "absent":
                needed_latents = spec.context
            needed_frames = WanLatentEncoder.required_frames(needed_latents)

            vae_started = time.perf_counter()
            clip = batch["image_clip"][:, :needed_frames].to(device, non_blocking=True)
            latents = vae(clip).to(dtype)
            vae_seconds = time.perf_counter() - vae_started

            model_started = time.perf_counter()
            pixel_values = None
            if spec.image:
                pixel_values = (
                    batch["image_t0"].to(device, non_blocking=True).to(dtype).div_(255.0)
                )
            text = tokenizer(
                [value or " " for value in batch["task"]],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=text_length,
            )
            text_ids = text["input_ids"].to(device, non_blocking=True)
            actions = batch["action"].to(device, non_blocking=True).to(dtype)
            domain_id = batch["dataset_id"].to(device, non_blocking=True)

            output = model_engine(
                latents=latents,
                pixel_values=pixel_values,
                text_ids=text_ids,
                actions=actions,
                domain_id=domain_id,
                task=task,
            )
            loss = output["loss"]
            boundary = model_engine.is_gradient_accumulation_boundary()
            model_engine.backward(loss)
            model_engine.step()
            step += 1
            model_seconds = time.perf_counter() - model_started

            should_log = boundary and cfg.log_freq > 0 and step % cfg.log_freq == 0
            if should_log:
                metrics = {
                    "loss": distributed_mean(loss.detach().item(), device, world_size),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "data_s": distributed_mean(data_seconds, device, world_size),
                    "vae_s": distributed_mean(vae_seconds, device, world_size),
                    "model_s": distributed_mean(model_seconds, device, world_size),
                    "task": task,
                }
                if "loss_video" in output:
                    metrics["loss_video"] = distributed_mean(
                        output["loss_video"].detach().item(), device, world_size
                    )
                if "loss_action" in output:
                    metrics["loss_action"] = distributed_mean(
                        output["loss_action"].detach().item(), device, world_size
                    )
                if rank == 0:
                    logger.info(
                        "step:%d task:%s loss:%.4f lr:%.2e data:%.3fs vae:%.3fs model:%.3fs",
                        step,
                        task,
                        metrics["loss"],
                        metrics["lr"],
                        metrics["data_s"],
                        metrics["vae_s"],
                        metrics["model_s"],
                    )
                    if wandb_logger:
                        wandb_logger.log_dict(metrics, step)

            should_save = boundary and cfg.save_checkpoint and (
                step == cfg.steps or (cfg.save_freq > 0 and step % cfg.save_freq == 0)
            )
            if should_save:
                logger.info("Saving MoT checkpoint after micro-step %d", step)
                model_engine.save_checkpoint(
                    save_dir=str(run_dir),
                    client_state={
                        "step": step,
                        "epoch": epoch,
                        "next_batch_in_epoch": batch_index + 1,
                        "training_geometry": training_geometry,
                    },
                )

            if step >= cfg.steps:
                break
            data_started = time.perf_counter()

        if step >= cfg.steps:
            break

    if dist.is_initialized():
        dist.barrier(device_ids=[local_rank])
    if rank == 0:
        logger.info("MoT training finished at micro-step %d", step)


if __name__ == "__main__":
    train()
