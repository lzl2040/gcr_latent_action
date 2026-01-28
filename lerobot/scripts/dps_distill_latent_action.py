import logging
import time
import os
import glob
import json
import functools
from pathlib import Path
from datetime import datetime
from pprint import pformat
from termcolor import colored
from typing import Any
from datetime import timedelta
from contextlib import nullcontext

import torch
from torch.utils.data import Subset
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)

FSDP.__repr__ = lambda self: "FSDP(...)"
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    always_wrap_policy
)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.api import StateDictType, FullStateDictConfig
import deepspeed

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer, Qwen2_5_VLVisionBlock
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.common.datasets.vt_dataset import MultiDatasetforDistTraining, extra_collate_fn
from lerobot.common.datasets.sampler import EpisodeAwareSampler, DistEpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy
from lerobot.common.policies.latent_action.distill_model import DistillModel

from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import math

def init_logger(cfg):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    if cfg.local_rank == 0:
        formatter = logging.Formatter(
            f'[%(asctime)s] [rank: {cfg.local_rank}] [%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # # 控制台Handler
        # console_handler = logging.StreamHandler()
        # console_handler.setFormatter(formatter)
        # logger.addHandler(console_handler)
        
        # 文件Handler
        log_path = Path(cfg.log_dir) / f"logs_with_pretrain/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0

        
def clip_grad_norm_low_mem(parameters, max_norm):
    """低内存版本的梯度裁剪"""
    grads = []
    for p in parameters:
        if p.grad is not None:
            # 分离梯度并复制，避免保持计算图
            grads.append(p.grad.detach().clone())

    # 逐个处理梯度，减少峰值内存
    total_norm = 0.0
    for grad in grads:
        grad_norm = grad.norm(2)
        total_norm += grad_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    # 应用裁剪
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for grad in grads:
            grad.mul_(clip_coef)
    
    # 将裁剪后的梯度复制回模型
    idx = 0
    for p in parameters:
        if p.grad is not None:
            p.grad.copy_(grads[idx])
            idx += 1
    
    return torch.tensor(total_norm, device=grads[0].device)

def train_step(model_engine, batch):
    batch = {k: v.to(model_engine.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # torch.cuda.empty_cache()
    # with torch.autocast("cuda"):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        loss, output_dict = model_engine(batch)

    model_engine.backward(loss)
    
    # for name, param in model_engine.module.model.paligemma_with_expert.qwen25vl.model.layers[-1].named_parameters():
    #     if param.grad is not None:
    #         print(f"{name} gradient norm: {param.grad.norm().item()}")
    
    model_engine.step()
    
   # torch.cuda.empty_cache()
    return loss, output_dict

@parser.wrap()
def train(cfg: TrainPipelineConfig):
    
    # 初始化配置
    cfg.validate()
    
    deepspeed.init_distributed()
    logger = init_logger(cfg)
    
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(pformat(cfg.to_dict()))
        if cfg.wandb.enable and cfg.wandb.project:
            wandb_logger = WandBLogger(cfg)
        else:
            wandb_logger = None
            logger.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
    else:
        wandb_logger = None

    if cfg.seed is not None:
        rank = int(os.environ.get('RANK', 0))
        set_seed(cfg.seed + rank)
    
    # 数据集初始化
    seed = cfg.seed + rank
    
    print(f"Seed is {seed}")
    image_transforms = ImageTransforms(cfg.dataset.image_transforms)
    wrist_image_transforms = ImageTransforms(cfg.dataset.wrist_image_transforms)
    print(f"image transforms:{image_transforms}")
    print(f"wrist image transforms:{wrist_image_transforms}")

    dataset = MultiDatasetforDistTraining(
        cfg=cfg, 
        image_transforms=image_transforms,
        wrist_image_transforms=wrist_image_transforms,
        seed=seed,
        data_mix=cfg.data_mix,
        vla2root_json="vla2root.json",
        # image_decoder_processor=img_gen_pipe.image_processor
        # vla2root_json="vla2root_bak_single.json"
    )
    
    # Policy setup
    logger.info("Creating policy...")
    if hasattr(cfg.policy, "tokenizer_max_length"):
        logger.info("Setting model's tokenizer_max_length to 100")
        cfg.policy.tokenizer_max_length=100
    logger.info("Still creating policy...")

    cfg.policy.set_token_idx(dataset.cp_act_token_idx, dataset.cp_sc_token_idx)
    teacher_policy = make_policy(
        cfg=cfg.policy,
        device="cpu",
        ds_meta=dataset.meta,
        weight_pt_path=cfg.policy.pretrained_path
    )

    stu_policy = make_policy(
        cfg=cfg.policy2,
        device="cpu",
        ds_meta=dataset.meta,
        weight_pt_path=cfg.policy2.pretrained_path
    )

    # with open("/home/v-zuoleili/Project/gcr_latent_action/scripts/model_pi.txt", "w") as f:
    #     for name, param in stu_policy.named_parameters():
    #         f.write(f"{name}:{param.shape}\n")

    policy = DistillModel(student_model=stu_policy, teacher_model=teacher_policy)
    
     # Optimizer and scheduler
    logger.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)


    logger.info("Setting model parameters to BF16...")
    for params in policy.parameters():
        # params.requires_grad = True
        params.data = params.data.bfloat16()

    teacher_trainable_params = sum(
            p.numel() for p in policy.teacher_model.parameters() if p.requires_grad
    )
    student_trainable_params = sum(
            p.numel() for p in policy.student_model.parameters() if p.requires_grad
    )
    logger.info(f"Teacher Trainable Params:{teacher_trainable_params / 1e6: .2f}M")
    logger.info(f"Student Trainable Params:{student_trainable_params / 1e6: .2f}M")
    
    
    # Dataloader setup
    if hasattr(cfg.policy, "drop_n_last_frames"):
        sampler = DistEpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
            num_replicas=int(os.environ.get('WORLD_SIZE', 1)),
            rank=int(os.environ.get('RANK', 0))
        )
    else:
        logger.info("Creating DistributedSampler")
        sampler = DistributedSampler(
            dataset,
            num_replicas=int(os.environ.get('WORLD_SIZE', 1)),
            rank=int(os.environ.get('RANK', 0)),
            shuffle=True,
            seed=cfg.seed
        )

    with open(cfg.deepspeed, 'r') as f:
        deepspeed_configs_in_dict = json.load(f)
    batch_size = deepspeed_configs_in_dict['train_micro_batch_size_per_gpu']
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            num_workers=4,
                            pin_memory=True,
                            collate_fn=extra_collate_fn,
                            # persistent_workers=True,
                            # prefetch_factor=2
                            )
    
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=policy,
        config=cfg.deepspeed,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )
    
    logger.info(f"Training batch size:{model_engine.train_batch_size()}") # micro_size * gradient_cum_size * gpu_num
    # Resume training state
    step = 0
    # cfg.output_dir = os.path.join(cfg.output_dir, cfg.job_name)
    if cfg.weight_resume:
        logger.info(f"Resuming training from {cfg.output_dir}")
        ckpt_path = cfg.output_dir
        # ckpt_list = os.listdir(ckpt_path)
        # latest_ckpt = sorted(ckpt_list, key=lambda x: int(x.split("step")[-1]))[-1]
        # checkpoint_path = os.path.join(ckpt_path, latest_ckpt)
        load_path, client_state = model_engine.load_checkpoint(
            ckpt_path,
            load_optimizer_states=True,
            load_lr_scheduler_states=True
        )
        if load_path is not None:
            step = client_state['step']
            logger.info(f"Resumed training from step {step}")
    else:
        client_state = {
            'step': step
        }
    
    if client_state is None:
        client_state = {
            'step': step
        }
    
    # Metrics setup
    train_metrics = {
        "loss": AverageMeter("loss", ":.4f"),
        "mse_loss": AverageMeter("mse_loss", ":.4f"),
        "kl_loss": AverageMeter("kl_loss", ":.4f"),
        "lg_loss": AverageMeter("lg_loss", ":.4f"),
        # "language_loss": AverageMeter("language_loss", ":.4f"),
        "grad_norm": AverageMeter("grdn", ":.4f"),
        "lr": AverageMeter("lr", ":0.01e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
        "optim_s": AverageMeter("optim_s", ":.3f"),
    }
    
    train_tracker = MetricsTracker(
        model_engine.train_batch_size(),
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=int(step/model_engine.gradient_accumulation_steps())
    )
    
    # Main training loop
    logger.info(f"Start training on {int(os.environ.get('WORLD_SIZE', 1))} devices")
    
    dataloader_iter = cycle(dataloader)
    
    fwd_bwd_time = 0.0
    dataloading_s = 0.0
    grad_norm_value = 0.0
    loss_value = 0.0
    mse_loss_value = 0.0
    kl_loss_value = 0.0
    lg_loss_value = 0.0
    
    if cfg.is_ft:
        cfg.job_type = "finetune"
    else:
        cfg.job_type = "pretrain"

    
    logger.info("Starting training loop...")
    # Main training loop
    logger.info(f"Start training on {int(os.environ.get('WORLD_SIZE', 1))} devices")
    total_steps = cfg.steps
    completed_steps = step
    fwd_bwd_time = 0
    dataloading_s = 0
    dist_step=10
    
    cfg.output_dir = os.path.join(cfg.output_dir, cfg.job_name)
    
    for step_idx in range(completed_steps, total_steps):
        
        
        start_time = time.perf_counter()
        batch = next(dataloader_iter)
        dataloading_time = time.perf_counter() - start_time
        dataloading_s += dataloading_time
        
        fwd_bwd_start = time.perf_counter()
        loss, output_dict = train_step(
            model_engine,
            batch
        )
        
        step += 1
        fwd_bwd_time += time.perf_counter() - fwd_bwd_start
        loss_value += loss.detach().mean().item()
        mse_loss_value += output_dict["mse"]
        kl_loss_value += output_dict["kl"]
        lg_loss_value += output_dict["lg_loss"]
        
        if model_engine.is_gradient_accumulation_boundary():
            
            train_tracker.dataloading_s = dataloading_s
            train_tracker.update_s = fwd_bwd_time
            train_tracker.loss = loss_value
            train_tracker.mse_loss = mse_loss_value
            train_tracker.kl_loss = kl_loss_value
            train_tracker.lg_loss = lg_loss_value
            train_tracker.grad_norm = grad_norm_value
            train_tracker.lr = optimizer.param_groups[0]["lr"]
            train_tracker.step()
            
            fwd_bwd_time=0
            dataloading_s=0
            grad_norm_value = 0.0
            loss_value = 0.0
            mse_loss_value = 0.0
            kl_loss_value = 0.0
            lg_loss_value = 0.0
        
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        
        if cfg.save_checkpoint and is_saving_step:
            logger.info(f"Checkpoint policy after step {step}")
            # checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            os.makedirs(cfg.output_dir, exist_ok=True)
            
            client_state['step'] = step
            if int(os.environ.get('RANK', 0)) == 0:
                torch.save(model_engine.module.state_dict(), os.path.join(cfg.output_dir, f"step{step}.pt"))
            dist.barrier(device_ids=[model_engine.local_rank])
            # model_engine.save_checkpoint(
            #     save_dir=cfg.output_dir,
            #     client_state=client_state
            # )
            # torch.save(client_state, os.path.join(checkpoint_dir, "metadata.pt"))
            # update_last_checkpoint(checkpoint_dir)
        
        if int(os.environ.get('RANK', 0)) == 0:
            if is_log_step:
                logger.info(train_tracker)
                if wandb_logger:
                    wandb_log_dict = train_tracker.to_dict()
                    if output_dict:
                        wandb_log_dict.update(output_dict)
                    wandb_logger.log_dict(wandb_log_dict, step)
                train_tracker.reset_averages()


        if step_idx % dist_step == 0:
            dist.barrier(device_ids=[model_engine.local_rank])
    logger.info("Training finished")

        
if __name__ == "__main__":
    os.environ['WANDB_API_KEY'] = '9e1c3ac77856b8ebb5573c4e1e250c84aabfb904'
    train()
