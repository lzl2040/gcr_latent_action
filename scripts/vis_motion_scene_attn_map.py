#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

from tqdm import tqdm
import math

from diffusers import SanaPipeline

def init_logger(cfg, rank):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO if rank == 0 else logging.WARN)
    
    if rank == 0:
        formatter = logging.Formatter(
            f'[%(asctime)s] [rank: {rank}] [%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        log_path = Path(cfg.log_dir) / f"fsdp_logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0

def save_fsdp_checkpoint(model, optim, output_dir, step):
    # 使用 StateDictType.FULL_STATE_DICT 替代 FSDP.FULL_STATE_DICT
    save_policy = StateDictType.FULL_STATE_DICT
    full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)


    # --- Step 1: 预同步，确保所有 rank 到达保存阶段 ---
    dist.barrier()

    # 所有进程统一进入状态字典收集阶段
    with FSDP.state_dict_type(model, save_policy, full_state_dict_config):
        model_state_dict = model.state_dict()
    
    # --- Step 3: 再次同步，防止部分 rank 提前退出通信 ---
    dist.barrier()

    # 仅主进程保存模型和优化器状态
    if get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        ckpt_path = os.path.join(output_dir, f"step{step}.pt")

        # 可选：保存优化器状态
        # optim_state_dict = FSDP.full_optim_state_dict(model, optim)

        # torch.save({
        #     'model': model_state_dict,
        #     'optimizer': optim_state_dict,
        #     'step': step,
        # }, ckpt_path)
        torch.save(model_state_dict, ckpt_path)

        logging.info(f"Checkpoint saved at {ckpt_path}")
    # 所有进程同步，防止部分进程提前退出
    # --- Step 5: 确保 rank0 保存完后其他 rank 再继续 ---
    dist.barrier()
        
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

def train_step(model, batch, scaler, cfg, sync_flag, step):
    """执行单个训练步骤"""
    # 前向传播
    sync_flag = True
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        
        if sync_flag:
            loss = model.visualize_attn_map(batch, step)
        else:
            with model.no_sync():
                loss = model(batch)
    
    return loss

@parser.wrap()
def train(cfg: TrainPipelineConfig):
    
    # dist.init_process_group(backend="nccl")
    # rank = dist.get_rank()
    # world_size = dist.get_world_size()
    # local_rank = rank
    local_rank = 0
    
    torch.cuda.set_device(local_rank)
    
    # 初始化配置
    cfg.validate()
    
    # 设置随机种子
    seed = cfg.seed
    if cfg.seed is not None:
        set_seed(cfg.seed)
    
    # 数据集初始化
    
    print(f"Seed is {seed}")
    image_transforms = ImageTransforms(cfg.dataset.image_transforms)
    wrist_image_transforms = ImageTransforms(cfg.dataset.wrist_image_transforms)
    print(f"image transforms:{image_transforms}")
    print(f"wrist image transforms:{wrist_image_transforms}")

    # img_gen_pipe = SanaPipeline.from_pretrained(
    #     cfg.policy.img_pred_model,
    #     variant="fp16",
    #     torch_dtype=torch.float16
    # ).to(f"cuda:{local_rank}")

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
    
    # 模型初始化
    cfg.policy.set_token_idx(dataset.cp_act_token_idx, dataset.cp_sc_token_idx)
    model = make_policy(
        cfg=cfg.policy,
        device="cpu",
        ds_meta=dataset.meta,
        weight_pt_path=cfg.policy.pretrained_path
    ).to("cuda")
    
    # for params in model.parameters():
    #     params.data = params.data.bfloat16()
        # params.data = params.data.to(dtype=torch.float16)
    
    
    # 优化器和学习率调度器
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, model)
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=2,
        collate_fn=extra_collate_fn,
        pin_memory=False,
    )
    
    # 混合精度scaler
    scaler = None
    # scaler = ShardedGradScaler()
    dataloader_iter = cycle(dataloader)
        
    step = 0
    while step < cfg.steps:
        sync_flag = (step % cfg.gradient_accumulation_steps == 0)
        batch = next(dataloader_iter)
        
        loss = train_step(model, batch, scaler, cfg, sync_flag, step)
        
        step += 1

if __name__ == "__main__":
    # 设置环境变量
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # os.environ["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
    # os.environ["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
    os.environ['WANDB_API_KEY'] = '9e1c3ac77856b8ebb5573c4e1e250c84aabfb904'
    
    # 启动训练
    train()