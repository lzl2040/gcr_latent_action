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

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.common.datasets.cosmos_policy_dataset import MultiDatasetforDistTraining, extra_collate_fn
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
            batch["step"] = step
            loss, output_dict = model(batch)
            loss = loss / cfg.gradient_accumulation_steps
            # 反向传播
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            # 梯度裁剪（可选）
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         grad_norm = param.grad.norm(2)
            #         if grad_norm.item() > 100:
            #             print(name)
            grad_norm = clip_grad_norm_low_mem(model.parameters(), max_norm=6.0)
            # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # 梯度平均，用于记录
            if dist.is_initialized():
                dist.all_reduce(grad_norm, op=dist.ReduceOp.SUM)
                grad_norm /= dist.get_world_size()
        else:
            with model.no_sync():
                loss, output_dict = model(batch)
                loss = loss / cfg.gradient_accumulation_steps
                # 反向传播
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            grad_norm = None
    
    return loss, grad_norm, output_dict

@parser.wrap()
def train(cfg: TrainPipelineConfig):
    # Initialize distributed environment
    os.environ["NODE_RANK"] = "0"
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_rank = int(os.environ["RANK"])
    node_rank = int(os.environ["NODE_RANK"])
    master_ip = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    master_uri = "tcp://%s:%s" % (master_ip, master_port)
    rank = world_rank
    dist.init_process_group(
        backend="nccl",
        init_method=master_uri,
        world_size=world_size,
        timeout=timedelta(minutes=60),
        rank=world_rank,
    )
    torch.cuda.set_device(local_rank)
    cfg.validate()
    
    # Prepare logger
    logger = init_logger(cfg, rank)
    logger.info(f"DIST INFO: world_size={world_size}, local_rank={local_rank}, world_rank={world_rank}, node_rank={node_rank}, master_uri={master_uri}")
    
    if rank == 0:
        logger.info(pformat(cfg.to_dict()))
        if cfg.wandb.enable and cfg.wandb.project:
            wandb_logger = WandBLogger(cfg)
        else:
            wandb_logger = None
            logger.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
    else:
        wandb_logger = None
    
    if cfg.seed is not None:
        set_seed(cfg.seed)
        
    
    # find resume weights
    step = 1
    if cfg.weight_resume:
        logger.info("Resume is set, will model from checkpoint...")
        os.makedirs(cfg.output_dir, exist_ok=True)
        pts = sorted(glob.glob(os.path.join(cfg.output_dir, "*.pt")))
        logger.info(f"Found {len(pts)} checkpoints, names are {pts}")
        if pts:
            steps = [int(os.path.basename(pt).split(".")[0].split("step")[1]) for pt in pts]
            step = sorted(steps)[-1] + 1
            # seed += (step-1)
    
    # prepare dataset
    seed = cfg.seed + rank
    print(f"Seed is {seed}")
    dataset = MultiDatasetforDistTraining(
        cfg=cfg, 
        seed=seed,
        data_mix=cfg.data_mix,
        vla2root_json="vla2root.json",
    )
    
    data_loader = DataLoader(dataset, batch_size=3, 
                             num_workers=2,
                            pin_memory=False)
    
    
    # prepare policy
    if rank ==0:
        logger.info("Creating policy...")
    if hasattr(cfg.policy, "tokenizer_max_length"):
        if rank ==0:
            logger.info("Setting model's tokenizer_max_length to 100")
        cfg.policy.tokenizer_max_length=100
    if rank ==0:
        logger.info("Still creating policy...")
        
    policy = make_policy(
        cfg=cfg.policy,
        device="cpu",
        ds_meta=dataset.meta,
        weight_pt_path=cfg.policy.pretrained_path
    )
    if rank == 0:
        model_params = sum(p.numel() for p in policy.parameters()) / 1e9
        logger.info(f"Model parameters: {model_params} B")
    
    # load resume weights
    if cfg.weight_resume:
        if pts:
            torch.cuda.empty_cache()
            cfg.resume = os.path.join(cfg.output_dir, f"step{step-1}.pt")
            logger.info(f"Resuming from checkpoint {cfg.resume} at step {step}")
            model_state_dict = torch.load(cfg.resume, map_location="cpu")
            key_to_remove = []
            for k, v in model_state_dict.items():
                if "awa_model.lm_head" in k or "qwen_expert.lm_head" in k:
                    key_to_remove.append(k)
            for k in key_to_remove:
                del model_state_dict[k]
            
            policy.load_state_dict(model_state_dict, strict=True)
            del model_state_dict
            del key_to_remove
            torch.cuda.empty_cache()
            logger.info("Checkpoint loaded successfully.")
        else:
            cfg.resume = False
            logger.info("No checkpoint found, starting from scratch.")
    
    logger.info("Setting model parameters to BF16...")
    for params in policy.parameters():
        params.data = params.data.bfloat16()
    
    # prepare fsdp model
    auto_wrap_policy = functools.partial(
        always_wrap_policy,
    )
    
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        # reduce_dtype=torch.float32,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
        keep_low_precision_grads=True
    )
    sharding_strategy = ShardingStrategy.HYBRID_SHARD
    model = FSDP(
        policy,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=sharding_strategy,
        device_id=local_rank,
        use_orig_params=True
    )
    # optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, model)
    
    # prepare dataloader
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=cfg.seed+rank,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=2,
        # collate_fn=extra_collate_fn,
        pin_memory=False,
    )
    
    # prepare metric recorder
    train_metrics = {
        "loss": AverageMeter("loss", ":.4f"),
        "grad_norm": AverageMeter("grdn", ":.4f"),
        "lr": AverageMeter("lr", ":0.01e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
        "optim_s": AverageMeter("optim_s", ":.3f"),
    }
    train_tracker = MetricsTracker(
        cfg.batch_size*world_size*cfg.gradient_accumulation_steps,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=int(step//cfg.gradient_accumulation_steps)
    )
    
    # prepare training
    if rank == 0:
        logger.info(f"Starting FSDP training on {world_size} devices")
        logger.info(pformat(cfg.to_dict()))
    
    model.train()
    dataloader_iter = cycle(dataloader)
    # to do: resume dataloader
    
    if rank == 0:
        logger.info("Starting training loop...")
    fwd_bwd_time = 0.0
    dataloading_s = 0.0
    grad_norm_value = 0.0
    loss_value = 0.0
    while step < cfg.steps:
        sync_flag = (step % cfg.gradient_accumulation_steps == 0)
        batch_start = time.perf_counter()
        batch = next(dataloader_iter)
        data_time = time.perf_counter() - batch_start
        dataloading_s += data_time
        
        step_start = time.perf_counter()
        
        model(batch)
    
    
    
    for data in data_loader:
        policy(data)
    

if __name__ == "__main__":
    # 设置环境变量
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # os.environ["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
    # os.environ["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
    os.environ['WANDB_API_KEY'] = '9e1c3ac77856b8ebb5573c4e1e250c84aabfb904'
    
    # 启动训练
    train()