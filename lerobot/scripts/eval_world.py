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
import torch
import torch.distributed as dist

from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.common.utils.random_utils import set_seed

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


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    rank = 0
    torch.cuda.set_device(rank)
    
    # 初始化配置
    cfg.validate()
    logger = init_logger(cfg, rank)
    # 设置随机种子
    if cfg.seed is not None:
        set_seed(cfg.seed)
    
    logger.info("Creating policy...")
        