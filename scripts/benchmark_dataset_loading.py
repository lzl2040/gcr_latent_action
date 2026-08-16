#!/usr/bin/env python

"""
数据集 DataLoader 遍历速度对比测试
对比 v2.1 和 v3.0 的 MultiDatasetforDistTraining 数据加载速度

运行命令:
cd /home/v-wangxiaofa/lzl/gcr_latent_action
source /home/v-wangxiaofa/anaconda3/etc/profile.d/conda.sh
conda activate lerobot_v2
python scripts/benchmark_dataset_loading.py
"""

import logging
import time
import os
import sys
from datetime import datetime

import torch
from torch.utils.data import DataLoader


def init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def test_v21_dataset(logger):
    """测试 v2.1 版本 MultiDatasetforDistTraining"""
    print("=" * 60)
    print("Testing LeRobot v2.1 MultiDatasetforDistTraining")
    print("=" * 60)
    
    from lerobot.common.datasets.lerobot_dataset_for_ace import MultiDatasetforDistTraining
    
    data_mix = "debug"
    vla2root_json = "vla2root.json"
    seed = 42
    dataset_size_one_epoch = 10000
    
    logger.info(f"data_mix: {data_mix}")
    logger.info(f"dataset_size_one_epoch: {dataset_size_one_epoch}")
    
    # 创建简单的 cfg 对象，包含 policy 配置
    class SimplePolicyConfig:
        observation_delta_indices = None
        
        def __getattr__(self, name):
            return None
    
    class SimpleConfig:
        def __init__(self):
            self.data_mix = data_mix
            self.seed = seed
            self.dataset_size_one_epoch = dataset_size_one_epoch
            self.dataset = self
            self.parent_dir = "/Data/lerobot_data_ort6d"  # v2.1 数据路径
            self.policy = SimplePolicyConfig()
            
        def __getattr__(self, name):
            return None
    
    cfg = SimpleConfig()
    
    logger.info("Loading v2.1 dataset...")
    start_time = time.time()
    
    dataset = MultiDatasetforDistTraining(
        cfg=cfg,
        image_transforms=None,
        wrist_image_transforms=None,
        seed=seed,
        data_mix=data_mix,
        vla2root_json=vla2root_json,
        dataset_size_one_epoch=dataset_size_one_epoch
    )
    
    load_time = time.time() - start_time
    logger.info(f"v2.1 Dataset loaded in {load_time:.2f} seconds")
    logger.info(f"Dataset length: {len(dataset)}")
    logger.info(f"Dataset num_frames: {dataset.num_frames}")
    logger.info(f"Dataset num_episodes: {dataset.num_episodes}")
    
    # 创建 DataLoader
    batch_size = 4
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    
    # 测试遍历速度
    logger.info("Testing DataLoader iteration (100 batches)...")
    start_time = time.time()
    batch_count = 0
    last_batch = None
    for batch in dataloader:
        last_batch = batch
        batch_count += 1
        if batch_count >= 100:
            break
    iteration_time = time.time() - start_time
    
    logger.info(f"Iterated 100 batches in {iteration_time:.2f} seconds")
    logger.info(f"Average time per batch: {iteration_time/100*1000:.2f} ms")
    logger.info(f"Average time per sample: {iteration_time/100/batch_size*1000:.2f} ms")
    
    # 打印 batch 内容
    if last_batch:
        logger.info(f"Batch keys: {list(last_batch.keys())}")
        for key, val in last_batch.items():
            if isinstance(val, torch.Tensor):
                logger.info(f"  {key}: {val.shape}, {val.dtype}")
    
    return load_time, iteration_time, len(dataset)


def test_v30_dataset(logger):
    """测试 v3.0 版本 MultiDatasetforDistTraining"""
    print("\n" + "=" * 60)
    print("Testing LeRobot v3.0 MultiDatasetforDistTraining")
    print("=" * 60)
    
    from lerobot.common.datasets_v30.lerobot_dataset import MultiDatasetforDistTraining
    
    data_mix = "debug"
    vla2root_json = "vla2root.json"
    seed = 42
    dataset_size_one_epoch = 10000
    
    logger.info(f"data_mix: {data_mix}")
    logger.info(f"dataset_size_one_epoch: {dataset_size_one_epoch}")
    
    # 创建简单的 cfg 对象，包含 policy 配置
    class SimplePolicyConfig:
        observation_delta_indices = None
        
        def __getattr__(self, name):
            return None
    
    class SimpleConfig:
        def __init__(self):
            self.data_mix = data_mix
            self.seed = seed
            self.dataset_size_one_epoch = dataset_size_one_epoch
            self.dataset = self
            self.parent_dir = "/Data/lerobot_data_ort6d/v30"  # v3.0 数据路径
            self.policy = SimplePolicyConfig()
            
        def __getattr__(self, name):
            return None
    
    cfg = SimpleConfig()
    
    logger.info("Loading v3.0 dataset...")
    start_time = time.time()
    
    dataset = MultiDatasetforDistTraining(
        cfg=cfg,
        image_transforms=None,
        seed=seed,
        data_mix=data_mix,
        vla2root_json=vla2root_json,
        dataset_size_one_epoch=dataset_size_one_epoch
    )
    
    load_time = time.time() - start_time
    logger.info(f"v3.0 Dataset loaded in {load_time:.2f} seconds")
    logger.info(f"Dataset length: {len(dataset)}")
    logger.info(f"Dataset num_frames: {dataset.num_frames}")
    logger.info(f"Dataset num_episodes: {dataset.num_episodes}")
    
    # 创建 DataLoader
    batch_size = 4
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    
    # 测试遍历速度
    logger.info("Testing DataLoader iteration (100 batches)...")
    start_time = time.time()
    batch_count = 0
    last_batch = None
    for batch in dataloader:
        last_batch = batch
        batch_count += 1
        if batch_count >= 100:
            break
    iteration_time = time.time() - start_time
    
    logger.info(f"Iterated 100 batches in {iteration_time:.2f} seconds")
    logger.info(f"Average time per batch: {iteration_time/100*1000:.2f} ms")
    logger.info(f"Average time per sample: {iteration_time/100/batch_size*1000:.2f} ms")
    
    # 打印 batch 内容
    if last_batch:
        logger.info(f"Batch keys: {list(last_batch.keys())}")
        for key, val in last_batch.items():
            if isinstance(val, torch.Tensor):
                logger.info(f"  {key}: {val.shape}, {val.dtype}")
    
    return load_time, iteration_time, len(dataset)


def main():
    print("=" * 60)
    print("LeRobot MultiDatasetforDistTraining Benchmark")
    print(f"data_mix = 'debug'")
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 设置环境变量
    os.environ["DECORD_LOG_LEVEL"] = "error"
    os.environ['DAV1D_LOG_LEVEL'] = '0'
    os.environ['FFMPEG_LOG_LEVEL'] = 'quiet'
    
    logger = init_logger()
    
    # 测试 v2.1
    try:
        v21_load_time, v21_iter_time, v21_len = test_v21_dataset(logger)
    except Exception as e:
        logger.error(f"v2.1 test failed: {e}")
        import traceback
        traceback.print_exc()
        v21_load_time, v21_iter_time, v21_len = None, None, None
    
    # 测试 v3.0
    try:
        v30_load_time, v30_iter_time, v30_len = test_v30_dataset(logger)
    except Exception as e:
        logger.error(f"v3.0 test failed: {e}")
        import traceback
        traceback.print_exc()
        v30_load_time, v30_iter_time, v30_len = None, None, None
    
    # 总结对比
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if v21_load_time and v30_load_time:
        print(f"\n数据集加载时间:")
        print(f"  v2.1: {v21_load_time:.2f} seconds")
        print(f"  v3.0: {v30_load_time:.2f} seconds")
        ratio = v30_load_time / v21_load_time
        print(f"  v3.0 is {ratio:.2f}x {'slower' if ratio > 1 else 'faster'} than v2.1")
    
    if v21_iter_time and v30_iter_time:
        print(f"\nDataLoader 遍历速度 (100 batches, batch_size=4):")
        print(f"  v2.1: {v21_iter_time:.2f} seconds ({v21_iter_time/100*1000:.2f} ms/batch)")
        print(f"  v3.0: {v30_iter_time:.2f} seconds ({v30_iter_time/100*1000:.2f} ms/batch)")
        ratio = v30_iter_time / v21_iter_time
        print(f"  v3.0 is {ratio:.2f}x {'slower' if ratio > 1 else 'faster'} than v2.1")
    
    print("\n" + "=" * 60)
    print("Benchmark finished")
    print("=" * 60)


if __name__ == "__main__":
    main()