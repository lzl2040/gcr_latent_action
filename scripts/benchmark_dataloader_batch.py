#!/usr/bin/env python
"""
Benchmark DataLoader speed for v2.1 and v3.0 with batch_size=64.

Usage:
    conda activate lerobot_v2
    python scripts/benchmark_dataloader_batch.py
"""

import time
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# =============================================
# Configuration
# =============================================
V21_DATA_ROOT = "/Data/lerobot_data_ort6d/taco_play"
V30_DATA_ROOT = "/Data/lerobot_data_ort6d/v30/taco_play"
DATASET_NAME = "taco_play"
BATCH_SIZE = 64
NUM_STEPS = 20
NUM_WORKERS = 4


def benchmark_v30_dataloader():
    """Benchmark v3.0 DataLoader with batch_size=64."""
    print("\n" + "=" * 70)
    print("Benchmarking v3.0 DataLoader")
    print("=" * 70)
    
    from lerobot.common.datasets_v30.lerobot_dataset import LeRobotDataset
    
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing v3.0 dataset...")
    start = time.time()
    dataset = LeRobotDataset(
        repo_id=DATASET_NAME,
        root=V30_DATA_ROOT,
        video_backend="torchcodec",
        dataset_name=DATASET_NAME,
    )
    init_time = time.time() - start
    print(f"  Dataset length: {len(dataset)}")
    print(f"  Init time: {init_time:.3f}s")
    
    # Create DataLoader
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Creating DataLoader (batch_size={BATCH_SIZE}, num_workers={NUM_WORKERS})...")
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    
    # Benchmark
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running {NUM_STEPS} steps...")
    
    step_times = []
    total_start = time.time()
    
    for i, batch in enumerate(dataloader):
        if i >= NUM_STEPS:
            break
        
        step_start = time.time()
        # Simulate some processing (just accessing the batch)
        _ = batch
        step_time = time.time() - step_start
        step_times.append(step_time)
        print(f"  Step {i+1}/{NUM_STEPS}: {step_time*1000:.1f} ms")
    
    total_time = time.time() - total_start
    
    # Calculate statistics
    avg_step_time = sum(step_times) / len(step_times) * 1000
    total_samples = NUM_STEPS * BATCH_SIZE
    throughput = total_samples / total_time
    
    print(f"\n" + "-" * 70)
    print("v3.0 DataLoader Results:")
    print("-" * 70)
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Avg step time: {avg_step_time:.1f} ms")
    print(f"  Throughput: {throughput:.1f} samples/s")
    print(f"  Time per sample: {total_time/total_samples*1000:.2f} ms")
    print("-" * 70)
    
    return {
        "total_time": total_time,
        "avg_step_time": avg_step_time,
        "throughput": throughput,
        "step_times": step_times,
    }


def benchmark_v21_dataloader():
    """Benchmark v2.1 DataLoader with batch_size=64."""
    print("\n" + "=" * 70)
    print("Benchmarking v2.1 DataLoader")
    print("=" * 70)
    
    from lerobot.common.datasets.lerobot_dataset_for_ace import LeRobotDataset
    
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing v2.1 dataset...")
    start = time.time()
    dataset = LeRobotDataset(
        repo_id=DATASET_NAME,
        root=V21_DATA_ROOT,
        video_backend="torchcodec",
        dataset_name=DATASET_NAME,
    )
    init_time = time.time() - start
    print(f"  Dataset length: {len(dataset)}")
    print(f"  Init time: {init_time:.3f}s")
    
    # Create DataLoader
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Creating DataLoader (batch_size={BATCH_SIZE}, num_workers={NUM_WORKERS})...")
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    
    # Benchmark
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running {NUM_STEPS} steps...")
    
    step_times = []
    total_start = time.time()
    
    for i, batch in enumerate(dataloader):
        if i >= NUM_STEPS:
            break
        
        step_start = time.time()
        # Simulate some processing (just accessing the batch)
        _ = batch
        step_time = time.time() - step_start
        step_times.append(step_time)
        print(f"  Step {i+1}/{NUM_STEPS}: {step_time*1000:.1f} ms")
    
    total_time = time.time() - total_start
    
    # Calculate statistics
    avg_step_time = sum(step_times) / len(step_times) * 1000
    total_samples = NUM_STEPS * BATCH_SIZE
    throughput = total_samples / total_time
    
    print(f"\n" + "-" * 70)
    print("v2.1 DataLoader Results:")
    print("-" * 70)
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Avg step time: {avg_step_time:.1f} ms")
    print(f"  Throughput: {throughput:.1f} samples/s")
    print(f"  Time per sample: {total_time/total_samples*1000:.2f} ms")
    print("-" * 70)
    
    return {
        "total_time": total_time,
        "avg_step_time": avg_step_time,
        "throughput": throughput,
        "step_times": step_times,
    }


def main():
    print("=" * 70)
    print("DataLoader Batch Benchmark: v2.1 vs v3.0")
    print("=" * 70)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Num steps: {NUM_STEPS}")
    print(f"Num workers: {NUM_WORKERS}")
    print(f"Total samples: {NUM_STEPS * BATCH_SIZE}")
    
    # Run v3.0 benchmark
    v30_results = benchmark_v30_dataloader()
    
    # # Run v2.1 benchmark
    # v21_results = benchmark_v21_dataloader()
    
    # # Summary comparison
    # print("\n" + "=" * 70)
    # print("COMPARISON SUMMARY")
    # print("=" * 70)
    # print(f"\n{'Metric':<30} {'v2.1':<15} {'v3.0':<15} {'Ratio':<10}")
    # print("-" * 70)
    
    # total_samples = NUM_STEPS * BATCH_SIZE
    
    # print(f"{'Total time (s)':<30} {v21_results['total_time']:<15.2f} {v30_results['total_time']:<15.2f} {v30_results['total_time']/v21_results['total_time']:<10.2f}x")
    # print(f"{'Avg step time (ms)':<30} {v21_results['avg_step_time']:<15.1f} {v30_results['avg_step_time']:<15.1f} {v30_results['avg_step_time']/v21_results['avg_step_time']:<10.2f}x")
    # print(f"{'Throughput (samples/s)':<30} {v21_results['throughput']:<15.1f} {v30_results['throughput']:<15.1f} {v21_results['throughput']/v30_results['throughput']:<10.2f}x")
    # print(f"{'Time per sample (ms)':<30} {v21_results['total_time']/total_samples*1000:<15.2f} {v30_results['total_time']/total_samples*1000:<15.2f} {v30_results['total_time']/v21_results['total_time']:<10.2f}x")
    
    # print("\n" + "-" * 70)
    # print("Summary:")
    # print("-" * 70)
    # speedup = v30_results['total_time'] / v21_results['total_time']
    # if speedup > 1:
    #     print(f"  v3.0 is {speedup:.2f}x SLOWER than v2.1")
    # else:
    #     print(f"  v3.0 is {1/speedup:.2f}x FASTER than v2.1")
    # print("-" * 70)


if __name__ == "__main__":
    main()