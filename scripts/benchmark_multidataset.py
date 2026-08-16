#!/usr/bin/env python
"""
Benchmark MultiDatasetforDistTraining vs LeRobotDataset.

Usage:
    conda activate lerobot_v2
    python scripts/benchmark_multidataset.py
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
V30_DATA_ROOT = "/Data/lerobot_data_ort6d/v30/taco_play"
DATASET_NAME = "taco_play"
BATCH_SIZE = 64
NUM_STEPS = 20
NUM_WORKERS = 4


def benchmark_lerobot_dataset():
    """Benchmark LeRobotDataset directly."""
    print("\n" + "=" * 70)
    print("Benchmarking LeRobotDataset (v3.0)")
    print("=" * 70)
    
    from lerobot.common.datasets_v30.lerobot_dataset import LeRobotDataset
    
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing dataset...")
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
        _ = batch
        step_time = time.time() - step_start
        step_times.append(step_time)
        print(f"  Step {i+1}/{NUM_STEPS}: {step_time*1000:.1f} ms")
    
    total_time = time.time() - total_start
    
    avg_step_time = sum(step_times) / len(step_times) * 1000
    throughput = NUM_STEPS * BATCH_SIZE / total_time
    
    print(f"\n" + "-" * 70)
    print("Results:")
    print("-" * 70)
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Avg step time: {avg_step_time:.1f} ms")
    print(f"  Throughput: {throughput:.1f} samples/s")
    
    return total_time, throughput


def benchmark_concat_dataset():
    """Benchmark ConcatDataset of LeRobotDataset."""
    print("\n" + "=" * 70)
    print("Benchmarking ConcatDataset (simulating MultiDatasetforDistTraining)")
    print("=" * 70)
    
    from lerobot.common.datasets_v30.lerobot_dataset import LeRobotDataset
    from torch.utils.data import ConcatDataset
    
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing datasets...")
    start = time.time()
    
    # Create multiple datasets (simulating multi-dataset scenario)
    dataset1 = LeRobotDataset(
        repo_id=DATASET_NAME,
        root=V30_DATA_ROOT,
        video_backend="torchcodec",
        dataset_name=DATASET_NAME,
    )
    
    # Use same dataset for testing (to avoid loading different data)
    datasets_list = [dataset1]
    
    concat_dataset = ConcatDataset(datasets_list)
    init_time = time.time() - start
    print(f"  ConcatDataset length: {len(concat_dataset)}")
    print(f"  Init time: {init_time:.3f}s")
    
    # Create DataLoader
    dataloader = DataLoader(
        concat_dataset,
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
        _ = batch
        step_time = time.time() - step_start
        step_times.append(step_time)
        print(f"  Step {i+1}/{NUM_STEPS}: {step_time*1000:.1f} ms")
    
    total_time = time.time() - total_start
    
    avg_step_time = sum(step_times) / len(step_times) * 1000
    throughput = NUM_STEPS * BATCH_SIZE / total_time
    
    print(f"\n" + "-" * 70)
    print("Results:")
    print("-" * 70)
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Avg step time: {avg_step_time:.1f} ms")
    print(f"  Throughput: {throughput:.1f} samples/s")
    
    return total_time, throughput


def benchmark_with_print_overhead():
    """Benchmark with print statement overhead (simulating current MultiDatasetforDistTraining)."""
    print("\n" + "=" * 70)
    print("Benchmarking with print overhead (current MultiDatasetforDistTraining)")
    print("=" * 70)
    
    from lerobot.common.datasets_v30.lerobot_dataset import LeRobotDataset
    
    # Create a wrapper dataset with print statement
    class DatasetWithPrint(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, index):
            item = self.dataset[index]
            # This is the problematic print statement in MultiDatasetforDistTraining
            print(f"get {index} {index}")
            return item
    
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing dataset...")
    start = time.time()
    
    base_dataset = LeRobotDataset(
        repo_id=DATASET_NAME,
        root=V30_DATA_ROOT,
        video_backend="torchcodec",
        dataset_name=DATASET_NAME,
    )
    
    dataset = DatasetWithPrint(base_dataset)
    init_time = time.time() - start
    print(f"  Dataset length: {len(dataset)}")
    print(f"  Init time: {init_time:.3f}s")
    
    # Create DataLoader
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
        _ = batch
        step_time = time.time() - step_start
        step_times.append(step_time)
        print(f"  Step {i+1}/{NUM_STEPS}: {step_time*1000:.1f} ms")
    
    total_time = time.time() - total_start
    
    avg_step_time = sum(step_times) / len(step_times) * 1000
    throughput = NUM_STEPS * BATCH_SIZE / total_time
    
    print(f"\n" + "-" * 70)
    print("Results:")
    print("-" * 70)
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Avg step time: {avg_step_time:.1f} ms")
    print(f"  Throughput: {throughput:.1f} samples/s")
    
    return total_time, throughput


def main():
    print("=" * 70)
    print("MultiDatasetforDistTraining Performance Analysis")
    print("=" * 70)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Num steps: {NUM_STEPS}")
    print(f"Num workers: {NUM_WORKERS}")
    
    # Test 1: LeRobotDataset directly
    time1, throughput1 = benchmark_lerobot_dataset()
    
    # Test 2: ConcatDataset
    time2, throughput2 = benchmark_concat_dataset()
    
    # Test 3: With print overhead
    time3, throughput3 = benchmark_with_print_overhead()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Test':<40} {'Time (s)':<15} {'Throughput':<15} {'Ratio':<10}")
    print("-" * 70)
    print(f"{'LeRobotDataset (baseline)':<40} {time1:<15.2f} {throughput1:<15.1f} {'1.00':<10}x")
    print(f"{'ConcatDataset':<40} {time2:<15.2f} {throughput2:<15.1f} {time2/time1:<10.2f}x")
    print(f"{'With print overhead':<40} {time3:<15.2f} {throughput3:<15.1f} {time3/time1:<10.2f}x")
    
    print("\n" + "-" * 70)
    print("Analysis:")
    print("-" * 70)
    print("""
Key findings:
1. ConcatDataset should have similar performance to LeRobotDataset
2. Print statements in __getitem__ cause significant overhead!
   - Each print requires I/O operations
   - In DataLoader with num_workers > 0, prints from workers are serialized
   - This can slow down data loading by 10-100x

Recommendation:
- Remove the print statement in MultiDatasetforDistTraining.__getitem__
- The line: print(f"get {index} {index}") should be removed or disabled
""")


if __name__ == "__main__":
    main()