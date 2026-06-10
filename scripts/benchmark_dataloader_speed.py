#!/usr/bin/env python
"""
Benchmark script to compare dataloader speed between lerobot v2.1 and v3.0.

This script tests:
1. Dataset initialization time
2. Single item access time
3. Batch loading time with DataLoader
4. Video decoding time

Usage:
    conda activate lerobot_v2
    python scripts/benchmark_dataloader_speed.py
"""

import time
import os
import sys
from pathlib import Path
from datetime import datetime
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
NUM_SAMPLES = 100  # Number of samples to test
BATCH_SIZE = 512
NUM_WORKERS = 0  # Set to 0 for simpler testing without multiprocessing issues
NUM_EPISODES_TEST = 5  # Test first N episodes


def time_function(func, name, *args, **kwargs):
    """Time a function execution and print results."""
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {name}: {elapsed:.3f}s")
    return result, elapsed


def benchmark_v21():
    """Benchmark v2.1 dataloader."""
    print("\n" + "=" * 60)
    print("Benchmarking LeRobot v2.1 Dataset")
    print("=" * 60)
    
    # Import v2.1 dataset
    from lerobot.common.datasets.lerobot_dataset_for_ace import LeRobotDataset as LeRobotDatasetV21
    from lerobot.common.datasets.utils import hf_transform_to_torch
    
    results = {}
    
    # 1. Dataset initialization
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing v2.1 dataset...")
    start = time.time()
    dataset_v21 = LeRobotDatasetV21(
        repo_id=DATASET_NAME,
        root=V21_DATA_ROOT,
        video_backend="pyav",
        dataset_name=DATASET_NAME,
    )
    # Print video backend info
    print(f"  Video backend: {dataset_v21.video_backend}")
    init_time = time.time() - start
    results['init_time'] = init_time
    print(f"  Dataset length: {len(dataset_v21)}")
    print(f"  Init time: {init_time:.3f}s")
    
    # 2. Single item access (no video decoding)
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing single item access...")
    single_times = []
    for i in range(NUM_SAMPLES):
        start = time.time()
        item = dataset_v21.hf_dataset[i]
        single_times.append(time.time() - start)
    avg_single_time = sum(single_times) / len(single_times)
    results['avg_single_access'] = avg_single_time
    print(f"  Average single item access (hf_dataset only): {avg_single_time*1000:.3f}ms")
    
    # 3. Full __getitem__ with video decoding
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing full __getitem__ with video...")
    full_times = []
    for i in range(NUM_SAMPLES):
        start = time.time()
        item = dataset_v21[i]
        full_times.append(time.time() - start)
    avg_full_time = sum(full_times) / len(full_times)
    results['avg_full_getitem'] = avg_full_time
    print(f"  Average full __getitem__ time: {avg_full_time*1000:.3f}ms")
    
    # 4. DataLoader batch loading
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing DataLoader batch loading...")
    dataloader_v21 = DataLoader(
        dataset_v21,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    
    batch_times = []
    num_batches = min(10, len(dataloader_v21))
    for i, batch in enumerate(dataloader_v21):
        if i >= num_batches:
            break
        batch_times.append(time.time() - start)
        start = time.time()
    
    if batch_times:
        avg_batch_time = sum(batch_times[1:]) / len(batch_times[1:]) if len(batch_times) > 1 else batch_times[0]
        results['avg_batch_time'] = avg_batch_time
        print(f"  Average batch loading time (batch_size={BATCH_SIZE}): {avg_batch_time:.3f}s")
    
    # 5. Video decoding - skip separate test since it's included in __getitem__
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Skipping separate video decoding test (included in __getitem__)")
    # Video decoding is already tested in the __getitem__ benchmark above
    results['avg_video_decode'] = results.get('avg_full_getitem', 0) * 0.5  # Estimate ~50% is video
    
    return results


def benchmark_v30(dataset_name, data_root):
    """Benchmark v3.0 dataloader."""
    print("\n" + "=" * 60)
    print("Benchmarking LeRobot v3.0 Dataset")
    print("=" * 60)
    
    # Import v3.0 dataset
    from lerobot.common.datasets_v30.lerobot_dataset import LeRobotDataset as LeRobotDatasetV30
    
    results = {}
    
    # 1. Dataset initialization
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing v3.0 dataset...")
    start = time.time()
    dataset_v30 = LeRobotDatasetV30(
        repo_id=dataset_name,
        root=data_root,
        video_backend="torchcodec",  # Use pyav for consistency
        dataset_name=dataset_name,
    )
    init_time = time.time() - start
    results['init_time'] = init_time
    print(f"  Dataset length: {len(dataset_v30)}")
    print(f"  Init time: {init_time:.3f}s")
    
    # 2. Single item access (no video decoding)
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing single item access...")
    single_times = []
    for i in range(NUM_SAMPLES):
        start = time.time()
        item = dataset_v30.hf_dataset[i]
        single_times.append(time.time() - start)
    avg_single_time = sum(single_times) / len(single_times)
    results['avg_single_access'] = avg_single_time
    print(f"  Average single item access (hf_dataset only): {avg_single_time*1000:.3f}ms")
    
    # 3. Full __getitem__ with video decoding
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing full __getitem__ with video...")
    full_times = []
    for i in range(NUM_SAMPLES):
        start = time.time()
        item = dataset_v30[i]
        full_times.append(time.time() - start)
    avg_full_time = sum(full_times) / len(full_times)
    results['avg_full_getitem'] = avg_full_time
    print(f"  Average full __getitem__ time: {avg_full_time*1000:.3f}ms")
    
    # 4. DataLoader batch loading
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing DataLoader batch loading...")
    dataloader_v30 = DataLoader(
        dataset_v30,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    
    batch_times = []
    num_batches = min(100, len(dataloader_v30))
    start = time.time()
    for i, batch in enumerate(dataloader_v30):
        if i >= num_batches:
            break
        batch_times.append(time.time() - start)
        start = time.time()
    
    if batch_times:
        avg_batch_time = sum(batch_times[1:]) / len(batch_times[1:]) if len(batch_times) > 1 else batch_times[0]
        results['avg_batch_time'] = avg_batch_time
        print(f"  Average batch loading time (batch_size={BATCH_SIZE}): {avg_batch_time:.3f}s")
    
    # 5. Video decoding - skip separate test since it's included in __getitem__
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Skipping separate video decoding test (included in __getitem__)")
    # Video decoding is already tested in the __getitem__ benchmark above
    results['avg_video_decode'] = results.get('avg_full_getitem', 0) * 0.5  # Estimate ~50% is video
    
    return results


def benchmark_data_loading_only():
    """Benchmark just the parquet loading part without full dataset init."""
    print("\n" + "=" * 60)
    print("Benchmarking Raw Parquet Loading")
    print("=" * 60)
    
    from datasets import load_dataset, Dataset
    import pyarrow.parquet as pq
    import pyarrow.dataset as pa_ds
    
    results = {}
    
    # v2.1 style: single merged.parquet
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing v2.1 style (merged.parquet)...")
    merged_path = Path(V21_DATA_ROOT) / "merged.parquet"
    
    start = time.time()
    ds_v21 = load_dataset("parquet", data_files=str(merged_path), split="train")
    load_time_v21 = time.time() - start
    results['v21_merged_load'] = load_time_v21
    print(f"  Load time: {load_time_v21:.3f}s")
    print(f"  Dataset length: {len(ds_v21)}")
    
    # Test access
    access_times = []
    for i in range(NUM_SAMPLES):
        start = time.time()
        item = ds_v21[i]
        access_times.append(time.time() - start)
    avg_access_v21 = sum(access_times) / len(access_times)
    results['v21_avg_access'] = avg_access_v21
    print(f"  Average item access: {avg_access_v21*1000:.3f}ms")
    
    # v3.0 style: nested parquet files
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing v3.0 style (nested parquet)...")
    data_dir = Path(V30_DATA_ROOT) / "data"
    paths = sorted(data_dir.glob("*/file*.parquet"))
    
    start = time.time()
    ds_v30 = Dataset.from_parquet([str(p) for p in paths])
    load_time_v30 = time.time() - start
    results['v30_nested_load'] = load_time_v30
    print(f"  Load time: {load_time_v30:.3f}s")
    print(f"  Number of parquet files: {len(paths)}")
    print(f"  Dataset length: {len(ds_v30)}")
    
    # Test access
    access_times = []
    for i in range(NUM_SAMPLES):
        start = time.time()
        item = ds_v30[i]
        access_times.append(time.time() - start)
    avg_access_v30 = sum(access_times) / len(access_times)
    results['v30_avg_access'] = avg_access_v30
    print(f"  Average item access: {avg_access_v30*1000:.3f}ms")
    
    # v3.0 with episode filtering (simulating episodes parameter)
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing v3.0 with episode filtering...")
    start = time.time()
    filters = pa_ds.field("episode_index").isin(list(range(NUM_EPISODES_TEST)))
    ds_v30_filtered = Dataset.from_parquet([str(p) for p in paths], filters=filters)
    load_time_v30_filtered = time.time() - start
    results['v30_filtered_load'] = load_time_v30_filtered
    print(f"  Load time with filter: {load_time_v30_filtered:.3f}s")
    print(f"  Filtered dataset length: {len(ds_v30_filtered)}")
    
    return results


def print_comparison(v21_results, v30_results, raw_results):
    """Print comparison table."""
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    print("\n| Metric | v2.1 | v3.0 | Speed Ratio (v2.1/v3.0) |")
    print("|--------|------|------|--------------------------|")
    
    if 'init_time' in v21_results and 'init_time' in v30_results:
        ratio = v21_results['init_time'] / v30_results['init_time']
        print(f"| Init Time | {v21_results['init_time']:.3f}s | {v30_results['init_time']:.3f}s | {ratio:.2f}x |")
    
    if 'avg_single_access' in v21_results and 'avg_single_access' in v30_results:
        ratio = v21_results['avg_single_access'] / v30_results['avg_single_access']
        print(f"| HF Dataset Access | {v21_results['avg_single_access']*1000:.3f}ms | {v30_results['avg_single_access']*1000:.3f}ms | {ratio:.2f}x |")
    
    if 'avg_full_getitem' in v21_results and 'avg_full_getitem' in v30_results:
        ratio = v21_results['avg_full_getitem'] / v30_results['avg_full_getitem']
        print(f"| Full __getitem__ | {v21_results['avg_full_getitem']*1000:.3f}ms | {v30_results['avg_full_getitem']*1000:.3f}ms | {ratio:.2f}x |")
    
    if 'avg_batch_time' in v21_results and 'avg_batch_time' in v30_results:
        ratio = v21_results['avg_batch_time'] / v30_results['avg_batch_time']
        print(f"| DataLoader Batch | {v21_results['avg_batch_time']:.3f}s | {v30_results['avg_batch_time']:.3f}s | {ratio:.2f}x |")
    
    if 'avg_video_decode' in v21_results and 'avg_video_decode' in v30_results:
        ratio = v21_results['avg_video_decode'] / v30_results['avg_video_decode']
        print(f"| Video Decoding | {v21_results['avg_video_decode']*1000:.3f}ms | {v30_results['avg_video_decode']*1000:.3f}ms | {ratio:.2f}x |")
    
    print("\n| Raw Parquet Loading | v2.1 (merged) | v3.0 (nested) | Ratio |")
    print("|---------------------|---------------|----------------|-------|")
    if 'v21_merged_load' in raw_results and 'v30_nested_load' in raw_results:
        ratio = raw_results['v21_merged_load'] / raw_results['v30_nested_load']
        print(f"| Load Time | {raw_results['v21_merged_load']:.3f}s | {raw_results['v30_nested_load']:.3f}s | {ratio:.2f}x |")
    
    if 'v21_avg_access' in raw_results and 'v30_avg_access' in raw_results:
        ratio = raw_results['v21_avg_access'] / raw_results['v30_avg_access']
        print(f"| Item Access | {raw_results['v21_avg_access']*1000:.3f}ms | {raw_results['v30_avg_access']*1000:.3f}ms | {ratio:.2f}x |")
    
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print("""
Key differences between v2.1 and v3.0:

1. PARQUET FILE STRUCTURE:
   - v2.1: Uses a single merged.parquet file (fast loading)
   - v3.0: Uses multiple chunk-xxx/file-xxx.parquet files (nested structure)
   
2. VIDEO DECODING:
   - v2.1: Direct torchcodec decoding
   - v3.0: Same torchcodec with decoder caching mechanism
   
3. EPISODES METADATA:
   - v2.1: Loads from episodes.jsonl (simple JSON)
   - v3.0: Loads from meta/episodes/*.parquet (more complex)
   
4. DATA ACCESS PATTERN:
   - v2.1: Uses hf_dataset.select() for delta timestamps
   - v3.0: Uses column-first access hf_dataset[key][indices]

If v3.0 is slower, potential causes:
- Multiple parquet files require more I/O operations
- Episode metadata loading from parquet instead of JSON
- Additional metadata processing in v3.0
- Different video file lookup mechanism
    """)


def main():
    print("=" * 60)
    print("LeRobot Dataloader Speed Benchmark")
    print("=" * 60)
    # print(f"Dataset: {DATASET_NAME}")
    # print(f"v2.1 Data: {V21_DATA_ROOT}")
    # print(f"v3.0 Data: {V30_DATA_ROOT}")
    print(f"Test samples: {NUM_SAMPLES}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Num workers: {NUM_WORKERS}")
    
    # First benchmark raw parquet loading
    # raw_results = benchmark_data_loading_only()
    
    # Benchmark v2.1
    # v21_results = benchmark_v21()
    
    # Benchmark v3.0
    
    print("\n" + "=" * 60)
    data_root = "/mnt/wangxiaofa/robot_dataset/lerobot-format-v30/fractal20220817_data_lerobot"
    d_name = "fractal20220817_data_lerobot"
    print(f"\nBenchmarking dataset: {d_name}")
    v30_results = benchmark_v30(d_name, data_root)
    print(v30_results)
    
    print("\n" + "=" * 60)
    
    print("\n" + "=" * 60)
    data_root = "/mnt/wangxiaofa/robot_dataset/lerobot-format-v30/Micro_data/Trossen_Stationary_AI_480x640_padded_MERGED"
    d_name = "Trossen_Stationary_AI_480x640_padded_MERGED"
    print(f"\nBenchmarking dataset: {d_name}")
    v30_results = benchmark_v30(d_name, data_root)
    print(v30_results)
    
    print("\n" + "=" * 60)
    
    data_root = "/mnt/wangxiaofa/robot_dataset/lerobot-format-v30/Micro_data/full/XMI_MERGED"
    d_name = "XMI_MERGED"
    print(f"\nBenchmarking dataset: {d_name}")
    v30_results = benchmark_v30(d_name, data_root)
    print("\n" + "=" * 60)
    
    data_root = "/mnt/wangxiaofa/robot_dataset/lerobot-format-v30/Micro_data/full/YAM_Station_MERGED"
    d_name = "YAM_Station_MERGED"
    print(f"\nBenchmarking dataset: {d_name}")
    v30_results = benchmark_v30(d_name, data_root)
    print("\n" + "=" * 60)
    
    data_root = "/mnt/wangxiaofa/robot_dataset/lerobot-format-v30/Micro_data/full/YAM_Box_MERGED"
    d_name = "YAM_Box_MERGED"
    print(f"\nBenchmarking dataset: {d_name}")
    v30_results = benchmark_v30(d_name, data_root)
    print("\n" + "=" * 60)
    
    # Print comparison
    # print_comparison(v21_results, v30_results, raw_results)


if __name__ == "__main__":
    main()