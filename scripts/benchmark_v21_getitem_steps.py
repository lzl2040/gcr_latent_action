#!/usr/bin/env python
"""
Detailed step-by-step timing analysis for LeRobotDataset v2.1 __getitem__.

This script measures the time spent in each step of the __getitem__ method to identify bottlenecks.

Usage:
    conda activate lerobot_v2
    python scripts/benchmark_v21_getitem_steps.py
"""

import time
import sys
import math
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# =============================================
# Configuration
# =============================================
V21_DATA_ROOT = "/Data/lerobot_data_ort6d/taco_play"
DATASET_NAME = "taco_play"
NUM_SAMPLES = 50  # Number of samples to test


class TimingContext:
    """Context manager for timing code blocks."""
    def __init__(self, name, timings_dict):
        self.name = name
        self.timings_dict = timings_dict
        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        elapsed = time.time() - self.start
        self.timings_dict[self.name].append(elapsed)


def benchmark_v21_getitem():
    """Benchmark each step of LeRobotDataset v2.1 __getitem__"""
    print("\n" + "=" * 70)
    print("Benchmarking LeRobotDataset v2.1 __getitem__ Step-by-Step")
    print("=" * 70)
    
    from lerobot.common.datasets.lerobot_dataset_for_ace import LeRobotDataset
    from lerobot.common.datasets.oxe_configs import OXE_DATASET_CONFIGS
    
    # Initialize dataset
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing v2.1 dataset...")
    start = time.time()
    dataset = LeRobotDataset(
        repo_id=DATASET_NAME,
        root=V21_DATA_ROOT,
        video_backend="pyav",
        dataset_name=DATASET_NAME,
    )
    init_time = time.time() - start
    print(f"  Dataset length: {len(dataset)}")
    print(f"  Init time: {init_time:.3f}s")
    print(f"  Video keys: {dataset.meta.video_keys}")
    print(f"  Camera keys: {dataset.meta.camera_keys}")
    
    # Timing storage
    timings = defaultdict(list)
    
    # Get primary obs key
    if OXE_DATASET_CONFIGS[dataset.dataset_name]["image_obs_keys"]["primary"] is not None:
        primary_obs_key = f"""observation.images.{OXE_DATASET_CONFIGS[dataset.dataset_name]["image_obs_keys"]["primary"]}"""
    else:
        primary_obs_key = "Zeus"
    
    # Test multiple samples
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing {NUM_SAMPLES} samples...")
    
    for i in range(NUM_SAMPLES):
        idx = i % len(dataset)
        
        # Step 1: hf_dataset[idx]
        with TimingContext("step1_hf_dataset_access", timings):
            item = dataset.hf_dataset[idx]
        
        # Step 2: Extract episode index
        with TimingContext("step2_extract_ep_idx", timings):
            ep_idx = item["episode_index"].item()
        
        # Step 3: Get query_indices (if delta_indices exists)
        query_indices = None
        if dataset.delta_indices is not None:
            with TimingContext("step3_get_query_indices", timings):
                query_indices, padding = dataset._get_query_indices(idx, ep_idx)
            
            # Step 4: _query_hf_dataset
            with TimingContext("step4_query_hf_dataset", timings):
                query_result = dataset._query_hf_dataset(query_indices)
            
            with TimingContext("step4b_merge_item", timings):
                item = {**item, **padding}
                for key, val in query_result.items():
                    item[key] = val
        else:
            timings["step3_get_query_indices"].append(0)
            timings["step4_query_hf_dataset"].append(0)
            timings["step4b_merge_item"].append(0)
        
        # Step 5: Video processing
        if len(dataset.meta.video_keys) > 0:
            with TimingContext("step5_get_timestamp", timings):
                current_ts = item["timestamp"].item()
            
            with TimingContext("step5b_get_query_timestamps", timings):
                query_timestamps = dataset._get_query_timestamps(current_ts, query_indices)
            
            # Step 6: _query_videos (the most time-consuming part expected)
            with TimingContext("step6_query_videos", timings):
                video_frames = dataset._query_videos(query_timestamps, ep_idx, primary_obs_key=primary_obs_key)
            
            with TimingContext("step6b_merge_video", timings):
                item = {**video_frames, **item}
        else:
            timings["step5_get_timestamp"].append(0)
            timings["step5b_get_query_timestamps"].append(0)
            timings["step6_query_videos"].append(0)
            timings["step6b_merge_video"].append(0)
        
        # Step 7: Image transforms
        if dataset.image_transforms is not None:
            with TimingContext("step7_image_transforms", timings):
                image_keys = dataset.meta.camera_keys
                for cam in image_keys:
                    item[cam] = dataset.image_transforms(item[cam])
        else:
            timings["step7_image_transforms"].append(0)
        
        # Step 8: Add task
        with TimingContext("step8_add_task", timings):
            task_idx = item["task_index"].item()
            task = dataset.meta.tasks[task_idx]
            item["task"] = task
        
        # Step 9: Add fps and dataset_name
        with TimingContext("step9_add_metadata", timings):
            item["fps"] = math.ceil(dataset.meta.info["fps"])
            item["dataset_name"] = dataset.dataset_name
    
    # Calculate and print statistics
    print("\n" + "-" * 70)
    print("Step-by-Step Timing Results (averaged over {} samples)".format(NUM_SAMPLES))
    print("-" * 70)
    print(f"{'Step':<40} {'Avg (ms)':<12} {'Total (ms)':<12} {'%':<8}")
    print("-" * 70)
    
    total_time = sum(sum(v) for v in timings.values())
    
    # Sort by total time
    sorted_timings = sorted(timings.items(), key=lambda x: sum(x[1]), reverse=True)
    
    for step_name, times in sorted_timings:
        if len(times) == 0:
            continue
        avg_time = sum(times) / len(times) * 1000  # Convert to ms
        total_step_time = sum(times) * 1000
        percentage = (sum(times) / total_time * 100) if total_time > 0 else 0
        print(f"{step_name:<40} {avg_time:<12.3f} {total_step_time:<12.3f} {percentage:<8.1f}%")
    
    print("-" * 70)
    print(f"{'TOTAL':<40} {total_time/NUM_SAMPLES*1000:<12.3f} {total_time*1000:<12.3f} {'100.0':<8}%")
    print("-" * 70)
    
    return timings


def benchmark_v21_video_decoding_detail():
    """Detailed benchmark of video decoding step for v2.1."""
    print("\n" + "=" * 70)
    print("Detailed Video Decoding Analysis for v2.1")
    print("=" * 70)
    
    from lerobot.common.datasets.lerobot_dataset_for_ace import LeRobotDataset
    from lerobot.common.datasets.video_utils import decode_video_frames_torchcodec
    
    dataset = LeRobotDataset(
        repo_id=DATASET_NAME,
        root=V21_DATA_ROOT,
        video_backend="pyav",
        dataset_name=DATASET_NAME,
    )
    
    # Test video decoding components
    video_timings = defaultdict(list)
    
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing video decoding details...")
    
    for i in range(min(NUM_SAMPLES, 20)):
        idx = i % len(dataset)
        item = dataset.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        
        if len(dataset.meta.video_keys) == 0:
            continue
        
        # Test each video key
        for vid_key in dataset.meta.video_keys:
            # Step 1: Get video file path
            with TimingContext("video_get_path", video_timings):
                video_path = dataset.root / dataset.meta.get_video_file_path(ep_idx, vid_key)
            
            # Step 2: Build query timestamps
            current_ts = item["timestamp"].item()
            with TimingContext("video_build_timestamps", video_timings):
                query_ts = [current_ts]
            
            # Step 3: Decode video frames
            with TimingContext("video_decode_frames", video_timings):
                frames = decode_video_frames_torchcodec(video_path, query_ts, dataset.tolerance_s, return_type="tensor")
    
    # Print video timing results
    print("\n" + "-" * 70)
    print("Video Decoding Sub-steps (averaged)")
    print("-" * 70)
    
    total_video_time = sum(sum(v) for v in video_timings.values())
    
    for step_name, times in sorted(video_timings.items(), key=lambda x: sum(x[1]), reverse=True):
        if len(times) == 0:
            continue
        avg_time = sum(times) / len(times) * 1000
        percentage = (sum(times) / total_video_time * 100) if total_video_time > 0 else 0
        print(f"  {step_name:<35} {avg_time:>8.3f} ms  ({percentage:>5.1f}%)")
    
    return video_timings


def compare_video_backends():
    """Compare different video backends for v2.1."""
    print("\n" + "=" * 70)
    print("Comparing Video Backends for v2.1")
    print("=" * 70)
    
    from lerobot.common.datasets.lerobot_dataset_for_ace import LeRobotDataset
    
    backends = ["pyav", "torchcodec"]
    results = {}
    
    for backend in backends:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing with backend: {backend}")
        try:
            dataset = LeRobotDataset(
                repo_id=DATASET_NAME,
                root=V21_DATA_ROOT,
                video_backend=backend,
                dataset_name=DATASET_NAME,
            )
            
            times = []
            for i in range(min(NUM_SAMPLES, 20)):
                idx = i % len(dataset)
                start = time.time()
                item = dataset[idx]
                times.append(time.time() - start)
            
            avg_time = sum(times) / len(times) * 1000
            results[backend] = avg_time
            print(f"  Average __getitem__ time: {avg_time:.3f} ms")
        except Exception as e:
            print(f"  Error with {backend}: {e}")
            results[backend] = None
    
    print("\n" + "-" * 70)
    print("Backend Comparison Summary")
    print("-" * 70)
    for backend, avg_time in results.items():
        if avg_time is not None:
            print(f"  {backend:<20} {avg_time:>10.3f} ms")
        else:
            print(f"  {backend:<20} {'FAILED':>10}")
    
    return results


def main():
    print("=" * 70)
    print("LeRobot v2.1 __getitem__ Step-by-Step Timing Analysis")
    print("=" * 70)
    print(f"Dataset: {DATASET_NAME}")
    print(f"v2.1 Data: {V21_DATA_ROOT}")
    print(f"Test samples: {NUM_SAMPLES}")
    
    # Run benchmarks
    timings_v21 = benchmark_v21_getitem()
    video_timings = benchmark_v21_video_decoding_detail()
    
    # Compare video backends
    try:
        backend_results = compare_video_backends()
    except Exception as e:
        print(f"\n[WARNING] Could not compare backends: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_v21 = sum(sum(v) for v in timings_v21.values())
    video_time = sum(timings_v21.get("step6_query_videos", [0]))
    
    print(f"""
Key Findings (v2.1):
--------------------
1. Total __getitem__ time (v2.1): {total_v21/NUM_SAMPLES*1000:.3f} ms

2. Video decoding accounts for {video_time/total_v21*100:.1f}% of total time

3. v2.1 uses decode_video_frames_torchcodec directly without additional overhead

Key Differences from v3.0:
--------------------------
1. v2.1 does NOT have from_timestamp shifting
   - v3.0 reads from_timestamp from episode metadata and shifts timestamps
   - This adds overhead in v3.0

2. v2.1 uses simpler video path construction
   - v3.0 uses meta.get_video_file_path() with more complex logic

3. v2.1 episode metadata is simpler (JSON-based)
   - v3.0 uses parquet-based episode metadata
""")


if __name__ == "__main__":
    main()