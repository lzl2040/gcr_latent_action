#!/usr/bin/env python
"""
Detailed step-by-step timing analysis for LeRobotDataset and MultiDatasetforDistTraining __getitem__.

This script measures the time spent in each step of the __getitem__ method to identify bottlenecks.

Usage:
    conda activate lerobot_v2
    python scripts/benchmark_getitem_steps.py
"""

import time
import sys
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
V30_DATA_ROOT = "/Data/lerobot_data_ort6d/v30/taco_play"
DATASET_NAME = "taco_play"
NUM_SAMPLES = 50  # Number of samples to test
VIDEO_BACKEND = "torchcodec"  # Use torchcodec for v3.0


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


def benchmark_lerobot_dataset_getitem():
    """Benchmark each step of LeRobotDataset.__getitem__"""
    print("\n" + "=" * 70)
    print("Benchmarking LeRobotDataset.__getitem__ Step-by-Step")
    print("=" * 70)
    
    from lerobot.common.datasets_v30.lerobot_dataset import LeRobotDataset
    
    # Initialize dataset
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing dataset...")
    start = time.time()
    dataset = LeRobotDataset(
        repo_id=DATASET_NAME,
        root=V30_DATA_ROOT,
        video_backend=VIDEO_BACKEND,
        dataset_name=DATASET_NAME,
    )
    init_time = time.time() - start
    print(f"  Dataset length: {len(dataset)}")
    print(f"  Init time: {init_time:.3f}s")
    
    # Timing storage
    timings = defaultdict(list)
    
    # Test multiple samples
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing {NUM_SAMPLES} samples...")
    
    for i in range(NUM_SAMPLES):
        idx = i % len(dataset)
        
        # Step 1: _ensure_hf_dataset_loaded
        with TimingContext("step1_ensure_loaded", timings):
            dataset._ensure_hf_dataset_loaded()
        
        # Step 2: hf_dataset[idx]
        with TimingContext("step2_hf_dataset_access", timings):
            item = dataset.hf_dataset[idx]
        
        # Step 3: Extract episode and absolute index
        with TimingContext("step3_extract_indices", timings):
            ep_idx = item["episode_index"].item()
            abs_idx = item["index"].item()
        
        # Step 4: _get_query_indices (if delta_indices exists)
        query_indices = None
        if dataset.delta_indices is not None:
            with TimingContext("step4_get_query_indices", timings):
                query_indices, padding = dataset._get_query_indices(abs_idx, ep_idx)
            
            # Step 5: _query_hf_dataset
            with TimingContext("step5_query_hf_dataset", timings):
                query_result = dataset._query_hf_dataset(query_indices)
            
            with TimingContext("step5b_merge_item", timings):
                item = {**item, **padding}
                for key, val in query_result.items():
                    item[key] = val
        else:
            timings["step4_get_query_indices"].append(0)
            timings["step5_query_hf_dataset"].append(0)
            timings["step5b_merge_item"].append(0)
        
        # Step 6: Video processing
        if len(dataset.meta.video_keys) > 0:
            with TimingContext("step6_get_timestamp", timings):
                current_ts = item["timestamp"].item()
            
            with TimingContext("step6b_get_query_timestamps", timings):
                query_timestamps = dataset._get_query_timestamps(current_ts, query_indices)
            
            # Step 7: _query_videos (the most time-consuming part expected)
            with TimingContext("step7_query_videos", timings):
                video_frames = dataset._query_videos(query_timestamps, ep_idx)
            
            with TimingContext("step7b_merge_video", timings):
                item = {**video_frames, **item}
        else:
            timings["step6_get_timestamp"].append(0)
            timings["step6b_get_query_timestamps"].append(0)
            timings["step7_query_videos"].append(0)
            timings["step7b_merge_video"].append(0)
        
        # Step 8: Image transforms
        if dataset.image_transforms is not None:
            with TimingContext("step8_image_transforms", timings):
                image_keys = dataset.meta.camera_keys
                for cam in image_keys:
                    item[cam] = dataset.image_transforms(item[cam])
        else:
            timings["step8_image_transforms"].append(0)
        
        # Step 9: Add task
        with TimingContext("step9_add_task", timings):
            task_idx = item["task_index"].item()
            item["task"] = dataset.meta.tasks.iloc[task_idx].name
        
        # Step 10: Add subtask (if available)
        if "subtask_index" in dataset.features and dataset.meta.subtasks is not None:
            with TimingContext("step10_add_subtask", timings):
                subtask_idx = item["subtask_index"].item()
                item["subtask"] = dataset.meta.subtasks.iloc[subtask_idx].name
        else:
            timings["step10_add_subtask"].append(0)
        
        # Step 11: Add fps and dataset_name
        with TimingContext("step11_add_metadata", timings):
            import math
            item["fps"] = math.ceil(dataset.meta.fps)
            item["dataset_name"] = dataset.dataset_name
        
        # Step 12: Handle missing action key
        with TimingContext("step12_handle_action", timings):
            if "action" not in item.keys():
                if "observation.ee_ort6d_pos" not in item.keys():
                    item["observation.state"] = item["observations.ee_ort6d_pos"]
                else:
                    item["observation.state"] = item["observation.ee_ort6d_pos"]
                item["action"] = item["action.ee_ort6d_pos"]
    
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


def benchmark_video_decoding_detail():
    """Detailed benchmark of video decoding step."""
    print("\n" + "=" * 70)
    print("Detailed Video Decoding Analysis")
    print("=" * 70)
    
    from lerobot.common.datasets_v30.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets_v30.video_utils import decode_video_frames
    
    dataset = LeRobotDataset(
        repo_id=DATASET_NAME,
        root=V30_DATA_ROOT,
        video_backend="torchcodec",
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
            # Step 1: Get episode metadata
            with TimingContext("video_get_episode_meta", video_timings):
                ep = dataset.meta.episodes[ep_idx]
            
            # Step 2: Get from_timestamp
            with TimingContext("video_get_from_timestamp", video_timings):
                from_timestamp = ep[f"videos/{vid_key}/from_timestamp"]
            
            # Step 3: Build shifted timestamps
            current_ts = item["timestamp"].item()
            with TimingContext("video_build_timestamps", video_timings):
                query_ts = [current_ts]  # Simplified
                shifted_query_ts = [from_timestamp + ts for ts in query_ts]
            
            # Step 4: Get video file path
            with TimingContext("video_get_path", video_timings):
                video_path = dataset.root / dataset.meta.get_video_file_path(ep_idx, vid_key)
            
            # Step 5: Decode video frames
            with TimingContext("video_decode_frames", video_timings):
                frames = decode_video_frames(video_path, shifted_query_ts, dataset.tolerance_s, dataset.video_backend)
    
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


def benchmark_multidataset_getitem():
    """Benchmark MultiDatasetforDistTraining.__getitem__"""
    print("\n" + "=" * 70)
    print("Benchmarking MultiDatasetforDistTraining.__getitem__")
    print("=" * 70)
    
    # Note: This requires actual config and data setup
    # For now, we'll just show the structure
    print("""
MultiDatasetforDistTraining.__getitem__ steps:
    
1. id2dataset[index] lookup          - O(1) dict/list access
2. dataset[data_id] call             - Calls LeRobotDataset.__getitem__
3. OXE_DATASET_CONFIGS lookup        - O(1) dict access
4. _fetch_data_dict()                - Post-processing

_fetch_data_dict() steps:
  - Image key remapping
  - Missing image handling
  - Action/state padding
  - Normalization with stats
  - Building return dict
    """)
    
    return None


def compare_with_v21():
    """Compare with v2.1 implementation."""
    print("\n" + "=" * 70)
    print("Comparison with v2.1 Implementation")
    print("=" * 70)
    
    from lerobot.common.datasets.lerobot_dataset_for_ace import LeRobotDataset as LeRobotDatasetV21
    
    V21_DATA_ROOT = "/Data/lerobot_data_ort6d/taco_play"
    
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing v2.1 dataset...")
    start = time.time()
    dataset_v21 = LeRobotDatasetV21(
        repo_id=DATASET_NAME,
        root=V21_DATA_ROOT,
        video_backend="pyav",
        dataset_name=DATASET_NAME,
    )
    init_time_v21 = time.time() - start
    print(f"  Dataset length: {len(dataset_v21)}")
    print(f"  Init time: {init_time_v21:.3f}s")
    
    # Test v2.1 __getitem__
    v21_timings = defaultdict(list)
    
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing v2.1 __getitem__...")
    
    for i in range(NUM_SAMPLES):
        idx = i % len(dataset_v21)
        
        # Full __getitem__ timing
        with TimingContext("full_getitem", v21_timings):
            item = dataset_v21[idx]
    
    avg_v21 = sum(v21_timings["full_getitem"]) / len(v21_timings["full_getitem"]) * 1000
    print(f"  v2.1 Average __getitem__: {avg_v21:.3f} ms")
    
    return v21_timings


def main():
    print("=" * 70)
    print("LeRobot v3.0 __getitem__ Step-by-Step Timing Analysis")
    print("=" * 70)
    print(f"Dataset: {DATASET_NAME}")
    print(f"v3.0 Data: {V30_DATA_ROOT}")
    print(f"Test samples: {NUM_SAMPLES}")
    
    # Run benchmarks
    timings_v30 = benchmark_lerobot_dataset_getitem()
    video_timings = benchmark_video_decoding_detail()
    
    # Try to compare with v2.1
    try:
        timings_v21 = compare_with_v21()
    except Exception as e:
        print(f"\n[WARNING] Could not benchmark v2.1: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 70)
    
    total_v30 = sum(sum(v) for v in timings_v30.values())
    video_time = sum(timings_v30.get("step7_query_videos", [0]))
    
    print(f"""
Key Findings:
-------------
1. Total __getitem__ time (v3.0): {total_v30/NUM_SAMPLES*1000:.3f} ms

2. Video decoding accounts for {video_time/total_v30*100:.1f}% of total time

3. Main bottleneck: step7_query_videos
   - This includes: episode metadata lookup, video path construction, 
     timestamp shifting, and actual video frame decoding

Recommendations:
----------------
1. Cache episode metadata in memory to avoid repeated lookups
2. Pre-compute video file paths during dataset initialization
3. Use video decoder caching (e.g., keep video file handles open)
4. Consider using torchcodec with caching instead of pyav

Code-level optimizations:
- self.meta.episodes[ep_idx] is called every __getitem__ - consider caching
- self.meta.get_video_file_path() is called every time - consider pre-computing
- from_timestamp lookup adds overhead
""")


if __name__ == "__main__":
    main()