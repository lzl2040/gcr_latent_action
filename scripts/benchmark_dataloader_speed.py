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
from torchvision import transforms as T

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# =============================================
# Configuration
# =============================================
V21_DATA_ROOT = "/Data/lerobot_data_ort6d/taco_play"
V30_DATA_ROOT = "/Data/lerobot_data_ort6d/v30/YAM_Station_MERGED"
DATASET_NAME = "taco_play"
NUM_SAMPLES = 100  # Number of samples to test
BATCH_SIZE = 512
NUM_WORKERS = 4  # Set to 0 for simpler testing without multiprocessing issues
NUM_EPISODES_TEST = 5  # Test first N episodes
V30_VIDEO_RETURN_TYPE = "uint8"  # Use uint8 to avoid moving multi-GB float32 image batches between workers.
V30_PIN_MEMORY = False  # Large v30 video batches can fail in CUDA's pin-memory thread.


def clear_v30_decoder_cache():
    """Drop torchcodec decoders before forking DataLoader workers."""
    try:
        from lerobot.common.datasets_v30 import video_utils
    except Exception as exc:
        print(f"  Warning: failed to import v3.0 video_utils while clearing cache: {exc}")
        return

    decoder_cache = getattr(video_utils, "_default_decoder_cache", None)
    if decoder_cache is None:
        return

    try:
        decoder_cache.clear()
    except Exception as exc:
        print(f"  Warning: decoder_cache.clear() failed ({exc}); clearing cache dict directly.")
        cache = getattr(decoder_cache, "_cache", None)
        lock = getattr(decoder_cache, "_lock", None)
        if cache is None:
            return
        if lock is None:
            cache.clear()
        else:
            with lock:
                cache.clear()


def dataloader_worker_init_fn(_worker_id):
    clear_v30_decoder_cache()


def lola_dataloader_worker_init_fn(_worker_id):
    from torch.utils.data import get_worker_info

    worker_info = get_worker_info()
    if worker_info is None:
        return
    dataset = worker_info.dataset
    for attr in ("video_decoder_cache", "_cuda_decoder_cache"):
        cache = getattr(dataset, attr, None)
        if cache is not None:
            try:
                cache.clear()
            except Exception:
                pass
        setattr(dataset, attr, None)


def clear_lola_decoder_cache(dataset):
    for attr in ("video_decoder_cache", "_cuda_decoder_cache"):
        cache = getattr(dataset, attr, None)
        if cache is not None:
            try:
                cache.clear()
            except Exception:
                pass
        setattr(dataset, attr, None)


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
    img_transforms = T.Compose([
        T.Resize((224, 224)),
    ])
    dataset_v21 = LeRobotDatasetV21(
        repo_id=DATASET_NAME,
        root=V21_DATA_ROOT,
        video_backend="pyav",
        dataset_name=DATASET_NAME,
        image_transforms=img_transforms
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


def benchmark_v30():
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
    img_transforms = T.Compose([
        T.Resize((224, 224)),
    ])
    # img_transforms = None
    dataset_v30 = LeRobotDatasetV30(
        repo_id=DATASET_NAME,
        root=V30_DATA_ROOT,
        video_backend="torchcodec",  # Use pyav for consistency
        # video_backend="torchcodec",
        dataset_name=DATASET_NAME,
        image_transforms=img_transforms,
        video_return_type=V30_VIDEO_RETURN_TYPE,
    )
    init_time = time.time() - start
    results['init_time'] = init_time
    print(f"  Dataset length: {len(dataset_v30)}")
    print(f"  Video return type: {V30_VIDEO_RETURN_TYPE}")
    print(f"  v30 pin_memory: {V30_PIN_MEMORY}")
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
    # Note: pin_memory=True with num_workers > 0 can cause CUDA errors in multiprocessing.
    # Use pin_memory=False when using multiprocessing, or set num_workers=0 for CUDA operations.
    if NUM_WORKERS > 0:
        # The full __getitem__ benchmark above opens torchcodec decoders in the main
        # process. If those decoder objects are inherited by forked workers, torchcodec
        # can fail with "Could not push packet to decoder".
        clear_v30_decoder_cache()
    dataloader_v30 = DataLoader(
        dataset_v30,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=V30_PIN_MEMORY,
        worker_init_fn=dataloader_worker_init_fn if NUM_WORKERS > 0 else None,
        # persistent_workers=True,
        # pin_memory=False,  # Must be False when using num_workers > 0 to avoid CUDA multiprocessing errors
        # multiprocessing_context="spawn" if NUM_WORKERS > 0 else None,  # Use spawn if workers > 0
    )
    
    batch_times = []
    num_batches = min(5, len(dataloader_v30))
    start = time.time()
    for i, batch in enumerate(dataloader_v30):
        if i >= num_batches:
            break
        batch_times.append(time.time() - start)
        start = time.time()
    
    if batch_times:
        avg_batch_time = sum(batch_times[1:]) / len(batch_times[1:]) if len(batch_times) > 1 else batch_times[0]
        results['avg_batch_time'] = avg_batch_time
        print(f"  Average batch loading time (batch_size={BATCH_SIZE}, w={NUM_WORKERS}): {avg_batch_time:.3f}s")

    # 4b. DataLoader with num_workers=0 (no worker contention)
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing DataLoader batch loading (w=0, no contention)...")
    dataloader_v30_w0 = DataLoader(
        dataset_v30,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    batch_times_w0 = []
    num_batches = min(5, len(dataloader_v30_w0))
    start = time.time()
    for i, batch in enumerate(dataloader_v30_w0):
        if i >= num_batches:
            break
        batch_times_w0.append(time.time() - start)
        start = time.time()

    if batch_times_w0:
        avg_batch_time_w0 = sum(batch_times_w0[1:]) / len(batch_times_w0[1:]) if len(batch_times_w0) > 1 else batch_times_w0[0]
        results['avg_batch_time_w0'] = avg_batch_time_w0
        print(f"  Average batch loading time (batch_size={BATCH_SIZE}, w=0): {avg_batch_time_w0:.3f}s")

    # 4c. DataLoader with num_workers=2 (reduce CPU contention)
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing DataLoader batch loading (w=2)...")
    clear_v30_decoder_cache()
    dataloader_v30_w2 = DataLoader(
        dataset_v30,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=V30_PIN_MEMORY,
        worker_init_fn=dataloader_worker_init_fn,
    )
    batch_times_w2 = []
    num_batches = min(5, len(dataloader_v30_w2))
    start = time.time()
    for i, batch in enumerate(dataloader_v30_w2):
        if i >= num_batches:
            break
        batch_times_w2.append(time.time() - start)
        start = time.time()

    if batch_times_w2:
        avg_batch_time_w2 = sum(batch_times_w2[1:]) / len(batch_times_w2[1:]) if len(batch_times_w2) > 1 else batch_times_w2[0]
        results['avg_batch_time_w2'] = avg_batch_time_w2
        print(f"  Average batch loading time (batch_size={BATCH_SIZE}, w=2): {avg_batch_time_w2:.3f}s")

    # 4d. DataLoader with num_workers=4 + persistent_workers + prefetch_factor=4
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing DataLoader batch loading (w=4, persistent, prefetch=4)...")
    clear_v30_decoder_cache()
    dataloader_v30_persistent = DataLoader(
        dataset_v30,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=V30_PIN_MEMORY,
        worker_init_fn=dataloader_worker_init_fn,
        persistent_workers=True,
        prefetch_factor=4,
    )
    batch_times_persistent = []
    num_batches = min(10, len(dataloader_v30_persistent))
    start = time.time()
    for i, batch in enumerate(dataloader_v30_persistent):
        if i >= num_batches:
            break
        batch_times_persistent.append(time.time() - start)
        start = time.time()

    if batch_times_persistent:
        # Skip first 2 batches for persistent workers (warmup)
        warm = batch_times_persistent[2:] if len(batch_times_persistent) > 2 else batch_times_persistent[1:]
        avg_batch_time_persistent = sum(warm) / len(warm) if warm else batch_times_persistent[0]
        results['avg_batch_time_w4_persistent'] = avg_batch_time_persistent
        print(f"  Average batch loading time (batch_size={BATCH_SIZE}, w=4 persistent, prefetch=4): {avg_batch_time_persistent:.3f}s")
        print(f"  (warmup excluded, measured {len(warm)} batches)")

    # 5. Video decoding - skip separate test since it's included in __getitem__
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Skipping separate video decoding test (included in __getitem__)")
    # Video decoding is already tested in the __getitem__ benchmark above
    results['avg_video_decode'] = results.get('avg_full_getitem', 0) * 0.5  # Estimate ~50% is video

    return results


def benchmark_lola_v30():
    """Benchmark v3.0 with LoLADataset/LoLAPretrainDataset."""
    print("\n" + "=" * 60)
    print("Benchmarking LoLA v3.0 Dataset")
    print("=" * 60)

    from lerobot.common.datasets_v30.efficient_lerobot_dataset import (
        LoLAPretrainDataset as LoLADataset,
        make_collate_fn,
    )

    results = {}

    # 1. Dataset initialization
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing LoLA v3.0 dataset...")
    start = time.time()
    dataset_lola = LoLADataset(
        repo_id=DATASET_NAME,
        root=V30_DATA_ROOT,
        max_history_length=100,
        action_chunk_size=10,
        decode_device="cuda",
    )
    init_time = time.time() - start
    results["init_time"] = init_time
    print(f"  Dataset length: {len(dataset_lola)}")
    print(f"  Init time: {init_time:.3f}s")

    # 2. Single item access
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing LoLA single item access...")
    single_times = []
    for i in range(NUM_SAMPLES):
        start = time.time()
        _ = dataset_lola[i]
        single_times.append(time.time() - start)
    avg_single_time = sum(single_times) / len(single_times)
    results["avg_full_getitem"] = avg_single_time
    print(f"  Average LoLA __getitem__ time: {avg_single_time*1000:.3f}ms")

    # 3. DataLoader batch loading
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing LoLA DataLoader batch loading...")
    if NUM_WORKERS > 0:
        clear_lola_decoder_cache(dataset_lola)
    dataloader_lola = DataLoader(
        dataset_lola,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        persistent_workers=NUM_WORKERS > 0,
        worker_init_fn=lola_dataloader_worker_init_fn if NUM_WORKERS > 0 else None,
        collate_fn=make_collate_fn(),
    )

    batch_times = []
    num_batches = min(5, len(dataloader_lola))
    start = time.time()
    for i, batch in enumerate(dataloader_lola):
        if i >= num_batches:
            break
        batch_times.append(time.time() - start)
        if i == 0:
            image_keys = [key for key in batch if key.startswith("observation.images.")]
            print(f"  LoLA image keys: {image_keys}")
            for key in image_keys[:2]:
                first_valid = next((img for img in batch[key] if img is not None), None)
                print(f"    {key}: batch list={len(batch[key])}, first_valid={type(first_valid).__name__}")
        start = time.time()

    if batch_times:
        avg_batch_time = sum(batch_times[1:]) / len(batch_times[1:]) if len(batch_times) > 1 else batch_times[0]
        results["avg_batch_time"] = avg_batch_time
        print(f"  Average LoLA batch loading time (batch_size={BATCH_SIZE}): {avg_batch_time:.3f}s")

    return results


def benchmark_streaming_v30():
    """Benchmark LoLAPretrainStreamingDataset (IterableDataset) on v3.0 data."""
    print("\n" + "=" * 60)
    print("Benchmarking LoLAPretrainStreamingDataset (v3.0 Streaming)")
    print("=" * 60)

    from lerobot.common.datasets_v30.pretrain_streaming_dataset import (
        LoLAPretrainStreamingDataset,
        AsyncDecodeDataLoader,
    )

    results = {}

    # 1. Dataset initialization
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing LoLAPretrainStreamingDataset...")
    start = time.time()
    dataset_streaming = LoLAPretrainStreamingDataset(
        repo_id=DATASET_NAME,
        root=V30_DATA_ROOT,
        max_history_length=100,
        action_chunk_size=10,
        delta_timestamps={
            "observation.images.top": [0],
            "observation.state": [0],
            "action": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        },
        deferred_video_decode=True,
        decode_device="cpu",
        decode_num_threads=1,
        shuffle=False,  # No shuffle for reproducible benchmark
        buffer_size=100,  # Small buffer for faster iteration
    )
    init_time = time.time() - start
    results["init_time"] = init_time
    print(f"  Init time: {init_time:.3f}s")

    # 2. Single item access via iterator
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing streaming single item access (with video decode)...")
    single_times = []
    it = iter(dataset_streaming)
    for i in range(NUM_SAMPLES):
        start = time.time()
        try:
            item = next(it)
        except StopIteration:
            print(f"  Dataset exhausted after {i} items")
            break
        single_times.append(time.time() - start)
    if single_times:
        avg_single_time = sum(single_times) / len(single_times)
        results["avg_single_access"] = avg_single_time
        print(f"  Average single item access (with video decode): {avg_single_time*1000:.3f}ms")
        print(f"  Items tested: {len(single_times)}")
    else:
        print("  No items could be retrieved!")

    # 3. DataLoader batch loading
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing Streaming DataLoader batch loading...")
    # For IterableDataset, we need to recreate the dataset for a fresh iterator
    clear_v30_decoder_cache()
    dataset_streaming2 = LoLAPretrainStreamingDataset(
        repo_id=DATASET_NAME,
        root=V30_DATA_ROOT,
        max_history_length=100,
        action_chunk_size=10,
        delta_timestamps={
            "observation.images.top": [0],
            "observation.state": [0],
            "action": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        },
        deferred_video_decode=True,
        decode_device="cpu",
        decode_num_threads=1,
        shuffle=False,
        buffer_size=100,
    )

    collate_fn = AsyncDecodeDataLoader.make_collate_fn()
    dataloader_streaming = DataLoader(
        dataset_streaming2,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        persistent_workers=NUM_WORKERS > 0,
        worker_init_fn=lola_dataloader_worker_init_fn if NUM_WORKERS > 0 else None,
        collate_fn=collate_fn,
    )

    # Wrap with AsyncDecodeDataLoader for deferred video decode
    async_loader = AsyncDecodeDataLoader(
        dataloader_streaming,
        dataset_streaming2,
        collate_fn=collate_fn,
    )

    batch_times = []
    num_batches = min(5, len(dataloader_streaming)) if hasattr(dataloader_streaming, '__len__') else 5
    start = time.time()
    for i, batch in enumerate(async_loader):
        if i >= num_batches:
            break
        batch_times.append(time.time() - start)
        start = time.time()

    if batch_times:
        avg_batch_time = sum(batch_times[1:]) / len(batch_times[1:]) if len(batch_times) > 1 else batch_times[0]
        results["avg_batch_time"] = avg_batch_time
        print(f"  Average batch loading time (batch_size={BATCH_SIZE}): {avg_batch_time:.3f}s")
        # Print batch shape info
        if i == 0 or len(batch_times) > 0:
            batch_keys = list(batch.keys()) if isinstance(batch, dict) else ["non-dict"]
            print(f"  Batch keys (first 10): {batch_keys[:10]}")
            for key in ["action", "observation.state"]:
                if key in batch:
                    val = batch[key]
                    if isinstance(val, torch.Tensor):
                        print(f"    {key}: shape={val.shape}, dtype={val.dtype}")

    # 4. Throughput measurement
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing streaming throughput (pure iteration speed)...")
    dataset_streaming3 = LoLAPretrainStreamingDataset(
        repo_id=DATASET_NAME,
        root=V30_DATA_ROOT,
        max_history_length=100,
        action_chunk_size=10,
        delta_timestamps={
            "observation.images.top": [0],
            "observation.state": [0],
            "action": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        },
        deferred_video_decode=False,  # No video decode for pure I/O throughput
        decode_device="cpu",
        shuffle=False,
        buffer_size=100,
    )
    start = time.time()
    count = 0
    for item in dataset_streaming3:
        count += 1
        if count >= NUM_SAMPLES:
            break
    elapsed = time.time() - start
    throughput = count / elapsed if elapsed > 0 else 0
    results["throughput_no_video"] = throughput
    print(f"  Throughput (no video decode): {throughput:.1f} items/s ({count} items in {elapsed:.3f}s)")

    # Cleanup
    dataset_streaming.shutdown_decode_pipeline() if hasattr(dataset_streaming, 'shutdown_decode_pipeline') else None
    dataset_streaming2.shutdown_decode_pipeline() if hasattr(dataset_streaming2, 'shutdown_decode_pipeline') else None
    dataset_streaming3.shutdown_decode_pipeline() if hasattr(dataset_streaming3, 'shutdown_decode_pipeline') else None

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


def benchmark_v30_cuda():
    """Benchmark v3.0 dataloader with CUDA video decoding (decode_device="cuda").

    CUDA decoder cannot be initialized in forked DataLoader workers, so we test:
    1. num_workers=0 (main process only) with decode_device="cuda"
    2. num_workers=4 with multiprocessing_context="spawn" + decode_device="cuda"
    """
    print("\n" + "=" * 60)
    print("Benchmarking LeRobot v3.0 Dataset (CUDA Decode)")
    print("=" * 60)

    from lerobot.common.datasets_v30.lerobot_dataset import LeRobotDataset as LeRobotDatasetV30

    results = {}

    # 1. Dataset initialization
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing v3.0 dataset (CUDA decode)...")
    start = time.time()
    img_transforms = T.Compose([
        T.Resize((224, 224)),
    ])
    dataset_v30_cuda = LeRobotDatasetV30(
        repo_id=DATASET_NAME,
        root=V30_DATA_ROOT,
        video_backend="torchcodec",
        dataset_name=DATASET_NAME,
        image_transforms=img_transforms,
        video_return_type=V30_VIDEO_RETURN_TYPE,
        # decode_device="cuda",
    )
    init_time = time.time() - start
    results['init_time'] = init_time
    print(f"  Dataset length: {len(dataset_v30_cuda)}")
    print(f"  Video return type: {V30_VIDEO_RETURN_TYPE}")
    print(f"  Decode device: cuda")
    print(f"  Init time: {init_time:.3f}s")

    # 2. Full __getitem__ with CUDA video decoding
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing full __getitem__ with CUDA video decode...")
    full_times = []
    for i in range(NUM_SAMPLES):
        start = time.time()
        item = dataset_v30_cuda[i]
        full_times.append(time.time() - start)
    avg_full_time = sum(full_times) / len(full_times)
    results['avg_full_getitem'] = avg_full_time
    print(f"  Average full __getitem__ time (CUDA): {avg_full_time*1000:.3f}ms")

    # 3. DataLoader with num_workers=0 (CUDA must run in main process)
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing DataLoader (num_workers=0, CUDA decode)...")
    clear_v30_decoder_cache()
    dataloader_cuda_w0 = DataLoader(
        dataset_v30_cuda,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    batch_times = []
    num_batches = min(5, len(dataloader_cuda_w0))
    start = time.time()
    for i, batch in enumerate(dataloader_cuda_w0):
        if i >= num_batches:
            break
        batch_times.append(time.time() - start)
        start = time.time()

    if batch_times:
        avg_batch_time = sum(batch_times[1:]) / len(batch_times[1:]) if len(batch_times) > 1 else batch_times[0]
        results['avg_batch_time_w0'] = avg_batch_time
        print(f"  Average batch loading time (w=0, CUDA): {avg_batch_time:.3f}s")

    # 4. DataLoader with num_workers=4 + spawn context (CUDA in workers)
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing DataLoader (num_workers=4, spawn, CUDA decode)...")
    clear_v30_decoder_cache()
    # CUDA decoder needs fresh dataset without CUDA decoders in main process
    # Workers will initialize CUDA context via spawn
    dataset_v30_cuda2 = LeRobotDatasetV30(
        repo_id=DATASET_NAME,
        root=V30_DATA_ROOT,
        video_backend="torchcodec",
        dataset_name=DATASET_NAME,
        image_transforms=img_transforms,
        video_return_type=V30_VIDEO_RETURN_TYPE,
        decode_device="cuda",
    )
    clear_v30_decoder_cache()

    try:
        dataloader_cuda_w4 = DataLoader(
            dataset_v30_cuda2,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=False,
            multiprocessing_context="spawn",
            worker_init_fn=dataloader_worker_init_fn if NUM_WORKERS > 0 else None,
        )

        batch_times = []
        num_batches = min(5, len(dataloader_cuda_w4))
        start = time.time()
        for i, batch in enumerate(dataloader_cuda_w4):
            if i >= num_batches:
                break
            batch_times.append(time.time() - start)
            start = time.time()

        if batch_times:
            avg_batch_time = sum(batch_times[1:]) / len(batch_times[1:]) if len(batch_times) > 1 else batch_times[0]
            results['avg_batch_time_w4_spawn'] = avg_batch_time
            print(f"  Average batch loading time (w=4, spawn, CUDA): {avg_batch_time:.3f}s")
    except Exception as e:
        print(f"  DataLoader with spawn + CUDA failed: {e}")
        results['avg_batch_time_w4_spawn'] = None

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LeRobot Dataloader Speed Benchmark")
    parser.add_argument("--benchmark", type=str, default="all",
                        choices=["all", "streaming", "v21", "v30", "v30_cuda", "lola", "raw"],
                        help="Which benchmark to run (default: all)")
    args = parser.parse_args()

    print("=" * 60)
    print("LeRobot Dataloader Speed Benchmark")
    print("=" * 60)
    print(f"Test samples: {NUM_SAMPLES}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Num workers: {NUM_WORKERS}")
    print(f"Benchmark mode: {args.benchmark}")

    if args.benchmark in ("all", "raw"):
        raw_results = benchmark_data_loading_only()
    else:
        raw_results = {}

    if args.benchmark in ("all", "v21"):
        v21_results = benchmark_v21()
    else:
        v21_results = {}

    if args.benchmark in ("all", "v30"):
        v30_results = benchmark_v30()
    else:
        v30_results = {}

    cuda_results = {}
    # if args.benchmark in ("all", "v30_cuda"):
    #     cuda_results = benchmark_v30_cuda()

    # if args.benchmark in ("all", "lola"):
    #     lola_results = benchmark_lola_v30()
    # else:
    #     lola_results = {}

    # if args.benchmark in ("all", "streaming"):
    #     streaming_results = benchmark_streaming_v30()
    # else:
    #     streaming_results = {}

    if args.benchmark == "all" and v21_results and v30_results:
        print_comparison(v21_results, v30_results, raw_results)

    if cuda_results:
        print("\n" + "=" * 60)
        print("CUDA DECODE vs CPU DECODE COMPARISON (v3.0)")
        print("=" * 60)
        if v30_results:
            print(f"\n| Metric | v3.0 CPU (w=4) | v3.0 CUDA (w=0) | v3.0 CUDA (w=4 spawn) |")
            print("|--------|----------------|------------------|------------------------|")
            if 'avg_full_getitem' in v30_results and 'avg_full_getitem' in cuda_results:
                print(f"| __getitem__ | {v30_results['avg_full_getitem']*1000:.3f}ms | {cuda_results['avg_full_getitem']*1000:.3f}ms | - |")
            if 'avg_batch_time' in v30_results and 'avg_batch_time_w0' in cuda_results:
                print(f"| Batch Time | {v30_results['avg_batch_time']:.3f}s | {cuda_results['avg_batch_time_w0']:.3f}s | {cuda_results.get('avg_batch_time_w4_spawn', 'N/A')} |")
            cpu_item = v30_results.get('avg_full_getitem', 0)
            cuda_item = cuda_results.get('avg_full_getitem', 0)
            if cpu_item and cuda_item:
                speedup = cpu_item / cuda_item
                print(f"\n  Per-item speedup (CUDA vs CPU): {speedup:.2f}x")
        else:
            for key, val in cuda_results.items():
                if "time" in key:
                    print(f"  {key}: {val:.3f}s" if val >= 1 else f"  {key}: {val*1000:.3f}ms")


if __name__ == "__main__":
    main()
