#!/usr/bin/env python
"""Debug script for LoLAStreamingDataset benchmark."""
import sys
import traceback

try:
    print("Step 1: Importing...", flush=True)
    from lerobot.common.datasets_v30.streaming_dataset import (
        LoLAStreamingDataset,
        _LoLAStreamingDatasetMSWrapper,
        AsyncDecodeDataLoader,
        benchmark_lola_streaming_dataset,
    )
    print("Step 1: Import OK", flush=True)

    print("\nStep 2: Creating dataset...", flush=True)
    ds = _LoLAStreamingDatasetMSWrapper(
        repo_id="Trossen_Stationary_AI_480x640_padded_MERGED",
        root="/Data/lerobot_data_ort6d/Trossen_Stationary_AI_480x640_padded_MERGED",
        deferred_video_decode=False,
        buffer_size=100,
        max_history_length=1,
        action_chunk_size=1,
    )
    print(f"Step 2: Dataset created. action_dim={ds.action_dim}, fps={ds.fps}", flush=True)

    print("\nStep 3: Iterating 5 items directly (no DataLoader)...", flush=True)
    for i, item in enumerate(ds):
        if i >= 5:
            break
        print(f"  Item {i}: keys={sorted(item.keys())}", flush=True)
        for k, v in item.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: shape={v.shape}, dtype={v.dtype}", flush=True)
            elif isinstance(v, str):
                print(f"    {k}: {v[:60]}", flush=True)
            elif v is None:
                print(f"    {k}: None", flush=True)
            else:
                print(f"    {k}: {type(v).__name__}", flush=True)

    print("\nStep 3: Direct iteration OK!", flush=True)

    print("\nStep 4: Running full benchmark...", flush=True)
    benchmark_lola_streaming_dataset()

except Exception as e:
    print(f"\nERROR: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

import torch
