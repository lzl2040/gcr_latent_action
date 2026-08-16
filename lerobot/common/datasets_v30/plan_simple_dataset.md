# Plan: SimpleStreamingDataset

## Goal
Create a simplified streaming dataset class that:
1. Loads a **single** dataset (not multi-dataset)
2. Returns per-frame items **without** history actions (`hist_actions_full`, `hist_actions_mask`, `hist_actions_length`)
3. Uses the dataset's own `meta/stats.json` for normalization (not per-sub-dataset normalization)
4. Handles ms data key mapping (`action.ee_ort6d_pos` → `action`, `observation.ee_ort6d_pos` → `observation.state`)
5. Still supports video decode (deferred or immediate)

## Design

### Class: `SimpleStreamingDataset(torch.utils.data.IterableDataset)`

Reuses from LoLAPretrainStreamingDataset:
- `EpisodeChunkReader` — parquet loading
- `BoundedVideoDecoderCache` — video decoder cache
- `_discover_parquet_files`, `_load_episodes_polars`, `_EpisodeAccessor` — metadata loading
- `_safe_concat_tables`, `_contiguous_ranges` — utility functions
- Video decode methods (`_query_videos`, `_decode_video_cuda`, `_make_timestamps_from_indices`, etc.)
- `DecodeProcessPipeline` + `AsyncDecodeDataLoader` — async decode pipeline

New/different behavior:
- No `dataset_to_episodes_path`, no sub-dataset normalization
- No `max_history_length`, no `action_chunk_size`, no history action construction
- Normalization: load stats from dataset's own `meta/stats.json`, apply z-score to action + observation.state
- Key mapping: after loading parquet data, remap `action.ee_ort6d_pos` → `action`, `observation.ee_ort6d_pos` → `observation.state` (if standard keys don't exist)
- `_process_episode_frames` simplified: no history actions, just current frame data + delta frames + video lookup
- No tier system (`tier_config_path`, `yield_tier`, `_episode_visual_cost`, etc.)

### Normalization
- Load `meta/stats.json` once in `__init__`
- Normalize `action` and `observation.state` using z-score: `(x - mean) / (std + 1e-8)`
- For action: only normalize translation dims (first 3 per arm, 2 arms = indices 0:3 and 10:13), leave rotation + gripper untouched

### Data flow
```
Parquet → EpisodeChunkReader → _process_episode_frames → Shuffle Buffer → yield
```

Each yielded item contains:
- `action`: Tensor [action_dim] — normalized
- `observation.state`: Tensor [state_dim] — normalized
- Delta frame data (if delta_timestamps set)
- Video frames (PIL.Image or deferred _video_lookup)
- `task`: str
- `episode_index`, `frame_index`, `timestamp`, `task_index`: scalars
- `camera_valid_mask`: dict
- `{video_key}_is_pad`: BoolTensor (if delta_timestamps)

### Test code
A `__main__` block that:
1. Instantiates `SimpleStreamingDataset` with the Trossen dataset
2. Iterates a few items, prints shapes and keys
3. Verifies key mapping and normalization work

## Files to modify
- `/home/v-wangxiaofa/lzl/gcr_latent_action/lerobot/common/datasets_v30/pretrain_streaming_dataset.py`
  - Add `SimpleStreamingDataset` class after `LoLAPretrainStreamingDataset`
  - Add `if __name__ == "__main__"` test block at the end
