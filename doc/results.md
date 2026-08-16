# RoboContrast — perception ↔ physical contrastive pre-training

## 1. Idea

Align, over a time window `[t, t+H]`:

| side | modalities | meaning |
| --- | --- | --- |
| perception | RGB(t), RGB(t+H), language (later: mask, optical flow) | *what visibly changed in the scene* |
| physical | canonical state, action chunk, tactile | *what the robot did* |

Both sides are projected to a shared 512-d space and pulled together with a symmetric
InfoNCE loss. The physical encoder is therefore forced to explain the visual change.

## 2. Design decisions

### 2.1 Language selects *which* change matters
`PerceptionEncoder` builds an evidence bank from the two frames:
`concat([v0 + e0, v1 + e1, (v1 - v0) + e2])`. A small set of latent **change queries**
(8 by default) is **seeded with the pooled sentence embedding** and cross-attends first to
the text tokens and then to the evidence bank, for `num_fusion_layers` rounds. Without the
text seed the queries are scene-agnostic and mostly recover camera motion.

Both SigLIP2 towers are frozen and run under `no_grad`. This is what makes batch ≥ 256 and
sub-second steps possible: only 75 M of 450 M parameters are trained.

### 2.2 Canonical action / state space (`canonical_space.py`)
40 slots, each sample carrying a `(40,)` validity mask:

```
[ 0: 3] arm0 eef xyz     [ 3: 9] arm0 rot6d      [ 9:10] arm0 gripper
[10:13] arm1 eef xyz     [13:19] arm1 rot6d      [19:20] arm1 gripper
[20:27] arm0 joints(7)   [27:28] arm0 joint gripper
[28:35] arm1 joints(7)   [35:36] arm1 joint gripper
[36:40] reserved
```

Every projection consumes `[value * mask, mask]`, so an unfilled slot is distinguishable
from a genuine zero. Datasets that only ship joints (`VisuoTactile`, arm_joint only) and
datasets that ship both (`YAM`, `sharpa`) coexist without any padding convention hack.

Observed masks: `fractal`/`RH20T` → `[0..9]`, `sharpa` → `[0..35]`,
`VisuoTactile` → `[20..35]`, `ms_data_xdof_3` → `[0..26, 28..34]`.

**Normalisation is per-dataset**, not mixture-aggregated. Mixture statistics leave a
dataset-scale signature in the vectors, which a contrastive model happily exploits as a
shortcut instead of learning motion.

### 2.3 Tactile without domination
Tactile arrives as a low-dimensional signal (RH20T: 6-d gripper torque) or as images
(sharpa: 6 streams, VisuoTactile: 4 streams). Raw tactile images are 4×3×64×64 ≈ 49 k
dimensions against 40 for the state, so three mechanisms keep them in check:

1. a deliberately shallow CNN pooled into a **single** token, so tactile never outnumbers
   the action tokens;
2. **zero-initialised learnable gates** (`tanh`), so training starts from a tactile-free
   model and the channel only opens if it lowers the loss;
3. **per-sample modality dropout** (tactile 0.3, state 0.15, action 0.1); dropped or
   absent modalities are replaced by learned `missing_embed` tokens, so a dataset without
   tactile is not a distinct distribution.

Tactile images are downsampled to 64×64 uint8 inside the dataloader worker.

### 2.4 Two-frame reading
Implemented with `delta_timestamps[primary_video_key] = [0, H/fps]` rather than
`observation_delta_indices`, which would have doubled the decode cost of *every*
observation key. `chunk_size = frame_horizon = 16`.

### 2.5 Negative sampling (`contrastive_sampler.py`)
Uniform batches make the task trivial (kitchen vs factory needs no motion understanding).
Per batch: `same_dataset_frac = 0.75` of the samples come from one dataset; half of those
are drawn as groups of `episode_group_size = 4` frames from the *same episode*, forced at
least `min_frame_gap = 32` frames apart by a stride-based draw. The remainder is filled
from the whole mixture to retain some easy negatives.

The loss additionally masks residual false negatives: same-episode candidates within
`false_negative_frame_gap = 32` frames are set to `-inf` (except the true positive).

## 3. Performance findings

| finding | before | after |
| --- | --- | --- |
| `hf_dataset.select(non_contiguous_idx)` on the 31 M-row YAM dataset | **2684 ms/item** | **17 ms/item** (plain list indexing) |
| decoding every camera of `sharpa` (9 streams) | 9 decodes | 1–5 decodes (`video_keys_to_decode`) |
| intra-op threads in dataloader workers | 1.31 s / 256-batch | 0.95 s / 256-batch (`torch.set_num_threads(1)`) |

`select()` materialises an on-disk arrow indices mapping whenever the index list is not
contiguous — which is exactly the two-frame query `[i, i+H]`. This was the single biggest
cost in the pipeline and is a **negative result worth remembering**: the "use `.select()`
for efficient column access" comment in the upstream code is wrong for large datasets.

Per-item cost after the fix (measured, single process):

| dataset | ms/item | mixture weight |
| --- | --- | --- |
| fractal20220817_data | 16 | 0.423 |
| taco_play | 13 | 0.024 |
| ms_data_xdof_3 (YAM) | 52 | 0.525 |
| ftp_1_RH20TCfg5Franka | 13 | 0.004 |
| ftp_1_sharpa_split_0 | 93 | 0.018 |
| ftp_1_VisuoTactile_D-WHEEL | 68 | 0.006 |

## 4. Verified run

`conda activate lerobot_v2 && bash train_ace_local.sh` (2 × RTX A6000, `CUDA_VISIBLE_DEVICES=0,3`):

```
params        : 74.9 M trainable / 450.1 M total
micro batch   : 256 per GPU (global 512)
peak memory   : ~3.1 GB at batch 128, comfortably < 48 GB at 256
step time     : updt_s ≈ 0.62 s, data_s ≈ 0.004 s (dataloading fully hidden)
throughput    : ≈ 410 samples/s/GPU
```

Metric trace (global batch 512, chance accuracy = 1/512 = 0.002):

| step | loss | retrieval_acc |
| --- | --- | --- |
| 20 | 6.240 | 0.003 |
| 60 | 6.171 | 0.006 |
| 100 | 5.747 | 0.008 |
| 140 | 5.611 | 0.013 |
| 180 | 5.495 | 0.018 |

Tactile gates stay at ~0 over the first 200 steps, i.e. the model has not yet found the
tactile channel useful — which is the intended cold start, not a bug.

## 5. Open items / negative results

- `pos_sim` sits at ~0.98 for the first hundreds of steps: the embeddings are still nearly
  collapsed onto a single direction. Loss and accuracy improve regardless, but if this
  persists past a few thousand steps, consider raising `logit_scale` initialisation or
  adding a uniformity term.
- `pair_is_valid` is produced by the dataset but not yet consumed by the loss. The sampler
  already excludes the last `horizon` frames of every episode, so it should always be 1;
  it is kept as a diagnostic.
- Mask and optical-flow perception modalities are designed for but not implemented.
