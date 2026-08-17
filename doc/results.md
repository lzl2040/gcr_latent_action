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
(sharpa: 6 streams, VisuoTactile: 4 streams). Raw tactile images are 4×3×112×112 ≈ 150 k
dimensions against 40 for the state, so four mechanisms keep them in check:

1. **one token per pad**, never more, so tactile cannot outnumber the action tokens
   (see §6.2 for why we rejected UniVTAC's 64-token-per-view alternative);
2. **zero-initialised learnable gates** (`tanh`), so training starts from a tactile-free
   model and the channel only opens if it lowers the loss;
3. **per-sample modality dropout** (tactile 0.3, state 0.15, action 0.1); dropped or
   absent modalities are replaced by learned `missing_embed` tokens, so a dataset without
   tactile is not a distinct distribution;
4. a **10× lower learning rate** for the tactile backbone (`tactile_lr_scale = 0.1`),
   following UniVTAC's downstream policy.

Tactile images are downsampled to 112×112 uint8 inside the dataloader worker.

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

### 3.1 The real ceiling is the disk, not the CPU

With the fixes above, `iostat` during training shows `sdc` (the `/Data` volume that holds
the videos) at **94.9 % utilisation, 95 ms average await, ~208 read IOPS, 13.6 MB/s**.
That is a spinning disk being asked for random seeks into 100-200 MB multi-episode mp4
files. CPU load sits at ~22 of 64 cores and 260 GB of RAM stay free, so neither is the
constraint.

The only in-code lever is **read locality**, and it happens to coincide with the science:
drawing more frames from the *same* episode gives both harder negatives and fewer seeks.
Raising `episode_group_size` 4 -> 8 and `episode_group_frac` 0.5 -> 0.75 measured
**80 -> 164 samples/s** in an A/B dataloader benchmark. A batch of 256 still contains
~18 distinct episodes plus 48 other frames of the same dataset plus 64 frames from the
rest of the mixture, so diversity is preserved.

Further speedups require infrastructure, not code: move `/Data/lerobot_data_ort6d` to an
SSD, or pre-transcode the AV1 videos to smaller per-episode files.

## 4. Verified run

`conda activate lerobot_v2 && bash train_ace_local.sh` (2 × RTX A6000, `CUDA_VISIBLE_DEVICES=0,3`):

```
params        : 74.9 M trainable / 450.1 M total
micro batch   : 256 per GPU (global 512)
peak memory   : ~3.1 GB at batch 128, comfortably < 48 GB at 256
step time     : data_s ≈ 0.004 s -- dataloading is fully hidden behind compute
                updt_s ≈ 0.5 s warm / ≈ 2 s sustained (blocked on the disk via the
                cross-rank all_gather, see 3.1)
throughput    : ≈ 250-500 samples/s/GPU depending on page-cache warmth
```

Metric trace (global batch 512, chance accuracy = 1/512 = 0.002):

| step | loss | retrieval_acc |
| --- | --- | --- |
| 20 | 6.240 | 0.003 |
| 60 | 6.171 | 0.006 |
| 100 | 5.747 | 0.008 |
| 140 | 5.611 | 0.013 |
| 180 | 5.495 | 0.018 |
| 280 | 5.458 | 0.021 |
| 320 | 5.401 | 0.019 |

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

## 6. Scale-up to 727M parameters

### 6.1 Where the capacity went
The first working model was 450M total / 75M trainable, and almost all of the "total" was
a frozen SigLIP2 text tower whose 256k-entry embedding table alone is 262M. Only 75M
parameters were actually learning. Target: 500–800M total, ~300–400M trainable.

| block | params | tokens it attends over |
|---|---|---|
| frozen SigLIP2 vision + text | 375.2M | — |
| evidence trunk (6 layers) | 75.6M | ~420 |
| change-query decoder (6 layers) | 100.8M | 16 queries × ~420 |
| physical trunk (12 layers) | 151.2M | ~15 |
| tactile ResNet-18 | 11.2M | — |
| tactile reconstruction head | 7.1M | — |
| **total / trainable** | **726.7M / 351.5M** | |

`hidden_dim` 768→1024, 16 heads, 16 change queries, action `group_size` 4→2.

Most of the new capacity sits where it is *cheap*: the physical trunk is the single
largest trainable block (151M) but reads only ~15 tokens, so it costs almost nothing per
step. The genuinely expensive addition is the 6-layer evidence trunk, and §6.3 is about
paying for it.

The other structural change: **language moved into the evidence bank**. Previously each
fusion block did text cross-attention and then visual cross-attention. Now `[v0, v1-v0,
text]` are concatenated with type embeddings and passed through a self-attention trunk, so
the instruction can select which patches matter *before* the change queries read anything.
`ChangeQueryBlock` correspondingly loses its text branch.

### 6.2 Tactile, after reading UniVTAC
`michaelyuancb/ftp1-policy/UniVTAC` turned out to be simpler than expected:
`encoder/network.py::Tactile` is literally `torchvision.models.resnet18(num_classes=512)`,
pretrained by **supervised reconstruction** (marked RGB, RGB, depth, 63 marker positions,
7-DoF contact pose), not contrastively. The decoders are discarded at inference.

Adopted:
- the **ResNet-18 / 512-d backbone**, ImageNet-initialised, replacing our 4-layer CNN;
- the **RGB reconstruction auxiliary loss** (weight 0.1). Of their five targets it is the
  only one whose labels we already have — marker, depth and contact pose all require their
  simulator;
- a **separate, 10× lower LR group** for the tactile backbone, as in their ACT policy;
- **frozen BatchNorm**, which they use when embedding the encoder into the policy. For us
  it is not optional: the number of tactile pads per batch is data dependent, so trainable
  BN statistics would drift with the mixture composition.

Deliberately **not** adopted:
- their default `tactile_type='full'`, which emits 8×8 = 64 tokens per view. With 4 pads
  that is 256 tactile tokens against ~15 physical ones — precisely the domination failure
  the gates and dropout exist to prevent. We keep one gated token per pad plus a view
  embedding;
- their lack of any sensor-type conditioning. UniVTAC handles only optical tactile sensors
  and normalises heterogeneity by training one CNN over all of them; our mixture also
  contains 6-d torque signals, which take the separate signal pathway.

There is **no standalone UniVTAC checkpoint published** — only full policies
(`MJJJJ1064/ftp1_v0426_50kstep`, `MJJJJ1064/ftp1_univtac_finetune`) that embed it — so the
backbone starts from ImageNet rather than from their tactile pretraining.

### 6.3 Making it affordable
The first scaled version needed **49.3 GB and 1.62 s/step** at batch 256 — over an A6000's
48 GB. Three changes brought it to **12.9 GB and 1.34 s/step**:

1. **Dropped the `v1` stream from the evidence bank.** It carried no information:
   `v1 == v0 + diff`, so it was 196 tokens of attention for an exact linear combination of
   its neighbours. The bank went 620 → ~420 tokens. The explicit `diff` stream is the one
   worth keeping — patches are spatially aligned across the two frames, so `diff[i]` is
   directly "what changed at location i", which attention would otherwise have to
   rediscover by matching patches.
2. **Activation checkpointing on the evidence trunk.** ~30% more compute for a large
   memory saving, which is free here: the GPU is idle waiting on the disk anyway (§3.1).
3. **The tactile ResNet only runs on pads that actually exist.** Most batches hold a
   handful of tactile samples among 256; encoding zero-filled placeholders wasted nearly
   all of its compute and polluted the reconstruction target.

### 6.4 The bug that only appears on 2 GPUs
Change 3 above introduced a rank desync. Skipping the tactile CNN when a batch has no
tactile data makes *the set of parameters receiving gradients data dependent*, and under
ZeRO-2 that set determines the gradient-reduction schedule. A rank whose batch happened to
be tactile-free simply never reduced the tactile gradients and its peers waited forever.

The symptom is not an exception but a 600 s NCCL timeout, with one rank stuck on the
351M-element ALLREDUCE while the other has already moved on to the next collective — worth
recognising, because it looks like a hardware or NCCL problem and is not.

The fix keeps the compute saving: the selection is padded to at least one row and that
row's contribution is multiplied by zero, so the schedule is data independent. Verified
with a two-rank test where rank 0 has tactile data and rank 1 has none.

Two smaller fixes came from the same run: `make_optimizer_and_scheduler` assumed
`get_optim_params()` returns a flat tensor list and broke on parameter groups, and the
launcher's hard-coded rendezvous port collided with orphaned workers from a killed run
(it now picks a free port).
