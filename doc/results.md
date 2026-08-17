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

### 6.5 Verified run of the scaled model
`CUDA_VISIBLE_DEVICES=0,3 bash train_ace_local.sh`, 2 × A6000, micro batch 256 → global
batch 512 (chance retrieval accuracy = 1/512 = 0.002). Ran to the wall-clock cutoff with
no errors and 16.4 GB per GPU.

| step | loss | recon | retrieval acc | pos_sim | tactile gates (sig / img) |
|---|---|---|---|---|---|
| 20  | 6.255 | 0.129 | 0.003 | 0.221 | 0.000 / 0.000 |
| 180 | 5.511 | 0.039 | 0.014 | 0.983 | 0.001 / 0.000 |
| 420 | 5.263 | 0.006 | 0.021 | 0.960 | 0.003 / 0.002 |
| 540 | 4.526 | 0.005 | 0.049 | 0.942 | 0.006 / 0.002 |
| 780 | 4.420 | 0.004 | 0.084 | 0.950 | 0.012 / 0.005 |
| 900 | 4.105 | 0.002 | **0.169** | 0.950 | 0.015 / 0.007 |

Against the 450M baseline the two models are level early (step 180: 0.014 vs 0.018) and the
scaled one pulls away as the LR reaches its plateau (step 540: 0.049 vs 0.045; step 900:
0.169, which is **87× chance**). The auxiliary reconstruction loss falls by two orders of
magnitude, and both tactile gates open slowly and monotonically from zero — the intended
behaviour, i.e. tactile is being adopted because it helps rather than because it is loud.

Throughput: `updt_s ≈ 1.67 s` at batch 256 with `data_s ≈ 0.004 s`, so the dataloader is
fully hidden in the steady state. The periodic excursions to 5–12 s are the HDD of §3.1,
not the model: they coincide with one GPU dropping to 0% utilisation while the other rank
waits on a disk stall inside the gradient allreduce. Note that `data_s` is measured on rank
0 only, so a stall on rank 1 surfaces as inflated `updt_s` rather than as `data_s`.

## 7. Branch `ace_plus/dinov3_perception`

Two concerns motivated this branch: the perception features came from SigLIP2, and the
change queries were supervised *only* by the contrastive loss.

### 7.1 Why DINOv3 for vision
SigLIP's visual features are trained to match a caption, so they preserve what language
describes and discard the rest. But the change between two frames a few hundred
milliseconds apart is mostly *not* describable — a gripper closing 2 cm has no caption —
so a language-aligned prior is the wrong one for the thing we are trying to measure.
DINOv3 is self-supervised and keeps far more spatial and geometric detail.

DINOv3 has no text tower, so language still comes from SigLIP2. That is deliberate rather
than a fallback: its text space is already aligned to a visual space, which is what makes
"the instruction selects which change matters" work at all. Only the text tower is kept,
which drops 93M of frozen vision parameters that DINOv3 now replaces.

Practical notes: DINOv3 emits 1 CLS + 4 register tokens before the 196 patches (all
stripped), and expects ImageNet normalisation rather than SigLIP's 0.5/0.5.

### 7.2 The reconstruction objective
`ChangePredictor` predicts the frame-`t+H` DINOv3 features from the frame-`t` features, the
instruction, and the change queries; the loss is smooth L1 against per-token normalised
targets (I-JEPA style).

The bottleneck is the entire point. The predictor reads the **raw projected `v0`**, never
the evidence trunk's output — the trunk has already attended over the `v1 - v0` stream and
therefore determines `v1` exactly, so a predictor reading it could ignore the queries
completely and the objective would be vacuous. As built, the 16 change queries are the only
route by which anything about frame `t+H` reaches the prediction. Targets come from the
frozen backbone and are detached, so there is no collapse mode.

### 7.3 The scale bug that nearly sank it
The first version trained badly: accuracy stuck at 0.009 through step 700, `pos_sim` pinned
near 0.98 — a nearly constant perception embedding. The cause was **not** architectural.

| tensor | per-token L2, before | after |
|---|---|---|
| `v0` (projected) | ~8 | 18.2 |
| `v1 - v0` | 7.75 | 32.0 |
| type embedding | ~32 | 0.65 |

`nn.Embedding` initialises to N(0, 1), which at width 1024 means the *tag* identifying each
evidence stream was four times larger than the content it tagged; the pre-norm blocks then
renormalised each token and read mostly the tag. SigLIP2 had been getting away with this by
coincidence — its features happened to be about the same size as the tags. And the
difference stream, the one signal the design exists to capture, was the faintest thing in
the bank.

Fixes: LayerNorm on the frozen vision/text features before projection (so the encoder is no
longer implicitly tuned to one backbone's scale), LayerNorm on the difference stream, and
std-0.02 init for the type embeddings.

**This is the lesson worth carrying:** when swapping a frozen backbone, check the output
magnitude against every learned constant it is summed with. Nothing errors, the loss still
decreases, and the model just quietly underperforms.

### 7.4 Parameter budget
Split evenly between the branches, as requested. Frozen towers are excluded from the split
since they are feature extractors, not capacity the model can use.

| | total | trainable |
|---|---|---|
| perception | 573.5M | **205.6M** (51%) |
| — frozen DINOv3 ViT-B/16 | 85.7M | — |
| — frozen SigLIP2 text | 282.3M | — |
| — evidence trunk (5) / decoder (5) / predictor (3) | | 63.0 / 84.0 / 55.4M |
| physical | 197.1M | **197.1M** (49%) |
| — trunk (14) / tactile | | 176.4 / 18.3M |
| **total** | **770.7M** | **402.7M** |

### 7.5 Verified run
2 × A6000, global batch 512, 13.4 GB per GPU, ~2.13 s/step.

| step | contrastive | precon | trecon | acc | pos_sim |
|---|---|---|---|---|---|
| 20   | 6.255 | 0.433 | 0.099 | 0.003 | 0.139 |
| 300  | 5.231 | 0.113 | 0.014 | 0.020 | 0.966 |
| 540  | 4.051 | 0.091 | 0.005 | 0.098 | 0.929 |
| 780  | 3.886 | 0.091 | 0.004 | 0.223 | 0.953 |
| 1060 | 3.622 | 0.087 | 0.003 | **0.271** | 0.950 |

Against the SigLIP2 branch at equal steps: **0.226 vs 0.169 at step 900**, with a lower
contrastive loss (3.83 vs 4.11) *while also* carrying the reconstruction objective.

### 7.6 Does the change query really encode the change?
Probed directly by re-scoring the reconstruction with the queries perturbed (250-step model,
batch 64):

| queries fed to the predictor | recon loss |
|---|---|
| true | 0.13480 |
| shuffled across the batch | 0.13826 |
| zeroed | 0.13865 |
| *reference:* copy `v0` unchanged | 0.13083 |

So the predictor demonstrably uses the queries — mismatching them hurts. But at 250 steps it
has not yet beaten the trivial "assume nothing moved" reference, which is the number to
watch: it is only above that line that the queries are carrying real motion rather than the
predictor learning the dataset's average frame. In the full run `precon` reaches 0.087,
comfortably below the ~0.131 reference, so the objective does become non-trivial — but this
probe is the right check to repeat on any future change to the predictor.

## 8. Reading the physical modalities over the window

### 8.1 What was wrong
Only the action chunk was read over `[t, t+H]`. State, tactile signal and tactile images were
all sampled at `t` alone, so the physical side described an *instant* while the perception side
described an *interval* — and the whole objective is to align the two.

For tactile this was close to a bug rather than a tuning choice. Tactile earns its place by
reporting contact events — grasp closure, slip, impact — and a contact is something that
*happens during* the window. If the grasp closes at `t+8`, the frame at `t` shows no contact at
all, so the most informative signal the sensor produces was systematically invisible.

### 8.2 How far each modality is now read
Decided by what it costs to read, since the pipeline is bound by a spinning disk:

| modality | storage | frames read | tokens |
| --- | --- | --- | --- |
| action | parquet | 16 | 8 |
| state | parquet | 16 | 8 |
| tactile signal | parquet | `t`, `t+H` | 1 |
| tactile image | **video** | `t`, `t+H` | 4 |

State is a parquet column in the same row group as the action, so the full chunk is nearly
free, and it is worth having on its own merits: the action chunk is what was *commanded*, the
state trajectory is what actually *happened*, and the two diverge exactly where the difference
becomes visible. Tactile is read at the two ends only because tactile cameras are video and
decoding 16 frames × 4 pads is precisely where throughput is already constrained; two frames
catch "before contact → after contact" at 2× the decode cost instead of 16×.

Both tactile streams fold their frame pair into their *existing* token count via
`[feat_t, feat_t1 - feat_t]`. The difference term is the point, and holding the token count
fixed stops tactile growing from 36% to 50% of the content tokens at the action chunk's
expense. State shares the action's positional embedding so group *i* of each covers the same
frames. Sequence 15 → 22 tokens; physical branch 197.1M → 197.8M; 13.4 → 20 GB per GPU at
global batch 512; 2.14 → 2.17 s/step.

Verified at random mid-episode frames — state drift 0.31–0.58, tactile image |t1−t0| 0.49–2.74
on a 0–255 scale, tactile signal delta 0.83. All of these read zero at *episode starts*, where
nothing has moved yet; that is a trap worth remembering when debugging window reads, because a
failed window read is silent (a single frame is broadcast across the chunk rather than raising).

### 8.3 The measurement problem — the main finding
**The aggregate retrieval accuracy cannot resolve a change of this size at ~1000 steps.**

Two runs of *identical code with the same seed* were compared:

| steps | run A | run B | mean \|difference\| |
| --- | --- | --- | --- |
| 500–800 | 0.101 | 0.106 | 0.011 |
| 800–1060 | 0.235 | 0.182 | **0.055** |

A 0.055 spread between two runs of the same code is as large as any difference measured
*between* code versions (the windowed delta against the §7 baseline was −0.079 for one run and
−0.015 for the other). Nondeterminism in dataloader worker ordering and video decoding changes
which datasets land in which batch, and batch composition dominates in-batch retrieval
accuracy. Any conclusion drawn from a single run at this horizon — in either direction — is
noise. This invalidates the "0.271 → 0.302" reading that a single step at 1060 initially
suggested.

The metric is also **structurally blind to tactile**: the three tactile datasets are 2.7% of
`debug_research_data` (0.0036 + 0.0178 + 0.0058, against fractal's 0.42 and YAM's 0.53). Even
a change that doubled tactile retrieval would move the headline number by about a point.

Hence `tactile_hits` / `tactile_rows` (§8.4), logged as counts rather than a ratio because most
batches contain no tactile at all and averaging a per-step ratio would fold in zeros meaning
"nothing to measure" rather than "got it wrong".

### 8.4 What can be said
Tactile-conditional retrieval accuracy, summed over windows (chance = 1/512 = 0.00195):

| steps | hits / rows | conditional acc | aggregate acc |
| --- | --- | --- | --- |
| 0–300 | 0.8 / 111.8 | 0.0072 | 0.013 |
| 300–600 | 1.5 / 117.1 | 0.0128 | 0.055 |
| 600–900 | 4.2 / 95.7 | 0.0439 | 0.134 |
| 900–1200 | 8.9 / 112.3 | **0.0793** | 0.238 |

The tactile path does learn — 40× chance by step 1200 — but the §7 baseline predates the metric,
so there is no A/B here yet.

The gates are a partial exception, being far more reproducible than accuracy:

| gate | noise floor (run A vs B) | effect vs §7 baseline | verdict |
| --- | --- | --- | --- |
| `tsig` | 0.00000 | 0.00486 | above noise |
| `timg` | 0.00436 | 0.00311 | **within noise** |

So the tactile *image* gate's apparent 3.6× faster opening is not a real result. The tactile
*signal* gate genuinely differs, but its magnitude is *smaller* than the baseline's
(0.0094 vs 0.0142), which does not support a simple "the new input is more useful" story.

### 8.5 Verdict and what is actually needed
The change is kept: it is principled (a contact that occurs mid-window is unobservable at `t`),
cheap (+0.7M parameters, +1.5% step time), and shows no harm above the noise floor. But its
benefit is **unproven at this horizon**, and honestly cannot be proven with the current tooling.

What would settle it is a **fixed held-out evaluation set** scored at checkpoints: same frames,
same batch composition, every time. That removes batch-composition variance entirely and would
make differences of 0.01 legible where 0.055 is currently invisible. Until that exists, prefer
comparing gate trajectories and tactile-conditional counts over aggregate accuracy, and treat
any single-step reading as meaningless.

---

## 9. Full-rate tactile signal, `group_size`, and a metric that actually works

Two changes and one measurement. The changes: the tactile signal is now read over the whole
16-frame window instead of at the two ends, and a fixed evaluation split was built to answer
"is this better?" at all. The measurement then said the `group_size` change shipped alongside
them was wrong, and it has been reverted.

### 9.1 Why the tactile signal is read at full rate

§8 gave both tactile streams two frames, `t` and `t+H`. That is right for the tactile cameras
and wrong for the tactile signal, for a reason that is specific to what a contact looks like:

- A contact transient is a few frames wide. Two samples at the ends of the window can
  **straddle** it and see almost nothing — a strictly worse failure than reading only `t`,
  because it looks like a successful read.
- The signal is a parquet column sharing a row group with state and action, so reading all 16
  frames costs nothing. The tactile cameras are video, and 16 frames x 4 pads lands squarely on
  the spinning disk this pipeline is already bottlenecked by.

So the two streams are now treated differently, and the reason is cost, not principle. The
signal is grouped like state and action; the cameras keep their `[feat_t, feat_t1 - feat_t]`
pair folded into one token per pad.

Verified on the three tactile datasets (mid-episode frames — see §8 for why that matters):
`ftp_1_RH20TCfg5Franka` returns `(16, 32)` with per-frame std 0.135, and the two camera
datasets return `(4, 2, 3, 112, 112)` with real `t1 - t0` change. The three are complementary:
RH20T is signal-only, the other two are camera-only.

### 9.2 The fixed evaluation split

§8 ended by noting that in-batch retrieval accuracy has a **run-to-run noise floor of 0.055**,
larger than any effect being measured, because batch composition dominates it. That made every
comparison in this document unfalsifiable, so `lerobot/common/datasets/contrastive_eval.py` now
draws a deterministic list of `(dataset_idx, frame_idx)` batches once and scores exactly those
in every run, with two splits: `mixture` (training weights) and `tactile` (only the tactile
datasets, because tactile is 2.7% of this mixture and the headline number cannot see it).
Accuracy is computed within each batch on one rank, never across an all-gather, so a 2-GPU run
and a 4-GPU run remain comparable.

**It helped, and it was not enough.** Two runs of identical code and identical seed, scored on
the same frames:

| step | gs4-a | gs4-b | \|diff\| |
| --- | --- | --- | --- |
| 250 | 0.0220 | 0.0283 | 0.0063 |
| 500 | 0.0308 | 0.0215 | 0.0093 |
| 750 | 0.0879 | 0.0674 | 0.0205 |
| 1000 | 0.1025 | 0.1191 | 0.0166 |
| 1250 | 0.2324 | 0.1289 | **0.1035** |

Fixing the frames removed batch-composition variance but not **training-trajectory** variance:
bf16 kernels are not bitwise reproducible, and 1250 steps is long enough for that to diverge
into genuinely different models. The noise floor grows with training, reaching 0.10 by step
1250 — larger than the 0.055 it replaced. Retrieval accuracy is simply the wrong statistic
here: it is a 0/1 decision per row, so it throws away almost everything the model computed.

### 9.3 The metric that does work: training loss

Averaged over a window, the contrastive loss is an order of magnitude more reproducible,
because every row contributes a continuous value rather than a coin flip:

| metric | window | noise floor (same config) | effect (gs4 - gs2) | verdict |
| --- | --- | --- | --- | --- |
| `mixture_acc` | 250–1250 | 0.0312 | −0.0287 | within noise |
| `tactile_acc` | 250–1250 | 0.0059 | −0.0023 | within noise |
| `contra_loss` | 600–900 | 0.0421 | **+0.3214** | 7.6x noise |
| `contra_loss` | 900–1250 | 0.0863 | **+0.4294** | 5.0x noise |

**Use windowed training loss to compare arms. Use the fixed eval splits to track progress.**
Retrieval accuracy could not resolve this difference even though the difference is large.

### 9.4 `group_size = 4` was worse, and the reasoning behind it was backwards

`group_size` was raised 2 -> 4 on the theory that the newly full-rate tactile signal would
otherwise take 8 tokens and crowd out the action chunk, and that 4 tokens each for state,
action, signal and cameras was "the most even split available".

Measured: `group_size=2` reaches contrastive loss **3.71 vs 4.14** over steps 900–1250, five
times the noise floor. Counting the tokens shows why:

| | state | action | signal | cameras | content total | tactile share |
| --- | --- | --- | --- | --- | --- | --- |
| `group_size=2` | 8 | 8 | 8 | 4 | 28 | **43%** |
| `group_size=4` | 4 | 4 | 4 | 4 | 16 | **50%** |

The tactile cameras contribute a **fixed** 4 tokens regardless of `group_size`, because they
are one token per pad and there are 4 pads. Raising `group_size` therefore shrinks only the
three chunked streams, and pushes tactile's share *up* rather than down — the exact opposite
of the stated goal — while also halving the temporal resolution of state, action and signal.
`group_size` is back to 2, physical sequence 29 tokens.

The general lesson: when a config is meant to rebalance a budget, count the budget after the
change instead of reasoning about it. This one took two 1250-step runs to catch and would have
taken one `print`.

### 9.5 Two process traps hit while running this

- **`train_ace_local.sh` did not forward `"$@"`.** `bash train_ace_local.sh
  --policy.group_size=2` silently dropped the override, and the "control" arm ran the identical
  config. Caught because its step-20 metrics were byte-identical to the other arm. Fixed.
- **GPU auto-selection changes the global batch.** The launcher picks every GPU with <5 GB in
  use, so when two neighbours' jobs finished, a replicate quietly ran on 4 GPUs at batch 1024
  instead of 2 at 512. **Always pin `CUDA_VISIBLE_DEVICES` when comparing runs**; the launcher
  now warns about it.

### 9.6 Status

Verified: `conda activate lerobot_v2 && bash train_ace_local.sh` runs end to end on 2x A6000
at ~2.15 s/step, global batch 512, 771M total / 403M trainable. Eval costs ~26 s at
`eval_freq=250` (~5% overhead) once the page cache is warm.

Unresolved: whether the full-rate tactile signal itself helps. `tactile_acc` cannot resolve it
(noise floor 0.0059, and the tactile datasets are 2.7% of the mixture), and windowed
`contra_loss` is dominated by the 97% of rows without tactile. Answering it needs the loss
restricted to tactile rows, which is not currently logged.
