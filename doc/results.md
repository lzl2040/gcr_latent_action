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

## 10. Reusing FTP-1's pretrained tactile weights

The question was whether the open-source weights at `michaelyuancb/ftp1-policy` could replace
the ImageNet ResNet-18 that currently encodes tactile camera pads.

### 10.1 The repo has two tactile encoders and the obvious one is the wrong one

`UniVTAC/encoder/network.py::Tactile` is the ResNet-18 our own `TactileImageEncoder` was
modelled on, so it looks like the thing to reuse. It is a dead end, for two independent
reasons: its training script writes checkpoints to a local `ablation/` path and **no weights
were ever published**, and it was only ever trained on *simulated GelSight Mini*, so it would
not transfer to SharpaWave or OpenLoongVTouch even if the weights existed. Anyone revisiting
this should not spend time looking again.

The usable artefact is the tower inside the shipped policy, `src/openpi/models_pytorch/`,
whose weights *are* published under MIT as `MJJJJ1064/ftp1_v0426_50kstep`. Per pad:

```
(3,224,224) --patch16--> 197 tokens
  --> per-sensor ViT, depth 3, width 768   (22.0M, one copy per sensor type)
  --> shared trunk, depth 9                 (64.2M)
  --> LayerNorm --> CLS --> Linear(768->512)
```

Two things made it drop in cleanly. The output is **1 token per pad**, matching the tokenisation
our physical encoder already used; and the projection is `768->512`, and 512 was already our
`tactile_feat_dim`, so the pretrained head is reused verbatim rather than thrown away. The
checkpoint's `normalization/` also happens to cover **our exact datasets** — `sharpa`,
`VisuoTactile_D-WHEEL`, `RDP_Bimanual`, `RH20TCfg5Franka`.

The key layout is exactly timm's. timm is not installed here, so `ftp1_tactile.py`
reimplements the layout by hand using **identical parameter names**, and loads with
`strict=True` on purpose, so any future drift fails loudly instead of silently loading a
partially-random tower.

Correction to an earlier note in this file: there is **no** `tactile_type='full'` /
64-tokens-per-pad mode anywhere in that repo. FTP-1 emits one token per function area, max 48.

### 10.2 Channel order was settled by measurement, not assumption

FTP-1's own eval feeds BGR, because its zarr was built with `cv2.imdecode`. Whether our
decoded frames matched was genuinely ambiguous, and getting it backwards would have silently
degraded the tower rather than crashed.

`VisuoTactile_D-WHEEL` is the decisive test — it is the only sensor whose three channel means
are far enough apart to distinguish the orders. Measured against the published stats:

| | mean abs error vs published |
| --- | --- |
| our frames as-is | **0.050** |
| our frames reversed | 0.124 |

So our decoding already matches FTP-1's order and **no flip is applied**.

### 10.3 Normalisation is per *dataset*, not per sensor

The same GelSight Mini has channel mean −0.175 in `Unit`, −0.398 in `VLA_touch`, −0.568 in
`RDP` and −0.849 in `RDP_Bimanual` — gel colour, lighting and camera gain differ per rig. So
`FTP1_TACTILE_DATASETS` is keyed by `(dataset, slot)`, with slot order mirroring
`tactile_image_keys(spec)` so pixels and their statistics cannot drift apart.

Independent cross-check that the transcribed table is right: the checkpoint's
`FlexivGripperForce` std is 16.284, matching the 16.3 measured from our own data in §8.

### 10.4 The tower is frozen, and one of the reasons is not obvious

Three reasons, any one sufficient. It is 152M parameters, which would blow the budget; tactile
is not a large enough fraction of the signal to determine that many parameters; and — the
subtle one — **per-sensor dispatch makes the set of gradient-receiving parameters
data-dependent**. A batch with no SharpaWave rows gives the SharpaWave ViT no gradient. Under
ZeRO-2 that does not raise; it desynchronises the reduction schedule and hangs for the full
600 s NCCL timeout. Frozen parameters never enter the schedule at all, so the hazard vanishes.

**Do not unfreeze without first making dispatch data-independent.** Verified explicitly: 246
parameters receive gradient both with and without tactile rows in the batch, and the sets are
identical.

Cost, measured back-to-back on the same GPUs at the same time: **2.90 -> 3.39 s/step, +17%**,
with trainable parameters *dropping* 403M -> 385M. An earlier reading of "+57%" was wrong — its
baseline came from a stale, less-loaded machine. Never compare against a baseline measured at a
different time.

### 10.5 Two of sharpa's six pads are dead, and this matters more than the encoder choice

While probing channel statistics, whole-video measurements turned up a data problem that is
independent of everything above:

| dataset / view | fraction of frames with ~zero spatial std | median std |
| --- | --- | --- |
| sharpa / tactile_left_0 | 0.58 | 0.0020 |
| sharpa / tactile_left_1 | 0.58 | 0.0013 |
| **sharpa / tactile_left_2** | **1.00** | 0.0000 |
| sharpa / tactile_right_0 | 0.46 | 0.0241 |
| sharpa / tactile_right_1 | 0.38 | 0.0517 |
| **sharpa / tactile_right_2** | **1.00** | 0.0000 |
| **RDP_Bimanual / tactile_right_0** | **1.00** | 0.0000 |
| RDP_Bimanual / tactile_left_0 | 0.00 | 0.2459 |
| VisuoTactile_D-WHEEL (all 4) | 0.00 | 0.19–0.23 |

Two sharpa pads and one RDP_Bimanual pad are constant for their entire file. The four live
sharpa pads are blank 38–58% of the time, and their median spatial std of 0.001–0.05 is roughly
**10x below** the 0.1999 that FTP-1 published for the same dataset — our copy of sharpa is far
lower contrast than the copy FTP-1 measured. D-WHEEL is fully healthy.

This is consequential because the mixture changed: tactile is now **~22%** of
`debug_research_data` (fractal 0.340, ms_data_xdof_3 0.421, **sharpa 0.143**, D-WHEEL 0.047,
RH20TCfg5Franka 0.029, RDP_Bimanual 0.002), not the 2.7% assumed in §9. Any argument in this
file that leans on tactile being a negligible slice needs redoing.

The obvious fix — set `mask=0` for a pad whose spatial std is ~0, so the model stops spending
tokens on a constant image — **is now implemented**; see §10.9.

Related, unresolved: our `RDP_Bimanual/tactile_left_0` measures mean (0.019, 0.014, 0.086),
which matches the **RDP** domain's MCTac (0.039, 0.053, 0.123) almost exactly and matches *none*
of RDP_Bimanual's own sensors. Our copy is probably not the copy FTP-1 measured. The sensor
assignment there is inferred from key ordering and is flagged in-code as unverified.

### 10.6 Offline proxies could not decide FTP-1 vs ResNet

Two cheap proxies were tried before committing to an A/B, and both were discarded:

- **Cosine spread.** FTP-1's features are *less* spread than the ImageNet ResNet's. On a dead
  stream that is the correct behaviour; on a live one it is ambiguous. Uninformative.
- **Spearman(|Δfeature|, |Δpixel|).** ResNet won on sharpa (0.92 vs 0.80), FTP-1 won on D-WHEEL
  (0.11 vs −0.03). Inconclusive, and in opposite directions.

Consistent with §9: proxies that are not the training objective do not predict the training
objective. The encoder is therefore a **config switch** (`policy.tactile_backbone`, default
`resnet`), to be decided by windowed `contra_loss` on a matched A/B rather than by argument.

### 10.7 A trap worth naming: draccus splits nested list literals into characters

`--policy.ftp1_tactile_sensors="['A','B']"` arrives as the tuple
`('[', "'", 'A', "'", ',', ...)` — one element per character. Bare `draccus.parse` does **not**
reproduce this; it only happens under the nested `--policy.` path, so it cannot be caught by
testing the parser in isolation. A first fix that checked `isinstance(value, str)` failed
because the value is already split by the time it is seen. `__post_init__` now rejoins the
elements and re-splits whenever any element is not a valid sensor name, and raises with a
readable message otherwise.

### 10.8 A/B verdict: the frozen FTP-1 tower is *worse* than the trainable ResNet

Matched A/B, 1300 steps, 2x A6000 each, launched simultaneously on pinned
`CUDA_VISIBLE_DEVICES` so neither arm got a quieter machine. The two arms consumed
**byte-identical data** — `smpl` and `tac_n` agree at every logged step — which means the
per-step differences can be **paired**, a far more powerful test than comparing two noisy means
against the §9 noise floor.

| window | ftp1 `contra_loss` | resnet `contra_loss` | diff | ftp1 `acc` | resnet `acc` |
| --- | --- | --- | --- | --- | --- |
| 600–1300 | 4.694 | **4.609** | +0.084 | 0.0442 | **0.0496** |
| 900–1300 | 4.503 | **4.345** | +0.158 | 0.0591 | **0.0682** |
| 1000–1300 | 4.400 | **4.275** | +0.124 | 0.0678 | **0.0737** |

Paired over identical data (lower is better, so positive = FTP-1 loses):

| window | n | mean diff | t | steps where ftp1 wins |
| --- | --- | --- | --- | --- |
| >=600 | 36 | +0.084 | +3.49 | 9 / 36 |
| >=900 | 21 | **+0.158** | **+6.20** | **1 / 21** |

The result is consistent and not marginal: over the second half of training the FTP-1 arm is
behind at 20 of 21 logged steps, the gap exceeds the 0.04–0.09 noise floor, and retrieval
accuracy agrees independently. Note the gap **grows** with training (+0.084 -> +0.158) — the
signature of a frozen encoder being overtaken by a trainable one that is still adapting, rather
than of a bad initialisation.

Honest confound: the arms differ in two ways, not one. `tactile_backbone=ftp1` also forces
`tactile_recon_weight=0`, so the ResNet arm gets an extra auxiliary reconstruction signal. That
term is tiny in the loss (0.1 x 0.003), and the widening-with-training pattern points at
frozen-vs-trainable rather than the aux loss, but the experiment does not separate them.

**Decision: `tactile_backbone` stays `resnet` by default.** The FTP-1 path is kept as a working
config switch, because the picture could change: it is frozen (see §10.4 for why unfreezing is
not free), and §10.5 shows a large share of our tactile pixels are dead or very low contrast,
which handicaps a pretrained encoder more than a randomly-initialised one that can simply learn
to ignore them. Cost is +17% step time for a worse result, so there is no reason to switch now.

The broader lesson, and the second time this file has recorded it: pretrained weights that are
architecturally perfect (same token count, same 512-d output, normalisation stats covering our
exact datasets) can still lose. The fit of the artefact says nothing about whether it helps —
only the training objective does.

## 11. Masking dead tactile pads

§10.5 found that roughly half our tactile pad-frames carry no signal at all. A constant image
is not a free input: it costs a CNN pass and a physical-sequence token, it is a trivially
satisfiable reconstruction target, and it trains the encoder to expect that "tactile present"
usually means "tactile says nothing". `MultiModalContrastiveDataset` now drops those pads at
load time by zeroing `tactile_image_mask`, controlled by `policy.tactile_dead_std`.

### 11.1 The threshold came from the measured distribution, not a guess

Per-frame spatial std over 700 sampled items, on a [0, 1] scale (one 8-bit grey level = 0.0039):

| dataset | slot | p05 | p25 | median | p95 |
| --- | --- | --- | --- | --- | --- |
| D-WHEEL | 0–3 | 0.097–0.117 | 0.101–0.121 | 0.102–0.141 | 0.108–0.157 |
| sharpa | 0 | 0.0000 | 0.0002 | 0.0228 | 0.2132 |
| sharpa | 1 | 0.0000 | 0.0000 | 0.0123 | 0.2051 |
| sharpa | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0835 |
| sharpa | 3 | 0.0000 | 0.0000 | 0.0078 | 0.2075 |
| sharpa | 4 | 0.0000 | 0.0000 | 0.0073 | 0.1965 |
| sharpa | 5 | 0.0000 | 0.0000 | 0.0000 | **0.0002** |
| RDP_Bimanual | 0 / 1 | 0.1186 / 0.0000 | | | |

The population is strongly bimodal and the gap is wide: dead frames sit at <=0.0002, live ones
at >=0.09, and only ~11% of pad-frames fall anywhere between 0.001 and 0.05. **0.002** — half a
grey level — sits inside that gap, so the threshold is not delicately placed.

This also corrects §10.5 on one point. `sharpa` slot 2 is *not* dead everywhere: its p95 is
0.0835, so it does have live frames. The earlier "100% dead" reading came from a single video
file. Slot 5 is genuinely dead throughout (p95 = 0.0002).

### 11.2 Two details that a naive implementation gets wrong

**Test spatial std per channel, not the std of the whole frame.** A pad stuck at a uniform
non-black colour is spatially dead but has a non-zero global std across the channel axis, so a
global test would wave it through. The check is `frame.reshape(2, 3, -1).std(-1).max()`.

**Drop a pad only when *both* frames are flat.** One flat and one live is a contact onset or
release — precisely the event this branch exists to capture — so requiring both to be dead
keeps it. Hence `max` over the frame axis rather than `min` or `any`.

### 11.3 Verified

Keep rates on real data, 700 sampled items, counting only slots that physically exist:

| dataset | pad slots | kept | keep rate |
| --- | --- | --- | --- |
| D-WHEEL | 112 | 112 | **1.000** |
| sharpa | 648 | 268 | 0.414 |
| RDP_Bimanual | 2 | 1 | 0.500 |
| total | 762 | 381 | 0.500 |

The important row is D-WHEEL at 1.000: the threshold removes half of all tactile pads without
touching the one dataset measured as healthy. The other two match their measured dead fractions.

- `tactile_dead_std=0` is an exact no-op (keep rate 1.000 everywhere), so the old behaviour is
  one flag away.
- End-to-end smoke run on 2x A6000 is clean, loss descending normally, no errors.
- **ZeRO-2 gradient path is invariant**: 610 parameters receive gradient both when every pad in
  the batch is masked and when some are live, and the sets are identical. This matters more
  than before — masking makes "no live tactile anywhere in this batch" a common event rather
  than a rare one, so the pre-existing dummy-row guard in `PhysicalEncoder` is now load-bearing.
  Do not remove it.

No model change was needed: the encoder already selected only unmasked pads, so this rides on
existing machinery and slightly *reduces* tactile CNN compute.

Not yet measured: whether masking improves `contra_loss`. It should at least not hurt, and the
compute saving is real, but the honest statement is that this is a correctness fix justified by
the input distribution rather than a result validated against the training objective.

## 12. The temporal window is a duration, not a frame count

`chunk_size=16` was raised on the theory that most datasets run above 15 fps, so 16 frames is
under a second and too short for the perception side to see meaningful change. The theory is
right for most of the mixture and badly wrong for the largest single dataset in it.

### 12.1 fps is bimodal *by sampling weight*, which is what actually matters

| dataset | weight | fps | median ep len | 16 frames | 32 frames |
| --- | --- | --- | --- | --- | --- |
| ms_data_xdof_3 | 0.421 | 30 | 2522 | 0.53 s | 1.07 s |
| **fractal** | **0.340** | **3** | **40** | **5.33 s** | **10.67 s** |
| sharpa | 0.143 | 30 | 756 | 0.53 s | 1.07 s |
| D-WHEEL | 0.047 | 30 | 1122 | 0.53 s | 1.07 s |
| RH20T | 0.029 | 30 | 496 | 0.53 s | 1.07 s |
| taco_play | 0.019 | 15 | 66 | 1.07 s | 2.13 s |

Counted by dataset, five of seven run at 30 fps. Counted by sampling weight, **fractal alone is
34% of the mixture and runs at 3 fps with 43-frame episodes**. A fixed frame count therefore
means something different on every dataset -- the same 16 frames is 5.3 s on fractal and 0.53 s
on everything else, a 10x spread caused by nothing but the recording rate.

### 12.2 What a flat move to `chunk_size=32` would have cost

Measured mean absolute pixel change between the two perception frames, and the fraction of
sampled rows whose `t+H` stays inside the episode:

| dataset | change @16 | change @32 | valid @16 | valid @32 |
| --- | --- | --- | --- | --- |
| ms_data_xdof_3 | 0.0302 | 0.0394 (+30%) | 0.994 | 0.989 |
| **fractal** | 0.0625 | 0.0667 (**+7%**) | **0.634** | **0.318** |
| sharpa | 0.0724 | 0.0929 (+28%) | 0.981 | 0.962 |
| D-WHEEL | 0.0217 | 0.0239 (+10%) | 0.986 | 0.973 |
| RH20T | 0.0429 | 0.0610 (+42%) | 0.973 | 0.945 |
| taco_play | 0.0344 | 0.0412 (+20%) | 0.758 | 0.515 |

fractal **saturates** -- across H = 2/4/8/16/32/48 its change goes 0.032/0.042/0.054/0.063/
0.067/0.074 while validity collapses 0.93/0.86/0.80/0.66/0.39/0.14. Doubling its window buys
7% more visual change and costs half its usable rows. Weighted over the mixture, a flat
`chunk_size=32` moves valid pairs **0.864 -> 0.746**, i.e. 221 -> 191 usable rows out of a
256-row batch. That directly undoes the cross-GPU negative gathering added earlier.

Note the token budget is *not* the issue here: `num_groups = chunk_size / group_size`, so
32/4 = 8 = the previous 16/2 and the physical sequence stays 31 tokens. This is the opposite of
the §9.4 failure and the §9.4 lesson still applies -- count the budget, don't reason about it.

### 12.3 Equal duration, resampled onto a fixed token grid

`chunk_size` now means *the number of resampled timesteps handed to the model*, and the raw
window is `clamp(round(chunk_seconds * fps), chunk_frames_min, chunk_frames_max)` frames,
resampled onto that grid. With `chunk_seconds=1.6`, `min=8`, `max=48`:

| dataset | fps | window | duration | change | valid |
| --- | --- | --- | --- | --- | --- |
| fractal | 3 | 8 | 2.67 s | 0.0536 | **0.802** |
| taco_play | 15 | 24 | 1.60 s | 0.0400 | 0.857 |
| ms_data_xdof_3 | 30 | 48 | 1.60 s | 0.0442 (+46%) | 0.986 |
| sharpa | 30 | 48 | 1.60 s | 0.1098 (+52%) | 0.967 |
| D-WHEEL | 30 | 48 | 1.60 s | 0.0299 (+38%) | 1.000 |
| RH20T | 30 | 48 | 1.60 s | 0.0690 (+61%) | 1.000 |

**Weighted valid pairs go 0.864 -> 0.920**, i.e. better than the 16-frame baseline rather than
worse, while the 30 fps datasets get 38–61% more visual change. fractal's floor of 8 frames is
why: it trades 14% less change for 21% more valid rows, which nets out slightly ahead
(change x valid 0.0413 -> 0.0430) and is the reason `chunk_frames_min` exists.

Three implementation points that are easy to get wrong:

- **Resampling is nearest-frame, not interpolation in time.** Every offset is a real frame
  index, so `delta_timestamps` is never asked for a timestamp between two frames, which would
  trip the loader's tolerance check. Short windows repeat offsets and long ones stride them.
- **The action keys had to be overridden explicitly.** `resolve_delta_timestamps` builds the
  action chunk from `action_delta_indices`, which is a consecutive-frame range. Left alone, the
  action would have sat on a different time base from the state and the image pair it is
  supposed to explain -- a silent misalignment, since all the shapes still match.
- **The batch sampler needs a horizon *per dataset*.** It trims each episode by `horizon` to
  keep `t+H` inside it. A single global horizon of 48 would trim fractal's 43-frame episodes to
  nothing, throwing away exactly what this change was meant to protect.

Equalising *duration* rather than *visual change* is deliberate. Pixel change at equal duration
still differs 4x across datasets (sharpa 0.110 vs D-WHEEL 0.030) because they are different
scenes moving at different rates; tuning per-dataset windows to equalise it would be fitting a
pixel-MAE proxy, which §9 and §10.6 both show does not predict the training objective.

### 12.4 Verified

- Physical sequence is 31 tokens, unchanged; shapes are `action (32,40)`, `state (32,40)`,
  `tactile_signal (32,32)`, `tactile_image (6,2,3,112,112)`.
- Resolved windows logged at startup: fractal 8, taco 24, everything else 48.
  (This line did not actually print until the logging fix in S12.6 -- see there.)
- End-to-end run clean through an eval cycle, 772M params, global batch 512 on 2x A6000.
- **No slowdown: 2.82 s/step vs 2.90 s/step for the 16-frame baseline**, despite reading 3x
  more rows. State, action and tactile signal share a parquet row group, so the extra rows are
  nearly free -- the video decode, which still reads exactly 2 frames, is the real cost.

Not yet measured: whether this improves `contra_loss`. The input-side argument is strong
(more visual change *and* more valid pairs, at no extra cost), but that is a statement about
the inputs, not a result against the objective.

### 12.5 The grid did not reach `t+H` (found while explaining it)

Writing up how the window works surfaced an off-by-two. The offsets were built as
`round(i * H / chunk_size)`, which spans the **half-open** `[0, H)`: at 30 fps the physical
grid stopped at frame 46 while the image pair was `(0, 48)`, at 15 fps it stopped at 23 of 24,
and at 3 fps it happened to land on 8. So the physical window described slightly *less* change
than the perception side saw, by a margin that varied per dataset -- and every shape still
matched, so nothing complained. That is precisely the failure mode §12.3 already lists three
examples of.

Fixed to `round(i * H / (chunk_size - 1))`, spanning the closed `[0, H]`. Verified that
`max(offsets) == H` and `len(offsets) == chunk_size` for fps in {0.5, 1, 3, 10, 15, 30, 60,
120}, and at runtime that the chunk grid and the image pair now both end on the same frame for
all seven datasets. The loss curve is unchanged to 3 decimals over the first 40 steps, which is
expected rather than suspicious: `H` itself did not move, so the image pair is bit-identical
and only intermediate physical samples shifted by a frame or two.

Two guards added at the same time, both about *arbitrary* datasets rather than today's mixture:

- **fps <= 0 or missing now raises** instead of dividing by zero and producing a degenerate grid.
- **The clamp now warns.** The equal-duration guarantee only holds while `chunk_seconds * fps`
  stays inside `[chunk_frames_min, chunk_frames_max]`, i.e. **fps in [5, 30]** at the current
  settings. Outside it the clamp wins and the dataset silently stops sharing the mixture's
  temporal receptive field -- 0.80 s at 60 fps, 0.40 s at 120 fps, 8.0 s at 1 fps. It is a
  deliberate cost cap (max bounds the rows read, min protects short episodes), and fractal
  already lives on it at 2.67 s, but it should be visible in the log rather than inferred.

Also removed a duplicate fps source: the dataset loop derived the sampler's horizon from
`info.json` while `_build_dataset` used `ds_meta.fps`. `_build_dataset` now returns the horizon
it actually built the timestamps with, so the sampler cannot disagree with the loader.

### 12.6 Duration windows are a stage switch, not the only mode

Duration equalisation is right for *this* stage and wrong for the next one. The contrastive
physical branch is never executed -- it is pooled into an embedding that has to explain a
visual change -- so resampling costs nothing and equal duration is what makes datasets
comparable. A downstream VLA that *emits* an action chunk needs the opposite: consecutive
frames at the dataset's own control rate, because the chunk is a sequence of commands the
robot executes back-to-back. Resampling there would skip commands at 30 fps and emit duplicate
ones on fractal, and "equal duration" is meaningless when the chunk length *is* the action
horizon.

So `window_mode` selects between them:

| | `"duration"` (default, stage 1) | `"frames"` (VLA) |
| --- | --- | --- |
| offsets | `round(i * H / (chunk_size - 1))` | `0..chunk_size-1` |
| `H` | `clamp(round(chunk_seconds * fps), min, max)` | `frame_horizon`, else `chunk_size - 1` |
| fractal window | 8 frames / 2.67 s | 31 frames / **10.33 s** |
| 30 fps window | 48 frames / 1.60 s | 31 frames / **1.03 s** |
| physical tokens | 31 | 31 |

The last two rows are the point: the token count is invariant across both modes and every
dataset, so nothing downstream changes, while the fps imbalance that motivated §12.1 is
plainly visible in `"frames"` (10x spread) and gone in `"duration"`.

Both modes verified end-to-end on `debug_research_data` (2 GPUs, global batch 512): grids
correct at fps in {3, 15, 30}, `max(offsets) == H` in duration mode, offsets exactly
consecutive in frames mode, fps <= 0 raising in *both* (the caller divides by fps regardless of
mode, so the guard could not live in the duration branch alone). `trecon` and `tac_n` differ
between the two runs, confirming the modes really do feed different data rather than silently
collapsing to the same grid.

**The startup log was invisible.** `init_logger` only ever configured the `__main__` logger, so
every record from `lerobot.*` fell through to logging's `lastResort` handler, which drops
anything below WARNING. The per-dataset "resolved window" line -- added precisely so this was
auditable -- had never once printed; the only reason the clamp warning showed up is that it is
a WARNING. Fixed by giving the `lerobot` package logger the same handlers (not the root logger,
which would drag in INFO spam from torch/deepspeed/PIL). Worth remembering when trusting any
other library-side INFO line in this repo: several of them have presumably never been seen.

## 13. Adding the OpenNeo datasets

`open_neo_aloha` (12210 episodes / 17.0M frames, 4 tactile pads) and `open_neo_arx5_single`
(5178 / 8.3M, 2 pads), both 30 fps and both on an external mount. Three things blocked them,
none of which announced itself clearly.

**The dataset root is nested one level deeper than `vla2root.json` says.** `OpenNeoData/aloha`
contains `aloha/`, which is the actual dataset. `_resolve_root` tested `os.path.exists` on the
directory, so it happily returned the outer one and the failure surfaced much later as
`FileNotFoundError: .../OpenNeoData/arx5_single/meta/info.json` -- which reads like missing data
rather than a wrong path. It now tests for `meta/info.json` and falls back to `<root>/<name>`,
so the resolution either finds a real dataset or reports the dataset as not found.

**The hard-coded extra root became `dataset.parent_dir_extra`.** A third mount had been added
inline in `_resolve_root`; it is now a comma-separated config field, set in `train_ace_local.sh`.

**`datasets` 3.3.2 cannot read parquet written by `datasets` 4.x.** 4.x renamed the list feature
`Sequence` -> `List` and embeds that name in the parquet's HuggingFace metadata. 3.x resolves an
unknown `_type` with `globals().get(_type)`, which for `"List"` finds `typing.List` -- not a
dataclass -- and dies with a bare `TypeError: must be called with a dataclass type or instance`
from inside `dataclasses.fields`, with nothing in the traceback naming the feature, the file or
the version. Our other v3.0 datasets carry *no* HuggingFace metadata at all, so they infer from
the arrow schema and were never affected; this only appears on newly written data. Fixed by
aliasing `List` to `Sequence` in `_FEATURE_TYPES` (`datasets_v30/io_utils.py`), which is what
`generate_from_dict` already special-cases. Upgrading `datasets` was rejected: LeRobot 2.1 and
3.0 paths both sit on 3.x, and this is a one-line name alias.

### 13.1 The gripper was being mapped onto a joint slot

Both OpenNeo arms ship `action`/`observation.state` as **6 joints + 1 gripper** per arm, not 7
joints. `meta/stats.json` settles it: the last dim of each arm spans `[-0.004, 0.115]` (a
gripper width in metres) while every other dim spans several radians.

Sliced as a flat 7 (`_seg("action", 0, 7, 20)`), the gripper lands on canonical index 26, the
`joint_6` slot, and the joint-space gripper slot 27 stays masked. That puts a gripper width on
top of a genuine 7-DoF arm's last joint -- exactly the collision the slotted canonical space
exists to prevent, and invisible downstream because the mask count is identical either way.

The same bug was already present, and already annotated `# a little error, because final pos is
gripper`, in `ms_data_xdof_2` (aliased by `ms_data_xdof_3`, which *is* in the mixture): its dims
6 and 13 span `[0, 1.009]`, a normalised gripper. Fixed all of them to
`joints -> [20:26]`, `gripper -> [27]`, `joints -> [28:34]`, `gripper -> [35]`. Verified the
grippers now occupy 27/35 with the `joint_6` slots empty, and that the total live-slot count is
unchanged (34 bimanual, 17 single-arm), so only the *meaning* of the layout moved.

Note this changes `ms_data_xdof_3`'s input layout, so `contra_loss` numbers from before this
commit are not exactly comparable to those after.

### 13.2 Verified

- All 9 datasets in `debug_research_data` load; the mixture trains 40 steps and runs an eval
  cycle cleanly, exit 0.
- `tac_n` rises from ~75 to ~154 per step, i.e. the new tactile pads are actually being read.
- Windows resolve as expected: both OpenNeo sets 30 fps -> 48 frames / 1.60 s.
- Sample shapes unchanged: `action (32,40)`, `state (32,40)`, `tactile_signal (32,32)`,
  `tactile_image (6,2,3,112,112)`; aloha masks 4 pads live, arx5 2.

Caveat on the run itself: the machine was busy (three other users' jobs holding 15-45 GB), so
this was validated **single-GPU at batch 256**, not the usual 2x512. That still clears the >=128
single-card requirement, but it is not a throughput measurement.

## 14. One fps was doing two jobs

FTP-1's `info.json` declares 30 fps, but the data was captured at 10-15 Hz. The stored
`timestamp` column is exactly 1/30 s apart, i.e. **synthesised from the declared rate rather
than measured**, and consecutive rows repeat in several of the sets -- mean runs of 2.34
identical frames in D-WHEEL and 2.10 in RDP_Bimanual, the signature of a slower stream written
onto a 30 Hz index. (exUMI and sharpa show no repeats in `eef_pose`, so they were presumably
interpolated instead; the duplication test cannot date them.)

The loader was using the single declared fps for two jobs that only coincide when the label is
right:

- **Index fps** -- the time base the timestamps were written on. Frame `i` is at `i/30`, so
  every `delta_timestamps` value *must* be built with 30 or the loader matches a different
  frame, or none and trips its tolerance check.
- **True fps** -- the capture rate. This decides how much wall-clock motion `H` frames cover,
  and it is what `sample_rate` should report to the model.

Conflating them meant `chunk_seconds=1.6` silently requested **3.2 s of robot motion** on every
FTP-1 dataset while claiming 1.6 s, so the mixture was not temporally aligned after all -- the
exact failure §12 was written to fix, hidden one level deeper in a mislabelled input.

`dataset_fps.py` now holds a `DATASET_TRUE_FPS` table (all FTP-1 sets at 15.0), and
`_build_dataset` keeps the two apart: `_window_offsets(true_fps)` for the frame count,
`offset / index_fps` for the timestamps. FTP-1 now reads 24 frames spanning 1.60 s of true
time, with the last stamp at 0.800 s -- a real stored frame.

### 14.1 `sample_rate` had never been right

`sample_rate` came from `item.get("fps", 10)`. The v3.0 loader sets `item["fps"]` from the
*declared* fps (30, wrong for FTP-1); **the v2.1 loader never sets it at all**, so every v2.1
dataset silently fell back to the default 10 -- fractal (3 fps) and taco_play (15 fps) included.
The `sample_rate_embed` bucket was therefore wrong for 7 of the 10 datasets in the mixture, and
constant across the two whose rates differ by 5x. It now comes from `self.true_fps[ds_idx]`.

Verified per dataset after the change: fractal 3, taco 15, ms_data 30, all five FTP-1 sets 15,
both OpenNeo sets 30.

### 14.2 Verified

- All 10 datasets of `debug_research_data` load, including the newly added `ftp_1_exUMI`;
  30 steps run and exit 0.
- Resolved windows: FTP-1 `fps=15 (declared 30) -> 24 frames (1.60s)`, OpenNeo and ms_data 48,
  taco 24, fractal 8.
- Timestamps still land on real frames: max offset 24 / 30 fps = 0.800 s.
- `tac_n` ~154 per step, i.e. tactile is still being read at the new window.

Not measured: whether 15.0 is the right number for each FTP-1 set individually. The duplication
statistics suggest they differ (D-WHEEL ~12.8, RDP_Bimanual ~14.3, RH20T ~18.9), but a single
15.0 was chosen deliberately to get the mixture running; `DATASET_TRUE_FPS` is per dataset, so
refining it later is a one-line change each.

## 15. Why a tactile-heavy mixture is slow

Reported symptom: on the cluster, raising `ftp_1_sharpa` and
`ftp_1_VisuoTactile_D-WHEEL` to 60% of the mixture gave `updt_s:124.5 data_s:38.6` at batch
512. The investigation produced one finding that explains it, and one that turned out to be
about a code path the training scripts do not use. Both are recorded, the second one
explicitly as a correction, because it was briefly believed to be the answer.

### 15.1 The answer: the loader is the bottleneck, and pads are video streams

Every tactile pad is a separate video stream, decoded at both ends of the window. So a
`ftp_1_sharpa` sample costs **7** decodes (1 RGB + 6 pads) where a `taco_play` sample costs
1. Sustained loader throughput, measured with `scripts/profile_contrastive_step.py
--loader_only` on a mixture that is 60% sharpa + D-WHEEL, batch 128, 12 workers, local SSD:

| `max_tactile_views` | streams/sample (sharpa) | samples/s | ms/sample |
|---|---|---|---|
| 6 | 7 | 48.7 | 20.5 |
| 3 | 4 | 111.7 | 9.0 |
| 1 | 2 | 166.0 | 6.0 |

The model needs 149 samples/s to stay busy at batch 128 (0.86 s/step). At 6 pads the loader
delivers a third of that, so **the step is loader-bound by ~3x even on local SSD**. At batch
512 that is ~10.5 s/step of loading against ~3.4 s of compute. The cluster's ~160 s implies
its storage is roughly 15x slower per sample than local SSD, which is the expected shape for
a blob/NFS mount: sharpa is 53 GB across 267 mp4 files, and a cold decoder open has to pull
the index of a file holding 1.2M frames.

Measurement trap, hit twice: `dataloading` in the training log (and in the profiler) reads
**0.000 s** for the first few steps no matter how slow the loader is, because
`num_workers x prefetch_factor` = 48 batches are already queued. Any throughput claim needs
more steps than the prefetch depth -- hence `--loader_only --steps 110 --warmup 60`. A
6-step run "proving" the loader keeps up proves nothing.

Second trap: `data_s` only measures rank 0's own wait. Under ZeRO the other ranks block in
the all-reduce instead, and that time is charged to `updt_s`. A loader problem therefore
presents as a slow *update*, which is exactly how this was reported.

Knobs, in order of leverage:

1. `max_tactile_views` (`--policy.max_tactile_views`): 6 -> 3 is 2.3x. Direct, and costs
   pads.
2. Stage the FTP-1 datasets on node-local disk. The gap between 20.5 ms and the cluster's
   implied ~300 ms is storage, not code.
3. `LEROBOT_VIDEO_DECODER_CACHE_SIZE` (already 256 in both launch scripts). With the real
   sampler's episode locality, a cache of 100 misses 12.1% of the time and spends 23% of
   load time on cold opens; 400 misses 4.4% and spends 6.9%. Each cached decoder costs
   ~6.5 MB of RSS **per worker**, so 400 x 12 workers is ~31 GB.
4. `--num_workers`.

### 15.2 Correction: the pyav findings do not apply to the training scripts

`DatasetConfig.video_backend` defaulted to `pyav`, and on that path:

- pyav rebuilds a `VideoReader` per call with no decoder cache, so `ftp_1_sharpa` cost
  2911 ms/sample against torchcodec's 202 ms (random access; ~65 ms under the real sampler's
  locality).
- pyav **ignores `video_return_type`**, so the uint8 that `LeRobotDatasetV30` asks for came
  back as float in `[0, 1]`. A frame already at the target size then reached
  `_to_pixel_values`, which divides by 255, so the backbone saw an image 255x too dark; a
  frame needing a resize was cast to uint8 and truncated to exactly **0**, and so was every
  tactile pad. Measured: `fractal20220817_data` `image_t0` uint8 with range `[0, 1]`, mean
  0.000 -- literally black.

Both are real and both are fixed (default is now `torchcodec`; `_as_uint8` converts on the
way in, keyed off dtype rather than observed range). **But neither ever affected a training
run**: `train_ace.sh` and `train_ace_local.sh` have always passed
`--dataset.video_backend=torchcodec`, and under torchcodec every dataset already returns
uint8, so `_as_uint8` is a no-op there. The fix matters for anything that does *not* go
through those two scripts -- `scripts/probe_contrastive_dataset.py`,
`scripts/profile_contrastive_step.py`, and any new entry point that takes the config default.

Related correction: `tac_n` rising from 57.7 to ~180 per 256-row batch was **not** caused by
this fix. `mixtures.py` was edited between the two runs, raising sharpa and D-WHEEL from 0.5
to 1.0.

## 16. A resolution mismatch that only bites after the v3.0 merge

Training on the cluster kept logging

```
Failed to read robomind_franka_3rgb[127063]:
    Expected pre-allocated tensor of shape 480x640x3, got [720, 1280, 3]
```

The message is easy to read backwards. From torchcodec's `CpuDeviceInterface.cpp`:

```cpp
"Expected pre-allocated tensor of shape ", outputDims.height, "x", outputDims.width,
"x3, got ", shape
```

`outputDims` comes from the **actual decoded frame**; `shape` is the tensor the **batch API
pre-allocated from stream metadata**. So it means metadata claims 720x1280 while the frame
really decodes at 480x640 — not the other way round. torchcodec says as much in a comment:
stream metadata cannot express a variable-resolution stream, and `get_frames_at` allocates
the whole batch up front from that metadata. Single-frame indexing sizes itself from the
frame and keeps working, which is why the file looks fine when poked by hand.

Reproduced exactly by concatenating one 1280x720 and one 640x480 episode with `ffmpeg -c copy`
and calling `get_frames_at` across the seam — same error string, character for character.

`RoboMind_full/franka_3rgb` turns out to have **376 of 3606 `camera_top` videos at 640x480**
while `info.json` declares 720x1280 and the other 3230 are 1280x720. The odd ones sit in
blocks: episodes 892-988, 1022-1078, 1169-1178, 1190-1401. Both frame indices in the warnings
land inside them (127063 -> episode 917, 194891 -> episode 1251).

Every individual file is internally consistent, so **the local v2.1 copy — one mp4 per episode —
decodes without complaint**. The v3.0 conversion packs many episodes into one mp4 per camera;
any output file spanning a resolution boundary becomes variable-resolution and fails. Same
bytes, different layout, different outcome. Mixed resolutions would also break collation.

`scripts/check_video_resolution_consistency.py` checks this. It handles both layouts (v2.1
`episodes.jsonl` plus the path template, v3.0 `meta/episodes/**/*.parquet`), and reports:

1. stream-level resolution histogram per video key, flagged against `info.json`;
2. per-file variable-resolution detection — torchcodec runs the same batch call training does,
   then falls back to single-frame decoding to show where the resolution changes.
   `--exhaustive` swaps sampling for a full ffprobe frame walk;
3. `--frames 127063 194891` maps the indices from a warning back to episode and video file.

10818 files in 72 s at `--workers 24`. Exit code 1 = a file is internally broken, 2 = only
cross-file inconsistency (survives v2.1, will break once merged).

The fix is upstream of the loader: re-encode the 376 episodes to the declared 720x1280 (or
re-declare and downscale the rest) **before** converting to v3.0.

### 16.1 Re-encoding the odd videos

`scripts/reencode_video_resolution.py` normalises the offenders. Defaults match the repo's
`encode_video_frames`: `libsvtav1`, `yuv420p`, `crf 30`, `g 2`. The keyframe density of the
output matches the source exactly (150 keyframes over 299 frames, i.e. one every 2 frames),
so random access costs the same as before.

Frame count and frame rate are re-probed after encoding and must match the source, because
`episodes.jsonl` lengths and the `timestamp` column are built on them; a mismatch aborts that
file with the original untouched. Output goes to a temp name in the same directory and is
only `os.replace`d in after the dimensions, frame count, fps and a torchcodec batch decode all
pass. Originals are moved to `<root>/.reencode_backup` unless `--no-backup`.

Direction: bring the 376 minority files **up** to the declared 720x1280 rather than pulling
3230 files down. `--strategy stretch` (default) ignores the 4:3 -> 16:9 aspect change on
purpose — the loader ends with a plain `F.interpolate(..., size=(size, size))` to a square, so
aspect ratio is discarded downstream anyway, and stretching makes the 376 geometrically
consistent with the other 3230 `camera_top` videos rather than consistent with nothing.
`--strategy pad` / `--crop` are there if that trade is not wanted.

Verified on this dataset:

- 376/376 re-encoded in 4m06s at `--workers 12`; every frame count preserved.
- Concatenating a fixed episode with a native 1280x720 one — the v3.0 merge, simulated —
  now decodes; the same concat with the backed-up original still throws.
- `check_video_resolution_consistency.py` returns 0: `camera_top` is 720x1280 x 3606.
- Through the real `MultiModalContrastiveDataset` path: 306 frames covering both reported
  indices and the whole 640x480 span, zero `Failed to read` warnings.
- Failure path: with a bogus `--vcodec`, all originals kept their md5 and the backup dir
  stayed empty.

Backups came to 283 MB. The re-encoded files are larger than the originals (1.39 MB vs
0.92 MB for a 299-frame episode) since they now carry 2.25x the pixels.

## 17. A resume path that only breaks on the second checkpoint

Saving on the cluster died with:

```
KeyError: 'client_state contains reserved checkpoint key: checkpoint_parallel_dimensions.
           This key is used internally by DeepSpeed checkpoint metadata.'
```

The trigger was `dps_train_contrast.py`, which on resume did `client_state = loaded_state`
and then re-saved that same dict at every checkpoint.

`DeepSpeedEngine._load_checkpoint` does not return the client state as it was written. It
returns *everything in the checkpoint that is not on a hardcoded blacklist*:

```python
deepspeed_states = ['module', 'sparse_tensor_module_names', 'skipped_steps',
                    'global_steps', 'dp_world_size', 'mp_world_size',
                    'data_sampler', 'random_ltd']
client_state = {k: v for k, v in checkpoint.items() if k not in deepspeed_states}
```

The blacklist is incomplete, so DeepSpeed's own bookkeeping leaks into the value handed to
the caller. Measured on a real engine (0.18.4, ZeRO-1): we passed in `{step, epoch}` and got
back ten keys — `buffer_names`, `ds_config`, `ds_version`, `frozen_param_fragments`,
`frozen_param_shapes`, `global_samples`, `param_shapes`, `shared_params` on top of ours.
Note `frozen_param_fragments`: that is tensor data, and the old code was re-saving it inside
`client_state` on every checkpoint.

Newer DeepSpeed writes `checkpoint_parallel_dimensions` into the checkpoint on save and
guards against receiving it on save, but never added it to the load blacklist. So it leaks
out, goes straight back in, and the guard fires. Grafting that behaviour onto the local
0.18.4 reproduces it exactly: the key does come back through `load_checkpoint`, the old
pattern raises the identical `KeyError`, and reading only `step` saves cleanly and resumes.

Why it never showed up locally: local DeepSpeed is 0.18.4, which has no such key, and it
only fires on the *second* save of a run that resumed — a fresh run never touches the path.

Fix: treat the return value of `load_checkpoint` as read-only, pull out only the fields the
script owns, and build a fresh dict for every save. Same bug was live in `dps_train_ace.py`;
`dps_train_ace_lam.py` has its save commented out and `ddp_train.py` already built a fresh
dict each time.

## 18. What the tactile normalisation warnings were hiding

A full-mixture run logged five warnings from `ftp1_tactile.py`:

```
ftp_1_RDP exposes 1 tactile view(s) but FTP-1 lists 2; using the first 1.
open_neo_arx5_single has 2 tactile image view(s) but no FTP-1 sensor entry; falling back
  to GelSightMini with an identity z-score. Add it to FTP1_TACTILE_DATASETS.
open_neo_aloha ...   open_neo_arx5 ...   open_neo_ur ...
```

Neither is cosmetic. The per-view sensor id selects which pretrained ViT tokenizer a pad is
dispatched to, and the z-score is applied before it, so a wrong entry means real tactile
pixels are pushed through the wrong tokenizer at the wrong scale.

`scripts/measure_tactile_stats.py` recomputes the statistics the way FTP-1 defines them
(per-channel mean/std of `uint8/255*2-1`, no channel flip) and ranks the result against every
published statistic, which is how an unlabelled pad gets identified. Validated against two
domains whose true values are known: `VisuoTactile_D-WHEEL` measures
`[-0.4093, -0.2086, -0.2254]` against a published `[-0.4103, -0.2010, -0.2186]` (mean absolute
error 0.005) and `sharpa` `-0.841` against `-0.868`; in both cases the nearest-neighbour
ranking puts the dataset's own sensor first.

### The OpenNeo pads are not GelSights

| dataset | measured mean | measured std | nearest published |
|---|---|---|---|
| `open_neo_aloha` | 0.083, 0.096, 0.073 | 0.341, 0.335, 0.320 | RDP / MCTac, 0.046 |
| `open_neo_arx5_single` | 0.132, 0.111, 0.087 | 0.337, 0.328, 0.306 | RDP / MCTac, 0.063 |
| `open_neo_ur` | 0.086, 0.083, 0.061 | 0.342, 0.331, 0.314 | RDP / MCTac, 0.047 |

Within each rig the pads agree to within 0.013, so one statistic per dataset is enough. Of all
seven published sensors, GelSight Mini -- the value the fallback was choosing -- is the
*farthest* from these at 0.34-0.37, about seven times worse than MCTac. `open_neo_arx5` is not
held locally and reuses `open_neo_arx5_single` (same rig, bimanual); `open_neo_flexiv` is not
held either and has no sibling, so it gets the average of the three measured rigs.

### `ftp_1_RDP` was being handed the wrong tokenizer

FTP-1 lists two pads for this domain, `[GelSightMini, MCTac]`, and our conversion keeps one.
The truncation path takes *the first*, so it picked the GelSight -- but `canonical_space.py`
documents that rig's single camera as the MCTac. Fixed by listing only the pad we have.

`ftp_1_RDP_Bimanual` is the same story with a measurement to back it: its `tactile_left_0`
measures `[0.013, 0.008, 0.090]` with std `0.243`, which is 0.034 from RDP's MCTac
(`[0.039, 0.053, 0.123]`, std `0.252`) and 0.81 from the GelSight statistics FTP-1 published
for this very domain. Our copy is not the copy FTP-1 measured, so its published numbers were
replaced with ours.

### One of its pads is dead

`ftp_1_RDP_Bimanual`'s `tactile_right_0` is uniform black for all 23201 frames. ffprobe's
`signalstats` reports Y constant at 16 -- limited-range black -- and the file is 2.8 MB against
16 MB for the live pad. It is now dropped from `canonical_space.py`. The right arm's
gripper-force *signal* is fine (range -0.66 to 57.01, 2116 distinct values) and is kept.

This mattered more than it looks: the dataset sits at weight 2.0 in the mixture, so a constant
image was being fed to the contrastive objective, and every sample paid for a second tactile
video decode -- the exact cost that section 15 identified as the loader bottleneck.

### A float32 trap in the measurement itself

The first run reported `tactile_right_0` as mean -0.557, std 0.830 rather than the obvious
-1.0, std 0.0. `np.ndarray.sum` inherits the array's dtype, and a float32 accumulator stops
growing once the running total passes `2**24`: summing 30105600 pixels that are all exactly
-1.0 saturates at -16777216, and `-16777216/30105600 = -0.5573`, matching the reported figure
to four decimals. It only appears with large per-call batches, which is why the smaller
validation runs looked correct. Summing with `dtype=np.float64` fixes it; the regression test
is that an all -1.0 array must give mean -1.0 and std 0.0 exactly.

### After

All 17 tactile datasets in `canonical_space.py` resolve with zero warnings, no degenerate
statistics, and view counts that match the registry. Through the real dataset,
`ftp_1_RDP_Bimanual` now yields one live view instead of two, dispatched to MCTac, with the
measured z-score, and pixel means around 130 rather than a black frame.

### 18.1 After RDP arrived

`ftp_1_RDP` was inferred rather than measured in the previous section because the data was not
held locally. It is now, and the inference holds: `tactile_left_0` measures
`[0.0340, 0.0427, 0.1176]` with std `[0.2491, 0.2497, 0.2521]` over 600 frames spanning all 140
episodes, against FTP-1's published RDP/MCTac `[0.0392, 0.0525, 0.1232]` / `[0.2522, 0.2524,
0.2519]`. Mean absolute error 0.0069; the published GelSight for the same domain is 0.55 away.
So the single pad our conversion keeps is the MCTac, and unlike `RDP_Bimanual` our copy of RDP
*is* the copy FTP-1 measured -- its published statistics are used unchanged.

The dead pad was re-checked with a stronger tool than the sampling used before. `ffmpeg
blackdetect` over the whole of `RDP_Bimanual`'s `tactile_right_0` returns a single interval,
`black_start:0 black_end:773.333333`, against a file duration of `00:12:53.37` (773.37 s): every
one of the 23201 frames, not a sampled subset. The same scan of `tactile_left_0` returns no
black interval at all. The key exists, but there is nothing behind it, so it stays out of
`canonical_space.py`. If upstream FTP-1 has real pixels for that pad, the fix is to re-convert
the dataset, not to re-list this file.

## 19. The tactile reconstruction loss was two orders of magnitude too small

`PhysicalEncoder` trains the tactile ResNet with a reconstruction head borrowed from UniVTAC.
The head was there from the start, but it was not teaching the encoder anything, and the
reason is arithmetic rather than architectural.

### Why it mattered more than it looks

`tactile_image_gate` is initialised to zero and the tokens are scaled by `tanh(gate)`, so
while the gate is closed `∂L_contrastive/∂view_feats` is *exactly* zero. That is deliberate --
the reconstruction loss is supposed to shape the features until they are worth attending to --
but it means the reconstruction term is the **only** gradient reaching the tactile encoder at
the start of training. Whatever that term is worth is what the encoder learns.

### What it was worth

Gel images occupy a narrow slice of `[0, 1]`. Measured per-channel pixel std across our
tactile datasets:

| dataset | pixel mean | pixel std |
| --- | --- | --- |
| sharpa | 0.089 | 0.110 |
| D-WHEEL | 0.289 / 0.410 / 0.416 | 0.107 / 0.094 / 0.102 |
| neo_aloha | 0.510 / 0.549 / 0.551 | 0.201 / 0.169 / 0.157 |
| neo_ur | 0.551 / 0.544 / 0.540 | 0.180 / 0.175 / 0.161 |

So an MSE computed on the raw `[0, 1]` target bottoms out at the variance of that slice.
Measured at the 28x28 the loss actually uses (`scripts/tactile_recon_floor.py`), the MSE a
decoder reaches by emitting **one fixed image** is 0.0015-0.015. At
`tactile_recon_weight=0.1` that is a contribution of ~1e-3 against a contrastive loss of ~7.6.
The `trecon:0.1997` seen at step 20 is not evidence to the contrary -- the decoder ends in a
bare `Conv2d`, so its initial output is ~0 against a target whose mean is ~0.45, giving
MSE ~0.2. It falls to ~0.01 within a few hundred steps and then stops mattering.

### The target is not the problem, the scale is

Worth checking before turning the weight up: subtracting each *episode's* own mean image
instead of the global one removes everything explained by episode identity (gel wear,
lighting, sensor). What survives can only come from the contact.

| dataset | fraction of the 28x28 objective that survives | 
| --- | --- |
| sharpa | 83-96% |
| D-WHEEL | 58-74% |
| neo_ur | 26-31% |

Downsampling to 28x28 does not destroy the contact information the way it looks like it
should. The objective is pointing at the right thing; there is simply almost none of it.

### The fix

`_tactile_recon_loss` now z-scores the target with the per-dataset per-channel statistics the
batch already carries (`tactile_img_mean` / `tactile_img_std`, shifted from FTP-1's `[-1, 1]`
convention into `[0, 1]`). No dataloader change was needed -- those fields were already
emitted unconditionally at `contrastive_dataset.py:854` and simply ignored on the ResNet path.

Measured end to end on a real 16-sample batch of `ftp_1_sharpa_split_0` +
`ftp_1_VisuoTactile_D-WHEEL_split_0` (53 live pads):

| | target variance | constant-predictor floor |
| --- | --- | --- |
| raw `[0, 1]` | 0.027 | 0.022 |
| z-scored | 2.374 | 1.834 |

That is 85x more gradient. The weighted term goes from ~1e-3 to 0.285 against a contrastive
loss of ~7.6, i.e. from 0.01% of the objective to ~3.6%. 22 `tactile_cnn` parameter tensors
receive gradient, total norm 0.154. A batch with every pad masked still returns exactly
`0.00000000` and back-propagates cleanly.

Because the target is now unbounded, the decoder's missing final `Sigmoid` (UniVTAC's
`RGBDecoder` has one) is correct rather than an oversight, and is now documented as such.

### Per dataset, not global

A single pooled tactile z-score does *not* work, and the reason is a nice trap. Pooled over
all tactile datasets the statistics are mean `[0.398, 0.426, 0.424]`, std
`[0.240, 0.228, 0.222]` -- almost exactly ImageNet's `[0.229, 0.224, 0.225]`, because most of
the pooled variance is *between* datasets rather than within them. A global z-score would
therefore leave sharpa's target at 0.23 variance and neo_aloha's at 0.55 instead of 1. It
recovers 19-55x of the available 85-100x; per-dataset recovers all of it.

The same arithmetic explains why the **encoder input** normalisation does not matter. Five
schemes measured on identical frames through the same frozen ResNet
(`scripts/tactile_feature_probe.py` metrics, averaged over 5 datasets):

| input normalisation | spread | rho | auc |
| --- | --- | --- | --- |
| raw `[0, 1]` (UniVTAC's choice) | 0.081 | 0.206 | 0.739 |
| ImageNet (ours) | 0.104 | 0.202 | 0.742 |
| one global tactile z-score | 0.104 | 0.203 | 0.743 |
| per-dataset | 0.112 | 0.216 | 0.748 |
| per-sample | 0.128 | 0.210 | 0.743 |

All within noise. The input only needs to be in roughly the right range and a pretrained
ResNet absorbs the rest; the reconstruction *target* needs unit variance within each dataset
because the gradient is directly proportional to it. Two different requirements that look
like the same knob.

### The FTP-1 ViT is not the OOD risk it appears to be

`config.__post_init__` forces `tactile_recon_weight = 0` for `tactile_backbone="ftp1"`,
because a frozen tower has nothing to shape, so none of the above applies to that path. The
open question there was different: the tower's tokenizers were trained on FTP-1's sensors,
and OpenNeo is not one of them. Measured with `scripts/tactile_feature_probe.py`:

| dataset | sensor | FTP-1 ViT spread / rho / auc | ImageNet ResNet spread / rho / auc |
| --- | --- | --- | --- |
| sharpa *(in-domain)* | SharpaWave | 0.033 / 0.053 / 0.500 | 0.210 / 0.056 / 0.505 |
| D-WHEEL *(in-domain)* | OpenLoongVTouch | 0.203 / 0.140 / 0.790 | 0.022 / 0.117 / 0.703 |
| neo_ur *(unseen)* | MCTac | 0.100 / 0.374 / 0.688 | 0.067 / 0.277 / 0.832 |

The ViT responds to OpenNeo at least as strongly and as structurally as to sensors it was
trained on. All these pads are marker-gel optical sensors with the same image formation, and
the tokenizers are evidently not as sensor-specific as the per-sensor dispatch implies.

The genuine finding is elsewhere in that table: **on sharpa both encoders sit at chance**
(auc 0.500 / 0.505, rho ~0.05). That is a property of the data, not the encoders -- sharpa's
tactile *images* barely move. Its signal channel is the one carrying information.

### A trap in measuring any of this

The first version of the probe sampled each episode's first 96 frames and reported that the
FTP-1 ViT had collapsed on its own sensor (spread `-0.0000`, auc 0.496). It had not. Episodes
open with a pre-contact idle stretch: sharpa's first 96 frames have a pixel std of **3e-4**
and a frame-to-frame std of 2e-5, i.e. a still image. Both scripts now sample with
`np.linspace` over the whole episode, and both say so in their docstrings.

## 20. Stage-1: pre-training perception on video alone

Action-bearing data is a minority of the video that exists, but the perception branch does
not need actions: its reconstruction objective (predict the frame-`t+H` patch features from
frame `t`, the instruction and the change queries) is self-supervised on any two frames. So
it is now trained first on plain video, and the physical branch is introduced afterwards.

`lerobot/common/datasets/perception_dataset.py` reads two frames of one camera and nothing
else -- no canonical projection, no normalisation statistics, no tactile, no `get_spec`. A
dataset with no `action`, no `observation.state` and no `OXE_DATASET_CONFIGS` entry is a
first-class citizen. Entry point: `train_perception_local.sh`.

### Read `qgain`, not the loss

The failure mode of this objective is that most of frame `t+H` is predictable from frame `t`
alone -- background, table, static objects -- so the predictor can drive the loss down while
ignoring the change queries, which are the only thing being pre-trained *for* stage 2. The
loss curve looks healthy either way. This is the same shape of problem as the tactile gate in
§19: a term that appears to be training something while contributing no gradient to it.

`percep_query_gain` measures it directly: rerun the prediction with each sample's change
queries replaced by another sample's (a `roll`, so no sample keeps its own), and report how
much worse the reconstruction gets. Measured over a 500-step run at batch 128:

| step | 25 | 50 | 100 | 200 | 300 | 400 | 500 |
|---|---|---|---|---|---|---|---|
| loss | 0.417 | 0.302 | 0.230 | 0.139 | 0.105 | 0.090 | 0.083 |
| qgain | 0.0001 | 0.0013 | 0.0155 | 0.0387 | 0.0421 | 0.0492 | **0.0525** |

The ratio `qgain/loss` goes from 0.0003 to **0.63**: by step 500, handing a sample another
sample's queries costs 63% of the remaining loss. The queries are carrying pair-specific
information, so the objective is doing what it was designed to do. Note that at step 25
`qgain` was 0.0001 -- indistinguishable from "the queries are dead". Judging this metric
early would have produced the wrong conclusion in either direction.

The probe costs one extra no-grad predictor pass, so it is sampled (`query_probe_freq=50`)
rather than run every step.

### Missing language is a modality, not an empty string

Plenty of video has no instruction, and converted datasets often fill the column with `""`,
`"none"`, `"n/a"` or the dataset's own name. Feeding those to the text tower is not harmless:
they embed to a perfectly ordinary but *constant* vector, so "this dataset has no language"
becomes a fingerprint the contrastive stage can read instead of looking at the images.

`lerobot/common/datasets/instruction_text.py` holds the single shared rule
(`is_real_instruction`). Samples that fail it carry `has_text=0`; the encoder then masks
their text tokens out of every attention and seeds the change queries from a learned
`null_text` vector, exactly as the physical branch already does for its missing modalities.
Per-dataset text coverage is printed in the dataset table.

Verified: `has_text=None` is bit-identical to all-ones (so stage 2 is unchanged); a no-text
sample's output is **exactly** independent of its caption (maxdiff 0.0, i.e. no leakage);
mixed batches are row-wise correct; `null_text` receives gradient and moved off zero
(norm 0.0124 after 500 steps). All current mixtures report `has_text=1.0` throughout, so the
change is a no-op on today's data and a guard for the video corpora stage 1 is aimed at.

### Degenerate pairs are sampled away, not masked

Frames near the end of an episode have no partner at `t+H`; clamping gives `image_t1 ==
image_t0`, a target the predictor hits by copying its input. Stage 1 samples only from frames
whose partner lands inside the same episode (episode drawn proportionally to its trimmed
length, then a uniform offset -- uniform over valid frames without materialising them).
Measured on the mixture: `pair_is_valid` 1.0, zero identical pairs.

### Two stages, one temporal window

The pair-horizon rule now lives in `resolve_pair_horizon` and is shared by both loaders. A
perception encoder pre-trained on 1.0 s pairs and fine-tuned against 2.0 s action chunks
would be describing a different physical event than the one it was taught to describe, and
nothing in either pipeline would report it. The extraction was verified identical to the
previous inline rule across 2304 parameter combinations.

### Cost, and the transfer

574M total / 206M learnable (the physical branch is not built at all under
`policy.perception_only=true`). At batch 128 on one GPU: **2.22 GB peak**, 1.13 s/step,
`data_s` 0.001 -- the loader is fully hidden behind the step. Batch 128 is nowhere near the
memory limit here.

The checkpoint is a strict subset of the stage-2 one. Loading it into a full `RoboContrast`:
0 unexpected keys, 388 missing keys (all `physical_encoder.*` plus `logit_scale`), and all
765 perception tensors transfer exactly, of which 715 differ from a fresh init. Pass it with
`--policy.pretrained_path=<...>/mp_rank_00_model_states.pt`.

### A latent bug found on the way

`_build_unified_meta`'s fallback image feature had `names: None`, which
`dataset_to_policy_features` indexes to decide HWC vs CHW. It had never fired because the
contrastive loader always passes a real dataset's features. Stage 1 hit it and got
`'NoneType' object is not subscriptable`. Fixed in the fallback, and stage 1 also passes
real features now.
