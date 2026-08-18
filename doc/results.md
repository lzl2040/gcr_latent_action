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
- End-to-end run clean through an eval cycle, 772M params, global batch 512 on 2x A6000.
- **No slowdown: 2.82 s/step vs 2.90 s/step for the 16-frame baseline**, despite reading 3x
  more rows. State, action and tactile signal share a parquet row group, so the extra rows are
  nearly free -- the video decode, which still reads exactly 2 frames, is the real cost.

Not yet measured: whether this improves `contra_loss`. The input-side argument is strong
(more visual change *and* more valid pairs, at no extra cost), but that is a statement about
the inputs, not a result against the objective.
