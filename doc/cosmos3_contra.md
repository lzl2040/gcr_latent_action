# `cosmos3_contra`: module reference

What this branch adds to the perception ↔ physical contrastive model, what each module
costs, and which parts are actually trained.

The perception-side knobs still default to the previous behaviour, and the tactile switch
also defaults to the existing ResNet path, so an existing config produces the same model:

| config | values | default |
|---|---|---|
| `vision_backbone` | `dinov3`, `cosmos3`, `qwen3vl` | `dinov3` |
| `perception_recon_target` | `vision`, `vae` | `vision` |
| `num_cls_tokens` | `1`, or any `K ≤ num_change_queries` | `1` |
| `tactile_backbone` | `resnet18`, `ftp1`, `anytouch` | `resnet18` |

Supporting knobs: `cosmos3_dir` (weights, default `/Data/lzl/huggingface/Cosmos3-Edge`),
`qwen3vl_dir` (default `/Data/lzl/huggingface/Qwen3-VL-4B-Instruct`) and
`vae_repeat_frames` (default `1`; see "Repeating frames is a no-op" below). AnyTouch adds
`anytouch_checkpoint` (default `/Data/lzl/huggingface/anytouch_encoder.pth`) and
`anytouch_forward_batch_size` (default `128` dynamic windows).

Regenerate every table here with `python scripts/dump_module_params.py`. Do not hand-edit
them — a stale parameter table is worse than no table.

---

## 1. Parameter budget

### `vision_backbone=dinov3, perception_recon_target=vision` (default)

| module | params | trainable |
|---|---:|---:|
| **perception_encoder** | **573.5M** | **205.6M** |
| &nbsp;&nbsp;`vision_backbone` (DINOv3 ViT-B/16) | 85.7M | 0.0M |
| &nbsp;&nbsp;`text_backbone` (SigLIP2 text tower) | 282.3M | 0.0M |
| &nbsp;&nbsp;`visual_proj` | 0.8M | 0.8M |
| &nbsp;&nbsp;`text_proj` | 0.8M | 0.8M |
| &nbsp;&nbsp;`evidence_blocks` (5 × self-attn over ~620 tokens) | 63.0M | 63.0M |
| &nbsp;&nbsp;`blocks` (5 × change-query cross-attn) | 84.0M | 84.0M |
| &nbsp;&nbsp;`out_proj` | 1.6M | 1.6M |
| &nbsp;&nbsp;`predictor` (reconstruction head) | 55.4M | 55.4M |
| **physical_encoder** | **200.9M** | **200.9M** |
| &nbsp;&nbsp;`state_proj` / `action_proj` / `signal_proj` | 0.3M each | 0.3M each |
| &nbsp;&nbsp;`tactile_cnn` | 11.2M | 11.2M |
| &nbsp;&nbsp;`tactile_temporal` | 3.2M | 3.2M |
| &nbsp;&nbsp;`tactile_img_proj` | 0.5M | 0.5M |
| &nbsp;&nbsp;`tactile_recon` | 7.1M | 7.1M |
| &nbsp;&nbsp;`sample_rate_embed` | 0.1M | 0.1M |
| &nbsp;&nbsp;`blocks` (14 × self-attn over ~40 tokens) | 176.4M | 176.4M |
| &nbsp;&nbsp;`out_proj` | 1.6M | 1.6M |
| **total** | **774.4M** | **406.5M** |

Switching `vision_backbone` changes only the `vision_backbone` row (and `predictor` by
±0.2M, since the reconstruction head's width follows the tower); every other row is
identical. The `vae` target adds a frozen 149.6M VAE encoder. See the summary table below
rather than repeating the breakdown four times.

### `vision_backbone=cosmos3` / `qwen3vl`

Only the perception branch changes; the physical branch is identical in every row.

| configuration | perception | physical | total | trainable | step @ bs128 |
|---|---:|---:|---:|---:|---:|
| `dinov3` + `vision` | 573.5M | 200.9M | **774.4M** | **406.5M** | 1.02 s |
| `cosmos3` + `vision` | 901.3M | 200.9M | **1102.2M** | **407.3M** | 1.71 s |
| `cosmos3` + `vae` | 1049.8M | 200.9M | **1250.7M** | **406.1M** | — |
| `qwen3vl` + `vision` | 794.6M | 200.9M | **995.5M** | **407.0M** | 1.70 s |
| `qwen3vl` + `vae` | 943.3M | 200.9M | **1144.2M** | **406.0M** | — |

The deltas are entirely frozen: `vision_backbone` 85.7M → 412.6M (cosmos3) or 306.2M
(qwen3vl), plus a 149.6M VAE encoder for the `vae` target. Trainable capacity stays at
~406M in every combination and is split almost exactly in half between the two branches
(205M / 201M), which is the budget the design targets. Total parameters exceed the original
500–800M envelope, but those are frozen feature extractors, not capacity the model is free
to use.

**The largest frozen module is the SigLIP2 text tower at 282.3M** — over three times
DINOv3's vision tower. It is kept because its text embedding is already aligned to a
visual space, which is what makes "the instruction selects which change matters" work.
Cosmos3's own text tower is a 2.2B LLM in `transformer/`; it was measured to add ~0.3s to
a 1.04s step, against the 0.013s the SigLIP2 tower currently costs, so it is not used and
its weights are not downloaded.

### Optional `tactile_backbone=anytouch`

Generated with:

```bash
python scripts/dump_module_params.py \
  --vision_backbone dinov3 \
  --perception_recon_target vision \
  --tactile_backbone anytouch
```

| module | params | trainable |
|---|---:|---:|
| **perception_encoder** | **573.5M** | **205.6M** |
| &nbsp;&nbsp;`vision_backbone` | 85.7M | 0.0M |
| &nbsp;&nbsp;`text_backbone` | 282.3M | 0.0M |
| &nbsp;&nbsp;`visual_proj` | 0.8M | 0.8M |
| &nbsp;&nbsp;`text_proj` | 0.8M | 0.8M |
| &nbsp;&nbsp;`evidence_blocks` | 63.0M | 63.0M |
| &nbsp;&nbsp;`blocks` | 84.0M | 84.0M |
| &nbsp;&nbsp;`out_proj` | 1.6M | 1.6M |
| &nbsp;&nbsp;`predictor` | 55.4M | 55.4M |
| **physical_encoder** | **485.1M** | **179.9M** |
| &nbsp;&nbsp;`state_proj` | 0.3M | 0.3M |
| &nbsp;&nbsp;`action_proj` | 0.3M | 0.3M |
| &nbsp;&nbsp;`signal_proj` | 0.3M | 0.3M |
| &nbsp;&nbsp;`tactile_cnn` (AnyTouch ViT-L/14) | 305.2M | 0.0M |
| &nbsp;&nbsp;`tactile_adapter` | 0.4M | 0.4M |
| &nbsp;&nbsp;`tactile_img_proj` | 0.5M | 0.5M |
| &nbsp;&nbsp;`sample_rate_embed` | 0.1M | 0.1M |
| &nbsp;&nbsp;`blocks` | 176.4M | 176.4M |
| &nbsp;&nbsp;`out_proj` | 1.6M | 1.6M |
| **total** | **1058.6M** | **385.4M** |

The optional tower therefore exceeds the original 500–800M total-parameter envelope, but
all 305.2M AnyTouch parameters are frozen. It removes the trainable ResNet, temporal
attention and tactile reconstruction head, then adds a trainable `768 -> 512` adapter.
Consequently trainable capacity falls from 406.5M to 385.4M rather than increasing.

---

## 2. Vision backbone

`build_cosmos3_vision()` in `lerobot/common/policies/ace/cosmos3_encoders.py`.

Cosmos-Reason3-Edge's vision tower is SigLIP-so400m-shaped (hidden 1152, 27 layers, patch
16, native 256×256 → a 16×16 = 256 patch-token grid) but is **not** loadable as a SigLIP
checkpoint without remapping:

- Weights are namespaced `model.visual.*`, not `vision_model.*`.
- `patch_embedding.weight` is stored `(1152, 768)` — a Linear over flattened 3×16×16
  patches. SigLIP wants a `(1152, 3, 16, 16)` Conv2d. Same values in the same order, so a
  `view` converts it.
- The checkpoint has 437 tensors against SigLIP's 448. The 11 missing ones are SigLIP's
  attention-pooling head, dropped with `vision_use_head=False`.
- The ~77M projector that follows the tower is deliberately discarded; it maps into the
  Reasoner LLM's space, which is not this model's space.

The loader **raises** on any missing or unexpected key other than `position_ids`. A
silently partial load of a frozen tower is invisible: the model trains and merely learns
less.

Two consequences handled elsewhere in the code:

| | DINOv3 | Cosmos3 |
|---|---|---|
| input resolution | 224 | **256** |
| normalisation | ImageNet mean/std | **0.5 / 0.5** (symmetric `[-1, 1]`) |
| prefix tokens to strip | 1 CLS + registers | **0** |
| patch tokens | 196 | **256** |

`_to_pixel_values` is now an instance method that reads `self.image_size` and
`self.pixel_mean/std`, so resolution and normalisation follow the *backbone* rather than
the dataset. Feeding a SigLIP-family tower 224px ImageNet-normalised input is silently
wrong — it still runs, it is just off-distribution for weights that are never trained.

### 2b. Qwen3-VL (`vision_backbone=qwen3vl`)

`build_qwen3vl_vision()` in `lerobot/common/policies/ace/qwen3vl_encoder.py`.

**First, a correction that motivated this backbone.** Cosmos3-Edge is *not* a Qwen3-VL
derivative. It is `model_type: cosmos3_edge`, a bespoke NVIDIA Mixture-of-Transformers; its
text tower uses `relu2` with vocab 131072, where Qwen3 uses `silu` with vocab 151936. The
NVIDIA line that *is* built on Qwen is **Cosmos-Reason** — HF's metadata for
`nvidia/Cosmos-Reason2-8B` carries `base_model:finetune:Qwen/Qwen3-VL-8B-Instruct`. So
there was no "Qwen3-VL increment" inside Cosmos3 to strip; this is a different backbone,
not a cleaner version of the same one. The block hyperparameters coincide because both
adopt the so400m recipe, but the weights are not interchangeable.

**Which size.** Qwen3-VL ships only two ViT variants: 2B/4B share a 1024-wide, 24-layer
tower, and 8B and above share a 1152-wide, 27-layer one (the same shape as Cosmos3-Edge's).
The 2B and 4B `vision_config`s differ *only* in `out_hidden_size` (2048 vs 2560), which
feeds the merger — and the merger MLP is dropped here, so **the trunk we use is 306.2M and
shape-identical in both**. Byte-comparing visual tensors from the two repos shows matching
names and shapes but different values, i.e. they were trained separately. 4B is the default
because it costs exactly the same to run and was co-trained with a stronger LLM. Neither
matches Cosmos3-Edge's 412.6M/1152-d; true parity would be the 8B tower, and switching is a
directory change since the loader reads the tower shape from the checkpoint.

Three properties of this tower are silent if handled wrong, so each is asserted by
`scripts/check_qwen3vl_vision.py` rather than assumed:

| trap | why it matters | check |
|---|---|---|
| forward takes **pre-flattened patches** `(seq, 3·2·16·16)` + a `grid_thw` table, not `pixel_values` | we must patchify by hand | vs Qwen's own image processor: 1.2e-07 |
| patches are ordered **by 2×2 merge block**, not row-major | token *i* would not be image location *i*, breaking the evidence stream's core property that `v1[i]-v0[i]` is "what changed at location *i*", and misaligning the VAE target | perturb one image cell, confirm the most-changed token is the matching row-major index |
| the tower ends in a **merger that pools 2×2 patches** (256→64 tokens) | destroys the spatial grid and the width | only `merger.norm` is kept |

Keeping `merger.norm` loses nothing: it is a `LayerNorm(hidden_size)` applied per token
*before* the pooling reshape, so it is the tower's own trained output norm at full
resolution. The DeepStack heads (81.8M) are dropped — they exist to inject intermediate
layers into the first three LLM layers, and there is no LLM here. That does mean the
features are not quite everything Qwen's pretraining optimised the tower to emit.

#### Why the original merger is removed

The merger is not an intrinsic part of spatial feature extraction; it is the adapter between
the ViT and Qwen's language model. For the 4B checkpoint it groups every 2×2 block of
1024-dimensional patches, concatenates the four vectors, and projects `4096→2560` so the
result matches the text model's hidden width. That is useful when visual tokens are inserted
into the LLM, but conflicts with this model in three ways:

1. The perception branch forms token-wise evidence such as `v1[i] - v0[i]`. Keeping the
   original merger would reduce the 16×16 grid to 8×8 and mix four spatial locations before
   the change-query transformer sees them.
2. The vision reconstruction target has a 16×16 patch grid (and the Wan VAE target is also
   16×16). Preserving 256 tokens gives an exact location-to-location target instead of asking
   the predictor to recover four cells from one already-pooled token.
3. The merger's 2560-dimensional output width exists only to match Qwen3-VL-4B's text
   decoder. This repository has no Qwen decoder; its own `visual_proj` maps the retained
   1024-dimensional vision features into the shared 512-dimensional perception space.

The whole merger is therefore not discarded blindly: its trained per-token LayerNorm is
kept, while only the task-specific 2×2 pooling and LLM-width MLP are removed. The trade-off is
that these are full-resolution ViT features rather than the exact 64 visual tokens consumed
by Qwen's LLM, which is intentional for spatial change reconstruction.

#### Why the code reads `pooler_output` even though no pooling remains

`pooler_output` is the fixed Hugging Face output-field name for the result of Qwen3-VL's
`merger`; the name describes the **original module**, not necessarily the replacement
installed at runtime. With the stock Qwen3-VL-4B merger, a 256×256 image follows:

```text
last_hidden_state: (N·256, 1024)
    -> LayerNorm
    -> group each 2×2 patch block: (N·64, 4096)
    -> merger MLP
pooler_output:     (N·64, 2560)
```

This repository replaces that merger with `_NormOnlyMerger(model.merger.norm)`. The forward
API still writes its result into `pooler_output`, but the replacement performs no reshape,
pooling, or 1024→2560 projection:

```text
last_hidden_state: (N·256, 1024)  # final ViT block, before output norm
pooler_output:     (N·256, 1024)  # after pretrained LayerNorm; not actually pooled here
wrapper output:    (N, 256, 1024) # restored batch and row-major spatial order
```

The perception encoder concatenates the current and future frames before this call, so for
a training batch of size `B`, `N=2B`; it then splits the wrapper output back into
`p0, p1: (B, 256, 1024)`. Reading `last_hidden_state` would preserve the token count but
silently omit the pretrained output LayerNorm. Transformers 4.57 returned the normalized
merger result as `tuple[0]`, so selecting `pooler_output` on Transformers 5 also keeps both
versions semantically aligned.

The patch-ordering trap is written up in full — why we cannot call Qwen's own processor in
the training loop, how the merge-block layout was derived, and how it is verified — in
[`doc/problem_and_solution.md` §1](problem_and_solution.md).

**Performance.** Out of the box this tower cost **1.95 s/step** against DINOv3's 1.02 s,
even though it is *smaller* than Cosmos3-Edge (306M vs 413M), which ran in 0.71 s. That
disproportion was the signal to profile inside the tower rather than accept it:

| stage | ms (256-image batch) | share |
|---|---:|---:|
| transformer blocks | 764.6 | 76.4% |
| `fast_pos_embed_interpolate` | 144.0 | 14.4% |
| `rot_pos_emb` | 64.9 | 6.5% |
| `patch_embed` | 22.0 | 2.2% |

Two fixes, both **algebraic no-ops verified bit-identical** (rel diff `0.0` in fp32 *and*
bf16) against the stock implementation:

1. **Batched attention.** The non-FlashAttention path splits the concatenated sequence into
   one SDPA call *per image* — ~6k tiny kernel launches per forward at batch 256. Every
   frame here is the same fixed resolution, so block-diagonal attention *is* batched
   attention; we reshape to `(B, H, n, d)` and issue one call. Non-uniform batches fall back
   to the stock loop, so mixed-resolution input stays correct, just slow.
2. **Grid caches.** `fast_pos_embed_interpolate` and `rot_pos_emb` are pure functions of a
   grid that never changes, yet are recomputed every step — 21% of the forward spent
   producing constants. Now memoised. The pos-embed cache **disables itself when
   `pos_embed.weight` is trainable**, so unfreezing the tower cannot silently pin the
   position embedding; a gradient check covers exactly that.

Result: vision **0.94 → 0.70 s**, step **1.95 → 1.70 s** at batch 256 — now on par with
Cosmos3-Edge per parameter. The remaining gap to DINOv3 (1.70 vs 1.02 s) is the honest cost
of a 306M tower at 256 tokens versus an 86M tower at 196.

**Environment note.** `grid_thw` is built on the CPU on purpose: `rot_pos_emb` calls
`torch.prod` on it, and an int64 `prod` on CUDA routes through torch's nvrtc jiterator,
which fails on this host (`libnvrtc-builtins.so.13.0` not found). The tensor is `(B, 3)`, so
computing it on the host is free.

| | DINOv3 | Cosmos3 | Qwen3-VL |
|---|---|---|---|
| trunk params | 86M | 412.6M | **306.2M** |
| hidden / layers | 768 / 12 | 1152 / 27 | **1024 / 24** |
| input resolution | 224 | 256 | **256** |
| normalisation | ImageNet | 0.5 / 0.5 | **0.5 / 0.5** |
| patch tokens | 196 | 256 | **256** |
| per-token feature L2 | 12.6 | 59.7 | **9.95** |

---

## 3. Reconstruction target

The predictor reconstructs frame `t+H` from frame `t`, the instruction and the change
queries. The queries are the only route by which anything about `t+H` can reach the
prediction, which is what forces them to encode the change.

| target | what is predicted | channels | extra frozen cost |
|---|---|---:|---:|
| `vision` | the vision tower's own `t+H` features | 768 (DINOv3) / 1152 (Cosmos3) | 0 |
| `vae` | the Cosmos3 (Wan2.2) VAE latent of `t+H` | 48 | 149.6M |

**The two grids coincide, which is why this is a drop-in swap.** Both the ViT and the VAE
tile the image into 16-pixel cells, so at 224px both produce 14×14 = 196 tokens and at
256px both produce 16×16 = 256. "Predict `t+H`" therefore stays a per-token regression in
either case; only the channel count changes. `_recon_loss` raises if the counts ever
diverge rather than letting a broadcast paper over it.

Three things the `vae` path gets right that are easy to get wrong:

1. **The decoder is dropped.** It is 555M of the checkpoint's 705M and only `encode` is
   ever called. Keeping it would cost more memory than the entire trainable perception
   trunk, for a module that never runs.
2. **Wan2.2's published per-channel latent statistics are applied.** Unnormalised, the 48
   channels differ in scale by over an order of magnitude and the loss simply follows
   whichever few are largest — the same failure §19 of `results.md` hit with unnormalised
   tactile targets. The per-token `target_norm` (non-affine LayerNorm, as in I-JEPA) is
   then applied on top, identically to the `vision` path.
3. **Repeating frames is a no-op.** The Wan VAE is *causal* and pads its own temporal
   history, so a single frame already yields one latent frame: measured, `T=1` and `T=4`
   (the same frame repeated) give bit-identical latents, cos-sim 1.0000, relative
   difference 0.0000. `vae_repeat_frames` remains configurable but raising it only
   multiplies VAE compute for the same target. Default 1.

### When to prefer which

`vision` is self-consistent and free — the features are already computed for the evidence
bank. `vae` ties the perception trunk to the *generator's* latent space, which is what a
stage-2 Cosmos world model actually consumes.

Note the tension worth measuring rather than assuming: the VAE is a generic video
compressor with no robotics knowledge, and a reconstruction latent by construction
preserves exactly the nuisance detail (lighting, texture, background) that the contrastive
objective is trying to discard. `vision` remains the default for that reason.

---

## 4. Multi-token alignment (`num_cls_tokens`)

At `K = 1` (default) nothing changes: the perception side pools its 16 change queries into
one embedding, the physical side reads one CLS token, and similarity is a dot product.

At `K > 1` neither side pools.

**Physical side.** `cls_token` becomes `(1, K, dim)`. The K vectors are independent
learned parameters, not tied, so they can specialise on different parts of the chunk — an
approach phase and a grasp, say — instead of being forced to average them. The read-out is
`out_proj(out_norm(tokens[:, :K]))`.

**Perception side.** The 16 change queries are mixed down to K vectors by a learned
`(Q, K)` matrix over the query axis, initialised to a uniform `1/Q`. Two properties matter:

- At `K = 1` this reproduces the old `queries.mean(dim=1)` **exactly at init**, so
  enabling the mechanism cannot by itself change the `K = 1` result.
- Each output token can learn to draw on a different subset of queries, rather than being
  handed an arbitrary slice of them.

**Similarity.** `pairwise_similarity` in `modeling_robo_contrast.py` dispatches on rank:

- `(N, D) × (M, D) → (N, M)`: the ordinary dot product.
- `(N, K, D) × (M, J, D) → (N, M)`: ColBERT-style late interaction — each of `a`'s K
  tokens finds its best match among `b`'s J tokens, and those matches are **averaged**.

The mean rather than ColBERT's sum is deliberate: it keeps the result in `[-1, 1]` exactly
as the `K = 1` dot product is, so `logit_scale` and `temperature` stay calibrated and a K
sweep does not silently rescale the loss.

Verified numerically: at `K = 1` the new path is bit-identical to `a @ b.t()` (max abs
difference 0.0), and a degenerate `K = 4` model whose four tokens are identical reduces to
the `K = 1` answer to float precision (5.6e-08). Late interaction is a strict
generalisation, not a different objective.

**Cost.** Similarity becomes an `O(K²)` einsum per pair instead of a matmul. At batch 256
and K = 4 the intermediate is 256 × 256 × 16 floats — negligible. What is *not* negligible
is interpretability: with K > 1 there is no single embedding to monitor, so read `gap`
(`pos_sim − neg_sim`) and `erank`, not `pos_sim`.

`contrastive_eval.py` uses the same two functions, so training and evaluation cannot drift
apart. Its effective-rank metric flattens the token axis, treating each token as its own
sample — a K > 1 model that gave all K tokens the same direction is collapsed in exactly
the sense that metric exists to catch.

---

## 5. AnyTouch tactile encoder

`lerobot/common/policies/ace/anytouch_tactile.py` reconstructs only the touch encoder from
the MIT-licensed [GeWu-Lab/AnyTouch](https://github.com/GeWu-Lab/AnyTouch) release, pinned
to commit `9c43a1a6eb38d904fd767712eb9dcb2d98b8d56b`. The local stage-2 checkpoint is a 2.9 GiB
full multimodal checkpoint; the loader deliberately maps only:

- the 24-layer CLIP ViT-L/14 touch transformer;
- the shared 3-D patch convolution;
- the five-token sensor embedding;
- the `1024 -> 768` touch projection.

Text, vision and MAE-decoder weights are never registered in this model. Loading is strict
for the retained CLIP subtree, so a missing or shape-incompatible encoder key raises instead
of silently leaving random weights. The checkpoint also contains
`video_position_embedding`, but the pinned AnyTouch dynamic forward does not read it; it
uses the CLIP touch position embedding for both static and dynamic inputs, and this
implementation preserves that behaviour.

### Input and temporal layout

Each frame is RGB, resized directly to 224 × 224, converted to `[0, 1]`, and normalized by
ImageNet mean/std. There is no center crop. The dynamic path preserves AnyTouch's unusual
released layout:

```text
(B, T=3, C=3, H, W) -> Conv3d(in_channels=3, kernel=(3,14,14))
```

This means the three frames occupy the convolution's input-channel axis and RGB occupies
its depth axis. Swapping to the conventional `(B,C,T,H,W)` interpretation looks cleaner but
would apply the pretrained weights to different dimensions.

The dataset supplies four samples per tactile pad. With the default
`tactile_tokens_per_pad=2`, they become two contiguous windows:

```text
token 0 <- frames [0,1,2]
token 1 <- frames [1,2,3]
```

Each window produces the post-LayerNorm CLS feature projected to 768 dimensions, exactly
the representation supervised by AnyTouch stage 2. A trainable linear + LayerNorm adapter
maps it to the existing 512-dimensional tactile interface; the physical transformer still
receives only two tokens per pad, not 256 patch tokens. Setting
`tactile_tokens_per_pad=1` uses only frames `[0,1,2]` and approximately halves AnyTouch
compute.

All current image-tactile datasets in `debug_research_data` use sensors outside AnyTouch's
five named training sensors, so the encoder uses sensor id `-1`, the learned universal
sensor token. AnyTouch explicitly trained that token by replacing known sensor identities
during pre-training; inventing a GelSight/DIGIT identity for Sharpa or OpenLoong would be
less defensible. Pad identity remains represented by this model's separate
`tactile_view_embed`.

The ViT, sensor token and pretrained projection are frozen and forced to eval mode. Only
the `768 -> 512` adapter, `tactile_img_proj`, modality gate and physical transformer learn.
`tactile_recon_weight` is forced to zero because the AnyTouch MAE decoder is not loaded and
a decoder attached after a frozen feature cannot shape the backbone. Frozen windows are
processed in chunks of `anytouch_forward_batch_size` (128 by default), bounding transient
attention memory for tactile-heavy batches without changing outputs.

There is still a temporal-domain mismatch to measure rather than hide: AnyTouch learned
from short clips of nearby historical frames, whereas this dataset samples four tactile
observations uniformly across the same ~1.6 s window as state/action/vision. The pretrained
spatial and sensor invariances remain useful, but its temporal filters are seeing larger
gaps than in much of the original corpus. Keep the ResNet baseline in every experiment and
judge AnyTouch by downstream contrastive/action metrics, not by its pre-training pedigree.
The repository code is MIT licensed; the checkpoint page does not state a separate weight
license, so commercial use should confirm the weight terms with the authors.

Enable it on the cluster launcher with:

```bash
bash train_ace.sh \
  --job_name anytouch_contrast \
  --tactile_backbone anytouch \
  --anytouch_checkpoint /mnt/wangxiaofa/pt_weights/anytouch_encoder.pth
```

The local launcher already forwards arbitrary policy overrides:

```bash
bash train_ace_local.sh \
  --policy.tactile_backbone=anytouch \
  --policy.anytouch_checkpoint=/Data/lzl/huggingface/anytouch_encoder.pth
```

### Measured cost

RTX A6000, bf16, real `debug_research_data`, batch 128, four tactile frames, two dynamic
windows per live pad:

| backbone | model params | trainable | representative step | tactile tower | CUDA peak |
|---|---:|---:|---:|---:|---:|
| ResNet-18 | 774.4M | 406.5M | 1.01 s | 0.074 s | 10.83 GiB |
| AnyTouch | 1058.6M | 385.4M | 1.62 s | 0.728 s | 9.66 GiB |

The matching uncontended runs spent ~0.33 s in the unchanged perception encoder. AnyTouch
processed about 176 live pads per timed step, or 352 three-frame windows. It is roughly
60% slower end to end at this tactile density, but the frozen/chunked implementation keeps
memory below the ResNet run and satisfies the single-GPU batch-128 requirement. These are
shared-machine measurements; unrelated GPU load can inflate absolute time, so use the
provided profiler for cluster-specific capacity planning:

```bash
python scripts/profile_contrastive_step.py \
  --mix debug_research_data --batch_size 128 --steps 6 \
  --tactile_backbone anytouch
```

This is a useful high-capacity reference encoder, not the default. If its online cost is too
high, first test `tactile_tokens_per_pad=1`; for large-scale pre-training, offline caching
or distillation into the existing smaller tower is more economical than unfreezing ViT-L.

---

## 6. Verification

| check | result |
|---|---|
| All 12 `backbone × target × K` combinations build, forward, backward | pass (`scripts/smoke_cosmos3_contrast.py`) |
| `dinov3/vision/K=1` reproduces the pre-change parameter split | 774.4M / 406.5M, exact |
| `K=1` similarity is bit-identical to the old formula | max abs diff 0.0 |
| Degenerate `K=4` reduces to `K=1` | max abs diff 5.6e-08 |
| End-to-end `train_ace_local.sh`, `cosmos3` + `vae` + `K=4` | 12 steps + eval + checkpoint, clean exit |
| Single-GPU batch size at the heaviest configuration | **256** (requirement: ≥128) |

Qwen3-VL adds its own suite (`scripts/check_qwen3vl_vision.py`), since that tower is driven
through a hand-written interface where a mistake would degrade features without ever
raising:

| check | result |
|---|---|
| Model normalisation vs the checkpoint's preprocessor config | 0.5/0.5, match |
| Our patchification vs Qwen's own image processor | max abs diff 1.2e-07 |
| Merge-block → row-major reorder | max abs diff 0.0 |
| Batch and order independence (no cross-image attention leak) | 5.9e-05 / 0.0 |
| Perturbing image cell *i* moves token *i* the most (4 cells probed) | pass |
| Optimized path vs stock, fp32 and bf16 | rel diff 0.0 (both) |
| Cached call is stable across repeats | rel diff 0.0 |
| `pos_embed` still receives gradient when unfrozen | pass |
| All 4 `target × K` combinations build, forward, backward | pass |
| End-to-end `train_ace_local.sh`, `qwen3vl` | 407.0M / 995.5M, batch **256**, clean exit |
| AnyTouch full checkpoint -> retained encoder strict load | all 391 CLIP keys matched |
| AnyTouch four frames -> two dynamic features | `(N,4,3,224,224) -> (N,2,768)`, pass |
| Window ordering | changing frame 3 leaves token 0 unchanged and changes token 1 |
| Frozen/trainable gradient split, with and without tactile rows | backbone none; adapter/projection present |
| Full DINOv3 RoboContrast forward/backward with AnyTouch | pass |
| Real-video `debug_research_data`, single-GPU batch 128 | pass, 9.66 GiB allocated peak |
| `train_ace.sh`, one-device ZeRO-2, real video, batch 128 | pass; 110 live pads, 3.54 s update |

---

## 7. Known pre-existing issues (not introduced here, not fixed here)

- **`MultiHeadAttention.norm_kv` is dead in self-attention.** The self-attention path uses
  `norm_q` for both query and key/value, so `norm_kv` never receives a gradient: 2
  parameters × 28 self-attention modules = 56 parameters, ~57k values. Harmless, but it is
  why the smoke test reports `grads 633/689`. Deleting it would change checkpoint keys and
  break loading of every existing checkpoint, so it is left alone deliberately.
- **`contrastive_eval` draws its batches i.i.d.** from `dataset.sample_weights`, while
  training uses `ContrastiveBatchSampler` with `same_dataset_frac=0.75`. The eval's
  negatives are therefore mostly cross-dataset, so it partly rewards scene recognition and
  cannot cleanly separate action alignment from dataset identity. Fix this before running
  any perception-side A/B on eval metrics.
