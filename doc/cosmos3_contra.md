# `cosmos3_contra`: module reference

What this branch adds to the perception ↔ physical contrastive model, what each module
costs, and which parts are actually trained.

Three knobs were added, each defaulting to the previous behaviour, so an existing config
run on this branch produces the same model it did before:

| config | values | default |
|---|---|---|
| `vision_backbone` | `dinov3`, `cosmos3` | `dinov3` |
| `perception_recon_target` | `vision`, `vae` | `vision` |
| `num_cls_tokens` | `1`, or any `K ≤ num_change_queries` | `1` |

Supporting knobs: `cosmos3_dir` (weights, default `/Data/lzl/huggingface/Cosmos3-Edge`)
and `vae_repeat_frames` (default `1`; see "Repeating frames is a no-op" below).

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

### `vision_backbone=cosmos3`

Only the perception branch changes; the physical branch is identical in every row.

| configuration | perception | physical | total | trainable |
|---|---:|---:|---:|---:|
| `dinov3` + `vision` | 573.5M | 200.9M | **774.4M** | **406.5M** |
| `cosmos3` + `vision` | 901.3M | 200.9M | **1102.2M** | **407.3M** |
| `cosmos3` + `vae` | 1049.8M | 200.9M | **1250.7M** | **406.1M** |

The deltas are entirely frozen: `vision_backbone` 85.7M → 412.6M, plus a 149.6M VAE
encoder for the `vae` target. Trainable capacity stays at ~406M in every combination and
is split almost exactly in half between the two branches (205M / 201M), which is the
budget the design targets. Total parameters exceed the original 500–800M envelope, but
those are frozen feature extractors, not capacity the model is free to use.

**The largest frozen module is the SigLIP2 text tower at 282.3M** — over three times
DINOv3's vision tower. It is kept because its text embedding is already aligned to a
visual space, which is what makes "the instruction selects which change matters" work.
Cosmos3's own text tower is a 2.2B LLM in `transformer/`; it was measured to add ~0.3s to
a 1.04s step, against the 0.013s the SigLIP2 tower currently costs, so it is not used and
its weights are not downloaded.

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

## 5. Verification

| check | result |
|---|---|
| All 8 `backbone × target × K` combinations build, forward, backward | pass (`scripts/smoke_cosmos3_contrast.py`) |
| `dinov3/vision/K=1` reproduces the pre-change parameter split | 774.4M / 406.5M, exact |
| `K=1` similarity is bit-identical to the old formula | max abs diff 0.0 |
| Degenerate `K=4` reduces to `K=1` | max abs diff 5.6e-08 |
| End-to-end `train_ace_local.sh`, `cosmos3` + `vae` + `K=4` | 12 steps + eval + checkpoint, clean exit |
| Single-GPU batch size at the heaviest configuration | **256** (requirement: ≥128) |

---

## 6. Known pre-existing issues (not introduced here, not fixed here)

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
