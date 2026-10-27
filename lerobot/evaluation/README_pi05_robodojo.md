# PyTorch PI05 on RoboDojo

This branch targets the repository's PyTorch PI05 implementation with the
same three RGB cameras, absolute 20D measured state, and 20D **delta**
end-effector targets used by the Qwen RoboDojo experiment. It is not the
official RoboDojo 14D joint-action/JAX recipe.

## Base model and tokenizer

The official PyTorch base is
[`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base), pinned here to
revision `22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0`.
Its raw `model.safetensors` contains 812 tensors, without the policy's
`model.` prefix or dataset normalization buffers. The tied PaliGemma
`lm_head.weight` is the token embedding; loading must restore that alias.
All model tensor names and shapes are checked, and incomplete/incompatible
base weights fail rather than continuing with random model parameters.
Dataset normalization buffers are retained instead of loading unrelated
pretraining statistics.

The official preprocessor specifies the
`google/paligemma-3b-pt-224` tokenizer. Its Hugging Face repository can require
access approval. Alternatively, pass the official public SentencePiece
vocabulary explicitly as `tokenizer_path`:
`https://storage.googleapis.com/big_vision/paligemma_tokenizer.model`.
The verified vocabulary has 257152 entries and SHA256
`8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`.
Do not substitute a Qwen tokenizer.

Install the extra tokenizer dependency with `pip install -e '.[pi05]'`.
The existing `wam` environment can be used without installing a second
upstream LeRobot package over this checkout.

## Correctness changes

PI05 now uses its configured camera list instead of only `primary`, rejects
missing real cameras, uses the configured token limit, and shares the same
normalized-state prompt between training and action prediction. The
PaliGemma prefix uses the local compatible Gemma implementation; both
training and sampling unpack the four prefix outputs correctly.

The shared `QUANTILES` normalizer previously applied the inverse transform.
It now maps `q01` to -1 and `q99` to +1, with the inverse restoring original
values. Invalid statistics fail explicitly. `MEAN_STD` and `MIN_MAX` are
unchanged. This is a behavior correction: old runs trained using the
incorrect quantile transform are not numerically interchangeable with new
runs. RoboDojo Qwen's mean/std normalization is unaffected.

For rollout deployment, delta predictions still require cumulative world
translations, left-multiplied rotation increments, clipped gripper
increments, and reanchoring at the next measured state. The Qwen-specific
online evaluator does not automatically become a PI05 deployment adapter.

## Fine-tuning entry point

Use the isolated `pi05_robodojo_finetune` worktree. The single-device entry
point reads the existing v3 dataset without rewriting its columns or stats:

```bash
conda activate wam
cd /home/jianf/lzl_project/gcr_latent_action_pi05_robodojo
DATA=/data2/lizuolei/data/lerobot_data/simulation_data/RoboDojo/data/stack_bowls_delta
BASE=/home/jianf/lzl_project/gcr_latent_action/outputs/evaluation/robodojo/runtime_assets/pretrained/pi05_base

# CPU-only metadata, three-camera decoding, and batched feature validation.
python -m lerobot.scripts.train_pi05_robodojo \
  --dataset-root "$DATA" --validate-data-only \
  --output-dir outputs/pi05_robodojo/data_check

# A bounded training smoke check; not a full fine-tuning experiment.
python -m lerobot.scripts.train_pi05_robodojo \
  --dataset-root "$DATA" \
  --pretrained-path "$BASE" \
  --tokenizer-path "$BASE/paligemma_tokenizer.model" \
  --device cuda:0 --steps 2 --batch-size 1 --chunk-size 32 \
  --output-dir outputs/pi05_robodojo/smoke
```

Output directories must be new. For a full run, choose the intended step
budget and batch size explicitly, e.g. `--steps 30000 --batch-size 1`.
This example is not a recommendation that 30000 steps is sufficient or
optimal. Default LR is `2.5e-5`, warmup is 1000 updates (shortened for a
shorter run), followed immediately by cosine decay to `2.5e-6` at the
requested update budget. There is no implicit 20000-step plateau.
The vision encoder is frozen; the language backbone and action expert are
trainable. CUDA uses BF16 autocast and gradient checkpointing. Images are
aspect-preserving resized/padded to 224x224; stochastic image augmentation
is not enabled by this entry point.

The loader maps `observation.ee_ort6d_pos` to the 20D state and
`delta_action.ee_ort6d_pos` to action targets, **not** the original 16D
`action`. Future actions start at the current row and never cross episode
boundaries; repeated tail rows are masked out of the training loss.
Camera order is head (`cam_high`), left wrist, right wrist. Numeric vectors
are quantile-normalized before padding to the base model's 32 dimensions;
only the 20 real action dimensions contribute to the loss. State prompts
contain the padded 32D state. The action horizon is 32 and deployment
configuration executes up to 10 actions per replan; PI05 retains its
official 10 inference denoising steps.

`data_contract.json` records feature semantics, normalization, input
shapes, source/statistics/tokenizer hashes, launch arguments and Git
revision. `training.jsonl` contains loss, gradient norm and LR per update.
Each checkpoint includes policy configuration, full model/normalizer
weights and `training_state.pt` (optimizer and scheduler). A completed run
writes `summary.json`; the current entry point starts fresh and does not
implement resume. A successful smoke check is not evidence of RoboDojo
task success; closed-loop PI05 deployment must be connected and evaluated
separately.

## FSDP launcher

`train_pi05_robodojo_fsdp.sh` launches the separate
`lerobot.scripts.train_pi05_robodojo_fsdp` entry through `torchrun`.
It supports single-node or multi-node CUDA training, distributed data
sampling, gradient accumulation, rank-zero JSONL logging, and checkpoint
resume. It does not change the existing single-device entry.

```bash
conda activate wam
cd /home/jianf/lzl_project/gcr_latent_action_pi05_robodojo

# Validate paths and print the command without starting training.
DRY_RUN=true bash train_pi05_robodojo_fsdp.sh

# Two GPUs, one sample per rank, effective batch size 8.
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 \
  BATCH_SIZE=1 GRAD_ACCUMULATION_STEPS=4 STEPS=30000 \
  JOB_NAME=pi05_stack_bowls_delta_v1 bash train_pi05_robodojo_fsdp.sh

# Resume the same job, with the same schedule and training geometry.
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 \
  BATCH_SIZE=1 GRAD_ACCUMULATION_STEPS=4 STEPS=30000 \
  JOB_NAME=pi05_stack_bowls_delta_v1 WEIGHT_RESUME=true \
  bash train_pi05_robodojo_fsdp.sh
```

Default paths point at this machine's verified delta dataset, official
base cache, and SentencePiece tokenizer. Override `DATASET_ROOT`,
`PRETRAINED_PATH`, and `TOKENIZER_PATH` on other machines. Other controls
include `OUTPUT_DIR` (or `OUTPUT_ROOT`/`JOB_NAME`), `LOG_DIR`, `PYTHON_BIN`,
`LR`, `WARMUP_STEPS`, `SAVE_FREQ`, `CHUNK_SIZE`, `NUM_WORKERS`, `SEED`,
and `VIDEO_BACKEND`. Defaults are 30000 optimizer updates, batch size 1 per
rank, accumulation 1, LR `2.5e-5`, warmup 1000, and checkpoint frequency
1000. Effective batch size is
`NNODES * NPROC_PER_NODE * BATCH_SIZE * GRAD_ACCUMULATION_STEPS`.
Check GPU headroom before launching; no long run is started by adding this
script. `CHECK_PATHS=false` is intended for command inspection, not as a
replacement for valid assets.

For two nodes, run this on both machines with the same shared output
directory, base/tokenizer/data content, and arguments, changing only
`NODE_RANK=0` to `NODE_RANK=1`:

```bash
NNODES=2 NODE_RANK=0 NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES=0,1 \
  MASTER_ADDR=training-node-0 MASTER_PORT=29500 \
  OUTPUT_DIR=/shared/experiments/pi05_stack_bowls_delta_v1 \
  bash train_pi05_robodojo_fsdp.sh
```

The implementation uses a **root FULL_SHARD unit**, FP32 master parameters
and optimizer states, BF16 autocast/parameter all-gathers, and activation
checkpointing. Dataset quantile buffers stay FP32. PI05 jointly computes
prefix/expert attention by accessing child-layer weights directly; generic
per-Gemma-layer wrapping would bypass FSDP forward hooks. A root unit
avoids that error, but gathers the full model during forward/backward:
this is not layerwise FSDP's best possible memory scaling. FP8 and
bitsandbytes are not enabled for this PI05 route.

Periodic saves use the repository's existing DCP format:
`checkpoint_XXXXXXXX/model/`, rank-local optimizer/RNG files, scheduler
state, metadata, and the atomically published `latest_checkpoint` pointer.
Resume requires the same world size, base/data/tokenizer, batch geometry,
seed, learning rate, and total step budget. `STEPS` is the total target,
not the number of additional updates. `MAX_UPDATES_THIS_RUN=1` can stop
after one update and save progress without changing the configured
schedule, for a bounded save/resume smoke check. Interrupted/incomplete
checkpoint directories are never overwritten.

`EXPORT_FINAL=true` (default) also exports the last update of this
invocation as `stepN/{config.json,model.safetensors}`, including dataset
normalization buffers, for ordinary `PI05Policy.from_pretrained` loading.
`EXPORT_FINAL=false` skips the full-model gather/export, not the DCP save.
Exports need sufficient rank-zero host RAM and disk space. Console logs
are written to `LOG_DIR` with per-node names, `training.jsonl` contains
globally averaged losses, and `launch_stepN.json` records provenance and
training geometry. `summary.json` distinguishes a bounded stop from
reaching the total training budget.

## Completed single-device smoke check

The official checkpoint was downloaded into the `BASE` directory above.
Its `model.safetensors` SHA256 is
`0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f`.
The 44774-frame, 100-episode, 25 FPS delta dataset passed three-camera
decoding and feature checks. At the last row of episode 0, the first target
is valid and the remaining 31 future targets are correctly masked, without
crossing into episode 1.

A real official-base run completed two single-sample GPU updates with
finite losses `0.4177229` and `0.2684192`, finite pre-clipping gradient norms
`24.3978` and `9.5307`, and learning rates `1.25e-5` and `2.5e-5`.
These are different batches/noise samples, not a convergence measurement.
The optimizer is standard PyTorch AdamW, not the repository-wide `adamw`
alias that selects bitsandbytes. Peak PyTorch allocated memory was
32732044288 bytes (30.48 GiB) for batch size 1; larger batches and concurrent
jobs require additional headroom.

The resulting `step2` checkpoint reloads with `strict=True`. State/action
quantile buffers exactly match the dataset, and standalone prediction from
a real three-camera observation produces a finite `(1, 32, 20)` action
chunk with 10 denoising steps. CUDA autocast is handled inside the policy,
so callers do not have to supply an external mixed-precision context.

Evidence and the checkpoint are retained at:

```text
/home/jianf/lzl_project/gcr_latent_action/outputs/evaluation/robodojo/pi05_delta_official_smoke_20261027_v2/
  data_contract.json
  training.jsonl
  summary.json
  reload_validation.json
  step2/config.json
  step2/model.safetensors
  step2/training_state.pt
```

The first attempt exposed the missing bitsandbytes dependency of the old
optimizer alias; it did not perform any updates. The subsequent reload
check exposed a BF16/FP32 standalone-inference mismatch, which was fixed
before the successful reload/prediction check. No long training job or
closed-loop PI05 RoboDojo success-rate evaluation has been started.
