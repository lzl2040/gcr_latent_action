# Native Cosmos3 Nano on this LeRobot version

This is the official native policy, not the existing Qwen3-VL MoT policy with a
Cosmos video encoder. The port stays on `qwen3vl_mot_pretrained_generation`.

## Provenance and environment

- Upstream: `huggingface/lerobot@81b8c5bfdd5591ec59e512cfc7ac6db8151c70ee`.
- Official EMA checkpoint: `lerobot/Cosmos3-Nano-Policy-LIBERO`,
  revision `29563132da0815a4e034435ca55aaaf061ecb5ba`.
- `model.safetensors`: 32,944,302,256 bytes,
  SHA256 `27d5ff955f156d48a97a4b9fe8e1124a49815054d384254221d896207fce336a`.
- Local copy: `/Data/lzl/huggingface/Cosmos3-Nano-Policy-LIBERO`.
- 16,454,213,580 total parameters; 6,982,401,216 trainable by default.
- Original Apache-2.0 code notices are retained; weights retain their upstream
  OpenMDW-1.1 license and NOTICE.

**Do not upgrade the shared `lerobot_v2` or MoT evaluation environment.**
Cosmos3 requires Transformers 5.22; MoT requires Transformers 4.57.
The isolated environment is `/Data/lzl/venvs/lerobot_cosmos3`, built with
`--system-site-packages` from `libero_lerobot_qwen3vl`.
It uses Python 3.12, PyTorch 2.9.1+cu130, Transformers 5.22.0, diffusers 0.38.0,
accelerate 1.15.0, TorchCodec 0.8.1 and bitsandbytes 0.50.2.
Install the `cosmos3` extra only in a separate environment, never together with
`contrast` or `pi0`.

TorchCodec also needs compatible FFmpeg shared libraries. On this machine the
training launcher can use the existing FFmpeg 7 libraries, read-only:

```bash
export PYTHON_BIN=/Data/lzl/venvs/lerobot_cosmos3/bin/python
export FFMPEG_LIB_DIR=/home/v-wangxiaofa/anaconda3/envs/lerobot_v2/lib
```

Run commands from the repository root. No editable installation is needed for
the module-based entry points.

## Fine-tuning with the existing FSDP trainer

```bash
CUDA_VISIBLE_DEVICES=1,2,3 \
MODEL_DIR=/Data/lzl/huggingface/Cosmos3-Nano-Policy-LIBERO \
OUTPUT_DIR=/Data/lzl/qwen3vl_mot_weights/cosmos3_libero \
DATA_MIX=libero BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=4 \
bash train_cosmos3_fsdp.sh
```

`MODEL_DIR` must contain the native config, safetensors, pre/postprocessor
statistics and tokenizer. The initial weights can instead be a Cosmos3 base
checkpoint; use `LOAD_ACTION_MODULES=false` when initializing new action modules.
Each newly saved run turns that flag back to true.

Resume the same run with the same batch/world/accumulation geometry:

```bash
CUDA_VISIBLE_DEVICES=1,2,3 \
OUTPUT_DIR=/Data/lzl/qwen3vl_mot_weights/cosmos3_libero \
DATA_MIX=libero BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=4 \
WEIGHT_RESUME=true STEPS=10000 bash train_cosmos3_fsdp.sh
```

The saved policy configuration is restored and explicit architectural changes
are rejected. Complete DCP model shards, rank-local optimizer state, scheduler,
RNG and sampler position are restored. An action width changed during a fresh
run can still resume: the shared backbone bootstraps first, then strict DCP
restoration replaces every action tensor.

The adapter reads 17 real consecutive frames from both cameras and 16 actions,
excludes padded episode tails, and recomputes native action quantiles over the
selected mixture. Our old 10D LIBERO actions encode controls as *Euler XYZ with
interleaved rotation columns*. The adapter reverses that encoding and produces
the official *axis-angle with concatenated rotation columns* representation.
Both are 10D but **are not interchangeable**. Gripper stays continuous.
`TRAINING_ACTION_LAYOUT=cosmos3_native` is only for already-native source data.

This is compatibility with our trainer, **not an exact upstream training recipe
reproduction**: FULL_SHARD/FSDP1, BF16 parameters and 8-bit AdamW; native timestep
projections remain FP32, with FP32 global gradient-norm reduction. The original
FP32-master/FSDP2/EMA training recipe is not implemented. MoT-specific generation
validation and FP8 conversion are disabled. The original MoT path is unchanged.

## Official-weight LIBERO evaluation

```bash
CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" -m lerobot.evaluation.eval_cosmos3_libero \
  --model-dir /Data/lzl/huggingface/Cosmos3-Nano-Policy-LIBERO \
  --task-suite libero_spatial --trials-per-task 20 --batch-size 10 \
  --render-gpu-device-id 1 \
  --output-dir /Data/lzl/cosmos3_evaluation/official_spatial
```

Repeat for `libero_object`, `libero_goal` and `libero_10`: 40 tasks x 20 trials =
800 episodes. Outputs must be new directories; existing results are never
deleted. Use `--task-ids 0 --trials-per-task 2 --batch-size 2` for a real full-horizon
smoke test, not a benchmark score.

Protocol: original fixed LIBERO initial states, task order 0, first 20 initial
states per task, seed 1000, 20Hz, 10 settling steps, dual 256 RGB rotated 180
degrees, native quantile normalization, 16 executed actions per chunk, 30 UniPC
steps, sampling shift 5, guidance 1. Spatial/Object/Goal/10 horizons are
220/280/300/520. Native rotations use SVD projection then axis-angle; gripper is
`clamp(1 - 2*g, -1, 1)`, not the MoT adapter's sign threshold.

Batched independent environments share each policy call. The active batch
shrinks after episodes terminate; the continuous sampling RNG is therefore
**not trajectory-identical to single-environment baseline evaluations**.
Native future video latents are sampled, but not decoded for prediction videos.
Every episode saves actual controls and native sampled chunks; failures also
save observed video. `results.json` is updated atomically after completed waves.
An optional `--checkpoint-receipt FILE.json` records an earlier full download
hash audit without rereading the 33GB file.

To evaluate our fine-tuned DCP, point `--model-dir` at the training run root
(saved config and processors) and add
`--dcp-checkpoint RUN/checkpoint_XXXXXXXX`.

## Validation

Real Object data: 454 episodes, 66,984 source frames, 59,720 unpadded windows.
The final valid anchors decode 17 camera frames and 16 finite native actions,
without read fallback. A real 3-A6000 smoke run completed two optimizer steps
and wrote all 1,356 model tensors plus optimizer/trainer state. Peak training
allocation was about 32GiB per card at batch 1, accumulation 1.
The same run restored step 2, optimizer/scheduler/RNG and batch position 2, then
completed step 3 and published another full checkpoint. Reading the 30.68GiB
DCP shards took about 445 seconds on this machine's disk; resume startup is
I/O-heavy even though steady-state updates took about 5 seconds.
The official weights also passed a real Spatial task-0 smoke: fixed initial
states 0/1 both succeeded, in 75/77 control steps, using the full 220-step
horizon and all 30 sampling steps. This **2/2 acceptance check is not a benchmark
score** and is excluded from the separate 800-episode evaluation.

```bash
/Data/lzl/venvs/lerobot_cosmos3/bin/python -m pytest -q \
  tests/test_cosmos3_policy.py tests/test_cosmos3_dataset.py \
  tests/test_cosmos3_libero_eval.py tests/test_fsdp_training.py
```

Synthetic-image inference is only a loader/finite-output check, not a real
rollout or success-rate measurement. Benchmark results must come from complete
`results.json` episode records; no upstream reported score is a local result.
