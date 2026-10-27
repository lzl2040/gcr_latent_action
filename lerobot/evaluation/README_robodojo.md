# Qwen3-VL MoT on RoboDojo

## Runtime and checkpoint contract

Use branch `qwen3vl_mot_pretrained_generation`. The policy runs in `wam`;
Isaac Sim runs in a separate simulator environment. Do not install Isaac Sim
into the policy environment or change the shared RoboDojo/Pi_05 installation.

This integration targets RoboDojo's XPolicyLab WebSocket interface
(`update_obs`, `get_action`, `reset`, and their batched variants), not a
LIBERO/Gym interface. Upstream reference:
<https://github.com/robodojo-benchmark/RoboDojo>.

The standard RoboDojo loader explicitly restores BF16, including the
precision arguments required by the shared LIBERO policy loader. The
optional `diagnose --diagnostic-fp8` probe is separate from native rollout
precision.

Local inputs:

| Input | Path |
|---|---|
| RoboDojo | `/data2/guoqi/robodojo_pi05_repro` |
| Simulation assets | `/data2/guoqi/robodojo_pi05_repro/Assets` |
| Experiment | `/data2/lizuolei/weights/qwen3vl_mot_weights/1016_ft_on_robodojo` |
| DCP checkpoint | Experiment's `step4k/model` directory |
| Qwen backbone | `/home/jianf/blob_data/pt_weights/Qwen3-VL-4B-Instruct` |
| Cosmos VAE | `/home/jianf/blob_data/pt_weights/Cosmos3-Edge` |
| Stage 1 text backbone | `/home/jianf/blob_data/pt_weights/siglip2-base-patch16-224` |
| Training metadata | `/home/jianf/blob_data/robot_dataset/lerobot-format-v30/RoboDojo/data/RoboDojo_ee_lerobot_v30_video/meta` |

The checkpoint uses Nano generation, embedding-space physical flow, three
cameras and joint video/action prediction. Native actions are **absolute
world-frame end-effector poses**, not joint targets or pose deltas. Both arms
use xyz + interleaved first-two rotation-matrix columns + gripper in canonical
slots 0:20; slots 20:40 are masked. RoboDojo native quaternions are wxyz.
Camera roles are primary=cam_head (named cam_high in converted training metadata), secondary=cam_left_wrist,
wrist=cam_right_wrist. Source rate is 25 Hz.

Normalization must use `observation.ee_ort6d_pos` and `action.ee_ort6d_pos`
statistics from the training dataset, with the training standard-deviation
floor of 1e-3. The user-specified dataset at
`/data2/lizuolei/data/lerobot_data/simulation_data/RoboDojo/data/RoboDojo_ee_lerobot_v30_video`
was rechecked on 2026-10-19 and currently contains both converted fields
and their statistics. The earlier statement that this local copy lacked
the statistics is no longer accurate. Both checked `info.json` files
declare the original 16D quaternion features and the converted 20D ort6d
features.

### User training-dataset comparison (2026-10-19)

The user-specified dataset above was compared directly with the default
dataset under `/home/jianf/blob_data/robot_dataset/lerobot-format-v30/`.
All three data parquets were checked, covering **1,856,102 rows**:

- `observation.ee_ort6d_pos` and `action.ee_ort6d_pos` are exactly equal,
  with maximum absolute difference 0. Both are float32, 20-dimensional:
  left xyz/ort6d/gripper, then right xyz/ort6d/gripper.
- Original `observation.state` and `action`, plus row, episode, frame and
  task indices and timestamps, are also exactly equal.
- Parsed `meta/info.json` and `meta/stats.json` are exactly equal,
  including converted-field mean/std. The adapter's effective
  normalization tensors are therefore identical.
- Reconstructing ort6d from original wxyz quaternions across all rows
  confirms the adapter's interleaved first-two-column convention:
  `[R00,R01,R10,R11,R20,R21]`. Maximum conversion errors are 7.01e-7
  for state and 3.03e-7 for action, consistent with float32 rounding.

The numeric report is retained at
`outputs/evaluation/robodojo/dataset_comparison_20261019/comparison.json`.
This comparison covers physical fields and normalization, not video-file
contents. It establishes no mismatch in these fields between the two
dataset paths, but does not establish the root cause of the failed rollout.
To explicitly use the supplied path, append this option to any evaluation
command:

```bash
--dataset-root /data2/lizuolei/data/lerobot_data/simulation_data/RoboDojo/data/RoboDojo_ee_lerobot_v30_video
```

## Policy environment

The existing `wam` environment is Python 3.12. From the repository root:

```bash
conda activate wam
python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r lerobot/evaluation/requirements_robodojo_qwen3vl.txt
```

Use the repository directly from its root; installing all LeRobot training
and hardware extras is unnecessary for this inference service.

The local simulator environment is `RoboDojo6` (Isaac Sim 6.0.0.1 with its
matching IsaacLab). This is a local compatibility setup, not the public
Isaac Sim 5.1 benchmark runtime. Results must identify this difference.

## Recorded-observation inference

```bash
conda activate wam
CUDA_VISIBLE_DEVICES=0 python -m lerobot.evaluation.eval_qwen3vl_mot_robodojo offline \
  --repeat 1 \
  --output-dir outputs/evaluation/robodojo/offline_step4k
```

This loads all eight DCP model shards with strict key/shape validation, then
predicts from the real `stack_bowls` HDF5 observation. No Stage 1 checkpoint,
optimizer or distributed launch is needed. The default saved 20 flow steps
and all three saved camera views are preserved. `sample_canonical_action`
automatically uses the trained joint-prediction task. This command does
**not** simulate the environment or measure success rate.

Use `--episode PATH --frame N` for another recorded observation and
`--inference-steps N` only for an explicitly labeled ablation.
`--replan-steps` defaults to 10 and cannot exceed the saved `n_action_steps`
(32 here). The full 32-step prediction is saved; only the first N actions
are returned to the simulator.

Every invocation requires a **new output directory**. Artifacts:
`settings.json`, `evaluation.log`, `summary.json`, and
`chunks/chunk_XXXXXX_env_YYY.npz` with normalized 40D predictions plus native
left/right wxyz poses and grippers. `chunks/inference.jsonl` records timing,
environment IDs, reset index and instruction for each prediction.

### Save matched future frames

Add `--save-predicted-video` to `offline`, `serve` or `eval`. For this
checkpoint it calls `generate_validation(seed=None)` once, using the same
joint sample for both executed actions and predicted video/state. It does
not run a second video prediction or reseed each replan. A checkpoint
with an enabled action-only sampler is rejected rather than silently
changing its action semantics. Decoder initialization/export preserves the
policy's torch RNG state.

In closed-loop mode, each replan writes:

```text
policy/predictions/reset_0000/env_000/replan_000000_step_0000/
  conditioning_primary.png       # actual resized model input (not generated)
  conditioning_secondary.png
  conditioning_wrist.png
  primary.mp4                   # same-sample prediction; analogous files for other views
  primary_frames/frame_000.png   # VAE reconstruction of the current conditioning frame
  primary_frames/frame_001.png   # first predicted future frame
  ... frame_008.png
  primary_filmstrip.png          # all 9 decoded frames, left to right
  action_chunk.npy               # all 32 native actions, shape (32,16)
  metadata.json
```

The 16 native action columns are left xyz/wxyz/gripper, then right
xyz/wxyz/gripper. Only the first `--replan-steps` actions are executed.
`metadata.json` gives timestamps and absolute artifact paths; the
corresponding `chunks/inference.jsonl` entry links this metadata to the
action NPZ and executed prefix. NPZs also include the actual canonical and
normalized current state, and normalized predicted future states when
export is enabled. Offline artifacts omit the `policy/` prefix.

There are **9 sparse frames including the current frame**, covering
31/25 = 1.24 seconds. Playback uses about 6.45 FPS, not the actual rollout's
25 FPS. Rounded training timestamps are recorded in metadata; compare
PNG frames at those timestamps, not equally numbered rollout frames.
With the default 10-action replan, only the first 0.4 seconds share the
original planned action prefix. Later actual observations incorporate
subsequent replans, whereas this prediction still describes the original
32-action sample; divergence beyond that point is not a controlled
open-loop video error measurement.
Export loads a separate Cosmos VAE decoder and adds memory/time overhead;
inference and export timing are recorded separately.

## Closed-loop evaluation

After reviewing and accepting the [Isaac Sim license terms](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html):

```bash
conda activate wam
export OMNI_KIT_ACCEPT_EULA=YES
python -m lerobot.evaluation.eval_qwen3vl_mot_robodojo eval \
  --task stack_bowls --eval-num 1 --num-envs 1 \
  --gpu 0 --sim-gpu 0 --seed 0 \
  --output-dir outputs/evaluation/robodojo/stack_bowls_step4k_smoke
```

`--gpu` and `--sim-gpu` are physical GPU indices; `--device` is the logical
policy device inside `CUDA_VISIBLE_DEVICES` (normally `cuda:0`).
`--sim-python` selects the simulator interpreter; its default is
`/home/jianf/miniforge3/envs/RoboDojo6/bin/python` on this machine.
The simulator interpreter's `bin/` directory is prepended to its child
process `PATH`, so native video recording finds `RoboDojo6`'s `ffmpeg`
even though the launcher itself runs in `wam`.
The launcher starts a policy server in `wam`, verifies its WebSocket
handshake, and starts RoboDojo's native evaluation client. Both child
process groups are stopped on completion, error, timeout or interruption.
Startup and simulation timeouts default to 1800 and 3600 seconds.
`--port` selects a free localhost policy port (default 19030); occupied
ports are rejected rather than attaching to someone else's server.

An isolated `simulator/` workspace links the supplied Assets and simulator
code, and registers only this policy's deployment module. Shared source,
assets and Pi_05 integrations are not edited. All simulator results and
videos stay under the requested output directory:

- `policy.log`, `simulator.log`, `kit.log`: component logs.
- `policy/chunks/`: predicted chunks and per-request timing.
- `simulator/eval_result/RoboDojo/<task>/Qwen3VLMoT/**/_result.json`:
  native success labels, partial scores and layout IDs.
- The native result directory also contains rollout MP4s and physics
  profiling artifacts.
- Top-level `summary.json` records native results only after episode counts,
  layout IDs and success rate agree. Interrupted/failed runs never report
  evaluation completion.

The deployment loop obtains observations at every control step to preserve
video sampling inside an action chunk. Parallel environments are supported
with `--num-envs N`; policy inference is sequential per active environment
to bound multi-view model memory, and returned chunks preserve requested
environment order. No task state or reward enters the policy input.

For native per-task evaluation use `--eval-num native` (usually 50, with
task-specific overrides). Run each desired task separately, keeping its
native horizon and layouts, and choose a new output directory each time.
The default one-episode invocation is only a smoke test, **not** an
aggregate RoboDojo benchmark score. The runner fails fast and rejects
skipped/replaced or duplicate layouts. No event-replan skills, physics
overrides or success-seeking retries are enabled.

## Adapter regression tests

```bash
conda activate wam
python -m pytest --noconftest -q \
  tests/test_robodojo_qwen3vl_evaluation.py \
  tests/test_libero_qwen3vl_evaluation.py
```

## Training/validation versus deployment diagnostics

The optional `diagnose` mode uses the actual `MultiModalContrastiveDataset`
loader and saved `train_config.json`, not a separate approximation of the
training preprocessing. Additional dependencies in `wam`:

```bash
conda install -n wam -c conda-forge 'ffmpeg>=7,<8'
python -m pip install 'polars>=1,<2' 'qwen-vl-utils>=0.0.14,<0.1' torchcodec==0.8.1
```

Run with the cached backbone options documented below, a new output
directory, and the checkpoint being investigated. Unless explicitly
overridden, `--training-config` selects `train_config.json` beside the
selected saved policy `config.json`, rather than the original experiment's
training configuration. The saved policy snapshots must agree exactly;
a mismatch is rejected before loading model weights. Runtime asset paths
and inference-step overrides should be supplied as CLI options, not by
editing the saved policy snapshot.

```bash
CUDA_VISIBLE_DEVICES=1 python -m lerobot.evaluation.eval_qwen3vl_mot_robodojo diagnose \
  --checkpoint outputs/evaluation/robodojo/runtime_assets/checkpoint/step8k \
  --training-config /data2/lizuolei/weights/qwen3vl_mot_weights/1016_ft_on_robodojo/train_config.json \
  --dataset-root /data2/lizuolei/data/lerobot_data/simulation_data/RoboDojo/data/RoboDojo_ee_lerobot_v30_video \
  --diagnostic-episodes 3000 3001 3002 --diagnostic-frames 0 64 \
  --output-dir outputs/evaluation/robodojo/step8k_training_diagnostics
```

It includes the two configured generation-validation samples and the
requested extra training episodes/frames. It compares masks, camera tensors,
instruction, sample rate and state normalization, then compares the actual
validation sampler and deployment sampler with the same current inputs and
sampling seed. Adapter state-roundtrip error is measured separately; exact
normalized current state is shared for sampler comparison.

`diagnostics.json` and per-sample `actions.npz`/`metrics.json` distinguish
free-running 20-step sampling, an explicitly labeled 50-step integration
ablation, and a hold-current-pose baseline. Physical xyz, angular and gripper
errors are reported separately from normalized MSE, including the executed
prefix. Fixed-noise probes at sigma 0.1, 0.5 and 0.9 measure the saved
training reconstruction/flow losses for joint prediction and inverse
dynamics. These probes use ground-truth-corrupted trajectories; inverse
dynamics also receives the ground-truth future video. They are not
free-running action errors or benchmark success rates.

The saved finetuning setup validates on **training-set diagnostic samples**,
not held-out episodes, and chooses only two episodes for the entire
RoboDojo mixture. The diagnostic reports their episode IDs/instructions.
By default, training FP8 transformations are not enabled in the BF16
restore; this numerical difference is recorded rather than assumed harmless.
For a paired precision ablation, install `torchao>=0.15,<0.16` and add
`--diagnostic-fp8` to reapply the saved native FP8 recipe and layer scope.
Use identical episodes, frames, seeds and flow steps for comparison.
On the current Torch 2.9.1/TorchAO 0.15 stack, this native FP8 probe failed
with `aten.reshape.default with axiswise scaling is not supported yet`;
even a matrix-input Linear smoke test hit the same error. No FP8 result
was obtained, no library monkey patch was retained, and the probe does
not silently fall back to BF16. A paired precision result requires a
compatible training-runtime stack; FP8/BF16 differences remain unresolved.

## Verified local setup

`wam` was installed and its dependency consistency check passed. Real `step4k`
inference restored 12.585B parameters and produced finite `(1,32,40)`
normalized predictions and native `(32,7)` poses / `(32,1)` grippers for both
arms. The first three-camera inference took 11.38 seconds on GPU 0.
Evidence: `outputs/evaluation/robodojo/offline_step4k_20261018/`.
This is recorded-observation evidence only, not a success-rate result.

The quaternion/ort6d mapping was additionally audited against 128 real
converted training parquet rows: maximum errors were 2.98e-7 for state,
1.49e-8 for action xyz and 1.20e-7 for action rotation matrices after
inverse normalization. The original 16D fields and converted 20D fields
agree with wxyz input and interleaved first-two matrix columns. Native
observations use environment-origin-relative, world-axis end-effector
poses (not robot-base poses); native IK accepts absolute targets in this
same convention. Camera roles, RGB decoding and deterministic resizing
were checked against the training/native interfaces. No coordinate or
normalization bug was established by these checks; they do **not** prove
correct model behavior or simulator/training image equivalence.

Source revisions inspected:
RoboDojo `726e9aabfaa642203722eb126f5eaf0f37f3e1ad`;
XPolicyLab `bb9a0b5f5136a74503b679af830bfd0a3a837d5c`.
The policy uses PyTorch 2.9.1+cu128; the separate simulator uses
Isaac Sim 6.0.0.1, IsaacLab 6.1.11 and PyTorch 2.11.0+cu128.

The initial mounted-asset/shared-disk load took over three hours. To avoid
repeating slow random reads, the required assets can be cached on the
repository's local NVMe (about 36 GB; no optimizer files):

```bash
CACHE="$PWD/outputs/evaluation/robodojo/runtime_assets"
mkdir -p "$CACHE/checkpoint" "$CACHE/pretrained/Cosmos3-Edge"
cp -a /data2/lizuolei/weights/qwen3vl_mot_weights/1016_ft_on_robodojo/config.json \
  "$CACHE/checkpoint/"
cp -a /data2/lizuolei/weights/qwen3vl_mot_weights/1016_ft_on_robodojo/step4k \
  "$CACHE/checkpoint/"
cp -a /home/jianf/blob_data/pt_weights/Qwen3-VL-4B-Instruct \
  /home/jianf/blob_data/pt_weights/siglip2-base-patch16-224 "$CACHE/pretrained/"
cp -a /home/jianf/blob_data/pt_weights/Cosmos3-Edge/vae \
  "$CACHE/pretrained/Cosmos3-Edge/"
```

Then append these options to `offline` or `eval`:

```bash
--checkpoint "$CACHE/checkpoint/step4k" \
--qwen3vl-dir "$CACHE/pretrained/Qwen3-VL-4B-Instruct" \
--cosmos3-dir "$CACHE/pretrained/Cosmos3-Edge" \
--stage1-text-dir "$CACHE/pretrained/siglip2-base-patch16-224"
```

Cached files are untracked evaluation artifacts; original model and
simulation assets remain untouched.

### Initial resource-blocked attempts (2026-10-18, before 22:11)

Local caching reduced model startup to roughly 25 seconds, including a
strict DCP restore. The real model WebSocket server and the native Isaac Sim
scene were both launched. The deployment keyword interface was corrected
to match RoboDojo's `TASK_ENV=...` calls, with regression coverage.

These initial attempts did not complete a closed-loop episode. A later
completed smoke evaluation is documented below.
During the smoke evaluation, unrelated jobs increased GPU memory use.
The same-GPU run failed in cuRobo warmup with CUDA OOM; a split-GPU attempt
then failed while moving the 12.585B model to GPU 0. At that point only
about 21 GiB was free on GPU 0 and 6 GiB on GPU 1. No other jobs were
stopped, and all evaluator-owned child processes were cleaned up.
The failed runs have `evaluation_complete: false`, not a fabricated 0%
success rate. Their logs and summaries are retained in:

```text
outputs/evaluation/robodojo/stack_bowls_step4k_smoke_20261018_v2/
outputs/evaluation/robodojo/stack_bowls_step4k_smoke_20261018_split_v2/
```

Reserve enough free GPU memory before running. Model parameters alone use
about 24 GiB, excluding inference workspace and the simulator. As practical
headroom, aim for at least 40 GiB free for a shared policy/simulator GPU,
or 30 GiB for the policy and 10 GiB for a separate simulator GPU; these are
recommendations, not validated minimums or reservations.

After GPU resources are available, run the same native one-episode check:

```bash
conda activate wam
CACHE="$PWD/outputs/evaluation/robodojo/runtime_assets"
OMNI_KIT_ACCEPT_EULA=YES python -m lerobot.evaluation.eval_qwen3vl_mot_robodojo eval \
  --task stack_bowls --eval-num 1 --num-envs 1 \
  --gpu 0 --sim-gpu 0 --seed 0 \
  --checkpoint "$CACHE/checkpoint/step4k" \
  --qwen3vl-dir "$CACHE/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$CACHE/pretrained/Cosmos3-Edge" \
  --stage1-text-dir "$CACHE/pretrained/siglip2-base-patch16-224" \
  --output-dir outputs/evaluation/robodojo/stack_bowls_step4k_smoke_ready
```

Choose a new output directory if this name already exists. A complete
smoke episode with a validated native `_result.json` is required before
attempting a full benchmark run.

### Completed closed-loop smoke evaluation (2026-10-18, 22:13-22:23)

The rerun completed after GPU memory became available. The first renewed
attempt reached video capture but exposed a simulator subprocess PATH
issue: `ffmpeg` was installed in `RoboDojo6`, while the subprocess inherited
`wam`'s PATH. The launcher now prepends the simulator's own `bin/` directory;
this fix and its regression test were committed separately as `6fded4d`.

| Setting / result | Value |
|---|---|
| Checkpoint | `1016_ft_on_robodojo/step4k`, strict restore from local cache |
| Task | `stack_bowls` |
| Seed / native layout ID | 0 / 0 |
| Episodes completed | 1 |
| Episode success | false (0/1) |
| Native partial score | 0.0 |
| Executed control steps | 800 (native horizon reached) |
| Flow integration steps | 20 (saved checkpoint setting) |
| Replan interval | 10 consecutive actions |
| Policy calls | 80 |
| Mean inference per chunk | 4.915 seconds |
| Total wall time, including startup | 584.58 seconds |
| Policy / rendering GPU | Physical GPU 0 |
| Simulator / physics | Isaac Sim 6.0.0.1 / native CPU PhysX configuration |

This was a **completed but unsuccessful** episode, not an infrastructure
failure or an incomplete run counted as a failure. Top-level
`summary.json` has `status: completed` and `evaluation_complete: true`.
The native result has one entry for layout 0, `success: false`,
`score: 0.0` and `eval_time: 1`. No unstable episodes were replaced.
One layout is not enough to estimate the model's overall success rate.
The Isaac Sim 6 compatibility runtime is not the official Isaac Sim 5.1
leaderboard runtime.

Artifacts are under:

```text
outputs/evaluation/robodojo/stack_bowls_step4k_smoke_20261018_2211_v2/
```

`summary.json` contains the exact commands and native result path.
`policy/chunks/` contains all 80 full prediction chunks; every saved
normalized/native value was checked to be finite.
The native result directory contains `_result.json`,
`_physics_profile.json` and three videos:

```text
episode_0000000_cam_head_fail.mp4
episode_0000000_cam_left_wrist_fail.mp4
episode_0000000_cam_right_wrist_fail.mp4
```

Each video is 640x480 at 25 FPS, with 801 frames / 32.04 seconds.
RoboDojo captures the terminal observation when the end flag changes, so
the final frame is included in addition to the 800 control-step
observations. Physics profiling independently records 800 executed steps,
one completed episode and zero unstable episodes. Evaluator-owned policy
and simulator processes exited and GPU memory was released.

### Matched future-frame rerun and inference audit (2026-10-18, 23:17-23:49)

The same checkpoint, task, seed/layout 0, 20 flow steps and 10-action
replan interval were rerun with `--save-predicted-video`. The native
episode completed all 800 control steps, again with success false and
score 0.0. All 80 joint samples were exported: **240 predicted MP4s and
2160 lossless PNG frames**, plus the actual conditioning images and
native action/state arrays. Every exported native action chunk was
checked against the corresponding adapter output; all saved numeric
outputs were finite. Mean model inference was 11.01 seconds and mean
export time was 8.48 seconds per chunk on the shared GPU; total wall time
was 1947.94 seconds. These timings are not directly comparable to the
earlier run under different shared-machine load.

Run directory:

```text
outputs/evaluation/robodojo/stack_bowls_step4k_futures_20261018_v1/
```

The first sample is in:

```text
policy/predictions/reset_0002/env_000/replan_000000_step_0000/
```

`reset_0002` reflects startup/native reset calls, not a third evaluated
episode. Each sample has `primary`, `secondary` and `wrist` MP4s,
`*_frames/frame_000.png` through `frame_008.png`, `*_filmstrip.png` and
`conditioning_*.png`. Replans continue through
`replan_000079_step_0790/`.

For a quick side-by-side inspection, the run's `diagnostics/` contains:

- `first_replan_primary_prediction_vs_actual.png` and `.mp4`: first
  sample's predictions versus native observations at matching timestamps.
  The PNG labels which observations occur after another policy replan.
- Equivalent `secondary` and `wrist` comparisons.
- `training_*_episode_3000_t0.png` and
  `training_episode_3000_t0.json`: actual training stack-bowls inputs,
  for checking camera roles and initial-state convention.
- `inference_audit.json`: artifact checks and measured end-effector
  tracking errors at all 79 consecutive replan boundaries.

The initial native canonical state agrees with training episode 3000 to
6.73e-8. At replan boundaries, the actual end effector follows the
preceding chunk's last executed target with median position error
2.25 mm / 2.55 mm and median angular error 0.54 / 0.72 degrees
(left/right). Maximum position errors were 6.23 / 24.75 mm; maximum
angular errors were 4.59 / 10.57 degrees. This argues against a gross
axis, quaternion or absolute/delta mismatch, but is not a complete
controller audit.

The first decoded conditioning-frame reconstruction has mean absolute
RGB error 2.51 / 1.92 / 1.95 on the uint8 scale across the three views.
It visually matches the actual input, supporting the latent decoding and
camera pipeline. The generated futures themselves do not show successful
stacking, and some later frames deform/blur the arms. The later rollout
also stays near the bowls without completing the task. These observations
do **not** establish whether checkpoint quality, remaining input/domain
differences or another inference issue is the root cause. No speculative
coordinate/channel swaps or action rescaling were applied.

### Completed step8k evaluation (2026-10-20, 00:18-00:48)

The newly supplied `1016_ft_on_robodojo/step8k` checkpoint was strictly
restored from its local cache. The run explicitly used the user's
`/data2/lizuolei/data/lerobot_data/simulation_data/RoboDojo/data/RoboDojo_ee_lerobot_v30_video`
dataset and the same task, seed/layout 0, 20 flow steps and 10-action
replan interval as the step4k future-export run.

| Setting / result | Value |
|---|---|
| Task / seed / native layout | `stack_bowls` / 0 / 0 |
| Completed episodes / successful episodes | 1 / 0 |
| Native partial score | 0.0 |
| Control steps / policy calls | 800 / 80 |
| Unstable episodes | 0 |
| Predicted MP4s / lossless PNG frames | 240 / 2160 |
| Actual videos | Three 640x480, 25 FPS videos, 801 frames each |
| Mean inference / future export per chunk | 9.50 / 6.77 seconds |
| Wall time including startup | 1854.45 seconds |

This is a completed unsuccessful **single-layout** episode, not an
infrastructure failure or an overall RoboDojo success-rate estimate.
The actual video shows the right arm moving around the bowls without
completing a stack. The first generated future likewise does not show
successful stacking; its arm motion already differs from the observed
execution before the first replan.

Artifacts:

```text
outputs/evaluation/robodojo/stack_bowls_step8k_futures_20261020_v1/
  summary.json
  policy/chunks/
  policy/predictions/reset_0002/env_000/replan_000000_step_0000/
  diagnostics/inference_audit.json
  diagnostics/native_rollout_primary_contact_sheet.png
  diagnostics/first_replan_primary_prediction_vs_actual.png
  diagnostics/first_replan_primary_prediction_vs_actual.mp4
```

The comparison PNG/MP4 also exists for `secondary` and `wrist`. Native
videos and `_result.json` are in:

```text
simulator/eval_result/RoboDojo/stack_bowls/Qwen3VLMoT/arx_x5/
0_action_type=ee,simulator=isaacsim6/stack_bowls_step8k_futures_20261020_v1/
```

The primary actual video is `episode_0000000_cam_head_fail.mp4`.
Predictions after 0.4 seconds are counterfactual: only the first ten
actions are executed before replanning, so later prediction-versus-actual
differences alone cannot establish model inconsistency.

All 80 saved normalized/native outputs were finite, and every exported
native action chunk exactly equaled the corresponding adapter output.
At the 79 saved replan boundaries, measured median end-effector tracking
errors were **1.76 / 1.56 mm** and **0.43 / 0.38 degrees** (left/right).
Maximum position errors were 14.90 / 10.19 mm. The initial canonical state
was exactly equal to the step4k run's initial state. The first decoded
conditioning frame's RGB MAE was 2.50 / 1.87 / 1.86 on the uint8 scale.
These checks argue against a gross pose/frame/action-export error, not
against every possible controller or visual-domain difference. The
evaluator-owned processes exited; unrelated GPU jobs were left untouched.

### Step8k training/validation/deployment comparison

Completed BF16 diagnostics are saved in:

```text
outputs/evaluation/robodojo/step8k_training_diagnostics_20261020_v1/
  diagnostics.json
  aggregate_analysis.json
  sample_*/metrics.json
  sample_*/actions.npz
  sample_*/conditioning_*.png
```

The eight samples comprise the two reconstructed configured generation
validation windows, plus stack-bowls episodes 3000, 3001 and 3002 at
offsets 0 and 64. These are diagnostic training windows, not a new
held-out test set. The original training validation artifacts/logs were
not supplied, so their actual contents and the exact metric behind the
reported `0.04` cannot be independently identified.

**Input and sampler alignment.** Using the actual saved training
configuration and training dataset loader, all eight samples had identical
camera tensors, masks, instructions and sample rates in the reconstructed
validation and deployment inputs. Maximum normalized state roundtrip error
through native xyz/wxyz/gripper was 7.75e-7. After separately measuring
that roundtrip and sharing the exact current normalized state, the actual
validation and deployment samplers produced **exactly identical action
tensors for the same seed** on all eight samples. Full training windows
contain ground-truth future frames/states, but `generate_validation`
selects current conditioning rather than consuming those futures.

This supports the existing 20D xyz/ort6d/gripper representation,
normalization, camera mapping and inference entry point. It does not prove
that online simulator images match the training capture domain, or that
the training and evaluation numerical runtimes are equivalent.

**Low reconstruction loss is not free-running action accuracy.**
`modeling_qwen3vl_mot.py::_flow_corrupt` constructs
`noisy = (1 - sigma) * clean + sigma * noise`. During training,
`forward` decodes `noisy - sigma * predicted_velocity` and computes the
masked physical reconstruction MSE against normalized ground-truth
actions. This is a one-step denoising problem with a ground-truth-corrupted
trajectory. At low sigma it already contains much of the true trajectory.
Inverse dynamics additionally conditions on the ground-truth future
video. Deployed joint prediction instead integrates video/state/action
from random noise using current images/state only.

For the six stack-bowls windows, the measured means were:

| Measurement | Value |
|---|---:|
| Joint reconstruction MSE, fixed sigma 0.1 | 0.02163 |
| Joint reconstruction MSE, fixed sigma 0.5 | 0.01433 |
| Joint reconstruction MSE, fixed sigma 0.9 | 0.13223 |
| Inverse-dynamics reconstruction MSE, fixed sigma 0.9 | 0.02458 |
| Free-running joint normalized action MSE, 20 steps / full 32 actions | 0.45012 |
| Free-running joint normalized action MSE, 50 steps / full 32 actions | 0.44206 |
| Free-running normalized action MSE, first 10 actions, 20 / 50 steps | 0.14835 / 0.15095 |
| First-10-action xyz L2 error versus demonstration, left / right | 2.55 / 6.85 cm |

For example, episode 3000 offset 0 has joint reconstruction MSE 0.01322
at sigma 0.5, inverse-dynamics reconstruction MSE 0.03096 at sigma 0.9,
but free-running full-chunk MSE 0.71372. Its right-arm xyz discrepancy
averages 19.65 cm over the full chunk and 7.82 cm over the executed
prefix. Thus a reconstruction loss near/below 0.04 can coexist with poor
free-running predictions even on recorded training observations, without
any simulator in the loop. This is a measured objective/sampling gap,
not proof that every deviation from a demonstration is invalid: multiple
successful trajectories can differ, and offline MSE is not task success.
Increasing flow integration from 20 to 50 steps did not meaningfully
resolve this gap; the baseline rollout remains at the saved 20 steps.

**The configured validation is weak coverage for this task.** The saved
finetuning setup keeps full training coverage and reports
`validation_split="training"`. Reconstructed configured validation selects
only episode 2975 (stack textured blocks) and episode 3435 (broom/dustpan
sweeping); neither is stack bowls. Their mean free-running normalized MSE
is 0.14978 at 20 steps, compared with 0.45012 on the six bowl windows.
Neither these two training samples nor an aggregate denoising loss
establish stack-bowls closed-loop competence or held-out generalization.

**Other differences and remaining uncertainty.** Training validation's
`generation_eval.py::_denormalize_action` clips displayed predictions to
empirical per-dimension min/max; deployment does not apply that arbitrary
pose clipping and only enforces native gripper limits. Reapplying the
display bounds to the six saved bowl predictions changes normalized MSE
only from 0.45012 to 0.44925, with **zero xyz values clipped**. It cannot
explain these measured centimeter-scale position discrepancies. No pose
clipping or speculative rescaling was added to deployment.

The saved training config enables native rowwise FP8 for the generation
and VLM layers; the completed diagnostics and native rollout restore BF16.
The paired native FP8 attempt was blocked by TorchAO's axiswise reshape
support, as documented above. Its failed output directory is
`outputs/evaluation/robodojo/step8k_fp8_diagnostics_20261020_v1/`;
it is not a completed diagnostic or evidence that precision is harmless.
Likewise, the local Isaac Sim 6 runtime differs from the official
Isaac Sim 5.1 setup, and the training capture's exact simulator version
and scene overlap remain unverified.

The strongest supported explanation is therefore **a gap between
conditional denoising loss and free-running joint sampling, combined with
nonrepresentative training-set validation**, rather than an established
xyz/ort6d adapter bug. Precision, visual-domain and simulator differences
are not fully ruled out. A useful subsequent training comparison would
track current-input-only free-running action/physical errors separately
from sigma/task-stratified denoising losses, validate held-out episodes
including stack bowls, and reproduce sampling in the exact training FP8
runtime before attributing the failure solely to model quality.

### Four-node step8k checkpoint setup (2026-10-24)

The new experiment is
`/data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_4nodes_8gpu`,
not the earlier `1016_ft_on_robodojo` experiment. The user describes this
checkpoint as trained for approximately four epochs.
Its `step8k/model` contains 32 DCP shards; all 32 files referenced by DCP
metadata are present. The saved standalone `config.json` exactly matches
`train_config.json`'s policy section.

The new experiment is cached separately under
`outputs/evaluation/robodojo/runtime_assets/robodojo_4nodes_8gpu/`.
All **35 files** (32 shards, DCP metadata and two configuration files)
were SHA256-verified against the source, totaling 25,227,532,685 bytes.
`checkpoint_provenance.json` in that cache records every hash and the
original source path. The new metadata hash is
`7900d88d239084a92cde00515e9cc34db034c9e3c8bada699213723ac40c50ce`;
the older step8k checkpoint's metadata hash is different.

Unlike the previous experiment, the new saved training config explicitly
sets `fsdp.fp8=false`, so standard BF16 inference matches its configured
training precision. It also changes the action reconstruction loss weight
from 5 to 10 and task weights from `[0.15,0.15,0.7]` to `[0.1,0.1,0.8]`.
Thus this is not an epoch-only controlled comparison. Architecture,
three-camera roles, 20D xyz/ort6d/gripper contract, 32-action horizon and
20 integration steps are unchanged.

To test three consecutive native stack-bowls layouts, without
success-seeking retries or physics changes:

```bash
conda activate wam
CACHE="$PWD/outputs/evaluation/robodojo/runtime_assets"
OMNI_KIT_ACCEPT_EULA=YES python -m lerobot.evaluation.eval_qwen3vl_mot_robodojo eval \
  --checkpoint "$CACHE/robodojo_4nodes_8gpu/step8k" \
  --policy-config "$CACHE/robodojo_4nodes_8gpu/config.json" \
  --dataset-root /data2/lizuolei/data/lerobot_data/simulation_data/RoboDojo/data/RoboDojo_ee_lerobot_v30_video \
  --qwen3vl-dir "$CACHE/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$CACHE/pretrained/Cosmos3-Edge" \
  --stage1-text-dir "$CACHE/pretrained/siglip2-base-patch16-224" \
  --task stack_bowls --eval-num 3 --num-envs 1 \
  --gpu 0 --sim-gpu 0 --seed 0 --replan-steps 10 \
  --save-predicted-video --timeout 10800 \
  --output-dir outputs/evaluation/robodojo/stack_bowls_4nodes_8gpu_step8k_20261024_v1
```

Use a fresh output directory when reproducing. Only a validated native
result with `evaluation_complete=true` counts as a completed evaluation.

### Completed four-node step8k results (2026-10-24, 18:54-20:03)

The three-layout run above completed with validated native results:
**0 successful episodes out of 3**, mean partial score **0.0**.
No layouts were skipped/replaced, and the native simulator reported zero
unstable episodes. All three reached the native 800-step horizon.
This is three layouts of **one task**, not a full RoboDojo benchmark or
evidence that the checkpoint cannot succeed on other tasks/layouts.

| Native layout | Success | Partial score | Control steps | Policy calls |
|---|---|---|---|---|
| 0 | false | 0.0 | 800 | 80 |
| 1 | false | 0.0 | 800 | 80 |
| 2 | false | 0.0 | 800 | 80 |

The completed run has `status="completed"` and
`evaluation_complete=true` in `summary.json`. It used the separately
verified new checkpoint/config, BF16, seed 0, 20 flow integration steps,
10-action replanning and the supplied training dataset statistics.
In contrast to the earlier experiment, this checkpoint's saved training
config disables FP8, removing that particular configured-precision
mismatch. No training-set diagnostic losses were measured for this new
checkpoint; the older experiment's offline loss numbers must not be
attributed to it.

Across all three episodes, **240 policy calls** produced **720 predicted
MP4s and 6480 lossless predicted PNGs**. Every saved normalized/native
output was finite, and every exported native action chunk exactly matched
the corresponding adapter output. All **nine actual rollout videos**
were decoded and checked: 640x480, 25 FPS, 801 frames each.
Mean model inference was 7.75 seconds and mean future export was
5.81 seconds per chunk; total wall time including startup was
4132.42 seconds (68.87 minutes) on the shared machine.

At the 79 saved replan boundaries per episode, actual end-effectors
continued to track the preceding executed target closely:

| Layout | Median xyz error, left / right | Median rotation error, left / right |
|---|---|---|
| 0 | 1.66 / 1.58 mm | 0.37 / 0.44 degrees |
| 1 | 1.64 / 1.64 mm | 0.41 / 0.38 degrees |
| 2 | 1.56 / 1.53 mm | 0.39 / 0.35 degrees |

Visual inspection of all three rollout contact sheets shows arms reaching
toward and sometimes displacing bowls, but no completed stack. Some first
sample predicted futures also show deformed/blurred arms; their motion
does not accurately match the actual execution. Predictions beyond the
first 0.4 seconds describe an unexecuted original action suffix, so those
later differences are not a controlled open-loop error measurement.
The tracking results argue against gross pose/coordinate export errors,
but do not establish the remaining failure's root cause.

All artifacts are under:

```text
outputs/evaluation/robodojo/stack_bowls_4nodes_8gpu_step8k_20261024_v1/
  summary.json
  checkpoint_provenance.json
  policy/chunks/inference.jsonl
  policy/predictions/
  diagnostics/inference_audit.json
  diagnostics/layout_00_native_primary_contact_sheet.png
  diagnostics/layout_00_primary_first_prediction_vs_actual.png
  diagnostics/layout_00_primary_first_prediction_vs_actual.mp4
```

Analogous comparisons/contact sheets exist for layouts `01` and `02`,
and prediction comparisons include all three camera views. Native videos
and `_result.json` are under:

```text
simulator/eval_result/RoboDojo/stack_bowls/Qwen3VLMoT/arx_x5/
0_action_type=ee,simulator=isaacsim6/stack_bowls_4nodes_8gpu_step8k_20261024_v1/
```

Primary actual video names are `episode_0000000_cam_head_fail.mp4`
through `episode_0000002_cam_head_fail.mp4`; left/right wrist videos
are alongside them. Prediction reset IDs `0002`, `0004`, `0006` correspond
to these three episodes, not seven evaluated episodes.
The evaluator-owned processes exited and GPU memory was released;
unrelated jobs and shared simulator/assets were not modified.

### Complete training demonstrations for comparison (2026-10-24)

The supplied dataset
`/data2/lizuolei/data/lerobot_data/simulation_data/RoboDojo/data/RoboDojo_ee_lerobot_v30_video`
contains **100 stack-bowls episodes**, indices **3000-3099**, with task
index **30** and instruction **"Stack the three bowls together."**
The first three complete episodes were extracted directly from its
original videos, not generated by the policy:

| Episode | Full frames | Duration at 25 FPS |
|---|---|---|
| 3000 | 240 | 9.60 seconds |
| 3001 | 366 | 14.64 seconds |
| 3002 | 341 | 13.64 seconds |

Exports are under:

```text
outputs/evaluation/robodojo/training_stack_bowls_20261024_v1/
  manifest.json
  episode_3000/
    three_views.mp4
    cam_high.mp4
    cam_left_wrist.mp4
    cam_right_wrist.mp4
    cam_high_contact_sheet.png
  episode_3001/
  episode_3002/
```

Each episode directory has the same files. `three_views.mp4` synchronizes
the original head, left-wrist and right-wrist cameras in that left-to-right
order. Individual views retain their original resolution and frame rate.
The contact sheet samples six points including the first and last frame;
the MP4s contain the entire episodes, not just those six frames or the
model's nine-frame prediction horizon.

Extraction uses each camera's own source file and `from_timestamp` in
`meta/episodes/chunk-000/file-000.parquet`, with exactly the episode's
`length` frames and an exclusive ending timestamp. Camera file boundaries
can differ: episode 3002's right-wrist video starts in a different source
file from the preceding two episodes. All nine individual camera clips
were encoded as lossless H264 and verified against their source segments
using decoded YUV hashes for **every frame**. All three synchronized clips
were checked for their full frame count, 25 FPS and exact duration.
`manifest.json` records source files, boundaries, frame counts, original
task/episode identities, output paths and extraction commands. Original
dataset videos were not modified; exported videos remain untracked.

### Absolute-action episode-tail correction (2026-10-24)

The first-action audit found a concrete error in the **current training
loader**, not a demonstrated simulator action-conversion error. Although
RoboDojo actions are absolute xyz/ort6d/gripper targets, canonical action
padding replaced positions with zero and rotations with identity, while
holding grippers. These synthetic timesteps remained active in the
physical reconstruction and flow targets.

On the supplied dataset's episode 3000, global frame 1627169 has one real
action and 31 padded actions. The real left target is approximately
`(-0.2994535, -0.3522613, 0.9217325)`; before correction, every padded left
target was approximately `(0, 0, 0)` with identity rotation. The same
problem affects the right arm. With all source frames used as anchors,
there are 1,736,000 padded action timesteps out of 59,395,264, or 2.923%.
That is not a measurement of this bug's contribution to checkpoint loss
or task failure.

RoboDojo's physical spec now explicitly selects
`action_padding="repeat_last"`, retaining the complete last real absolute
pose and grippers. Other datasets retain their existing `zero_control`
default. Both v2.1 and v3.0 window paths are covered, including short
episodes and terminal anchors. On the same real terminal sample, all 31
padded targets now match the last real command within 9.38e-9 after
normalization and denormalization.

Before/after evidence is saved in
`outputs/evaluation/robodojo/four_node_first_action_audit_20261024_v1/`
as `action_padding_before.json` and `action_padding_after.json`. This
changes future training targets, **not existing checkpoint weights or
deployment actions**. The original training checkout/logs were not
provided, so whether this checkpoint was trained with exactly this
padding implementation is not independently established. It is not
proof of the cause of its initial-action error; the separate initial
training-window probes contain no padded timesteps.

The `diagnose` output now also reports each arm's first predicted and
target xyz, first position/rotation errors, and (for embedding-flow
policies) `clean_action_autoencoding`: ground-truth action embeddings
passed directly through the physical decoder, without generation or
diffusion noise. `actions.npz` includes `current_state_raw` and
`decoded_ground_truth_embedding`. This separates decoder reconstruction
error from free-running sampling error; it is a ground-truth-assisted
diagnostic, not a deployable policy or a bound on achievable accuracy.

### Four-node first-action and official WAM audit (2026-10-24)

This audit concerns **`robodojo_4nodes_8gpu/step8k`**, not the earlier
`1016_ft_on_robodojo/step8k`. It addresses the observation that initial
predicted actions differ sharply from the demonstrations. Small simulator
tracking errors only validate following the requested target; they do
**not** validate whether that target was inferred correctly.

**Saved training contract.** The parent `config.json` equals the policy
snapshot inside `train_config.json`. The actual RoboDojo spec selects
`action.ee_ort6d_pos` and `observation.ee_ort6d_pos`: absolute dual-arm
20D xyz/interleaved-ort6d/gripper, first 20 slots of the 40D canonical
space. Training and inference use mean/std normalization, three ordered
RGB cameras, 256-pixel resizing, and a 32-action window spanning offsets
0 through 31 at 25 Hz. Nine video frames span that window. Although the
saved config contains `use_amp=false`, the FSDP trainer explicitly moves
the policy to BF16; this checkpoint has `fsdp.fp8=false`. BF16 inference
therefore matches the configured model precision.

The task mixture is forward dynamics 0.1, inverse dynamics 0.1, and joint
prediction 0.8, with no trained `action_prediction` task. Deployment
correctly selects joint sampling rather than switching to an untrained
action-only task. The physical encoder is frozen; its action decoder is
learned. Reconstruction weights are action 10 and state 5.
`initialize_from_stage1=false` in a saved, self-contained configuration
is not evidence that training skipped initialization: the constructor
updates that flag for subsequent restoration.

**Official WAM comparison.** Sources were checked at XPolicyLab revision
`e08ba78888b233b77ed2bb2d92886a52360a1081`:

- [X-WAM adapter](https://github.com/XPolicyLab/XPolicyLab/blob/e08ba78888b233b77ed2bb2d92886a52360a1081/policy/X_WAM/model.py)
  and [inner inference wrapper](https://github.com/XPolicyLab/XPolicyLab/blob/e08ba78888b233b77ed2bb2d92886a52360a1081/policy/X_WAM/X-WAM/evaluation/deploy_policy.py):
  its 14D delta actions, quantile normalization, and fixed orientation
  reframe belong to its own training representation. Its adapter
  accumulates those deltas into native absolute targets. Applying that
  reframe or delta accumulation to our already-absolute 20D labels would
  change the meaning of the learned targets.
- [FastWAM adapter](https://github.com/XPolicyLab/XPolicyLab/blob/e08ba78888b233b77ed2bb2d92886a52360a1081/policy/FastWAM/model.py)
  and [relative-action transforms](https://github.com/XPolicyLab/XPolicyLab/blob/e08ba78888b233b77ed2bb2d92886a52360a1081/policy/FastWAM/FastWAM/src/fastwam/datasets/lerobot/transforms/relative_action.py):
  preprocessing and normalization are tied to its saved processor/config.
  Relative-pose and relative-joint transforms, where configured, have
  paired training/inverse transformations; they are not generic WAM
  requirements. Its configured execution prefix is taken from the
  beginning of the generated chunk.

Our control loop also executes actions starting at chunk index zero,
then replans on a fresh observation. No extra first-action offset,
left/right swap, orientation reframe, relative conversion, normalization
change, or arbitrary action rescaling was justified by this comparison.

**Real-label transport positive control.** The first 32 true commands
from episode 3000 were passed through the authentic loader,
normalization, and the exact deployment `native_actions` conversion,
then compared against the dataset's original 16D xyz/wxyz/gripper
commands. Maximum xyz component error was 1.50e-8 m, rotation error
1.04e-5 degrees, and gripper error zero. These commands approximately
match the following recorded states (maximum raw component difference
1.19e-4), consistent with absolute next-state targets, not deltas.
This control does not replay contacts or establish simulator task success.

**Controlled first-window sampling.** Complete, unaugmented training
inputs from episodes 3000, 3001 and 3002 at offset zero were sampled with
seeds 0 and 1, using the new checkpoint, BF16 and the saved 20 flow steps.
There are no padded targets in these initial windows.

| Episode | Free-running action MSE, seed 0 | Seed 1 | Clean action embedding -> decoder MSE |
|---|---:|---:|---:|
| 3000 | 1.27542 | 1.18505 | 0.07233 |
| 3001 | 1.20145 | 0.89352 | 0.08895 |
| 3002 | 0.64350 | 0.30611 | 0.06623 |

All MSEs cover the 20 active normalized dimensions over 32 actions.
Across these six samples, first-10-action position discrepancy against
the demonstrations averages 4.67 cm for the left arm and 7.24 cm for the
right. On episode 3000, seed zero, the **first command's displacement
from the current state** is:

| Arm | Demonstration delta xyz, mm | Predicted delta xyz, mm |
|---|---|---|
| Left | `(0.029, -0.0001, -0.0001)` | `(12.418, 33.128, -4.953)` |
| Right | `(0.286, -0.447, 0.624)` | `(-22.695, 20.176, -3.146)` |

This discrepancy is present **before simulator execution**, even with
the same training images/state as the demonstration. At the same episode
and seed, joint reconstruction loss is 0.006747 at sigma 0.5, versus
1.27542 for free-running actions; inverse reconstruction at sigma 0.9
is 0.010767. These are different conditioning problems, not interchangeable
measures of policy accuracy. Even decoding clean ground-truth physical
embeddings is imperfect, so the discrepancy should not be attributed
exclusively to the flow integrator.

The current visual conditioning tokens from encoding the full training
video versus just its first frames are **exactly equal** on all three
windows. Current-state embeddings from the training and deployment
physical batches are also exactly equal after sharing the separately
checked normalized current state. Reloading the checkpoint and supplying
the saved native rollout's first conditioning images/state reproduces
that rollout's complete first normalized action tensor with **zero
maximum error**.

Evidence is under:

```text
outputs/evaluation/robodojo/four_node_first_action_audit_20261024_v1/
  report.json
  native_initial_reproduction.npz
  action_padding_before.json
  action_padding_after.json
  episode_3000/{conditioning_*.png,actions.npz,metrics.json}
  episode_3001/
  episode_3002/
```

`report.json` identifies the checkpoint/training config, seeds, input
parity, current visual/state embedding comparisons, conditional losses,
and first/chunk-end action directions. `action_padding_after.json` also
records the real-label transport control. The controlled two-seed probe
used explicit seeds per episode; the standard `diagnose` CLI additionally
includes the two configured validation windows and uses `seed + index`.

**Bounded conclusion.** The absolute-target padding and diagnostic
configuration-selection errors were fixed separately. Neither correction
changes the existing model's sampled first actions. The initial
training-input discrepancies do not require simulator execution or
different object layouts to appear; however, offline deviation can also
reflect a different valid plan, and these measurements alone are not a
task-success criterion. No root cause of all three native failures, and
no successful corrected rollout, has been established.

The reconstructed configured validation still selects two training
windows, neither from stack bowls. Original training initialization
logs/code revision, original remote-dataset normalization metadata, VAE
configuration identity, capture runtime, and exact scene overlap remain
unverified. Accordingly, this audit does not certify the entire training
setup or rule out every sampler/runtime issue.

## Non-destructive delta-action dataset export

Run from the repository root in `conda activate wam`:

```bash
python -m lerobot.evaluation.scripts.add_robodojo_delta_actions \
  --source /data2/lizuolei/data/lerobot_data/simulation_data/RoboDojo/data/RoboDojo_ee_lerobot_v30_video \
  --output outputs/evaluation/robodojo/dataset_tools_20261024_v1/RoboDojo_ee_lerobot_v30_delta
```

The output must not exist and must be separate from the source. The tool builds
in a sibling staging directory and publishes only a complete dataset. It never
overwrites the original `action`, `action.ee_ort6d_pos`, or state columns.
Instead, it appends `delta_action` (16D) and `delta_action.ee_ort6d_pos` (20D),
registers both in `meta/info.json`, and computes their own global/per-episode
statistics, including exact numeric quantiles. Existing absolute statistics
are preserved. `meta/delta_action_manifest.json` records the conversion contract.

The final chosen contract matches X-WAM's RoboDojo **measured-state increments**
at `action_skip=1`, not differences between consecutive raw action targets:

```text
delta_position[t] = state_position[t+1] - state_position[t]
delta_rotation[t] = R_state[t+1] @ R_state[t].T
delta_gripper[t]  = state_gripper[t+1] - state_gripper[t]
```

Positions use world axes. Each 16D arm block is
`[dx,dy,dz,dqw,dqx,dqy,dqz,dgripper]`; quaternion order is wxyz. Each 20D arm
block replaces the quaternion with the interleaved first two rotation-matrix
columns `[R00,R01,R10,R11,R20,R21]`. Quaternion/ort6d components are **not**
subtracted. Episode ends retain one zero-motion target: zero translation,
identity rotation, and zero gripper increment, including single-frame episodes.
This preserves the original row count; X-WAM's own transformed action sequence
instead has length `T-1`.

The reference is XPolicyLab revision
`e08ba78888b233b77ed2bb2d92886a52360a1081`,
`policy/X_WAM/transform_robodojo_to_xwam.py` and
`policy/X_WAM/X-WAM/data/robot_dataset.py::_build_delta_action_tensor`.
Its fixed right-side orientation reframe cancels from
`(R_next @ P) @ (R_current @ P).T`, so these relative rotations need no extra
reframe. X-WAM itself stores axis-angle deltas in a 14D action; these new
16D/20D columns share its increment semantics, not its feature representation
or released normalization statistics.

**Exporting does not switch model training or evaluation to delta actions.**
The `delta_robodojo_stack_bowl` training mix explicitly selects the new feature;
the ordinary evaluation entry still defaults to absolute actions. For a model
trained with that mix, use the delta evaluation entry documented below. It
cumulatively adds translation and gripper increments, left-multiplies rotations
(`R = delta_R @ R`), clips the integrated gripper at every step, and restarts
integration from the current observed state at each replan. Do not run delta
predictions through an absolute-action adapter.

By default videos are symlinked to avoid copying approximately 65 GB. Use
`--video-mode copy` for an independent export. Encoded files and original
camera timestamps are preserved, without re-encoding. Existing source video
intervals that differ from the data length are explicitly logged and recorded
in the manifest, not silently repaired. The full source contains 26
camera/episode intervals one frame shorter than its tabular length; this is a
separate source-video metadata issue, not an ort6d conversion error.

The command above has completed for all 3,500 episodes / 1,856,102 rows.
`outputs/evaluation/robodojo/dataset_tools_20261024_v1/delta_verification.json`
records an independent full-row check: original columns are unchanged, source
parquet SHA256 hashes still match the pre-export audit, both delta
representations match measured-state increments, and new global moments match
the generated rows. Maximum relative-rotation matrix error was `1.24e-8`.

## Task-based v3 subsets

List IDs, available task-name aliases, and exact instruction strings:

```bash
python -m lerobot.evaluation.scripts.subset_robodojo_tasks --list-tasks
```

Select one or several tasks with `--task` / `--tasks`. Each argument can be a
task name, a source task ID, or an exact quoted instruction:

```bash
# Standalone bowls subset, retaining both absolute and new delta columns.
python -m lerobot.evaluation.scripts.subset_robodojo_tasks \
  --source outputs/evaluation/robodojo/dataset_tools_20261024_v1/RoboDojo_ee_lerobot_v30_delta \
  --output outputs/evaluation/robodojo/dataset_tools_20261024_v1/stack_bowls_delta \
  --tasks stack_bowls

# Multiple tasks from the original absolute-action dataset.
python -m lerobot.evaluation.scripts.subset_robodojo_tasks \
  --output /path/to/new_bowls_and_blocks_subset \
  --tasks stack_bowls stack_blocks
```

The default source is the original full dataset. Task-name aliases are read
from literal instruction templates in the local RoboDojo checkout, without
importing simulator code; override its location using `--robodojo-root`.
Numeric IDs and quoted instructions do not require the simulator checkout.
`stack_bowls` and `stack_bowls_random` share instruction/task ID 30 in this
dataset: they select the same 100 episodes, not separately identifiable layout
variants. Mixed-task episodes are rejected rather than silently truncated.

The output contains complete selected episodes with contiguous episode, task,
and global row indices. Per-episode frame indices, timestamps, original
action/state values, and any delta columns are preserved. `meta/info.json`,
tasks, episode metadata, and numeric statistics/quantiles are rebuilt for the
selected rows; `train` covers the complete subset. Image statistics are
aggregated from selected source episode statistics using the repository's
existing weighted-moment/approximate-quantile convention, not recomputed by
decoding every video. `meta/subset_manifest.json` records source-to-output
episode/task mappings and any source video-interval warnings.

Videos are **copied by default**, including when the source uses symlinks, so
the subset is standalone. `--video-mode symlink` explicitly opts into sharing
source video storage. Both modes preserve original video files and per-camera
offsets without re-encoding; shared encoded files can retain unused frames
from episodes not selected in the tabular subset. They are not exposed as
extra dataset rows. Existing outputs are never overwritten.

The bowls command above has completed: **100 episodes, 44,774 rows**, original
episodes 3000 through 3099, with 2,261,418,695 bytes of copied video. Video
SHA256 hashes match the source. The repository's native v3 `LeRobotDataset`
successfully loaded both delta fields and decoded first/last rows of subset
episodes 0, 50, and 99 across all three cameras; decoded pixels match the
original videos using the loader's nearest-frame timestamp convention.
Evidence: `outputs/evaluation/robodojo/dataset_tools_20261024_v1/subset_verification.json`.

## Full quaternion-to-ort6d audit

```bash
python -m lerobot.evaluation.scripts.audit_robodojo_ort6d \
  --source /data2/lizuolei/data/lerobot_data/simulation_data/RoboDojo/data/RoboDojo_ee_lerobot_v30_video \
  --report /path/to/new_ort6d_audit.json
```

Reports must not already exist. The audit reads every physical row, validates
global/episode/frame indexing, and compares all 20 converted dimensions with
an independent SciPy quaternion-to-matrix conversion of each original 16D
action/state. It checks finite values, nonzero/unit quaternions, rotation-column
norms and orthogonality, records per-dimension maximum errors and source file
SHA256 hashes, and exits nonzero on failure. Quaternion sign equivalence is
handled through rotation matrices. Default conversion/rotation tolerance is
`2e-6`; source quaternion norm deviations greater than `1e-4` also fail.

The complete original dataset audit passed: **3,500 episodes, 1,856,102 rows**
for each feature pair, across all three physical parquet files.

| Original -> converted feature | Invalid rows | Conversion mismatch rows | Maximum absolute error |
|---|---:|---:|---:|
| `action` -> `action.ee_ort6d_pos` | 0 | 0 | `2.980232238769531e-7` |
| `observation.state` -> `observation.ee_ort6d_pos` | 0 | 0 | `7.152557373046875e-7` |

Both arms use the correct interleaved rotation-column ordering; positions and
grippers were included in the comparison. No ort6d repair was needed or made.
The full report is
`outputs/evaluation/robodojo/dataset_tools_20261024_v1/ort6d_audit.json`.
Subsequent full delta-export verification confirmed that all three original
parquet SHA256 hashes still match this report.

Focused regression checks for the three tools:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  python -m pytest --noconftest -q tests/test_robodojo_dataset_scripts.py
```

`--noconftest` avoids unrelated robot-hardware fixtures in the repository-wide
conftest; the dataset tools do not require robot hardware or a simulator.

## Evaluating a delta-trained stack-bowls checkpoint

Use the dedicated entry, which defaults to the supplied
`robodojo_stack_bowl_2nodes_8gpu/step8k` checkpoint and the local
`/data2/lizuolei/data/lerobot_data/simulation_data/RoboDojo/data/stack_bowls_delta`
subset. The existing absolute-action entry keeps its previous defaults.

```bash
conda activate wam
CACHE="$PWD/outputs/evaluation/robodojo/runtime_assets"
OMNI_KIT_ACCEPT_EULA=YES python -m lerobot.evaluation.eval_qwen3vl_mot_robodojo_delta eval \
  --checkpoint "$CACHE/robodojo_stack_bowl_2nodes_8gpu/step8k" \
  --qwen3vl-dir "$CACHE/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$CACHE/pretrained/Cosmos3-Edge" \
  --stage1-text-dir "$CACHE/pretrained/siglip2-base-patch16-224" \
  --task stack_bowls --eval-num 3 --num-envs 1 \
  --gpu 0 --sim-gpu 1 --replan-steps 10 --seed 0 \
  --save-predicted-video --port 19039 --timeout 7200 \
  --output-dir outputs/evaluation/robodojo/stack_bowls_delta_step8k_20261025_v1
```

The cache is a separately copied checkpoint, not an alias to another
experiment. `checkpoint_provenance.json` verifies all 19 source files,
including 16 DCP shards, by SHA256. The policy is restored strictly.
The delta entry defaults to a two-hour simulator timeout: exporting futures
at every replan and sharing GPUs can make three full 800-step layouts exceed
one hour. The ordinary absolute entry keeps its existing timeout.
To use the source weights directly, omit `--checkpoint`; the supplied source
experiment is the delta entry's default. Use a new output directory for every run.

The evaluator checks that `train_config.json` exactly matches the saved policy
snapshot and declares `data_mix=delta_robodojo_stack_bowl`, then verifies the
dataset's measured-state increment contract. It uses
`delta_action.ee_ort6d_pos` mean/std, including the training loader's `1e-3`
std floor, while states retain their own absolute
`observation.ee_ort6d_pos` statistics. Neither absolute-action statistics nor
X-WAM's released 14D quantile statistics are substituted.

Integration follows `policy/X_WAM/model.py::_integrate_deltas`: world-axis
translation accumulation, left-multiplied rotation accumulation, and stepwise
clipped gripper accumulation. Our 20D relative rotations are projected from
interleaved ort6d columns; they are not Euler-angle increments. Anchors are
each environment's latest measured state, not the preceding predicted endpoint.
No X-WAM absolute-frame reframe is needed because this dataset's states and
targets use the native environment frame.

The policy writes `evaluation_contract.json` with the normalization metadata
hash. Every chunk saves its action mode, unnormalized increments, observed
state, normalized prediction, and integrated absolute native targets. Native
videos are under the run's `simulator/eval_result/` tree; matching model futures
are under `policy/predictions/`. Native success remains the benchmark criterion;
small prediction loss or offline reconstruction error is not task success.

Before model rollouts, all **44,774 real labels across 100 episodes** were
round-tripped through the deployment adapter in chunks of up to 32, reanchoring
also in the middle of episodes. Integrated outputs match the next measured
state, with maximum errors `7.46e-9 m` in position, `5.87e-6 degrees` in rotation,
and `2.99e-8` in gripper. The evidence is
`outputs/evaluation/robodojo/stack_bowls_delta_step8k_20261025_preflight/ground_truth_transport.json`.

The actual training loader was also checked using the saved policy configuration
on subset episodes 0, 1, and 6. All three current camera tensors, masks, and
sample rates agree exactly with deployment; maximum normalized current-state
roundtrip error is `7.75e-7`. Normalized delta targets agree exactly with the
adapter's own-feature mean/std calculation, including the `1e-3` floor. See
`outputs/evaluation/robodojo/stack_bowls_delta_step8k_20261025_preflight/training_loader_parity.json`.
These checks establish local label transport and input parity, not identity of
unavailable remote training dataset/statistics snapshots or native task success.

The same entry supports `offline`, `serve`, and `diagnose`. Diagnostic episode
IDs refer to the **subset** (defaults 0, 1, 2), not original IDs 3000, 3001, 3002.
Its physical action errors compare cumulatively integrated targets from the
same anchor; the hold-current baseline uses zero-motion deltas. As before,
`diagnose` uses authentic training windows, not a held-out task-success test.

### Delta checkpoint: authentic training-window diagnostics

The new checkpoint was independently restored for a diagnostic run at
`outputs/evaluation/robodojo/stack_bowls_delta_step8k_20261025_diagnostics_v1/`.
It used the same cached weights/backbones and subset as the native command,
`--diagnostic-episodes 0 6 --diagnostic-frames 0
--diagnostic-inference-steps 20 --seed 0`, on physical GPU 1 in a separate
process. It includes the two saved configured validation windows and two
additional initial windows. Seeds are 0, 1, 2, and 3 in sample order.

| Subset episode | Global frame | Sample type | Free-running normalized action MSE | Joint conditioned reconstruction loss at sigma 0.5 |
|---|---:|---|---:|---:|
| 6 | 2299 | Configured validation | 0.861748 | 0.024446 |
| 85 | 39103 | Configured validation | 0.211980 | 0.008558 |
| 0 | 0 | Additional initial window | 1.298735 | 0.055249 |
| 6 | 1884 | Additional initial window | 1.339103 | 0.023299 |

In each case, generation using the authentic training batch and deployment
conditioning produced **exactly the same entire normalized action tensor**
after sharing the exact current normalized state (max difference 0). These
are real training images/states, with no simulator, layout, or visual-domain
change needed to observe the free-running errors. The compared task, masks,
cameras, sample rate, and target normalization match the verified loader.

Mean first-10 integrated position errors against demonstrations were,
respectively, left/right **0.31/0.72 cm**, **0.20/1.18 cm**, **0.13/2.30 cm**,
and **1.83/4.76 cm**. Integration of predicted deltas is included in these
physical errors; this is not merely a per-step increment comparison.

Low reconstruction loss at an intermediate ground-truth-conditioned noise
level is not free-running accuracy. Clean frozen physical-codec action
autoencoding also had nonzero normalized MSEs (0.210356, 0.037166, 0.457732,
0.105241), but these are not a proven irreducible loss floor or a complete
explanation for rollout behavior. A different valid plan can differ from a
demonstration, so offline deviation alone does not establish native failure.
The full measurements are in `diagnostics.json` and per-sample `metrics.json`.

The native layout-0 and training episode-0 initial camera comparison is saved
in the native run as `training_vs_native_initial.jpg`. Bowl positions,
appearance, and apparent size differ; therefore direct native-versus-training
action MSE is not used to attribute errors to the adapter or sampler.

### Delta checkpoint: native stack-bowls result

The supplied experiment's `step8k` was evaluated on native layouts 0, 1, and 2
with one environment, seed 0, 20 denoising steps, a 32-action prediction chunk,
and replanning every 10 actions. All three completed layouts failed: **0/3
successes, mean score 0**. Each ran 800 control steps; the native physics
profiles report no unstable episodes. This uses the same local Isaac Sim 6
compatibility runtime as the earlier absolute-action evaluation, with CPU
physics/tensor view, no fabric/GPU dynamics, physics dt 0.004 and rendering
every 10 physics steps.

| Layout | Native success | Native score | Completed control steps |
|---|---|---:|---:|
| 0 | false | 0 | 800 |
| 1 | false | 0 | 800 |
| 2 | false | 0 | 800 |

**Timeout/restart provenance matters.** The first attempt explicitly used
`--timeout 3600`, before the default was increased. Layouts 0 and 1 completed,
but the coordinator timed out during layout 2. Its original `summary.json`
correctly remains `failed`, `evaluation_complete=false`; it has not been
rewritten as a completed run. A separate policy process on port 19040 reused
the official native layout-resume manifest and the original isolated simulator
workspace/run ID. It retained layouts 0 and 1 and restarted only incomplete
layout 2, with policy seed 0 again. There were no abandoned/replacement layouts.
This is **not an uninterrupted three-layout sampling RNG stream**.

Artifacts are rooted at
`outputs/evaluation/robodojo/stack_bowls_delta_step8k_20261025_v1/`:

- `summary.json`: original timeout, wall time 3657.61 seconds.
- `resume_attempt1/stack_bowls_delta_step8k_20261025_v1/summary.json`:
  completed native evaluation, wall time 1356.25 seconds, including startup.
- `resume_attempt1/stack_bowls_delta_step8k_20261025_v1/resume_provenance.json`:
  original native manifest snapshot and explicit policy RNG restart.
- `simulator/eval_result/RoboDojo/stack_bowls/Qwen3VLMoT/arx_x5/0_action_type=ee,simulator=isaacsim6/stack_bowls_delta_step8k_20261025_v1/`:
  final `_result.json`, `_physics_profile.json`, and all nine native camera MP4s.
- `diagnostics/episode_000000{0,1,2}_cam_head_fail_contact_sheet.jpg`:
  head-camera frames 0, 100, 200, 400, 600, and 800 for each completed layout.
- `final_adapter_audit.json`: independent target integration, tracking, and
  decoded native-video checks across both attempts.

Every native MP4 decodes to 801 RGB frames at 640x480, 25 FPS (32.04 seconds).
Video names are `episode_0000000_cam_head_fail.mp4`, with episode numbers
0/1/2 and camera names `cam_head`, `cam_left_wrist`, and `cam_right_wrist`.
The head-camera sheets show approach/contact and bowl movement but no completed
three-bowl stack in these layouts.

All logged chunks and futures were preserved separately: the original attempt
has 220 chunks (80 layout-0, 80 layout-1, and 60 from incomplete layout-2),
660 predicted-camera MP4s and 7260 PNGs; the resumed attempt has 80 layout-2
chunks, 240 predicted-camera MP4s and 2640 PNGs. Thus the artifact totals include
one incomplete attempt and must not be described as just 240 uninterrupted
chunks. Futures are in each attempt's own `policy/predictions/` directory.

Independent NumPy integration checked all 300 logged chunks, grouped by
attempt/reset/environment so anchors from different attempts never mix.
Inverse normalization agrees exactly. Maximum differences from the saved
native targets are `5.96e-8 m` in translation, `1.05e-7` in rotation-matrix
entries, and `2.99e-8` in gripper. Across 296 within-group replan transitions,
median previous-target-to-next-measured position differences are 0.095 mm
left and 0.215 mm right, but maxima are 5.11 mm and 128.20 mm respectively.
Small median tracking error is not proof of good action targets, and the
large right-arm deviation must not be hidden. These numerical/input checks
rule out the tested transport/normalization discrepancies, not every possible
evaluation issue or the training failure's root cause. Three failed layouts
also do not establish a task-wide success rate.

### Delta checkpoint: step10k comparison

The next checkpoint from the same `robodojo_stack_bowl_2nodes_8gpu`
experiment is cached separately under
`outputs/evaluation/robodojo/runtime_assets/robodojo_stack_bowl_2nodes_8gpu/step10k/`.
Its policy/training configurations are byte-identical to the verified step8k
configuration. All 19 files (25,200,797,460 bytes, including 16 DCP shards)
were SHA256-verified against the supplied source. All 16 model shard hashes
differ from step8k; the DCP metadata contains 2560 state-dict entries and all
16910 storage references fit within their referenced files. The evidence is
`checkpoint_provenance_step10k.json` alongside the cache's original, unchanged
step8k provenance.

The matched native command retains seed 0, layouts 0/1/2, one environment,
replan 10, the saved 20 denoising steps, delta normalization/integration,
three cameras, and predicted-future export. It explicitly permits two hours:

```bash
conda activate wam
CACHE="$PWD/outputs/evaluation/robodojo/runtime_assets"
OMNI_KIT_ACCEPT_EULA=YES python -m lerobot.evaluation.eval_qwen3vl_mot_robodojo_delta eval \
  --checkpoint "$CACHE/robodojo_stack_bowl_2nodes_8gpu/step10k" \
  --qwen3vl-dir "$CACHE/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$CACHE/pretrained/Cosmos3-Edge" \
  --stage1-text-dir "$CACHE/pretrained/siglip2-base-patch16-224" \
  --task stack_bowls --eval-num 3 --num-envs 1 \
  --gpu 0 --sim-gpu 1 --replan-steps 10 --seed 0 \
  --save-predicted-video --port 19041 --timeout 7200 \
  --output-dir outputs/evaluation/robodojo/stack_bowls_delta_step10k_20261026_v1
```

Use a fresh output directory when repeating this command. The earlier step8k
layout-2 policy RNG was restarted after timeout, as documented above; that
caveat must remain explicit when comparing it against an uninterrupted run.

#### Step10k native completion and visual-input caveat

The native run completed uninterrupted in **3286.71 seconds** (54.78 minutes,
including startup), with no timeout or policy RNG restart. The native result
records **0/3 successes, mean score 0**; layouts 0, 1, and 2 each ran 800
control steps. The physics profiles record zero unstable episodes.

**This is not a clean model-quality comparison.** Inspection of the actual
native videos and saved policy-conditioning PNGs found a visual-input anomaly:
the head-camera framing changes during layouts 0 and 1, and layout 2 starts
with the head view pointed at the floor instead of the task table. Its initial
wrist views are also displaced. This is present in
`policy/predictions/reset_0006/env_000/replan_000160_step_0000/conditioning_primary.png`,
so it is not just an exported-video/contact-sheet problem.

All three initial canonical robot states exactly match their corresponding
step8k initial states. Initial per-camera pixel MAEs against the original 8k
attempt are only 0.29-0.40 for layout 0 and 0.58-0.77 for layout 1, but
70.25-86.03 for layout 2 (uint8 units). The image evidence, not an arbitrary
pixel-MAE threshold, establishes the bad framing. The cause of the changed
camera/scene view and its persistence across reset is **unresolved**; no
camera, scene, shared runtime, or asset repair was applied. The original
native labels are preserved unchanged, but must not be used alone to conclude
that 10k training is worse, insufficient, or otherwise responsible for all
three failures. Zero native instability flags do not certify camera validity.

The run root is
`outputs/evaluation/robodojo/stack_bowls_delta_step10k_20261026_v1/`.
Key artifacts:

- `summary.json`: native completion and the unchanged native 0/3 result.
- `simulator/eval_result/RoboDojo/stack_bowls/Qwen3VLMoT/arx_x5/0_action_type=ee,simulator=isaacsim6/stack_bowls_delta_step10k_20261026_v1/`:
  `_result.json`, `_physics_profile.json`, and nine native camera MP4s.
- `diagnostics/episode_000000{0,1,2}_cam_head_fail_contact_sheet.jpg`:
  head-camera frames 0/100/200/400/600/800.
- `diagnostics/8k_vs_10k_initial_head.jpg` and `visual_input_audit.json`:
  initial-view comparison and observed-state/pixel checks.
- `final_adapter_audit.json`: `benchmark_complete=true` means native execution
  completed, while `visual_input_sanity_passed=false` and
  `clean_model_comparison=false` explicitly qualify interpretation.

All nine native MP4s decode to 801 frames, 640x480, 25 FPS. There are 240
logged/saved prediction chunks, 720 predicted-camera MP4s, and 7920 PNGs.
Independent NumPy checks across every chunk confirm exact inverse
normalization and maximum integration differences of `5.96e-8 m` in
translation, `9.33e-8` in rotation-matrix entries, and `2.99e-8` in gripper.
These numerical checks do not validate the camera inputs.

Across 237 within-episode replan transitions, median target-to-next-measured
position differences are 0.107 mm left and 0.210 mm right; maxima are 2.65 mm
and 311.92 mm. The largest right-arm discrepancy is in layout 0, between the
step-619 target and step-620 measured anchor; its exact chunk paths and poses
are preserved in `worst_right_target_tracking.json`. Neither small medians nor
correct delta integration justify hiding the large execution deviation or the
visual-input anomaly. A trustworthy 8k/10k success comparison requires resolving
the camera/scene/reset validity issue first.

#### Camera-integrity follow-up: idle native reset control

At the start of the follow-up inspection on 2026-10-26, the original
`stack_bowls_delta_step10k_20261026_v1/` output directory was no longer
present. The preceding section records historical findings, not a claim that
those ignored artifacts remain available. No original files were removed by
this investigation.

A separate, no-policy diagnostic exercised the native layouts 0/1/2 and the
same close/recreate/reset lifecycle, with 100 idle control intervals per
layout (10 physics steps per interval). All three initial head/wrist views
show the expected task table. Head-camera world transforms remained exactly
unchanged in all nine snapshots; both wrist-camera transforms also remained
unchanged while idle. Each capture manager had exactly three active views.
The retained evidence is under
`outputs/evaluation/robodojo/camera_integrity_idle_20261026_v1/`:
`snapshots.json`, `initial_views.jpg`, `probe_summary.json`, and
`simulator.log`. This diagnostic loads no model and is not a success-rate
evaluation.

The same `Stage opening or closing already in progress` errors occurred at
all three teardown points despite correct subsequent initial views. They are
therefore not, by themselves, evidence explaining the earlier floor-facing
images. This idle control does not exclude a longer-duration, action-dependent,
or renderer-buffer failure.

#### Camera-integrity follow-up: full-duration controls and passive 10k rerun

The follow-up did **not reproduce the original camera anomaly**. This does
not establish its cause or constitute an environment repair: the original
run's raw artifacts remain unavailable, and no shared simulator, asset, or
production evaluator change was made.

The native head sensor is the unmounted
`/World/envs/env_0/Camera_0/cam_head`, at approximately
`[0, -0.41, 1.308]` with a 30-degree X rotation. It is not mounted on the
visual `camera_stand` asset. Native end-effector actions do not directly
control it; the material warnings alone do not establish that a robot
physically moved the camera.

The completed no-policy controls are under
`outputs/evaluation/robodojo/`:

| Directory | Control intervals per layout | Snapshots | Head world-matrix change |
| --- | --- | --- | --- |
| `camera_integrity_idle_20261026_v1` | 100 | 9 | Exactly zero |
| `camera_integrity_idle_long_20261026_v1` | 800 | 51 | Exactly zero |
| `camera_integrity_action_replay_20261026_v1` | 800 | 51 | Exactly zero |

Each control completed layouts 0/1/2. The action replay uses 450 cached
native targets from the earlier layout-0 rollout, then holds the final target;
it does not load or evaluate a policy. In the replay, the maximum wrist-to-EE
relative-position changes are `6.27e-8 m` left and `8.60e-8 m` right, with
rotation-matrix changes below `3.10e-7`. Camera render-product bindings match
their intended prims in every saved snapshot. Each directory retains
`snapshots.json`, `camera_integrity_report.json`, `probe_summary.json`, and
`simulator.log`. The JSON summaries were generated from the retained snapshots
and verified completed native results; these are diagnostics, not policy
success-rate measurements.

The fresh, uninterrupted **step10k** policy run is retained at:

```text
outputs/evaluation/robodojo/stack_bowls_delta_step10k_camera_audit_20261026_v2/
```

It uses the same checkpoint, dataset statistics, seed 0, delta integration,
32-step predictions, and 10 executed steps per replan as the earlier 10k
attempt, with port 19048 and a 7200-second timeout. Session-local workspace
instrumentation only reads USD transforms and saves existing observations;
it adds no RGB accessor or render calls. A first diagnostic attempt at
`stack_bowls_delta_step10k_camera_audit_20261026_v1` was interrupted and is
not included in the result. Its extra RGB accessor was not an independent
render control because Replicator can reuse an existing render product.

The completed v2 run took **4487.23 seconds** including startup and records
**0/3 native successes, mean score 0**, with 800 steps in every layout and
zero native unstable-episode flags. All 240 saved policy-conditioning head
images were reviewed via contact sheets, together with native head-video
samples: the task-table view remains normal, including the layout-2 reset;
the original floor-facing anomaly is not observed.

Across 51 passive snapshots (steps 0/50/.../800 in each layout), head world
matrices are exactly identical across layouts. World, environment, robot-base,
room, and table transforms remain unchanged within each layout. Room/table
instance-name suffixes change during native recreation, so their transforms
are checked against each layout's own initial prims rather than requiring
identical prim paths across resets. Wrist-to-EE relative-position errors are
at most `1.69e-7 m` left and `1.35e-7 m` right; corresponding rotation-matrix
errors are below `3.55e-7`. All saved camera render bindings are correct.
These observations support current-run camera integrity, not a diagnosis of
the missing original run.

The numerical audit covers all 240 chunks: inverse normalization is exact,
and maximum delta-integration errors are `5.96e-8 m` in translation,
`1.02e-7` in rotation-matrix entries, and `2.99e-8` in gripper. All nine native
videos decode to 801 frames at 640x480, 25 FPS. There are also 720 predicted
camera videos and 7920 prediction PNGs.

Correct conversion does **not** imply perfect physical target tracking.
Across 237 within-layout replan transitions, median target-to-next-measured
position differences are 0.148 mm left and 0.325 mm right; maxima are
3.24 mm and **413.87 mm**. The largest right-arm discrepancy is in layout 0,
between the step-519 target and step-520 measured anchor. Its cause is not
established, and it must not be hidden by the small medians or integration
errors. This run still does not, by itself, prove undertraining; the earlier
8k layout-2 RNG-restart caveat also remains relevant to comparisons.

Key artifacts relative to the v2 root:

- `summary.json` and `settings.json`: unchanged native labels and run settings.
- `camera_audit/camera_observations.jsonl` and
  `camera_audit/camera_integrity_report.json`: passive geometry evidence.
- `diagnostics/layout_{0,1,2}_all_conditioning_heads.jpg`: all 80 actual
  head-conditioning images per layout.
- `diagnostics/episode_000000{0,1,2}_cam_head_fail_contact_sheet.jpg`:
  native head frames 0/100/200/400/600/800.
- `final_adapter_audit.json`: full numerical, artifact, and video audit;
  `benchmark_complete=true` alone is not a visual-integrity guarantee.
- `worst_right_target_tracking.json`: largest target/measurement discrepancy.
- `simulator/eval_result/RoboDojo/stack_bowls/Qwen3VLMoT/arx_x5/0_action_type=ee,simulator=isaacsim6/stack_bowls_delta_step10k_camera_audit_20261026_v2/`:
  native result, physics profile, and nine camera MP4s.

### Delta checkpoint: step12k comparison

The supplied `step12k` checkpoint from
`/data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_stack_bowl_2nodes_8gpu`
is cached under the same experiment's `runtime_assets` directory. All 19
files (25,200,797,460 bytes) were SHA256-verified against the source, which
remained unchanged during copying. Policy and training configurations are
byte-identical to the verified 10k configuration; all 16 model shard hashes
differ from 10k. The 2560 DCP state-dict entries have matching names, shapes,
and dtypes, and all 16910 storage references fit within their files.
The retained provenance is `checkpoint_provenance_step12k.json`.

The comparison retains seed 0, native layouts 0/1/2, one environment, the
saved 20 denoising steps, delta integration, 32-step predictions with 10
executed steps per replan, and actual/predicted video export:

```bash
conda activate wam
CACHE="$PWD/outputs/evaluation/robodojo/runtime_assets"
OMNI_KIT_ACCEPT_EULA=YES python -m lerobot.evaluation.eval_qwen3vl_mot_robodojo_delta eval \
  --checkpoint "$CACHE/robodojo_stack_bowl_2nodes_8gpu/step12k" \
  --qwen3vl-dir "$CACHE/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$CACHE/pretrained/Cosmos3-Edge" \
  --stage1-text-dir "$CACHE/pretrained/siglip2-base-patch16-224" \
  --task stack_bowls --eval-num 3 --num-envs 1 \
  --gpu 0 --sim-gpu 1 --replan-steps 10 --seed 0 \
  --save-predicted-video --port 19049 --timeout 7200 \
  --output-dir outputs/evaluation/robodojo/stack_bowls_delta_step12k_20261026_v1
```

Use a fresh output directory for any repeat. The initial 12k attempt also
uses the same session-local passive camera/scene-observation wrapper as the
completed 10k v2 audit, without additional RGB accessors or rendering.
The command above reproduces the supported policy/evaluation settings but
does not itself install that diagnostic wrapper. This preparation records
no success result; completion, native labels, and visual integrity must be
verified separately. The earlier 8k layout-2 RNG-restart caveat still applies
to comparisons against 8k.

#### Step12k native completion and matched 10k comparison

The 12k run completed uninterrupted in **2467.87 seconds** (41.13 minutes,
including startup). All three layouts ran 800 control steps, with zero
native unstable-episode flags. Native success remains **0/3**, but layout 0
receives partial task credit:

| Layout | Success | Per-layout normalized score |
| --- | --- | --- |
| 0 | false | 0.15 |
| 1 | false | 0.00 |
| 2 | false | 0.00 |

The native aggregate `score` is **5.0**, calculated as
`mean([0.15, 0, 0]) * 100`; this is **not a 5% success rate**. In
`task/RoboDojo/tasks/stack_bowls.py`, the 15-point partial stage requires
open grippers and a valid two-bowl stack. Full success additionally requires
the three-bowl stack with the required orientations and the robots returned
to their initial poses. Thus the result shows partial two-bowl-stack progress,
not a completed three-bowl task.

The reference is the camera-audited 10k v2 run, not the missing original
10k attempt:

| Checkpoint | Native successes | Native aggregate score | Per-layout normalized scores |
| --- | --- | --- | --- |
| 10k v2 | 0/3 | 0.0 | 0 / 0 / 0 |
| 12k | 0/3 | 5.0 | 0.15 / 0 / 0 |

Both runs are uninterrupted, and initial canonical robot states match
exactly for every layout. Checkpoint, output directory, and policy port are
the only differing recorded settings. This small matched comparison shows
additional partial credit, not a statistically established success-rate
improvement or proof that undertraining alone explains the failures.

All 240 actual policy-conditioning head images and native head-video samples
were visually reviewed. Table framing remains normal across all layouts;
the previous floor-facing anomaly was not observed. Across 51 passive
snapshots, the head world matrix is exactly unchanged, tracked scene
transforms remain fixed within each layout, and all camera render bindings
match their intended prims. Maximum wrist-to-EE relative-position errors are
`1.39e-7 m` left and `1.11e-7 m` right, with rotation-matrix errors below
`3.73e-7`.

All 240 prediction chunks pass inverse-normalization and delta-integration
checks: maximum differences are `5.96e-8 m` in translation, `9.56e-8` in
rotation-matrix entries, and `2.99e-8` in gripper; inverse normalization is
exact. All nine native camera MP4s decode to 801 frames, 640x480, 25 FPS.
The 720 predicted-camera videos and 7920 prediction PNGs are retained.

Physical target tracking is not perfect. Across 237 within-layout replan
transitions, median target-to-next-measured position differences are 0.077 mm
left and 0.207 mm right; maxima are 4.30 mm and **155.02 mm**. The largest
right-arm discrepancy occurs in layout 0, between the step-459 target and
the step-460 measured anchor. Its cause remains unestablished; correct
delta conversion and normal camera inputs do not eliminate this execution
deviation or diagnose the training failure.

Artifacts under
`outputs/evaluation/robodojo/stack_bowls_delta_step12k_20261026_v1/`:

- `summary.json`, `settings.json`, and `comparison_with_step10k.json`:
  completion, native scores, matched settings, and scoring interpretation.
- `final_adapter_audit.json` and
  `camera_audit/camera_integrity_report.json`: numerical, video, geometry,
  and completed visual-review evidence.
- `diagnostics/layout_{0,1,2}_all_conditioning_heads.jpg` and native
  head-video contact sheets: actual input and rollout image review.
- `worst_target_tracking.json`: largest left/right target deviations.
- `policy/predictions/`: predicted-camera videos and conditioning/future PNGs.
- `simulator/eval_result/RoboDojo/stack_bowls/Qwen3VLMoT/arx_x5/0_action_type=ee,simulator=isaacsim6/stack_bowls_delta_step12k_20261026_v1/`:
  native result, physics profile, and all nine actual camera MP4s.

#### Matched step16k evaluation preparation

The `step16k` checkpoint from
`/data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_stack_bowl_2nodes_8gpu`
is cached under the same runtime-assets experiment directory as 12k.
All 19 files (25,200,797,460 bytes) have matching source/destination SHA256
hashes, with source sizes and modification times unchanged during copying.
The parent policy/training configurations match 12k byte-for-byte, all 16
model shards differ from 12k, and the DCP metadata preserves all 2560 state
entries with matching tensor shapes/dtypes and 16910 valid storage references.
Provenance is retained in
`runtime_assets/robodojo_stack_bowl_2nodes_8gpu/checkpoint_provenance_step16k.json`.

This is another **three-layout matched diagnostic**, not the full
75-episode, three-seed `stack_bowls` evaluation. It retains layouts 0/1/2,
seed 0, one environment, the saved 20 denoising steps, 32 predicted actions,
10 executed actions per replan, and actual/predicted video export.

```bash
conda activate wam
CACHE="$PWD/outputs/evaluation/robodojo/runtime_assets"
OMNI_KIT_ACCEPT_EULA=YES python -m lerobot.evaluation.eval_qwen3vl_mot_robodojo_delta eval \
  --checkpoint "$CACHE/robodojo_stack_bowl_2nodes_8gpu/step16k" \
  --qwen3vl-dir "$CACHE/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$CACHE/pretrained/Cosmos3-Edge" \
  --stage1-text-dir "$CACHE/pretrained/siglip2-base-patch16-224" \
  --task stack_bowls --eval-num 3 --num-envs 1 \
  --gpu 0 --sim-gpu 1 --replan-steps 10 --seed 0 \
  --save-predicted-video --port 19050 --timeout 7200 \
  --output-dir outputs/evaluation/robodojo/stack_bowls_delta_step16k_20261027_v1
```

The initial run also uses the session-local passive camera wrapper used for
10k v2 and 12k; the command above reproduces supported evaluation settings
without installing that diagnostic wrapper. Use a fresh output directory
for repeats. This preparation does not establish a success result.

#### Step16k native completion and matched 12k comparison

The run completed uninterrupted in 4313.25 seconds (71.89 minutes including
startup), with 800 control steps per layout and no native unstable-episode
flags. Full success is **0/3**, with normalized scores **0 / 0 / 0** and a
native aggregate score of **0.0**, compared with 12k's 0/3 successes and
aggregate partial-credit score of 5.0. This three-layout diagnostic does not
establish a population-level checkpoint ranking.

All 240 actual conditioning head images and native head-video contact sheets
were reviewed: table framing is normal, with no floor-facing anomaly.
The 51 passive snapshots preserve the head and within-layout scene matrices
exactly, correct camera bindings, and wrist-to-EE errors below `4e-7`.
Initial canonical states match 12k exactly. Recorded settings differ only in
checkpoint, output directory, and port; policy sampling remains uninterrupted.

All 240 chunks pass inverse-normalization and delta-integration checks
(translation maximum `5.96e-8 m`, rotation-matrix maximum `9.82e-8`,
gripper maximum `2.98e-8`). Nine native camera videos, 720 prediction videos,
and 7920 prediction PNGs are retained. The maximum left-arm target-to-next-
measured position discrepancy is **126.50 mm**, versus **1.26 mm** right;
medians are 0.081 mm and 0.202 mm. The left-arm outlier's cause remains
unestablished and is not explained by numerical conversion checks.

Artifacts are under
`outputs/evaluation/robodojo/stack_bowls_delta_step16k_20261027_v1/`:
`summary.json`, `final_adapter_audit.json`, `comparison_with_step12k.json`,
`camera_audit/camera_integrity_report.json`, `worst_target_tracking.json`,
and `diagnostics/`. Actual videos are in the native
`simulator/eval_result/` subtree; predicted videos remain in
`policy/predictions/`.

#### Step16k ten-step sampling ablation (2026-10-27)

The requested sampling ablation changes only the flow integration count
from the saved 20 steps to **10 steps**. It retains the same step16k
checkpoint, layouts 0/1/2, seed 0, one environment, 32 predicted actions,
10 executed actions per replan, and actual/predicted video export.
The checkpoint configuration is not modified; `--inference-steps 10`
overrides sampling for this evaluation only. Ten sampling steps are
distinct from the unchanged ten-action execution interval.

```bash
conda activate wam
CACHE="$PWD/outputs/evaluation/robodojo/runtime_assets"
OMNI_KIT_ACCEPT_EULA=YES python -m lerobot.evaluation.eval_qwen3vl_mot_robodojo_delta eval \
  --checkpoint "$CACHE/robodojo_stack_bowl_2nodes_8gpu/step16k" \
  --qwen3vl-dir "$CACHE/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$CACHE/pretrained/Cosmos3-Edge" \
  --stage1-text-dir "$CACHE/pretrained/siglip2-base-patch16-224" \
  --task stack_bowls --eval-num 3 --num-envs 1 \
  --gpu 0 --sim-gpu 1 --replan-steps 10 --seed 0 \
  --inference-steps 10 --save-predicted-video --port 19060 --timeout 7200 \
  --output-dir outputs/evaluation/robodojo/stack_bowls_delta_step16k_sampling10_20261027_v1
```

The instrumented run uses the same session-local passive camera wrapper
as the 20-step baseline. Its settings, logs, native result, and videos are
kept in the fresh output directory above. This is a three-layout
diagnostic, not the full benchmark; preparation alone establishes no result.

The instrumented run started successfully and produced finite `(32, 20)`
physical-action predictions with ten executed actions per replan. The saved
coordinator settings differ from the 20-step baseline only in the sampling
override, output directory, and port. Startup evidence is retained in
`startup_comparison_with_sampling20.json`.

The run completed uninterrupted with 800 control steps per layout and no
native unstable-episode flags. Reducing sampling to ten steps did not
improve success or partial-credit score on these three layouts:

| Metric | 20 sampling steps | 10 sampling steps |
|---|---|---|
| Full success | 0/3 | 0/3 |
| Layout 0/1/2 normalized scores | 0 / 0 / 0 | 0 / 0 / 0 |
| Native aggregate score | 0.0 | 0.0 |
| Mean model inference per chunk | 8.25 s | 7.16 s |
| Mean prediction-video export per chunk | 5.86 s | 5.96 s |
| Total wall time, including startup | 71.89 min | 68.43 min |

Observed model inference time fell by 13.3%, while total wall time fell by
4.8%; shared-machine load differs between runs, so these are not isolated
kernel-speed measurements. The task videos still show reaching/contact
without completing the bowl stack.

All 240 saved chunks contain finite numeric arrays and pass delta-integration
and inverse-normalization checks. Initial states match the 20-step baseline
exactly. All 240 conditioning head images and the three native head-video
contact sheets were reviewed: table framing is normal, with no floor-facing
anomaly. The 51 passive snapshots preserve head/scene transforms and camera
bindings; wrist-to-EE errors remain below `4e-7`. Target-to-next-measured
position discrepancy reaches 160.03 mm on the left and 1.81 mm on the right;
this tracking outlier's cause remains unestablished.

The run retains nine actual camera videos, 720 prediction videos, and 7920
prediction PNGs. See `comparison_with_sampling20.json`,
`final_adapter_audit.json`, `camera_audit/camera_integrity_report.json`, and
`diagnostics/` in the output directory. Actual videos are under
`simulator/eval_result/`; predicted videos are under `policy/predictions/`.
These three layouts do not establish equivalence across the full benchmark.

#### Step16k noise-to-action diagnostic

Closed-loop tracking error measures whether the robot follows its generated
target, not whether that target matches a demonstration. To measure the
latter, compare pure-noise generation at 10 and 20 sampling steps against
real labels, using the existing training/deployment input-parity diagnostic:

```bash
conda activate wam
CACHE="$PWD/outputs/evaluation/robodojo/runtime_assets"
CUDA_VISIBLE_DEVICES=0 python -m lerobot.evaluation.eval_qwen3vl_mot_robodojo_delta diagnose \
  --checkpoint "$CACHE/robodojo_stack_bowl_2nodes_8gpu/step16k" \
  --qwen3vl-dir "$CACHE/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$CACHE/pretrained/Cosmos3-Edge" \
  --stage1-text-dir "$CACHE/pretrained/siglip2-base-patch16-224" \
  --diagnostic-episodes 0 6 --diagnostic-frames 0 \
  --diagnostic-inference-steps 10 20 --replan-steps 10 --seed 0 \
  --output-dir outputs/evaluation/robodojo/stack_bowls_delta_step16k_action_error_20261027_v1
```

This selects the two configured generation-validation windows plus the
initial windows of episodes 0 and 6. The saved split is `training`, so these
four windows do not estimate held-out generalization. Each window uses the
same seed for both sampling counts. Metrics cover the 20 active normalized
action dimensions and separately compare integrated native targets over
the first 10 actions and full 32-action horizon. Clean ground-truth action
embedding reconstruction and ground-truth-conditioned denoising losses are
separate diagnostics, not substitutes for pure-noise sampling accuracy.

The four-window measurement completed in BF16. Training/deployment sampling
parity is exact for every window. Both step counts use identical per-window
seeds (0/1/2/3); saved predictions independently reproduce the reported MSEs.

| Metric, averaged equally across four windows | 10 sampling steps | 20 sampling steps |
|---|---|---|
| Full 32-action normalized MSE | 0.201318 | 0.202351 |
| First 10-action normalized MSE | 0.208998 | 0.196754 |
| Full 32-action normalized RMSE | 0.448685 | 0.449834 |
| First 10 target position mean L2, left / right | 5.15 / 6.99 mm | 5.20 / 6.96 mm |
| First 10 target rotation mean error, left / right | 1.22 / 1.08 deg | 1.19 / 1.17 deg |
| Full 32 target position mean L2, left / right | 15.02 / 18.79 mm | 12.94 / 19.45 mm |
| Full 32 target rotation mean error, left / right | 3.52 / 3.69 deg | 3.08 / 3.90 deg |
| Full 32 target clipped gripper MAE, left / right | 0.0122 / 0.0268 | 0.0114 / 0.0268 |

Normalized MSE averages squared error across 20 active dimensions; it has
no single physical unit. Position/rotation metrics compare predicted and
demonstrated delta sequences integrated from the same measured anchor,
using the exact deployment conversion. They measure execution-target
accuracy, not actual simulator tracking.

Passing clean ground-truth action embeddings through the physical decoder
alone gives mean normalized MSE 0.186810 (first ten: 0.217971). Thus decoder
reconstruction is also imperfect. This ground-truth-assisted diagnostic
is neither a lower bound nor an additive error decomposition: subtracting
it from sampling MSE does not isolate a causal sampling-error component.
The hold-current baseline gives full-horizon MSE 1.243064. These limited
windows show similar 10/20-step errors, not statistical equivalence.

Raw per-window metrics, predictions, and aggregate results are retained in
`diagnostics.json`, `sample_*/{actions.npz,metrics.json}`, and
`aggregate_action_error.json` in the output directory above.
