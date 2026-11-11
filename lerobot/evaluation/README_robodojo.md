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

## Qwen all-task step16k put-bottles-into-dustbin (2026-11-04)

Evaluate `put_bottles_into_dustbin` with the **same uploaded all-task
step16k** used in the latest ordinary-bowls rollout:

```text
/data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_2nodes_8gpu_high_noise_shift_5_decay_25k/step16k
```

Preflight rehashed all19 source files and the saved full-task
metadata/parquet fingerprints against that evaluation's provenance.
All match. Continue using the full1,856,102-frame delta dataset and its
own normalization, not the bowl-only subset. This task is present in
training:100 demonstrations, full-task episodes2700..2799,
41,816frames. Native instruction:
`Pick up the bottles and throw them into the dustbin, using handover when needed.`

Use native layouts0/1/2, seed0, one environment, BF16,
sampling10/prediction32/execution10 and continuous policy RNG.
Run all three even if the first fails. Preserve the task's own
**700-control-step limit** (not the bowl task's800); no reward, horizon,
physics or skill modifications. Each saved layout contains four bottles,
the dustbin and the camera stand. Their native task/config/layout and
used asset files were verified and fingerprinted; native USD and USDZ
asset formats are both accepted. Preflight:

```text
outputs/evaluation/robodojo/runtime_assets/robodojo_2nodes_8gpu_high_noise_shift_5_decay_25k/put_bottles_into_dustbin_step16k_preflight_20261104.json
```

The previous bowl rollout is only a settings reference. These are
different tasks/layouts: robot initial-state differences are descriptive,
not an equality requirement, and success rates are not matched-layout
checkpoint comparisons. Three layouts are not a full-task benchmark.

```bash
RUN=outputs/evaluation/robodojo/put_bottles_into_dustbin_delta_alltask_shift5_decay25k_step16k_sampling10_20261104_v1
CACHE=outputs/evaluation/robodojo/runtime_assets
unset ROBODOJO_STOP_AFTER_FIRST_FAILURE
export PYTHONPATH="$PWD" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export OMNI_KIT_ACCEPT_EULA=YES HF_HUB_OFFLINE=1
export ROBODOJO_EVAL_MODULE=lerobot.evaluation.eval_qwen3vl_mot_robodojo_delta
/home/jianf/miniforge3/envs/wam/bin/python \
  /home/jianf/lzl_project/.copilot/session-state/1341fac2-578e-4665-a582-dc35f01c3d5b/files/inspect_robodojo_camera_run.py eval \
  --checkpoint /data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_2nodes_8gpu_high_noise_shift_5_decay_25k/step16k \
  --dataset-root /data2/lizuolei/data/lerobot_data/simulation_data/RoboDojo/data/RoboDojo_ee_lerobot_v30_video \
  --qwen3vl-dir "$CACHE/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$CACHE/pretrained/Cosmos3-Edge" \
  --stage1-text-dir "$CACHE/pretrained/siglip2-base-patch16-224" \
  --task put_bottles_into_dustbin --eval-num 3 --num-envs 1 \
  --gpu 0 --sim-gpu 1 --replan-steps 10 --seed 0 \
  --inference-steps 10 --save-predicted-video \
  --port 19187 --timeout 7200 --output-dir "$RUN"
```

Save actual and predicted three-camera videos and passive camera
snapshots; independently audit delta denormalization/integration, action
exports, physical target tracking, episode completeness and file
fingerprints.

**Verified actual startup:** the12.585B restored policy is executing
`joint_prediction` with three RGB views and10 denoising steps. The first
finite32x20 delta chunk uses full-task inverse normalization and its
anchor exactly matches the measured native camera-state snapshot.
The first three conditioning views were visually checked: the native
four bottles and left-side dustbin are present, wrist views match their
arms, and no black input or obvious camera-role swap is visible.
Policy port19187 is live and native control has advanced beyond the
first chunk. Only task/output/port differ from the prior bowl command.
Native task/layout/asset hashes match preflight. Startup alone did not
establish a success rate; the subsequently completed results and final
audits are recorded below.

**Final result:1/3 successful, success rate33.33%, aggregate score50/100.**
The subsequent completed run preserved the original three layouts and
continuous RNG, with no first-failure stopping or policy restart.
Native wall time was3155.48s (52.59min).

| Layout | Success | Native score | Controls | Visual observation |
| --- | --- | ---: | ---: | --- |
| 0 | No | 0.25 | 700 | Pink/yellow bottles are removed toward the bin; white-bottle manipulation/transfer does not finish, and two bottles remain outside the bin. |
| 1 | Yes | 1.00 | 619 | Sequential bottle pickup/disposal includes a two-arm transfer of the last yellow bottle; final table is clear and wrist view shows bin contents. |
| 2 | No | 0.25 | 700 | Blue/pale-pink bottles are removed; green-bottle transfer/disposal remains unfinished, with green and cream bottles left on the table. |

Reviewed all202 conditioning head images and exact final frames of
all nine actual videos. Native scoring and the visual observations
are consistent: one completed task and two partial-progress episodes.
The same checkpoint can perform pickup, transfer and disposal on this
task; the prior bowl0/3 result does not mean all learned actions are
ineffective. Conversely, these three layouts do not establish this
task's benchmark success rate or broader generalization.

All202 chunks are finite; independent denormalization error is0,
translation integration discrepancy is at most5.97e-8, rotation-matrix
discrepancy at most1.01e-7 and gripper discrepancy below3e-8.
All prediction-export actions exactly match adapter outputs. All source
checkpoint, full-task metadata/parquet and selected native task/layout/
asset SHA256 fingerprints remain unchanged.

The43 passive camera snapshots have zero head/static-world drift,
correct render-product bindings and wrist-to-EE errors below3.2e-7.
Physical target tracking has left/right median XYZ errors0.906/0.146mm
and maxima30.956/11.617mm. Both maxima occur in layout0 at
controls310->320; numerical adapter correctness is not a guarantee of
perfect physical tracking, and the failure cause is not established.
The evaluator exited successfully and owned policy port19187 closed.

Saved202 chunks,606 predicted videos and6666 prediction/conditioning
PNGs. The nine native videos are640x480 at25FPS; layouts0/1/2 have
701/620/701 frames per view, exactly controls+1. Actual video directory:

```text
outputs/evaluation/robodojo/put_bottles_into_dustbin_delta_alltask_shift5_decay25k_step16k_sampling10_20261104_v1/simulator/eval_result/RoboDojo/put_bottles_into_dustbin/Qwen3VLMoT/arx_x5/0_action_type=ee,simulator=isaacsim6/put_bottles_into_dustbin_delta_alltask_shift5_decay25k_step16k_sampling10_20261104_v1/
```

Successful head video: `episode_0000001_cam_head_success.mp4`.
The run root also contains `final_adapter_audit.json`,
`diagnostics/native_final_frames.jpg`, all conditioning-head sheets and
predicted videos under `policy/predictions/`.

## Qwen all-task step16k training-action error diagnostic (2026-11-04)

This diagnostic investigates the same uploaded all-task step16k that scored
0/3 on ordinary bowls. No model parameters or training data are modified.
The existing `diagnose` mode uses the authentic saved training loader and
full-task delta statistics, without random RGB augmentation. All 19 source
files were rehashed against the completed native evaluation's provenance;
all hashes match. The full-task statistics SHA256 is unchanged:
`94d978e4177fa8db22a1a48daedcffda28c8d24a325210a488d4c9006c1778e1`.

The fixed selection has20 bowl windows: full-task episodes3000,3025,3050,
3075,3099, at offsets0,64,128,192. It also retains the eight configured
generation-validation windows (episodes102,609,1663,2017,2339,2970,3174,
3429). All28 windows have32 real, unpadded action targets. Finetuning
retains all3500 episodes, so these are **training examples**, not a
held-out split. They are a diagnostic sample, not the whole training set;
the fixed offsets do not cover every phase of the longer demonstrations.
Representative three-camera/mask/instruction/sample-rate inputs match
deployment exactly; normalized current-state roundtrip errors are below
5e-7.

Run BF16 as in the native evaluation,10-step sampling and a paired20-step
ablation, each with prediction32 and measured prefix10. Seed0 plus sample
index is shared between training/deployment input comparisons and between
the10/20-step samples. Compare free-running action errors against both a
zero-motion hold-current-pose baseline and clean physical-codec
autoencoding. Also report ground-truth-conditioned joint losses at
sigma0.1/0.5/0.9 separately: they are not free-running prediction errors.
Delta targets and predictions are integrated from the same measured
anchor before measuring position/rotation/gripper error.

```bash
RUN=outputs/evaluation/robodojo/alltask_shift5_decay25k_step16k_training_error_sampling10_20261104_v1
CACHE=outputs/evaluation/robodojo/runtime_assets
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  HF_HUB_OFFLINE=1 PYTHONPATH="$PWD" \
  /home/jianf/miniforge3/envs/wam/bin/python \
  -m lerobot.evaluation.eval_qwen3vl_mot_robodojo_delta diagnose \
  --checkpoint /data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_2nodes_8gpu_high_noise_shift_5_decay_25k/step16k \
  --dataset-root /data2/lizuolei/data/lerobot_data/simulation_data/RoboDojo/data/RoboDojo_ee_lerobot_v30_video \
  --qwen3vl-dir "$CACHE/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$CACHE/pretrained/Cosmos3-Edge" \
  --stage1-text-dir "$CACHE/pretrained/siglip2-base-patch16-224" \
  --inference-steps 10 --replan-steps 10 --seed 0 \
  --diagnostic-episodes 3000 3025 3050 3075 3099 \
  --diagnostic-frames 0 64 128 192 --diagnostic-inference-steps 20 \
  --output-dir "$RUN/sampling"
```

Preflight: `$RUN/training_error_preflight.json`. The CLI requires a fresh
output directory, so sampling writes into the new `$RUN/sampling/`
subdirectory. An initial launch stopped at this directory guard before
loading a model; its log is retained separately.

**Verified startup:** the restored policy completed the first real
training window (episode102, global frame59928), including10/20-step
sampling, hold-current and clean-codec baselines, and conditioned losses.
Training-batch versus deployment-input action sampling has exact tensor
parity. The owned policy process was verified live on physical GPU1.

**Completed results:** all28 windows finished; training/deployment action
and future-state sampling tensors match exactly in every window. Source
checkpoint SHA256 and all dataset metadata SHA256 fingerprints match
preflight after sampling. Independent cumulative-translation calculations
also reproduce the reported physical action errors. All28 primary
conditioning images were visually reviewed: the eight configured windows
show their respective tasks; the20 added windows show bowl demonstrations
at multiple phases, with varying bowl appearances and no black primary
images. This visual check does not claim review of all wrist images.
The owned sampling process exited and released its GPU allocation.
Saved `fsdp.fp8=false`; no FP8 transformation was applied.

The following are equal-weight means over the **20 bowl training windows**.
Position errors are predicted versus demonstrated integrated absolute
targets, not simulator target-tracking errors. Prefix10 is the portion
that would be executed before replanning.

| Method | Normalized action MSE,32 steps | Prefix10 XYZ error, left/right cm | All32 XYZ error, left/right cm |
| --- | ---: | --- | --- |
| Free-running,10 denoising steps | 0.20664 | 1.174 / 1.274 | 2.692 / 3.412 |
| Free-running,20 denoising steps | 0.20300 | 1.254 / 1.322 | 2.721 / 3.565 |
| Hold current pose, zero-motion deltas | 0.45826 | 1.989 / 2.735 | 5.784 / 7.578 |
| Clean GT action embedding -> current decoder | 0.10583 | 1.057 / 1.187 | 2.581 / 2.478 |

For10-step sampling, prefix normalized MSE is0.18438, prefix angular
errors average2.555/2.979degrees and clipped gripper MAEs0.00808/0.03090
(native0..1 units). Across the200 prefix positions per arm, XYZ-error
medians are0.320/0.414cm,90th percentiles3.482/3.807cm and maxima
7.005/9.418cm. Thus the mean does not imply every position is wrong by
several centimeters. The largest window-mean prefix XYZ errors are
4.041cm on the left (episode3000,offset128) and3.691cm on the right
(episode3099,offset64).

Sampling beats hold-current prefix normalized MSE on17/20 windows;
prefix XYZ is better on8/20 left and13/20 right windows. Restricting
each arm to windows whose demonstrated mean prefix motion exceeds5mm
leaves9 left and14 right windows. Mean XYZ errors are then
**2.373/1.730cm**, versus hold-current4.419/3.907cm. This supports
learning useful motion while retaining meaningful training-example
deviations; it is not evidence that the policy is simply motionless.
Initial bowl windows (five demonstrations) have prefix XYZ errors
0.510/2.086cm; the15 noninitial windows have1.396/1.004cm.

The eight configured training-validation windows are **other tasks**,
not extra bowl demonstrations. Their mean action MSE is0.40133 and
prefix XYZ0.520/1.028cm at10 steps, versus hold-current
MSE1.31112 and XYZ1.408/2.228cm. Do not mix their MSE with the bowl
subset or label them a35-task benchmark.

**Interpretation and limits.** Doubling sampler steps did not reduce
mean prefix physical error on this paired selection. More importantly,
clean ground-truth actions passed through the frozen physical encoder
and the current action decoder already have centimeter-scale integrated
reconstruction error. This decoder is the separate
`physical_decoders["action"]` head; `physical_tuning_mode="frozen"`
freezes the encoder, not this head. The clean-codec probe does not run
the denoising model, but its result is **not a proven irreducible error
floor**, nor proof that all free-running or native-rollout errors come
from the codec.

At sigma0.5, the bowl windows' mean joint ground-truth-conditioned action
reconstruction loss is0.01537 and embedding flow loss0.92210. These are
different objectives/conditioning from free-running action MSE0.20664;
a small training reconstruction number does not demonstrate accurate
generation from noise. These offline deviations exist with genuine
training images/states and no simulator, but alternative valid actions,
single-seed sampling, limited windows and closed-loop distribution shifts
prevent assigning the native0/3 result to one cause.

Artifacts in the run root:
`sampling/diagnostics.json`, per-sample `sampling/sample_*/actions.npz`
and `metrics.json`, `training_error_summary.json`,
`training_conditioning_sheet.jpg`, `diagnostic.log` and `summary.log`.
The summary separates bowl/other-task, initial/noninitial, moving-arm
groups, normalized components, means/medians/tails and baselines.

## Qwen all-task shift5 decay25k step16k ordinary bowls (2026-11-04)

Evaluate the user's newly uploaded full-task checkpoint:
`/data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_2nodes_8gpu_high_noise_shift_5_decay_25k/step16k`.
This is the same all-task experiment as the evaluated step12k, not a
bowl-subset shift5 or pi05 checkpoint. Run all three ordinary `stack_bowls`
layouts0/1/2, even if the first fails, with BF16, seed0, one environment,
sampling10, prediction32/execution10 and continuous policy RNG.

Fresh preflight verified all 19 source files, 16 DCP shards,
25,200,777,551 bytes, 2560 tensors and 16,908 valid storage references.
All 16 shard hashes differ from the evaluated step12k, while saved policy
and training-config hashes match exactly. Policy equals saved training
policy: finetuning on `robodojo`, 1,856,102 frames, shifted_uniform shift5,
LR decay25k; saved beta fields are inactive. Size/mtime stayed stable and
the verified source is loaded directly without a full checkpoint copy.

Full-task dataset:
`/data2/lizuolei/data/lerobot_data/simulation_data/RoboDojo/data/RoboDojo_ee_lerobot_v30_video`.
Reverified all 1,856,102 rows and 3500 episode boundaries against the
measured-state delta contract; maximum discrepancy was
`5.960464477539063e-8`. State/delta means and standard deviations match
the saved statistics. Use these full-task statistics, not bowl-subset
statistics. SHA256 matches the evaluated step12k:
`94d978e4177fa8db22a1a48daedcffda28c8d24a325210a488d4c9006c1778e1`.
Checkpoint and dataset fingerprints are in:
`outputs/evaluation/robodojo/runtime_assets/robodojo_2nodes_8gpu_high_noise_shift_5_decay_25k/checkpoint_provenance_step16k.json`.

The instrumented evaluation command is:

```bash
conda activate wam
cd /home/jianf/lzl_project/gcr_latent_action
unset ROBODOJO_STOP_AFTER_FIRST_FAILURE
export PYTHONPATH="$PWD" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export OMNI_KIT_ACCEPT_EULA=YES HF_HUB_OFFLINE=1
export ROBODOJO_EVAL_MODULE=lerobot.evaluation.eval_qwen3vl_mot_robodojo_delta
CACHE="$PWD/outputs/evaluation/robodojo/runtime_assets"
ENTRY=/home/jianf/lzl_project/.copilot/session-state/1341fac2-578e-4665-a582-dc35f01c3d5b/files/inspect_robodojo_camera_run.py
python "$ENTRY" eval \
  --checkpoint /data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_2nodes_8gpu_high_noise_shift_5_decay_25k/step16k \
  --dataset-root /data2/lizuolei/data/lerobot_data/simulation_data/RoboDojo/data/RoboDojo_ee_lerobot_v30_video \
  --qwen3vl-dir "$CACHE/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$CACHE/pretrained/Cosmos3-Edge" \
  --stage1-text-dir "$CACHE/pretrained/siglip2-base-patch16-224" \
  --task stack_bowls --eval-num 3 --num-envs 1 \
  --gpu 0 --sim-gpu 1 --replan-steps 10 --seed 0 \
  --inference-steps 10 --save-predicted-video --port 19186 --timeout 7200 \
  --output-dir outputs/evaluation/robodojo/stack_bowls_delta_alltask_shift5_decay25k_step16k_sampling10_20261104_v1
```

Save actual/future videos and passive camera observations. Compare with
the same all-task experiment's step12k: 0/3 successful, scores0/0/0,
800 controls each. Three fixed layouts are a checkpoint diagnostic,
not the full benchmark or a causal inference-speed comparison.
Preparation is verified; actual native startup and final results are pending.

The actual native rollout subsequently started successfully. The loader
reported 12.585B parameters, joint_prediction, all three RGB views and
ten denoising steps. Independent first-chunk checks verified finite
`[32,20]` deltas, full-task inverse normalization and a measured-state
integration anchor exactly matching the native camera snapshot.
Layout0's initial state matches the evaluated all-task step12k exactly;
only checkpoint, output and port differ in the CLI settings.
Primary and both wrist conditioning images were visually reviewed and
show the expected bowls, grippers and camera roles without black input
or obvious role swap. Owned policy port19186 is reachable and native
inference is active. Startup report:
`outputs/evaluation/robodojo/stack_bowls_delta_alltask_shift5_decay25k_step16k_sampling10_20261104_v1/startup_validation.json`.
Successful startup alone was not a success-rate result. The subsequent
completed run and final audits are recorded below.

**Final result: 0/3 successful, scores0/0/0, aggregate0/100.**
All three layouts reached 800 controls without a policy restart or
first-failure stopping. Native wall time was2607.11s (43.45min).
On these same three layouts, step16k does not improve the success rate
or score over this all-task experiment's step12k (also0/3, scores0/0/0).
This is not a full-benchmark or generalization conclusion.

| Layout | Success | Score | Controls | Visual observation |
| --- | --- | --- | --- | --- |
| 0 | No | 0 | 800 | Left arm contacts/repositions the left bowl, then remains nearby; bowls stay separate. |
| 1 | No | 0 | 800 | Right bowl is brought next to the middle bowl, but no retained nested pair forms; final bowls are separate. |
| 2 | No | 0 | 800 | Right arm contacts/repositions the right bowl; no valid nesting and three separate final bowls. |

Reviewed all 240 conditioning head images and exact final frames from
all nine actual videos. Camera/input sanity passed. All three initial
states match the evaluated all-task step12k exactly. The 51 passive
camera snapshots show zero head/static-scene drift and wrist-to-EE
transform errors below1e-6; render bindings match their camera prims.

All 240 chunks are finite; full-task delta denormalization error is
zero, independent integration discrepancies are below1.1e-7, and all
prediction-export actions exactly match adapter outputs. All source
checkpoint and full-task dataset/metadata SHA256 fingerprints remain
unchanged. Physical tracking errors (previous chunk step9 target versus
next measured anchor) have left/right medians0.181/0.191mm and
maxima51.785/5.327mm. **The left maximum occurs in layout1 at
controls490->500; passing numerical adapter checks does not mean
perfect physical tracking.** The cause and its contribution to failure
are not established. The run exited successfully and owned port19186 closed.

Artifacts: 720 predicted videos, 7920 prediction/conditioning PNGs and
nine actual videos, each801frames,640x480,25FPS. Final audit:
`outputs/evaluation/robodojo/stack_bowls_delta_alltask_shift5_decay25k_step16k_sampling10_20261104_v1/final_adapter_audit.json`.
Actual videos:

```text
outputs/evaluation/robodojo/stack_bowls_delta_alltask_shift5_decay25k_step16k_sampling10_20261104_v1/simulator/eval_result/RoboDojo/stack_bowls/Qwen3VLMoT/arx_x5/0_action_type=ee,simulator=isaacsim6/stack_bowls_delta_alltask_shift5_decay25k_step16k_sampling10_20261104_v1/
```

Predicted videos are in the same run root's `policy/predictions/`.
Final nine-view sheet: `diagnostics/native_final_frames.jpg`.

## Qwen all-task shift5 decay25k step12k ordinary bowls (2026-11-03)

Requested full-task checkpoint:
`/data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_2nodes_8gpu_high_noise_shift_5_decay_25k/step12k`.
The actual experiment directory ends in `decay_25k`, not `decay_25`.
Saved policy and training policy match, declaring finetuning on `robodojo`,
1,856,102 frames, shifted_uniform shift5 and LR decay25k. Saved beta fields
are inactive. Evaluate ordinary `stack_bowls` layouts0/1/2 once each, with
BF16, seed0, one environment, sampling10, prediction32/execution10 and one
uninterrupted policy RNG. Do not stop after a failed first episode.

The user confirmed **delta actions** and the full-task dataset:
`/data2/lizuolei/data/lerobot_data/simulation_data/RoboDojo/data/RoboDojo_ee_lerobot_v30_video`.
Its newly synchronized metadata and parquet now contain the 20D
`delta_action.ee_ort6d_pos` feature and the measured-next-state increment
contract. Full-task statistics SHA256:
`94d978e4177fa8db22a1a48daedcffda28c8d24a325210a488d4c9006c1778e1`.
**Use its own absolute-state and delta-action statistics, never the
44,774-frame bowl-subset statistics.** Do not infer absolute semantics from
`data_mix=robodojo`: both older absolute and newer full-task delta recipes
use that name. Explicitly select the action mode matching training.

The evaluator now accepts `data_mix=robodojo` with `--action-mode delta`
only when the dataset has the verified delta contract, 20D delta feature,
usable state/delta statistics, and the saved training frame count.
Existing absolute and bowl-subset delta behavior is preserved. Regression
validation: 46 evaluator tests passed; Ruff lint and format checks passed.
Fresh preflight verified all 19 checkpoint files, 16 DCP shards,
25,200,777,551 bytes, 2560 tensors and 16,908 valid storage references.
Source size/mtime remained unchanged; no full-size checkpoint duplicate
was made. Independently checked all 1,856,102 full-task rows and all 3500
episode boundaries against measured-state delta integration, including
identity terminal rotations; maximum delta discrepancy was
`5.960464477539063e-8`. Recomputed state/delta means and standard
deviations match the saved full-task statistics (preserved state uses
sample std, new delta uses population std). Metadata and all data parquet
SHA256 fingerprints are retained alongside checkpoint fingerprints:
`outputs/evaluation/robodojo/runtime_assets/robodojo_2nodes_8gpu_high_noise_shift_5_decay_25k/checkpoint_provenance_step12k.json`.
Preparation is verified; actual native startup and final results are pending.

The actual instrumented command was:

```bash
conda activate wam
cd /home/jianf/lzl_project/gcr_latent_action
unset ROBODOJO_STOP_AFTER_FIRST_FAILURE
export PYTHONPATH="$PWD" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export OMNI_KIT_ACCEPT_EULA=YES HF_HUB_OFFLINE=1
export ROBODOJO_EVAL_MODULE=lerobot.evaluation.eval_qwen3vl_mot_robodojo_delta
CACHE="$PWD/outputs/evaluation/robodojo/runtime_assets"
ENTRY=/home/jianf/lzl_project/.copilot/session-state/1341fac2-578e-4665-a582-dc35f01c3d5b/files/inspect_robodojo_camera_run.py
python "$ENTRY" eval \
  --checkpoint /data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_2nodes_8gpu_high_noise_shift_5_decay_25k/step12k \
  --dataset-root /data2/lizuolei/data/lerobot_data/simulation_data/RoboDojo/data/RoboDojo_ee_lerobot_v30_video \
  --qwen3vl-dir "$CACHE/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$CACHE/pretrained/Cosmos3-Edge" \
  --stage1-text-dir "$CACHE/pretrained/siglip2-base-patch16-224" \
  --task stack_bowls --eval-num 3 --num-envs 1 \
  --gpu 0 --sim-gpu 1 --replan-steps 10 --seed 0 \
  --inference-steps 10 --save-predicted-video --port 19185 --timeout 7200 \
  --output-dir outputs/evaluation/robodojo/stack_bowls_delta_alltask_shift5_decay25k_step12k_sampling10_20261103_v1
```

Save actual/future videos and passive camera observations. The bowl-subset
shift5_1102 step12k run is only an initial-layout/evaluation-protocol
reference; training data, normalization and LR schedule differ. Three
fixed layouts do not estimate full-benchmark performance or establish a
causal effect of full-task finetuning.

The actual native rollout started successfully. The first finite
`[32,20]` delta chunk executed ten controls; independent denormalization
uses the verified full-task statistics, and its integration anchor
exactly matches the native camera snapshot's measured state. Layout0's
initial state is bitwise identical to the bowl-subset reference.
The only CLI differences are checkpoint, dataset root, output and port.
Primary/left-wrist/right-wrist conditioning images were visually checked:
all three bowls and the expected robot/camera roles are present, with no
black input or obvious role swap. Policy port19185 is reachable and
inference is active. Startup report:
`outputs/evaluation/robodojo/stack_bowls_delta_alltask_shift5_decay25k_step12k_sampling10_20261103_v1/startup_validation.json`.
Startup alone was not a success-rate result; the subsequently completed
three-episode evaluation and final audits are recorded below.

**Final result: 0/3 successful, scores0/0/0, aggregate0/100.**
All three ordinary layouts reached 800 controls, with no first-failure
stopping or policy restart. Native wall time was2439.76s (40.66min);
shared-machine timing is not a causal inference-speed comparison.

| Layout | Success | Score | Controls | Visual observation |
| --- | --- | --- | --- | --- |
| 0 | No | 0 | 800 | Left arm contacts the left bowl, then stays around it; all bowls remain separate. |
| 1 | No | 0 | 800 | Arms approach the two outer bowls; no sustained transfer or valid nesting. |
| 2 | No | 0 | 800 | Left arm contacts/repositions the left bowl; three bowls remain separate. |

Visual review covered all 240 conditioning head images and exact final
decoded frames from all nine native videos. No completed stack or valid
retained pair was observed. Input/camera sanity passed: 51 passive
snapshots, zero head/static-scene drift, wrist-to-EE transform errors
below1e-6, and render bindings matching their camera prims.
Initial measured states match the bowl-subset reference in all three
layouts. The reference shift5_1102 step12k scored1/0/0.15 with1/3 success;
this full-task checkpoint performed worse on these specific three
layouts, not evidence that full-task finetuning generally hurts.

All 240 chunks are finite, full-task delta denormalization error is zero,
independent integration discrepancies are below9e-8, and all saved
prediction actions exactly match adapter outputs. Checkpoint and full-task
data/metadata SHA256 fingerprints remained unchanged. Physical target
tracking position errors (previous chunk step9 target versus next measured
anchor) had left/right medians0.284/0.256mm and maxima8.588/15.986mm.
The native process exited successfully and owned policy port19185 closed.
Final audited report:
`outputs/evaluation/robodojo/stack_bowls_delta_alltask_shift5_decay25k_step12k_sampling10_20261103_v1/final_adapter_audit.json`.

Artifacts: 720 predicted videos, 7920 prediction/conditioning PNGs and
nine actual videos, each801frames,640x480,25FPS. Actual videos:

```text
outputs/evaluation/robodojo/stack_bowls_delta_alltask_shift5_decay25k_step12k_sampling10_20261103_v1/simulator/eval_result/RoboDojo/stack_bowls/Qwen3VLMoT/arx_x5/0_action_type=ee,simulator=isaacsim6/stack_bowls_delta_alltask_shift5_decay25k_step12k_sampling10_20261103_v1/
```

Predicted videos are under the same run root's `policy/predictions/`.
Final nine-view sheet: `diagnostics/native_final_frames.jpg`.

## Qwen shift5_1102 step16k ordinary bowls (2026-11-02)

Evaluate the same newly supplied shift5_1102 experiment's checkpoint
`/data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_stack_bowl_2nodes_8gpu_decay_20k_high_noise_shift_5_1102/step16k`
on ordinary `stack_bowls`, layouts0/1/2 once each. **Run all three episodes
regardless of the first outcome.** Use the same BF16, seed0, one environment,
sampling10, prediction32/execution10, continuous policy RNG, actual/future
videos and passive camera audits as this experiment's step12k.

Fresh preflight verified all 19 files, 16 DCP shards, 25,200,777,652 bytes,
2560 tensors and 16,908 valid storage references. All 16 shard hashes differ
from step12k, while saved policy/training config hashes match exactly.
Source size/mtime remained stable during verification. The policy equals
the saved training policy: shifted_uniform, shift5, finetuning on
`delta_robodojo_stack_bowl`, 44,774 frames. Saved beta fields are inactive.
The matching bowl-delta statistics SHA256 remains
`cd7e1005b9f99678f036d24bb07626140f3771f2da36b6d15bc57a1887605e2a`.
No full checkpoint duplicate is made. Manifest:
`outputs/evaluation/robodojo/runtime_assets/robodojo_stack_bowl_2nodes_8gpu_decay_20k_high_noise_shift_5_1102/checkpoint_provenance_step16k.json`.

The actual instrumented command is session-specific:

```bash
conda activate wam
cd /home/jianf/lzl_project/gcr_latent_action
unset ROBODOJO_STOP_AFTER_FIRST_FAILURE
export PYTHONPATH="$PWD" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export OMNI_KIT_ACCEPT_EULA=YES HF_HUB_OFFLINE=1
export ROBODOJO_EVAL_MODULE=lerobot.evaluation.eval_qwen3vl_mot_robodojo_delta
CACHE="$PWD/outputs/evaluation/robodojo/runtime_assets"
ENTRY=/home/jianf/lzl_project/.copilot/session-state/1341fac2-578e-4665-a582-dc35f01c3d5b/files/inspect_robodojo_camera_run.py
python "$ENTRY" eval \
  --checkpoint /data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_stack_bowl_2nodes_8gpu_decay_20k_high_noise_shift_5_1102/step16k \
  --qwen3vl-dir "$CACHE/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$CACHE/pretrained/Cosmos3-Edge" \
  --stage1-text-dir "$CACHE/pretrained/siglip2-base-patch16-224" \
  --task stack_bowls --eval-num 3 --num-envs 1 \
  --gpu 0 --sim-gpu 1 --replan-steps 10 --seed 0 \
  --inference-steps 10 --save-predicted-video --port 19184 --timeout 7200 \
  --output-dir outputs/evaluation/robodojo/stack_bowls_delta_shift5_1102_step16k_sampling10_20261102_v1
```

Final validation covers all three native results, unchanged source SHA256
fingerprints, numerical/action parity, cameras, videos and visual review.
Compare against the same experiment's step12k (1/3 success, scores1/0/0.15,
aggregate38.33/100). This is a three-layout checkpoint diagnostic, not a
full-benchmark estimate; shared load prevents causal timing comparisons.
Preparation is verified; startup and results are pending.

The actual native rollout subsequently started successfully. The loader
explicitly restores `high_noise_shift_5_1102/step16k/model`, reporting
12.585B parameters, BF16, joint prediction and ten denoising steps.
First actual inference took 6.723 seconds, with matched prediction export
13.403 seconds. The `(32,20)` action chunk is finite and configured for
ten-action execution; the policy service is responsive on port19184.

Saved evaluation settings differ from this experiment's step12k only in
checkpoint, output directory and port; layout0's initial measured state
matches exactly. All three actual conditioning images were visually
checked: ordinary white bowls/wooden table and expected
head/left-wrist/right-wrist roles. Actual predictions and passive camera
observations are saved. There is no first-failure gate marker: all three
layouts remain requested regardless of layout0's outcome. Startup evidence:
`startup_validation.json`. Preparation commit: `b9056c8`.
Evaluation is running; final success counts are not yet available.

The shift5_1102 step16k run subsequently completed: **1/3 success**,
native scores **1.0/0.0/0.0**, aggregate **33.33/100**. Layout0 succeeded
at 490 control steps; layouts1/2 reached 800 steps. All three requested
layouts ran in one uninterrupted policy invocation.

Visual review covered all conditioning-head contact sheets, actual
head-video timelines and exact final frames from all nine actual videos.
In layout0, the right arm forms a pair, then the left arm places the
third bowl, releases and withdraws; the exact final native video frame
shows the completed three-bowl stack. Layout1 brings the bowls together
without valid nesting, followed by repeated contact/repositioning and
tipping. Layout2 initially brings two bowls into overlapping contact,
but does not retain a valid pair; the final bowls are separate and tilted.

Compared with this experiment's step12k, full success remains 1/3 on the
same layout0. Layout0 takes 490 rather than 467 control steps, and layout2
loses its previous 0.15 partial credit, reducing the aggregate from
38.33 to 33.33/100. There is no improvement on these three layouts;
this is not a full-benchmark estimate or statistical ranking.

Wall time was 3926.61 seconds (65.44 minutes), including startup.
The run saved 209 action chunks, 627 predicted videos, 6897 prediction
PNGs and nine actual rollout videos. Each actual video is 640x480 at
25 FPS: 491 frames for layout0, 801 for layouts1/2. Mean inference/export
time was 7.002/5.583 seconds per chunk; shared load and slower startup
prevent attributing wall-time differences to checkpoint weights.

All numerical/independent-integration/export-parity, camera, video and
post-run source-SHA256 checks passed. All three initial measured states
match step12k exactly. Across 44 passive camera snapshots, head and
static-scene matrix drift were zero; wrist transforms/render bindings
were consistent. No episode was flagged physically unstable.
Left/right target tracking medians were 0.160/0.083 mm, maxima
2.263/6.929 mm. These checks do not imply grasping or task success.

Artifacts:

```text
outputs/evaluation/robodojo/stack_bowls_delta_shift5_1102_step16k_sampling10_20261102_v1/
  summary.json
  final_adapter_audit.json
  camera_audit/camera_integrity_report.json
  diagnostics/native_final_frames.jpg
  policy/predictions/
```

Actual videos are below the same root:

```text
simulator/eval_result/RoboDojo/stack_bowls/Qwen3VLMoT/arx_x5/0_action_type=ee,simulator=isaacsim6/stack_bowls_delta_shift5_1102_step16k_sampling10_20261102_v1/
```

All three requested episodes and final audits are complete, not the
full benchmark. The owned policy service shut down; port19184 is closed.
Startup commit: `e1ef3e7`.

## Qwen shift5_1102 step12k ordinary bowls (2026-11-02)

Evaluate the newly supplied checkpoint
`/data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_stack_bowl_2nodes_8gpu_decay_20k_high_noise_shift_5_1102/step12k`
on ordinary `stack_bowls`, layouts0/1/2 once each. **Run all three
episodes regardless of the first outcome.** Use the same BF16, seed0,
one environment, sampling10, prediction32/execution10, continuous policy
RNG, actual/future videos and passive camera audits as previous runs.

Fresh preflight verified all 19 files, 16 DCP shards, 25,200,777,652 bytes,
2560 tensors and 16,908 valid storage references. All tensor keys, shapes
and dtypes match the previously evaluated architecture, and source
size/mtime remained stable during verification. All 16 shard hashes
differ from Beta025 step12k. The policy equals the saved training policy,
finetuned on `delta_robodojo_stack_bowl`, 44,774 frames.
The saved noise recipe is **shifted_uniform, shift5**, not Beta025:
saved beta alpha/beta/probability fields are inactive in this mode.
The matching bowl-delta statistics SHA256 remains
`cd7e1005b9f99678f036d24bb07626140f3771f2da36b6d15bc57a1887605e2a`.
No full checkpoint duplicate is made. Manifest:
`outputs/evaluation/robodojo/runtime_assets/robodojo_stack_bowl_2nodes_8gpu_decay_20k_high_noise_shift_5_1102/checkpoint_provenance_step12k.json`.

The actual instrumented command is session-specific:

```bash
conda activate wam
cd /home/jianf/lzl_project/gcr_latent_action
unset ROBODOJO_STOP_AFTER_FIRST_FAILURE
export PYTHONPATH="$PWD" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export OMNI_KIT_ACCEPT_EULA=YES HF_HUB_OFFLINE=1
export ROBODOJO_EVAL_MODULE=lerobot.evaluation.eval_qwen3vl_mot_robodojo_delta
CACHE="$PWD/outputs/evaluation/robodojo/runtime_assets"
ENTRY=/home/jianf/lzl_project/.copilot/session-state/1341fac2-578e-4665-a582-dc35f01c3d5b/files/inspect_robodojo_camera_run.py
python "$ENTRY" eval \
  --checkpoint /data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_stack_bowl_2nodes_8gpu_decay_20k_high_noise_shift_5_1102/step12k \
  --qwen3vl-dir "$CACHE/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$CACHE/pretrained/Cosmos3-Edge" \
  --stage1-text-dir "$CACHE/pretrained/siglip2-base-patch16-224" \
  --task stack_bowls --eval-num 3 --num-envs 1 \
  --gpu 0 --sim-gpu 1 --replan-steps 10 --seed 0 \
  --inference-steps 10 --save-predicted-video --port 19182 --timeout 7200 \
  --output-dir outputs/evaluation/robodojo/stack_bowls_delta_shift5_1102_step12k_sampling10_20261102_v1
```

Final validation covers all three native results, unchanged source
SHA256 fingerprints, numerical/action parity, cameras, videos and visual
review. Beta025 step12k is the settings/initial-state reference, not an
isolated causal training-noise comparison or full-benchmark estimate.
Shared load prevents causal timing comparisons.
Preparation is verified; startup and results are pending.

The actual native rollout subsequently started successfully. The loader
explicitly restores `high_noise_shift_5_1102/step12k/model`, reporting
12.585B parameters, BF16, joint prediction and ten denoising steps.
First actual inference took 6.723 seconds, with matched prediction export
5.938 seconds. The `(32,20)` action chunk is finite and configured for
ten-action execution; the policy service is responsive on port19182.

Saved evaluation settings differ from Beta025 step12k only in checkpoint,
output directory and port; layout0's initial measured state matches
exactly. All three actual conditioning images were visually checked:
ordinary white bowls/wooden table and expected head/left-wrist/right-wrist
roles. Actual predictions and passive camera observations are saved.
There is no first-failure gate marker: all three layouts remain requested
regardless of layout0's outcome. Startup evidence:
`startup_validation.json`. Preparation commit: `d54bd25`.
Evaluation is running; final success counts are not yet available.

The shift5_1102 step12k run subsequently completed: **1/3 success**,
native scores **1.0/0.0/0.15**, aggregate **38.33/100**. Layout0 succeeded
at 467 control steps; layouts1/2 reached 800 steps. All three requested
layouts ran in one uninterrupted policy invocation.

Visual review covered all conditioning-head contact sheets, actual
head-video timelines and exact final frames from all nine actual videos.
In layout0, the right arm forms a pair, then the left arm places the
third bowl, releases and withdraws; the exact final native video frame
shows the completed three-bowl stack. Layout1 brings the bowls alongside
one another but never obtains valid nesting despite later contact and
repositioning. Layout2 retains a two-bowl partial stack, with the remaining
bowl beside it rather than successfully added.

Wall time was 2964.02 seconds (49.40 minutes), including startup.
The run saved 207 action chunks, 621 predicted videos, 6831 prediction
PNGs and nine actual rollout videos. Each actual video is 640x480 at
25 FPS: 468 frames for layout0, 801 for layouts1/2. Mean inference/export
time was 5.391/4.475 seconds per chunk; shared load prevents causal
timing comparisons.

All numerical/independent-integration/export-parity, camera, video and
post-run source-SHA256 checks passed. All three initial measured states
match the reference exactly. Across 44 passive camera snapshots, head
and static-scene matrix drift were zero; wrist transforms/render bindings
were consistent. No episode was flagged physically unstable.
Left/right target tracking medians were 0.242/0.212 mm, maxima
12.411/8.091 mm. These checks do not imply perfect physical following.

For context, Beta025 step12k and step17k each scored 0/3 with
0.15/0/0.15 (10/100) on these layouts. This new experiment improves
layout0 to a full success, while the other layout scores remain the same.
Different experiments and only three layouts do not establish that the
training noise distribution caused the improvement.

Artifacts:

```text
outputs/evaluation/robodojo/stack_bowls_delta_shift5_1102_step12k_sampling10_20261102_v1/
  summary.json
  final_adapter_audit.json
  camera_audit/camera_integrity_report.json
  diagnostics/native_final_frames.jpg
  policy/predictions/
```

Actual videos are below the same root:

```text
simulator/eval_result/RoboDojo/stack_bowls/Qwen3VLMoT/arx_x5/0_action_type=ee,simulator=isaacsim6/stack_bowls_delta_shift5_1102_step12k_sampling10_20261102_v1/
```

All three requested episodes and final audits are complete, not the
full benchmark. The owned policy service shut down; port19182 is closed.
Startup commit: `d2ae5b6`.

## Qwen Beta(0.25,0.25) step17k ordinary bowls (2026-11-01)

Evaluate the same Beta025 experiment's checkpoint
`/data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_stack_bowl_2nodes_8gpu_decay_20k_beta_0.25/step17k`
on ordinary `stack_bowls`, layouts0/1/2 once each. **Run all three
episodes regardless of the first outcome.** Use the same BF16, seed0,
one environment, sampling10, prediction32/execution10, continuous policy
RNG, actual/future videos and passive camera audits as step12k.

Fresh preflight verified all 19 files, 16 DCP shards, 25,200,777,632 bytes,
2560 tensors and 16,908 valid storage references. All 16 shard hashes
differ from step12k, while saved policy/training config hashes match.
The policy equals the saved training policy: pure Beta(0.25,0.25),
finetuning on `delta_robodojo_stack_bowl`, 44,774 frames. The matching
bowl-delta statistics SHA256 remains
`cd7e1005b9f99678f036d24bb07626140f3771f2da36b6d15bc57a1887605e2a`.
No full checkpoint duplicate is made. Manifest:
`outputs/evaluation/robodojo/runtime_assets/robodojo_stack_bowl_2nodes_8gpu_decay_20k_beta_0.25/checkpoint_provenance_step17k.json`.

The actual instrumented command is session-specific:

```bash
conda activate wam
cd /home/jianf/lzl_project/gcr_latent_action
unset ROBODOJO_STOP_AFTER_FIRST_FAILURE
export PYTHONPATH="$PWD" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export OMNI_KIT_ACCEPT_EULA=YES HF_HUB_OFFLINE=1
export ROBODOJO_EVAL_MODULE=lerobot.evaluation.eval_qwen3vl_mot_robodojo_delta
CACHE="$PWD/outputs/evaluation/robodojo/runtime_assets"
ENTRY=/home/jianf/lzl_project/.copilot/session-state/1341fac2-578e-4665-a582-dc35f01c3d5b/files/inspect_robodojo_camera_run.py
python "$ENTRY" eval \
  --checkpoint /data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_stack_bowl_2nodes_8gpu_decay_20k_beta_0.25/step17k \
  --qwen3vl-dir "$CACHE/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$CACHE/pretrained/Cosmos3-Edge" \
  --stage1-text-dir "$CACHE/pretrained/siglip2-base-patch16-224" \
  --task stack_bowls --eval-num 3 --num-envs 1 \
  --gpu 0 --sim-gpu 1 --replan-steps 10 --seed 0 \
  --inference-steps 10 --save-predicted-video --port 19180 --timeout 7200 \
  --output-dir outputs/evaluation/robodojo/stack_bowls_delta_beta025_step17k_sampling10_20261101_v1
```

Final validation covers all three native results, unchanged source
SHA256 fingerprints, numerical/action parity, cameras, videos and visual
review. Compare against the same experiment's step12k (0/3 success,
scores0.15/0/0.15, aggregate10/100), not the user-reported problematic
step16k. This is a three-layout checkpoint diagnostic, not a full-benchmark
estimate; shared load prevents causal timing comparisons.
Preparation is verified; startup and results are pending.

The actual native rollout subsequently started successfully. The loader
explicitly restores `beta_0.25/step17k/model`, reporting 12.585B parameters,
BF16, joint prediction and ten denoising steps. First actual inference
took 6.390 seconds, with matched prediction export 5.441 seconds.
The `(32,20)` action chunk is finite and configured for ten-action
execution; the policy service is responsive on port19180.

Saved evaluation settings differ from Beta025 step12k only in checkpoint,
output directory and port; layout0's initial measured state matches
exactly. All three actual conditioning images were visually checked:
ordinary white bowls/wooden table and expected head/left-wrist/right-wrist
roles. Actual predictions and passive camera observations are saved.
There is no first-failure gate marker: all three layouts remain requested
regardless of layout0's outcome. Startup evidence:
`startup_validation.json`. Preparation commit: `d5bddac`.
Evaluation is running; final success counts are not yet available.

The step17k run subsequently completed: **0/3 success**, native scores
**0.15/0.0/0.15**, aggregate **10/100**, all three episodes reaching
800 control steps. These outcomes match step12k on the same layouts:
there is no success/score improvement in this three-layout diagnostic.
No first-failure stopping occurred.

Visual review covered all conditioning-head contact sheets, actual
head-video timelines and exact final frames from all nine actual videos.
Layout0 forms a partial pair; the left arm attempts the remaining-bowl
placement without completing the task. Late arm motions retreat outside
the head view; the final head image has only one bowl-shaped silhouette,
not evidence of a complete three-bowl stack. Layout1 brings the bowls
alongside one another but never obtains valid nesting. Layout2 retains a
two-bowl partial stack while the left arm stalls near the remaining bowl.
Partial credit must not be reported as task success.

Wall time was 3007.72 seconds (50.13 minutes), with 240 action chunks,
720 predicted videos, 7920 prediction PNGs and nine actual rollout videos.
Each actual video is 640x480, 25 FPS, 801 frames. Mean inference/export
time was 5.051/4.199 seconds per chunk; shared load prevents causal
timing comparisons.

All numerical/independent-integration/export-parity, camera, video and
post-run source-SHA256 checks passed. All three initial measured states
match step12k exactly. The 51 passive camera snapshots show zero head or
static-scene matrix drift; wrist render bindings/transforms are consistent.
No episode was flagged physically unstable.

**Physical tracking has material outliers despite these checks.**
Left/right median target-to-measured position errors are 0.188/0.441 mm,
but maxima are 228.904/255.795 mm, both in layout0: left670->680,
right610->620. Step12k maxima were only 4.871/8.505 mm. The independent
adapter and camera audits do not certify accurate physical following;
the large errors' cause is not established by this evaluation.

Artifacts:

```text
outputs/evaluation/robodojo/stack_bowls_delta_beta025_step17k_sampling10_20261101_v1/
  summary.json
  final_adapter_audit.json
  worst_target_tracking.json
  camera_audit/camera_integrity_report.json
  diagnostics/native_final_frames.jpg
  policy/predictions/
```

Actual videos are below the same root:

```text
simulator/eval_result/RoboDojo/stack_bowls/Qwen3VLMoT/arx_x5/0_action_type=ee,simulator=isaacsim6/stack_bowls_delta_beta025_step17k_sampling10_20261101_v1/
```

All three requested episodes and final audits are complete, not the
full benchmark. The owned policy service shut down; port19180 is closed.
Startup commit: `2645db1`.

## Qwen Beta(0.25,0.25) step12k ordinary bowls (2026-10-31)

The user corrected the requested checkpoint to **step12k**, reporting a
problem with step16k. Retain the previous step16k artifacts as historical
evidence, not as the result of this corrected request; the reported
step16k problem has not been independently diagnosed.

Evaluate
`/data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_stack_bowl_2nodes_8gpu_decay_20k_beta_0.25/step12k`
on ordinary `stack_bowls`, layouts0/1/2 once each, **all three episodes
regardless of the first outcome**. Fresh preflight verified all 19 files,
16 DCP shards, 25,200,777,632 bytes, 2560 tensors and 16,908 valid storage
references. All 16 shard hashes differ from step16k. Policy equals the
saved training policy: pure Beta(0.25,0.25), finetuning on
`delta_robodojo_stack_bowl`, 44,774 frames. The existing matching
bowl-delta statistics SHA256 is
`cd7e1005b9f99678f036d24bb07626140f3771f2da36b6d15bc57a1887605e2a`.
No full checkpoint duplicate is made. Manifest:
`outputs/evaluation/robodojo/runtime_assets/robodojo_stack_bowl_2nodes_8gpu_decay_20k_beta_0.25/checkpoint_provenance_step12k.json`.

Use the same BF16, seed0, one environment, sampling10,
prediction32/execution10, continuous policy RNG, actual/future videos and
passive camera audits as previous ordinary runs. Beta05 step17k remains
the initial-state/settings reference, not an isolated noise-distribution
comparison or full-benchmark estimate.

The actual instrumented command is session-specific:

```bash
conda activate wam
cd /home/jianf/lzl_project/gcr_latent_action
unset ROBODOJO_STOP_AFTER_FIRST_FAILURE
export PYTHONPATH="$PWD" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export OMNI_KIT_ACCEPT_EULA=YES HF_HUB_OFFLINE=1
export ROBODOJO_EVAL_MODULE=lerobot.evaluation.eval_qwen3vl_mot_robodojo_delta
CACHE="$PWD/outputs/evaluation/robodojo/runtime_assets"
ENTRY=/home/jianf/lzl_project/.copilot/session-state/1341fac2-578e-4665-a582-dc35f01c3d5b/files/inspect_robodojo_camera_run.py
python "$ENTRY" eval \
  --checkpoint /data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_stack_bowl_2nodes_8gpu_decay_20k_beta_0.25/step12k \
  --qwen3vl-dir "$CACHE/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$CACHE/pretrained/Cosmos3-Edge" \
  --stage1-text-dir "$CACHE/pretrained/siglip2-base-patch16-224" \
  --task stack_bowls --eval-num 3 --num-envs 1 \
  --gpu 0 --sim-gpu 1 --replan-steps 10 --seed 0 \
  --inference-steps 10 --save-predicted-video --port 19178 --timeout 7200 \
  --output-dir outputs/evaluation/robodojo/stack_bowls_delta_beta025_step12k_sampling10_20261031_v1
```

Final validation covers all three native results, unchanged source
SHA256 fingerprints, numerical/action parity, cameras, videos and visual
review. Preparation is verified; startup and results are pending.

The actual native rollout subsequently started successfully. The loader
explicitly restores `beta_0.25/step12k/model`, reporting 12.585B parameters,
BF16, joint prediction and ten denoising steps. First actual inference
took 5.386 seconds, with matched prediction export 4.671 seconds.
The `(32,20)` action chunk is finite and configured for ten-action
execution; the policy service is responsive on port19178.

Saved evaluation settings differ from Beta05 step17k only in checkpoint,
output directory and port; layout0's initial measured state matches
exactly. All three actual conditioning images were visually checked:
ordinary white bowls/wooden table and expected head/left-wrist/right-wrist
roles. Actual predictions and passive camera observations are saved.
There is no first-failure gate marker: all three layouts remain requested
regardless of layout0's outcome. Startup evidence:
`startup_validation.json`. Preparation commit: `ff65ccb`.
Evaluation is running; final success counts are not yet available.

The corrected **step12k** evaluation subsequently completed: **0/3
success**, native scores **0.15/0.0/0.15**, aggregate **10/100**. All
three episodes ran to 800 control steps, with no first-failure stopping.
The partial scores are not successful three-bowl stacks.

Visual review covered all conditioning-head contact sheets, actual
head-video timelines and exact final frames from all nine actual videos.
Layout0 forms a two-bowl partial stack; the left arm brings the remaining
bowl toward it but fails to complete the stack. Layout1 brings the bowls
together without valid nesting, ending with zero credit. Layout2 retains
a two-bowl partial stack while the left arm approaches the remaining bowl
and stalls. Thus the main failure remains completing the third-bowl
placement, with layout1 also failing the first valid pair.

Wall time was 3163.83 seconds (52.73 minutes), including startup.
The run saved 240 chunks, 720 predicted videos, 7920 prediction PNGs and
nine actual videos (640x480, 25 FPS, 801 frames each). Mean inference and
prediction-export times were 5.132 and 4.057 seconds per chunk; shared
load prevents causal timing comparisons.

All numerical/independent-integration/export-parity, camera, video and
post-run source-SHA256 checks passed. All three initial measured states
match the reference exactly. Across 51 passive camera snapshots, head
and static-scene matrix drift were zero; wrist transforms/render bindings
were consistent. No episode was flagged physically unstable.
Left/right target tracking medians were 0.205/0.232 mm, maxima
4.871/8.505 mm. These checks do not imply grasping or task success.

Artifacts:

```text
outputs/evaluation/robodojo/stack_bowls_delta_beta025_step12k_sampling10_20261031_v1/
  summary.json
  final_adapter_audit.json
  camera_audit/camera_integrity_report.json
  diagnostics/native_final_frames.jpg
  policy/predictions/
```

Actual videos are below the same root:

```text
simulator/eval_result/RoboDojo/stack_bowls/Qwen3VLMoT/arx_x5/0_action_type=ee,simulator=isaacsim6/stack_bowls_delta_beta025_step12k_sampling10_20261031_v1/
```

All three requested episodes and final audits are complete, not the
full benchmark. The owned policy service shut down; port19178 is closed.
Startup commit: `837b073`. Results here are specifically for step12k;
the previous, user-reported problematic step16k run remains separate.

## Qwen Beta(0.25,0.25) step16k ordinary bowls (2026-10-31)

Evaluate
`/data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_stack_bowl_2nodes_8gpu_decay_20k_beta_0.25/step16k`
on ordinary `stack_bowls`, layouts0/1/2, once each. **Run all three
episodes; do not inherit the previous step15k first-failure stopping rule.**
The first upload was rejected before launching because its saved training
mix was inconsistent. After the user replaced it, a fresh preflight
verified `delta_robodojo_stack_bowl`, 44,774 training frames, exact
policy/training-policy equality, and pure Beta(0.25,0.25). Saved shift and
mixture-probability fields are inactive in pure-beta mode.

All 19 source files were rehashed: 16 DCP shards, 25,200,777,632 bytes,
2560 tensors and 16,908 valid storage references. Tensor keys, shapes and
dtypes match the previously evaluated architecture. Source size/mtime
remained stable during verification. Read the verified source directly
without copying the full checkpoint. Manifest:
`outputs/evaluation/robodojo/runtime_assets/robodojo_stack_bowl_2nodes_8gpu_decay_20k_beta_0.25/checkpoint_provenance_step16k.json`.

Use the matching bowl-delta dataset and its existing 20D mean/std delta
adapter, BF16, seed0, one environment, sampling10, prediction32/execution10,
actual/future videos and passive camera audits. Keep the same policy
process/RNG across all layouts. Relative to Beta(0.5,0.5) step17k, the
saved policy differs only in beta alpha/beta, but experiment and checkpoint
also differ: this is not an isolated causal noise-distribution comparison
or a full-benchmark estimate.

The actual instrumented command is session-specific:

```bash
conda activate wam
cd /home/jianf/lzl_project/gcr_latent_action
unset ROBODOJO_STOP_AFTER_FIRST_FAILURE
export PYTHONPATH="$PWD" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export OMNI_KIT_ACCEPT_EULA=YES HF_HUB_OFFLINE=1
export ROBODOJO_EVAL_MODULE=lerobot.evaluation.eval_qwen3vl_mot_robodojo_delta
CACHE="$PWD/outputs/evaluation/robodojo/runtime_assets"
ENTRY=/home/jianf/lzl_project/.copilot/session-state/1341fac2-578e-4665-a582-dc35f01c3d5b/files/inspect_robodojo_camera_run.py
python "$ENTRY" eval \
  --checkpoint /data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_stack_bowl_2nodes_8gpu_decay_20k_beta_0.25/step16k \
  --qwen3vl-dir "$CACHE/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$CACHE/pretrained/Cosmos3-Edge" \
  --stage1-text-dir "$CACHE/pretrained/siglip2-base-patch16-224" \
  --task stack_bowls --eval-num 3 --num-envs 1 \
  --gpu 0 --sim-gpu 1 --replan-steps 10 --seed 0 \
  --inference-steps 10 --save-predicted-video --port 19176 --timeout 7200 \
  --output-dir outputs/evaluation/robodojo/stack_bowls_delta_beta025_step16k_sampling10_20261031_v1
```

Final validation must cover all three native results, unchanged source
SHA256 fingerprints, numerical/action parity, cameras, videos and visual
review. Preparation is verified; startup and results are pending.

The actual native rollout subsequently started successfully. The loader
restores this exact step16k DCP checkpoint as 12.585B parameters, BF16,
joint prediction and ten denoising steps. First actual inference took
5.493 seconds, with matched prediction export 4.742 seconds. The
`(32,20)` action chunk is finite and configured for ten-action execution;
the policy service is responsive on port19176.

Saved evaluation settings differ from Beta05 step17k only in checkpoint,
output directory and port; layout0's initial measured state matches
exactly. All three actual conditioning images were visually checked:
ordinary white bowls/wooden table and expected head/left-wrist/right-wrist
roles. Actual predictions and passive camera observations are saved.
There is no first-failure gate marker: all three layouts remain requested
regardless of layout0's outcome. Startup evidence:
`startup_validation.json`. Preparation commit: `d27908d`.
Evaluation is running; final success counts are not yet available.

The three-layout run subsequently completed: **0/3 success**, native
scores **0.0/0.0/0.0**, aggregate **0/100**. All three episodes reached
800 control steps; no first-failure stopping occurred. Wall time was
3151.13 seconds (52.52 minutes), with 240 action chunks, 720 predicted
videos, 7920 prediction PNGs and nine actual rollout videos. Each actual
video is 640x480, 25 FPS, 801 frames. Mean inference/export time was
5.316/4.313 seconds per chunk; shared-machine timing is not a causal
performance comparison.

Visual review covered all conditioning-head contact sheets, actual
head-video timelines and exact final frames from all nine actual videos.
Layout0 brings the right bowl near the middle bowl, then repeats
near-contact poses without nesting. Layout1 approaches/slightly moves
the left bowl and stalls. Layout2 nudges the left bowl toward the middle
bowl but likewise stalls; the right arm changes pose without finishing
a stack. The final bowls remain separate, consistent with zero scores.

All numerical/independent-integration/export-parity, camera, video and
post-run source-SHA256 checks passed. All initial measured states match
Beta05 step17k exactly. The 51 passive camera snapshots show zero head or
static-scene matrix drift; wrist render bindings/transforms are consistent.
No native episode was flagged physically unstable. Left/right target
tracking medians were 0.134/0.228 mm, maxima 3.075/8.295 mm.
These checks do not imply grasping or task success.

For context, Beta(0.5,0.5) step17k succeeded 1/3 on the same layouts,
scores 1.0/0.0/0.15 (38.33/100). This Beta(0.25,0.25) step16k diagnostic
is worse on these three layouts, but different experiments/update counts
and only three layouts do not establish that the noise distribution
caused the difference.

Artifacts:

```text
outputs/evaluation/robodojo/stack_bowls_delta_beta025_step16k_sampling10_20261031_v1/
  summary.json
  final_adapter_audit.json
  camera_audit/camera_integrity_report.json
  diagnostics/native_final_frames.jpg
  policy/predictions/
```

Actual videos are below the same root:

```text
simulator/eval_result/RoboDojo/stack_bowls/Qwen3VLMoT/arx_x5/0_action_type=ee,simulator=isaacsim6/stack_bowls_delta_beta025_step16k_sampling10_20261031_v1/
```

All three requested episodes and final audits are complete, not the
full benchmark. The owned policy service shut down; port19176 is closed.
Startup commit: `c46837d`.

## Qwen Beta(0.5,0.5) step15k conditional evaluation (2026-10-31)

Evaluate the same experiment's Qwen3VL-MoT checkpoint
`/data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_stack_bowl_2nodes_8gpu_decay_20k_beta_0.5/step15k`
on ordinary `stack_bowls`. **Stop if the first episode (layout0) fails**;
otherwise continue layouts1/2 in the same process, without reseeding the
policy. A later layout1 failure does not trigger this first-episode gate.
An early stop reports only the one evaluated episode, not three failures.

Preflight verifies 19 files, 16 DCP shards, 25,200,777,625 bytes,
2560 tensors and 16,908 storage references. All 16 shard hashes differ
from Beta17k; policy/training config hashes match exactly. Noise remains
pure Beta(0.5,0.5). No full checkpoint copy is made. Manifest:
`outputs/evaluation/robodojo/runtime_assets/robodojo_stack_bowl_2nodes_8gpu_decay_20k_beta_0.5/checkpoint_provenance_step15k.json`.

The session-local camera wrapper implements the requested gate at native
next-seed selection: on first failure, return no next layout and allow
native result/video saving and graceful shutdown. On first success,
normal seed selection continues. `first_episode_gate.json` records that
decision. The coordinator validates exactly one failed layout0 if stopped,
otherwise all three layouts, and records requested versus actual episode
counts explicitly. The CPU auditor checks the actual count rather than
inventing results for unrun layouts. This changes neither shared simulator
sources nor policy actions.

Three focused gate tests passed: first failure prevents requesting a
second seed; first success continues even if the second episode fails;
multiple environments are rejected. Evaluation otherwise matches Beta17k:
seed0, one environment, sampling10, prediction32/execution10, BF16,
native/future videos and passive cameras. Early-stopped aggregates must
not be compared with unconditional three-layout aggregates.

The actual instrumented command is session-specific:

```bash
conda activate wam
cd /home/jianf/lzl_project/gcr_latent_action
CACHE="$PWD/outputs/evaluation/robodojo/runtime_assets"
ENTRY=/home/jianf/lzl_project/.copilot/session-state/1341fac2-578e-4665-a582-dc35f01c3d5b/files/inspect_robodojo_camera_run.py
PYTHONPATH="$PWD" ROBODOJO_STOP_AFTER_FIRST_FAILURE=1 OMNI_KIT_ACCEPT_EULA=YES \
python "$ENTRY" eval \
  --checkpoint /data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_stack_bowl_2nodes_8gpu_decay_20k_beta_0.5/step15k \
  --qwen3vl-dir "$CACHE/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$CACHE/pretrained/Cosmos3-Edge" \
  --stage1-text-dir "$CACHE/pretrained/siglip2-base-patch16-224" \
  --task stack_bowls --eval-num 3 --num-envs 1 \
  --gpu 0 --sim-gpu 1 --replan-steps 10 --seed 0 \
  --inference-steps 10 --save-predicted-video --port 19174 --timeout 7200 \
  --output-dir outputs/evaluation/robodojo/stack_bowls_delta_beta05_step15k_sampling10_first_failure_stop_20261031_v1
```

`--eval-num 3` is the maximum, not a promise that three episodes run.
This conditional stopping rule is supplied by the session wrapper/env
variable, not a new public evaluator CLI option. Final audits include
source hashes, numerical/action parity, cameras, videos and visual review.
Preparation is verified; actual startup and the first episode outcome
are pending.

The actual native rollout subsequently started successfully. The loader
restores this exact Beta15k checkpoint as 12.585B parameters, BF16,
joint prediction and ten denoising steps. First actual inference took
6.352 seconds, with matched prediction export 4.890 seconds. The
`(32,20)` action chunk is finite and configured for ten-action execution;
the policy service is responsive on port19174.

Saved evaluation settings differ from Beta17k only in checkpoint,
output directory and port; layout0's initial measured state matches
exactly. All three actual conditioning images were visually checked:
ordinary white bowls/wooden table, expected head/left-wrist/right-wrist
roles. Actual predictions and passive camera observations are saved.
The installed native gate reports `waiting_for_first_episode` in
`first_episode_gate.json`; the first episode has not finished yet.
Its eventual failure will stop before requesting layout1, while success
will continue without restarting the policy. Startup evidence:
`startup_validation.json`. Preparation commit: `43b162b`.

The first episode subsequently **failed at 800 control steps, native
score0.0**, and the evaluator stopped as requested. **Only layout0 ran:
0/1 success. Layouts1/2 were not evaluated**, not counted as failures.
`first_episode_gate.json` records `first_failure_stopped`; the summary
records `requested_max_episodes=3`, `evaluated_episodes=1` and
`stopped_early=true`. The conditional protocol is complete, not the
unconditional three-layout benchmark.

The right arm brings two bowls into overlap/contact, then the left arm
transports the remaining bowl beside them and stalls without a valid
completed stack. Native partial credit remains zero. All 80 conditioning
head thumbnails, the actual head-video timeline/final frame and both final
native wrist snapshots were reviewed; the ordinary scene and camera
roles remain coherent. The same layout0 succeeded at Beta17k in 520 steps,
but this single conditional result does not establish a general ranking.

Wall time was 1120.98 seconds (18.68 minutes); mean inference/prediction
export were 4.811/4.165 seconds per chunk. All 80 chunks are finite and
pass inverse normalization, independent 32-step integration and exact
exported-action parity. Denormalization error is zero; translation/
rotation-matrix/gripper errors are below 6.0e-8/8.5e-8/3.0e-8.
The initial measured state matches Beta17k exactly. Left/right tracking
median errors are 0.166/0.633 mm and maxima 1.103/3.724 mm, measuring
command following rather than expert-action prediction quality.

All three actual videos decode at 640x480/25 FPS with 801 frames each;
240 prediction videos and 2640 PNGs are saved. All 17 passive camera
snapshots pass with zero head/static-scene transform drift, correct render
bindings and wrist-to-EE errors below 2.35e-7. Native unstable-physics
flags are zero and all 19 source-file SHA256 hashes match preflight.
The evaluator/chained auditor exited successfully and policy port19174
is closed.

Final report: the output's `final_adapter_audit.json`; camera report:
`camera_audit/camera_integrity_report.json`. Visual evidence:
`diagnostics/layout_0_all_conditioning_heads.jpg`,
`diagnostics/episode_0000000_cam_head_fail_contact_sheet.jpg` and
`camera_audit/layout_0_step_0800_cam_{left,right}_wrist.png`.
Video/result subdirectory:

```text
simulator/eval_result/RoboDojo/stack_bowls/Qwen3VLMoT/arx_x5/0_action_type=ee,simulator=isaacsim6/stack_bowls_delta_beta05_step15k_sampling10_first_failure_stop_20261031_v1/
```

Actual videos are `episode_0000000_cam_{head,left_wrist,right_wrist}_fail.mp4`;
matched futures remain under `policy/predictions/reset_0002/env_000/`.

## Qwen Beta(0.5,0.5) step17k ordinary bowls (2026-10-31)

Evaluate our Qwen3VL-MoT checkpoint
`/data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_stack_bowl_2nodes_8gpu_decay_20k_beta_0.5/step17k`
on ordinary `stack_bowls`, not random or PI05. Compare against the same
experiment's completed step12k (0/3, scores0.15/0/0.15, aggregate10/100).

Preflight passed for all 19 source files, including 16 DCP shards,
25,200,777,625 bytes, 2560 model tensors and 16,908 storage references.
All tensor keys/shapes/dtypes match the previously evaluated architecture,
storage ranges fit their shards, and source sizes/mtimes remained stable
during SHA256 hashing. All 16 weight-shard hashes differ from step12k;
policy/training config hashes match step12k exactly. Noise remains
**pure Beta(0.5,0.5)**, not a 50% uniform mixture; the saved shift and
mixture-probability fields are inactive in beta mode. No checkpoint copy
is made. Manifest:
`outputs/evaluation/robodojo/runtime_assets/robodojo_stack_bowl_2nodes_8gpu_decay_20k_beta_0.5/checkpoint_provenance_step17k.json`.

Keep layouts0/1/2, seed0, one environment, BF16, sampling10,
prediction32/execution10, native/future videos and passive camera checks.
Only checkpoint, output directory and port may differ from step12k's
saved evaluation settings. Three layouts are diagnostic, not a full
benchmark estimate. Final audits include initial-state parity, all-action
integration/export parity, video decoding, camera geometry and post-run
source SHA256 verification.

```bash
conda activate wam
cd /home/jianf/lzl_project/gcr_latent_action
CACHE="$PWD/outputs/evaluation/robodojo/runtime_assets"
OMNI_KIT_ACCEPT_EULA=YES python -m lerobot.evaluation.eval_qwen3vl_mot_robodojo_delta eval \
  --checkpoint /data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_stack_bowl_2nodes_8gpu_decay_20k_beta_0.5/step17k \
  --qwen3vl-dir "$CACHE/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$CACHE/pretrained/Cosmos3-Edge" \
  --stage1-text-dir "$CACHE/pretrained/siglip2-base-patch16-224" \
  --task stack_bowls --eval-num 3 --num-envs 1 \
  --gpu 0 --sim-gpu 1 --replan-steps 10 --seed 0 \
  --inference-steps 10 --save-predicted-video --port 19172 --timeout 7200 \
  --output-dir outputs/evaluation/robodojo/stack_bowls_delta_beta05_step17k_sampling10_20261031_v1
```

The instrumented session uses the existing passive camera wrapper and
chains the CPU auditor without editing shared simulator sources.
Preparation is verified; actual startup and final results remain pending.

The actual native rollout subsequently started successfully. The loader
restores this exact Beta17k DCP checkpoint and reports 12.585B parameters,
BF16, joint prediction and ten denoising steps. First real simulator
inference took 5.581 seconds; matched prediction export took 5.056 seconds.
The first `(32,20)` physical action chunk is finite and configured for
ten-action execution; policy port19172 is responsive.

Complete saved evaluation settings differ from Beta12k only in
checkpoint, output directory and port. Layout0's initial measured state
matches exactly. All three actual conditioning views were visually
checked: ordinary white bowls/wooden table and expected head/left-wrist/
right-wrist roles. The first passive camera snapshot and three matched
future videos are persisted. Evidence is in the output's
`startup_validation.json`, `policy/chunks/`, `policy/predictions/` and
`camera_audit/`. Preparation commit: `70ccbf4`. Final success scores and
full audits remain pending at this startup check.

The run subsequently completed with **1/3 success (33.33%)** and native
aggregate **38.33/100**.

| Layout | Success | Native score | Control steps | Visual outcome |
|---|---|---|---|---|
| 0 | Yes | 1.00 | 520 | Right arm stacks two bowls, then left arm adds/releases the third. |
| 1 | No | 0.00 | 800 | Bowls brought together, but remain separate without nesting. |
| 2 | No | 0.15 | 800 | Two-bowl partial stack remains; third bowl moves outside the head view. |

| Same Beta(0.5,0.5) experiment | Success | Layout0/1/2 scores | Aggregate /100 |
|---|---|---|---|
| step10k | 0/3 | 0.15 / 0.00 / 0.15 | 10.00 |
| step12k | 0/3 | 0.15 / 0.00 / 0.15 | 10.00 |
| step17k | 1/3 | 1.00 / 0.00 / 0.15 | 38.33 |

Step17k improves on these three fixed layouts, not a reliable full-benchmark
ranking. All initial measured states match Beta12k exactly, and saved
evaluation settings differ only in checkpoint/output directory/port.
Wall time was 2581.73 seconds (43.03 minutes); mean inference and
prediction-video export were 4.777 and 4.123 seconds per chunk.
Shared-machine load prevents causal runtime comparisons.

All 212 chunks are finite and pass inverse normalization, independent
32-step physical integration and exact prediction-export action parity.
Denormalization error is zero; translation/rotation-matrix/gripper errors
are below 6.0e-8/1.06e-7/3.0e-8. All nine actual videos decode at
640x480/25 FPS, with 521 frames per layout0 view and 801 per layout1/2
view, matching actual control steps plus the initial frame.
There are 636 prediction videos and 6996 prediction PNGs.

All 45 passive camera snapshots pass: zero head/static-scene transform
drift, correct render bindings and wrist-to-EE errors below 3.94e-7.
All 19 source-file SHA256 hashes match the preflight manifest.
Native physics-instability flags are zero. However, physical tracking
is not perfect: left/right median errors are 0.191/0.514 mm and maxima
196.996/155.218 mm. Both maxima occur in layout2, at control360->370
and660->670 respectively. These are command-following discrepancies,
not expert-action prediction errors; the adapter/camera checks do not
eliminate contact, reachability or controller limitations.

Visual review covered all 212 conditioning-head thumbnails, three native
head-video timelines and exact final frames from all nine actual videos.
The successful final head frame shows three nested bowls with the
grippers withdrawn. The ordinary scene/camera roles remain coherent;
late layout2 wrist views point away from the table as the arms move,
not because of camera-binding or transform drift.

Final report: the output's `final_adapter_audit.json`; camera report:
`camera_audit/camera_integrity_report.json`; final-view overview:
`diagnostics/native_final_frames.jpg`. Conditioning sheets and native
timelines are under `diagnostics/`. Video/result subdirectory:

```text
simulator/eval_result/RoboDojo/stack_bowls/Qwen3VLMoT/arx_x5/0_action_type=ee,simulator=isaacsim6/stack_bowls_delta_beta05_step17k_sampling10_20261031_v1/
```

Successful head video: `episode_0000000_cam_head_success.mp4`;
layouts1/2: `episode_000000{1,2}_cam_head_fail.mp4`, each with matching
left/right wrist videos. Model futures remain under `policy/predictions/`.
The evaluator/chained auditor exited successfully and policy port19172
is closed.

## Qwen robodojo_decay_20k step17k ordinary bowls (2026-10-31)

Evaluate our Qwen3VL-MoT checkpoint
`/data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_decay_20k/step17k`
on ordinary `stack_bowls`, not random or PI05. The reference is this
same experiment's completed step14k (0/3, aggregate0/100).

Preflight passed for all 19 source files, including 16 DCP shards,
25,200,777,591 bytes, 2560 model tensors and 16,908 storage references.
Tensor keys/shapes/dtypes match the evaluated architecture, storage
ranges fit their shards, and source sizes/mtimes remain stable during
SHA256 verification. **All 16 weight-shard hashes differ from step14k**;
policy and training config SHA256 hashes match step14k exactly.
The noise recipe remains 50% Beta(1.5,1.0) plus 50% uniform.
Weights are read directly without a full checkpoint copy. Manifest:
`outputs/evaluation/robodojo/runtime_assets/robodojo_decay_20k/checkpoint_provenance_step17k.json`.

Keep layouts0/1/2, seed0, one environment, BF16, sampling10,
prediction32/execution10, native/future videos and passive camera
auditing. Only checkpoint, output directory and port may differ from
step14k's saved evaluation settings. This is a three-layout diagnostic,
not a full-benchmark estimate. Final audits include initial-state parity,
independent action integration, exported-action parity, video decoding,
camera geometry and all source SHA256 hashes.

```bash
conda activate wam
cd /home/jianf/lzl_project/gcr_latent_action
CACHE="$PWD/outputs/evaluation/robodojo/runtime_assets"
OMNI_KIT_ACCEPT_EULA=YES python -m lerobot.evaluation.eval_qwen3vl_mot_robodojo_delta eval \
  --checkpoint /data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_decay_20k/step17k \
  --qwen3vl-dir "$CACHE/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$CACHE/pretrained/Cosmos3-Edge" \
  --stage1-text-dir "$CACHE/pretrained/siglip2-base-patch16-224" \
  --task stack_bowls --eval-num 3 --num-envs 1 \
  --gpu 0 --sim-gpu 1 --replan-steps 10 --seed 0 \
  --inference-steps 10 --save-predicted-video --port 19170 --timeout 7200 \
  --output-dir outputs/evaluation/robodojo/stack_bowls_delta_decay20k_step17k_sampling10_20261031_v1
```

The instrumented session uses the existing passive camera wrapper and
chains the CPU auditor without changing shared simulator sources.
Preparation is verified; actual startup and final results remain pending.

The real native rollout subsequently started successfully. The loader
restores this exact step17k DCP checkpoint and reports 12.585B parameters,
BF16, joint prediction and ten denoising steps. First actual simulator
inference took 5.703 seconds and matched prediction export 4.788 seconds.
The returned `(32,20)` physical action chunk is finite; the simulator
advances beyond its first ten-action reply, with policy port19170 live.

Saved evaluation settings differ from same-experiment step14k only in
checkpoint, output directory and port. Layout0's initial measured state
matches step14k exactly. All three actual conditioning views were
visually checked: expected ordinary white bowls/wooden table and correct
head/left-wrist/right-wrist roles. Initial camera observations, actions
and matched future videos are persisted. Evidence is in the output's
`startup_validation.json`, `policy/chunks/`, `policy/predictions/` and
`camera_audit/`. Preparation commit: `ebdd847`. Final success scores and
full audits remain pending at this startup check.

The run subsequently completed with **0/3 successful episodes** and
native aggregate **5.0/100**. Layout0 earns partial credit0.15; layouts1/2
score zero. All three run 800 control steps with zero unstable-physics flags.

| Same robodojo_decay_20k experiment | Success | Layout0/1/2 scores | Aggregate /100 |
|---|---|---|---|
| step14k | 0/3 | 0.00 / 0.00 / 0.00 | 0.0 |
| step17k | 0/3 | 0.15 / 0.00 / 0.00 | 5.0 |

Step17k shows partial progress on layout0, but full success remains zero.
These three fixed layouts are insufficient for a general performance
ranking. Wall time was 3532.23 seconds (58.87 minutes); mean inference
and prediction export were 6.172 and 4.766 seconds per chunk. Shared load
prevents a causal runtime comparison with 14k.

All 240 chunks are finite and pass inverse normalization, independent
32-step physical integration and exact prediction-export action parity.
Denormalization error is zero; translation/rotation-matrix/gripper errors
are below 6.0e-8/9.3e-8/3.0e-8. All three initial measured states match 14k
exactly. Left/right target-tracking median errors are 0.082/0.261 mm,
maxima 2.477/38.839 mm; these are command-following errors rather than
expert-action prediction errors.

All nine actual videos decode at 640x480/25 FPS with 801 frames each;
720 prediction videos and 7920 prediction PNGs are retained. All 51 passive
camera snapshots pass with zero head/static-scene transform drift,
correct render bindings and wrist-to-EE errors below 3.61e-7.
All 19 source-file SHA256 hashes match their preflight records.
The evaluator and chained auditor exited successfully, and owned policy
port19170 is closed.

Visual review covered all 240 conditioning-head thumbnails, three native
head-video timelines/final frames and six final native wrist snapshots.
Layout0 forms a two-bowl partial stack; the left arm transports the third
bowl beside it rather than completing the stack. Layout1 spends most of
the episode near the right bowl without stacking. Layout2 brings/tips a
bowl into contact with the center bowl but does not retain a valid stack.
The reviewed inputs/cameras look coherent.

Final report: the output's `final_adapter_audit.json`; camera report:
`camera_audit/camera_integrity_report.json`. Conditioning sheets:
`diagnostics/layout_{0,1,2}_all_conditioning_heads.jpg`; native timelines:
`diagnostics/episode_000000{0,1,2}_cam_head_fail_contact_sheet.jpg`;
final wrist snapshots: `camera_audit/layout_{0,1,2}_step_0800_cam_{left,right}_wrist.png`.
Video/result subdirectory:

```text
simulator/eval_result/RoboDojo/stack_bowls/Qwen3VLMoT/arx_x5/0_action_type=ee,simulator=isaacsim6/stack_bowls_delta_decay20k_step17k_sampling10_20261031_v1/
```

Head videos are `episode_000000{0,1,2}_cam_head_fail.mp4`, with matching
left/right wrist videos. Matched model futures are under
`policy/predictions/reset_*/env_000/replan_*_step_*/{primary,secondary,wrist}.mp4`.

## Qwen robodojo_decay_20k step14k ordinary bowls (2026-10-30)

This evaluates **our Qwen3VL-MoT, not PI05**, from
`/data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_decay_20k/step14k`.
The requested scope is ordinary `stack_bowls` only, not random.
Changes and results belong to `qwen3vl_mot_pretrained_generation`.

Preflight verified all 19 source files (16 DCP shards), totaling
25,200,777,591 bytes, with stable sizes/mtimes during SHA256 hashing.
All 2560 tensor keys/shapes/dtypes match the previously evaluated Qwen
architecture, and all 16,908 storage references fit their shard files.
The saved training data mix is `delta_robodojo_stack_bowl`. Source weights
are loaded directly; no full-size checkpoint duplicate is made.
Manifest:
`outputs/evaluation/robodojo/runtime_assets/robodojo_decay_20k/checkpoint_provenance_step14k.json`.

This experiment's saved noise distribution is **mixed: 50% Beta(1.5,1.0)
and 50% uniform** over the saved sigma range. It is not pure Beta(0.5,0.5)
or shifted-uniform; the saved shift5 field is inactive in mixed mode.
Evaluation does not change this configuration. Saved inference steps are
20, overridden to 10 to match earlier diagnostics.

Use layouts 0/1/2, seed0, one environment, BF16, sampling10,
prediction32/execution10, three actual cameras, matched future-prediction
videos, and passive camera auditing. Compare with the earlier shift5
step14k ordinary run (2/3, aggregate66.67/100), but these are different
training experiments: matched evaluation alone cannot isolate the effect
of the training noise distribution. Three layouts are diagnostic, not a
full-benchmark success-rate estimate.

```bash
conda activate wam
cd /home/jianf/lzl_project/gcr_latent_action
CACHE="$PWD/outputs/evaluation/robodojo/runtime_assets"
OMNI_KIT_ACCEPT_EULA=YES python -m lerobot.evaluation.eval_qwen3vl_mot_robodojo_delta eval \
  --checkpoint /data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_decay_20k/step14k \
  --qwen3vl-dir "$CACHE/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$CACHE/pretrained/Cosmos3-Edge" \
  --stage1-text-dir "$CACHE/pretrained/siglip2-base-patch16-224" \
  --task stack_bowls --eval-num 3 --num-envs 1 \
  --gpu 0 --sim-gpu 1 --replan-steps 10 --seed 0 \
  --inference-steps 10 --save-predicted-video --port 19168 --timeout 7200 \
  --output-dir outputs/evaluation/robodojo/stack_bowls_delta_decay20k_step14k_sampling10_20261030_v1
```

The instrumented session uses the existing passive camera wrapper without
editing shared simulator sources. Full numerical/action-export/video/
camera checks and a post-run source rehash are chained automatically.
Preparation is verified; actual startup and final results remain pending.

The actual native rollout subsequently started successfully. The loader
restored this exact `robodojo_decay_20k/step14k` DCP checkpoint and reports
12.585B parameters, joint prediction, BF16 and ten denoising steps. First
real simulator inference took 4.976 seconds; prediction export took
4.274 seconds. The returned `(32,20)` physical action chunk is finite,
and the native simulator has advanced beyond its first ten-action reply.
The policy service is responsive on port 19168.

Complete evaluation settings differ from the earlier shift5 step14k only
in checkpoint, output directory and port; layout0's initial measured state
matches exactly. All three actual conditioning views were visually checked:
ordinary white bowls on the wooden table with correct head/left-wrist/
right-wrist roles. Matched prediction videos and the initial passive
camera snapshot are persisted. Evidence: `startup_validation.json`,
`policy/chunks/inference.jsonl`, `policy/predictions/` and `camera_audit/`
under the output directory above. Preparation commit: `81d5668`.
Evaluation remains running at this startup check; final success scores
and complete audits are not yet available.

The run subsequently completed with **0/3 successful episodes**, all
three native scores **0/0/0**, aggregate **0/100**. Layouts 0/1/2 each
reached 800 control steps with zero unstable-physics flags.

| Ordinary stack_bowls checkpoint | Success | Layout 0/1/2 scores | Aggregate /100 |
|---|---|---|---|
| Earlier high_noise_shift5 step14k | 2/3 | 1.0 / 0.0 / 1.0 | 66.67 |
| This robodojo_decay_20k step14k | 0/3 | 0.0 / 0.0 / 0.0 | 0.00 |

This checkpoint performs worse on these three fixed layouts, not a
population-level ranking or proof that mixed noise caused the failures.
Wall time was 2922.56 seconds (48.71 minutes); mean inference and
prediction-video export were 4.944 and 3.983 seconds per chunk.

All 240 chunks are finite and pass inverse normalization, independent
32-step physical integration and prediction-export action parity checks.
Denormalization error is zero; translation, rotation-matrix and gripper
errors are below 6.0e-8, 9.8e-8 and 3.0e-8. All three initial measured
states match the earlier shift5 step14k exactly. Left/right target-tracking
median errors are 0.070/0.180 mm and maxima 1.700/4.271 mm; these measure
command following, not expert-action prediction quality.

All nine actual videos decode at 640x480/25 FPS with 801 frames each;
720 prediction videos and 7920 prediction PNGs are saved. All 51 passive
camera snapshots pass with zero head/static-scene transform drift,
correct render bindings and wrist-to-EE errors below 4.14e-7.
Post-run SHA256 hashes of all 19 source files match the preflight manifest.
The evaluation and chained auditor exited successfully; policy port
19168 is closed.

Visual review covered all 240 conditioning-head thumbnails, three native
head-video timelines/final frames and six final native wrist snapshots.
Layout0 reaches the right bowl then makes little progress near its rim;
layout1 spends a long time at the right bowl and later approaches the
outer bowls with both arms; layout2 moves/tips the right bowl toward the
center before approaching the left bowl. No valid nested stack forms.
The reviewed inputs/cameras look coherent; low tracking errors do not
mean correct learned task decisions.

Final report: the output's `final_adapter_audit.json`; camera report:
`camera_audit/camera_integrity_report.json`. Complete conditioning sheets:
`diagnostics/layout_{0,1,2}_all_conditioning_heads.jpg`; native timelines:
`diagnostics/episode_000000{0,1,2}_cam_head_fail_contact_sheet.jpg`;
final wrist snapshots: `camera_audit/layout_{0,1,2}_step_0800_cam_{left,right}_wrist.png`.
Video/result subdirectory:

```text
simulator/eval_result/RoboDojo/stack_bowls/Qwen3VLMoT/arx_x5/0_action_type=ee,simulator=isaacsim6/stack_bowls_delta_decay20k_step14k_sampling10_20261030_v1/
```

Head videos are `episode_000000{0,1,2}_cam_head_fail.mp4`, with matching
left/right wrist videos. Matched model futures remain under
`policy/predictions/reset_*/env_000/replan_*_step_*/{primary,secondary,wrist}.mp4`.

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

#### Beta(0.5, 0.5) step12k evaluation (2026-10-30)

The source is `step12k` in the same
`/data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_stack_bowl_2nodes_8gpu_decay_20k_beta_0.5`
experiment. Policy and training config SHA256 fingerprints match the
completed beta step10k evaluation exactly; sampling remains pure
Beta(0.5, 0.5), not shifted-uniform or a uniform mixture.

Preflight passed for 19 files, including 16 DCP shards, totaling
25,200,777,625 bytes. Source files remained stable throughout SHA256
verification; all 2560 tensor keys/shapes/dtypes match step10k, and all
16,908 storage references fit their shard files. All 16 weight-shard hashes
differ from step10k. The manifest is saved in
`outputs/evaluation/robodojo/runtime_assets/robodojo_stack_bowl_2nodes_8gpu_decay_20k_beta_0.5/checkpoint_provenance_step12k.json`.
No full checkpoint duplicate is made.

The matched protocol is ordinary **`stack_bowls`, not random**: layouts
0/1/2, seed0, one environment, sampling10, prediction32/execution10, BF16,
actual/prediction videos, and passive camera auditing. The reference is
the same experiment's step10k (0/3 success, native aggregate score10/100).
Only three layouts are requested, not the full native task benchmark;
preparation alone is not an evaluation result.

```bash
conda activate wam
CACHE="$PWD/outputs/evaluation/robodojo/runtime_assets"
OMNI_KIT_ACCEPT_EULA=YES python -m lerobot.evaluation.eval_qwen3vl_mot_robodojo_delta eval \
  --checkpoint /data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_stack_bowl_2nodes_8gpu_decay_20k_beta_0.5/step12k \
  --qwen3vl-dir "$CACHE/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$CACHE/pretrained/Cosmos3-Edge" \
  --stage1-text-dir "$CACHE/pretrained/siglip2-base-patch16-224" \
  --task stack_bowls --eval-num 3 --num-envs 1 \
  --gpu 0 --sim-gpu 1 --replan-steps 10 --seed 0 \
  --inference-steps 10 --save-predicted-video --port 19152 --timeout 7200 \
  --output-dir outputs/evaluation/robodojo/stack_bowls_delta_beta05_step12k_sampling10_20261030_v1
```

The instrumented run uses the existing passive camera wrapper without
changing shared simulator tasks, cameras, or physics. Final audits compare
against beta step10k and retain checkpoint stability, initial-state,
numerical, video, and camera checks. Shared-machine load prevents causal
runtime comparisons.

Startup is verified with a real finite `(32, 20)` action chunk executing
ten steps. The live policy and simulator select this exact step12k
checkpoint and ordinary `stack_bowls`. Saved evaluation settings differ
from beta step10k only in checkpoint, output directory, and port; layout0's
initial robot state matches exactly. The initial head-camera input was
visually inspected and shows the same ordinary uncluttered three-bowl
scene. `startup_validation.json`, the initial camera snapshot, and actual/
prediction video outputs are being retained. At this check the run is
still active; final success scores are pending.

The run subsequently completed all three layouts with **0/3 successful
episodes**, exactly matching beta step10k's native result. Layouts0/1/2
scored **0.15 / 0 / 0.15**, for an aggregate **10.0/100**. All three ran
800 control steps with zero native unstable-episode flags. Wall time was
3920.84s (65.35min); mean inference and prediction export were 6.9153s and
5.5515s per chunk. This is not a controlled runtime comparison.

All 240 chunks are finite. Inverse normalization and prediction-export
action parity are exact; maximum translation, rotation-matrix, and gripper
integration errors are below `6.0e-8`, `9.5e-8`, and `3.0e-8` respectively.
Source files still match their recorded sizes/mtimes. All three initial
robot states match beta step10k exactly. Fifty-one camera snapshots show
zero head/static-scene transform drift, correct render-product bindings,
and wrist-to-EE transform errors below `3.5e-7`.

All nine actual videos are 640x480 at 25 FPS, with 801 frames each;
720 prediction videos and 7920 PNGs were also saved. Visual review covered
all 240 conditioning head thumbnails, three sampled native head timelines,
and nine final native views. The policy approaches and transports bowls.
Layouts 0/2 retain a nested pair, but the third bowl remains separate near/
in the left gripper instead of completing a released three-bowl stack.
Layout 1 brings bowls together with tipping/contact but does not finish
stacking. The scores match step10k, not necessarily its exact trajectories.

Left/right median target-tracking errors were 0.158/0.489mm and maxima
2.137/47.461mm. The right-arm maximum occurs in layout 2 at control510→520.
Its maximum is smaller than step10k's 117.757mm, but the commands differ,
so this alone does not establish improved policy accuracy. No task-score
improvement was observed on these three layouts; this is not evidence of
a full-benchmark training plateau.

Final reports are `final_adapter_audit.json` and
`camera_audit/camera_integrity_report.json`. Actual videos are under:

```text
outputs/evaluation/robodojo/stack_bowls_delta_beta05_step12k_sampling10_20261030_v1/
  simulator/eval_result/RoboDojo/stack_bowls/Qwen3VLMoT/arx_x5/
    0_action_type=ee,simulator=isaacsim6/
      stack_bowls_delta_beta05_step12k_sampling10_20261030_v1/
        episode_000000{0,1,2}_cam_{head,left_wrist,right_wrist}_fail.mp4
```

Prediction videos remain under `policy/predictions/`; reviewed contact
sheets are under `diagnostics/`. Owned policy/simulator processes exited
normally; no unrelated jobs were stopped.

#### Beta(0.5, 0.5) step10k evaluation (2026-10-30)

The requested source is
`/data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_stack_bowl_2nodes_8gpu_decay_20k_beta_0.5/step10k`.
Its saved policy selects pure `beta` sampling with alpha=beta=0.5 and
sigma bounds `[0.001, 0.999]`. The saved `time_sampling_shift=5` is inactive
for this sampler; `time_sampling_beta_probability=0.5` only affects
`mixed`, so this is not a 50% uniform mixture.

The protocol uses ordinary **`stack_bowls`, not `stack_bowls_random`**:
layouts0/1/2, seed0, one environment, ten inference steps, prediction32/
execution10, BF16, actual/prediction videos, and passive camera auditing.
Source files are read directly without duplicating the full checkpoint.
SHA256/file-stability and DCP tensor/storage checks are recorded in
`outputs/evaluation/robodojo/runtime_assets/robodojo_stack_bowl_2nodes_8gpu_decay_20k_beta_0.5/checkpoint_provenance_step10k.json`.

Relative to the shift5 policy config, the differences are the noise
distribution, beta alpha/beta parameters, and scheduler plateau100 rather
than200. Comparisons with ordinary shift5 step14k also change checkpoint
update, so they cannot isolate the effect of the noise distribution.
This is a three-layout diagnostic, not the complete native task benchmark.
Preparation alone is not an evaluation result.

```bash
conda activate wam
CACHE="$PWD/outputs/evaluation/robodojo/runtime_assets"
OMNI_KIT_ACCEPT_EULA=YES python -m lerobot.evaluation.eval_qwen3vl_mot_robodojo_delta eval \
  --checkpoint /data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_stack_bowl_2nodes_8gpu_decay_20k_beta_0.5/step10k \
  --qwen3vl-dir "$CACHE/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$CACHE/pretrained/Cosmos3-Edge" \
  --stage1-text-dir "$CACHE/pretrained/siglip2-base-patch16-224" \
  --task stack_bowls --eval-num 3 --num-envs 1 \
  --gpu 0 --sim-gpu 1 --replan-steps 10 --seed 0 \
  --inference-steps 10 --save-predicted-video --port 19150 --timeout 7200 \
  --output-dir outputs/evaluation/robodojo/stack_bowls_delta_beta05_step10k_sampling10_20261030_v1
```

The instrumented run uses the existing session-local passive camera wrapper
around this entry; no shared simulator task, camera, or physics code changes
are required.

Preflight passed: 19 source files (including 16 DCP shards), totaling
25,200,777,625 bytes, remained stable through SHA256 verification. All
2560 tensor keys/shapes/dtypes match the previously evaluated architecture,
and all 16,908 storage references fit within their shard files.

Startup is verified: the policy restored this exact `step10k/model`,
reports BF16 and ten denoising steps, and the live simulator command selects
ordinary `stack_bowls`. Relative to ordinary shift5 step14k, saved evaluation
settings differ only in checkpoint, output directory, and port. The first
`(32, 20)` action chunk is finite and executes ten steps; layout0's initial
robot state matches exactly. The initial head-camera input was visually
inspected and shows three bowls on the ordinary uncluttered wooden table.
`startup_validation.json` and the initial passive camera snapshot are saved.
At this startup check, final success results were still pending.

The evaluation subsequently completed with **0/3 successful episodes** on
ordinary `stack_bowls`. Layouts0/1/2 scored **0.15 / 0 / 0.15** respectively;
the native aggregate score is **10.0/100**. All three ran 800 control steps
with zero native unstable-episode flags. Wall time was 3788.95s (63.15min),
including loading/startup; mean inference and prediction export took
6.3318s and 5.0782s per chunk. Shared-machine load and different episode
lengths prevent causal runtime comparisons.

All 240 chunks are finite. Inverse normalization and prediction-export
action parity are exact; maximum translation, rotation-matrix, and gripper
integration errors are below `6.0e-8`, `1.0e-7`, and `3.0e-8` respectively.
The source files still match their recorded sizes/mtimes. All three initial
robot states match ordinary shift5 step14k exactly. Fifty-one passive camera
snapshots show no head/static-scene transform drift, correct render-product
bindings, and wrist-to-EE transform errors below `4.2e-7`.

All nine actual videos are 640x480 at 25 FPS, with 801 frames each; 720 prediction
videos and 7920 PNGs were also saved. Visual review covered all 240
conditioning head thumbnails, three sampled native head timelines, and
all nine final native views. The policy does approach and transport bowls.
Layouts 0/2 show a partially nested pair, consistent with score 0.15, but
the third bowl is not stacked. Layout 1 brings bowls together with tipping/
contact but does not complete the stack, scoring 0. This is not the same
largely unproductive motion seen in the preceding random-task evaluation.

Left/right median target-tracking errors were 0.168/0.449mm and maxima
2.865/117.757mm. The right-arm maximum occurs in layout 2 at control350→360;
this measures a previous chunk's executed-step target versus the next
measured EE position, not demonstration or denoising error.

For context, ordinary shift5 step9k succeeded 1/3 and step14k succeeded 2/3.
This beta10k diagnostic did not improve on those observed outcomes, but
only three layouts were tested and training update/scheduler settings
differ: it does not establish that beta sampling is inherently worse.
Final reports are `final_adapter_audit.json` and
`camera_audit/camera_integrity_report.json`. Actual videos are under:

```text
outputs/evaluation/robodojo/stack_bowls_delta_beta05_step10k_sampling10_20261030_v1/
  simulator/eval_result/RoboDojo/stack_bowls/Qwen3VLMoT/arx_x5/
    0_action_type=ee,simulator=isaacsim6/
      stack_bowls_delta_beta05_step10k_sampling10_20261030_v1/
        episode_000000{0,1,2}_cam_{head,left_wrist,right_wrist}_fail.mp4
```

Prediction videos remain under `policy/predictions/`; reviewed contact
sheets are under `diagnostics/`. Owned policy/simulator processes exited
normally, and no unrelated jobs were stopped.

#### Random bowls, shift5 step14k evaluation (2026-10-29)

This diagnostic uses the same verified `step14k` checkpoint as the ordinary
bowls evaluation below, but selects the native `stack_bowls_random` task.
Checkpoint file sizes and modification times still match the completed
step14k provenance. The native seed0 layout files 0/1/2 exist; their bowl
model IDs are 14/11/11 respectively, with additional clutter in every scene.
Task-config and layout SHA256 fingerprints are retained in
`outputs/evaluation/robodojo/runtime_assets/random_bowls_step14k_preflight.json`.

The protocol retains three layouts, seed0, one environment, ten sampling
steps, 32 predicted actions, ten executed actions, BF16, actual/prediction
videos, and the passive camera audit. Unlike checkpoint-to-checkpoint
comparisons, the scenes intentionally differ from ordinary `stack_bowls`;
robot initial states are recorded but are not required to match. This is
not the complete 25-layout native task evaluation or the full benchmark.
Preparation alone is not a task result.

```bash
conda activate wam
CACHE="$PWD/outputs/evaluation/robodojo/runtime_assets"
OMNI_KIT_ACCEPT_EULA=YES python -m lerobot.evaluation.eval_qwen3vl_mot_robodojo_delta eval \
  --checkpoint /data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_stack_bowl_2nodes_8gpu_decay_20k_high_noise_shift_5/step14k \
  --qwen3vl-dir "$CACHE/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$CACHE/pretrained/Cosmos3-Edge" \
  --stage1-text-dir "$CACHE/pretrained/siglip2-base-patch16-224" \
  --task stack_bowls_random --eval-num 3 --num-envs 1 \
  --gpu 0 --sim-gpu 1 --replan-steps 10 --seed 0 \
  --inference-steps 10 --save-predicted-video --port 19141 --timeout 7200 \
  --output-dir outputs/evaluation/robodojo/stack_bowls_random_delta_high_noise_shift5_step14k_sampling10_20261029_v1
```

The instrumented run uses the existing session-local camera wrapper around
this entry, without editing the shared simulator. Its final audit explicitly
validates `stack_bowls_random` results and allows the intentional task change
relative to ordinary step14k; it does not substitute ordinary-task scores.

Startup has been verified against a real first action chunk. The policy
and simulator processes are live, and the simulator command selects
`stack_bowls_random`. Relative to ordinary step14k, saved settings differ
only in `task`, `output_dir`, and `port`; the checkpoint and model assets
are unchanged. The first `(32, 20)` action chunk contains only finite
numeric values and executes ten steps. Layout0's initial robot state
happens to match the ordinary run exactly, without requiring the object
scenes to match.

The initial head-camera input was visually inspected: three white bowls
are visible among substantial distractor clutter on the randomized scene
surface. This confirms that the first real model input is not the ordinary
uncluttered bowls scene. Initial camera transforms are recorded and stable.
`startup_validation.json` records these numerical/configuration checks;
final native success scores and full-run audits are still pending.

The run subsequently completed all three requested layouts. Native success
is **0/3 (0%)**, versus **2/3 (66.67%)** for the same checkpoint on ordinary
`stack_bowls`. Random layouts 0/1/2 each scored 0, ran the full 800 control
steps, and had zero native unstable-episode flags. Wall time was 4415.78s
(73.60min); mean policy inference and prediction export were 8.1422s and
5.2051s per chunk. Different episode lengths and shared-machine load mean
this is not a controlled inference-speed comparison.

All 240 chunks contain finite numerical arrays. Inverse normalization and
prediction-export action parity are exact; maximum translation, rotation
matrix, and gripper integration errors are respectively `5.97e-8`,
`1.02e-7`, and `2.99e-8`. The checkpoint's recorded sizes/mtimes and random
task-config/layout SHA256 fingerprints remain unchanged. The final CPU
audit initially lacked the repository `PYTHONPATH`; rerunning only that
audit with the repository import path passed, without restarting inference.

All nine actual videos are 640x480 at 25 FPS, with 801 frames each.
There are also 720 prediction videos and 7920 exported PNGs. Fifty-one
passive camera snapshots show zero head/world or within-layout static-scene
transform drift, correct render bindings, and wrist-to-EE transform errors
below `3.3e-7`. All 240 conditioning head thumbnails, three sampled native
head timelines, and nine final native views were visually reviewed: the
bowls remain separate in all three scenes. Arms move, retract, and rotate
without completing stacking; some later right-wrist views face the room
or ceiling, consistent with the arm pose rather than camera drift.

Target tracking is also worse than in the ordinary run. Left/right
median position errors are 3.24/4.50mm and maxima 62.71/548.94mm.
The largest right-arm deviation occurs in layout0 at control590→600;
this measures commanded target versus subsequent physical EE position,
not denoising or demonstration error. The numerical adapter checks pass,
but that does not establish that the generated targets are reachable or
that physical tracking is adequate.

These three cases show a substantial drop on the combined randomization
(bowl models, clutter, and scene appearance), not an isolated causal test
of any one factor or a reliable full-task success-rate estimate.
Final reports are `final_adapter_audit.json` and
`camera_audit/camera_integrity_report.json` under the output directory.
Actual videos are under:

```text
outputs/evaluation/robodojo/stack_bowls_random_delta_high_noise_shift5_step14k_sampling10_20261029_v1/
  simulator/eval_result/RoboDojo/stack_bowls_random/Qwen3VLMoT/arx_x5/
    0_action_type=ee,simulator=isaacsim6/
      stack_bowls_random_delta_high_noise_shift5_step14k_sampling10_20261029_v1/
        episode_000000{0,1,2}_cam_{head,left_wrist,right_wrist}_fail.mp4
```

Prediction videos remain under `policy/predictions/`, and the reviewed
contact sheets are under `diagnostics/`. Owned policy/simulator processes
have exited; no unrelated jobs were stopped.

#### High-noise shift5 step14k evaluation (2026-10-29)

The requested checkpoint is
`/data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_stack_bowl_2nodes_8gpu_decay_20k_high_noise_shift_5/step14k`.
All 11 files (25,187,233,961 bytes, including eight DCP shards) were
SHA256-fingerprinted with unchanged sizes and modification times throughout
verification. All 9,260 storage references fit their shard files. The 2,560
tensor keys, shapes, and dtypes match step9k; policy and training config
SHA256 hashes also match step9k. All eight weight-shard hashes differ from
step9k, confirming this is not the previous checkpoint.

The run reads verified weights directly from `/data2` without duplicating
25 GB on the local filesystem. Provenance is retained as
`runtime_assets/robodojo_stack_bowl_2nodes_8gpu_decay_20k_high_noise_shift_5/checkpoint_provenance_step14k.json`
under `outputs/evaluation/robodojo`.

The matched protocol is `stack_bowls`, layouts 0/1/2, seed0, one environment,
ten sampling steps, 32 predicted actions, ten executed actions, BF16,
actual/prediction videos, and the same passive camera wrapper used for
step9k. This is a three-layout diagnostic, not the full benchmark.
Preparation does not establish a task result.

```bash
conda activate wam
CACHE="$PWD/outputs/evaluation/robodojo/runtime_assets"
OMNI_KIT_ACCEPT_EULA=YES python -m lerobot.evaluation.eval_qwen3vl_mot_robodojo_delta eval \
  --checkpoint /data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_stack_bowl_2nodes_8gpu_decay_20k_high_noise_shift_5/step14k \
  --qwen3vl-dir "$CACHE/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$CACHE/pretrained/Cosmos3-Edge" \
  --stage1-text-dir "$CACHE/pretrained/siglip2-base-patch16-224" \
  --task stack_bowls --eval-num 3 --num-envs 1 \
  --gpu 0 --sim-gpu 1 --replan-steps 10 --seed 0 \
  --inference-steps 10 --save-predicted-video --port 19140 --timeout 7200 \
  --output-dir outputs/evaluation/robodojo/stack_bowls_delta_high_noise_shift5_step14k_sampling10_20261029_v1
```

The command above is the standard entry; this instrumented run uses the
existing session-local camera wrapper without changing the shared simulator.

The instrumented evaluation started successfully: the policy server loaded
12.585B parameters, enabled only joint prediction, and used ten denoising
steps. The first native action chunk is finite with shape `(32, 20)`.
Settings differ from step9k only in checkpoint, output directory, and port;
layout0 initial state matches step9k exactly. The initial head camera has
zero transform drift, and visual inspection shows the expected table,
three bowls, and robot framing. Evidence is retained in
`startup_validation.json` and `camera_audit/`. The detached run is still
in progress at this startup check; no success or score result is available.
Numerical, video, and passive-camera audits are chained automatically
after native evaluation completes.

The run subsequently completed uninterrupted in **1871.12 seconds
(31.19 minutes)**. The matched checkpoint comparison is:

| Metric | Shift5 step7k | Shift5 step9k | Shift5 step14k |
|---|---|---|---|
| Sampling steps | 10 | 10 | 10 |
| Full success | 0/3 | 1/3 (33.33%) | **2/3 (66.67%)** |
| Layout 0/1/2 success | no / no / no | no / no / yes | **yes / no / yes** |
| Layout 0/1/2 normalized scores | 0 / 0 / 0 | 0.15 / 0 / 1.0 | **1.0 / 0 / 1.0** |
| Native aggregate score (0--100) | 0.0 | 38.33 | **66.67** |
| Layout 0/1/2 control steps | 800 / 800 / 800 | 800 / 800 / 367 | **495 / 800 / 381** |

Layout0 now completes the stack, and layout2 remains successful. The final
native head frames show nested bowls after release in both successful
layouts. Layout1 still leaves the bowls separate and reaches the 800-step
limit. All three native unstable-episode counts are zero. This demonstrates
improvement on these three fixed layouts only, not full-benchmark success
or statistical evidence of a general improvement.

All 169 action chunks are finite, pass inverse-normalization and cumulative
delta-integration checks, and exactly match their prediction action exports.
Inverse-normalization error is zero; integration errors are below `1e-7`.
Initial states match step9k exactly in every layout. All nine actual videos
decode at 640x480/25 FPS, with 496/801/382 frames per layout/view. The run
also saves 507 prediction videos and 5,577 prediction PNGs. All 35 passive
camera snapshots have stable head/scene transforms and correct render
bindings; wrist-to-EE transform errors remain below `3.1e-7`. Source
checkpoint sizes and modification times are unchanged throughout the run.

Visual review covers all 169 conditioning head-image thumbnails, sampled
native head-video contact sheets, and final head/left-wrist/right-wrist
frames for all layouts; see `diagnostics/native_final_frames.jpg`.
The persisted `final_adapter_audit.json` records numerical, camera, and
visual checks. Previous step9 target versus next measured anchor has left
median/max tracking discrepancy 0.24/4.18 mm and right 0.11/13.42 mm;
these are physical tracking checks, not demonstration prediction errors.
Mean model inference and prediction export times are 4.43 s and 3.72 s
per chunk. Early successful termination and different shared-machine load
both affect wall time; do not attribute the speed difference to weights.

#### High-noise shift5 step9k evaluation (2026-10-28)

The newly uploaded checkpoint is
`/data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_stack_bowl_2nodes_8gpu_decay_20k_high_noise_shift_5/step9k`.
All 11 files (25,187,233,961 bytes, including eight DCP shards) were
SHA256-fingerprinted; their sizes and modification times remained unchanged
through verification. All 9,260 DCP storage references fit their shard files.
The 2,560 tensor keys, shapes, and dtypes match step7k, and the saved policy
configuration is unchanged: joint prediction only, shifted-uniform shift5,
32 predicted actions, and 20 saved inference steps.

The run reads these verified weights directly from `/data2` without creating
another 25 GB local copy. Provenance is retained as
`runtime_assets/robodojo_stack_bowl_2nodes_8gpu_decay_20k_high_noise_shift_5/checkpoint_provenance_step9k.json`
under `outputs/evaluation/robodojo`.

The matched evaluation uses `stack_bowls`, layouts 0/1/2, seed0, one
environment, ten sampling steps, 32 predicted actions, ten executed actions,
BF16, actual/prediction videos, and the same passive camera audit as step7k.
The output directory is
`outputs/evaluation/robodojo/stack_bowls_delta_high_noise_shift5_step9k_sampling10_20261028_v1`.
Preparation alone is not a task-success result; this is a three-layout
diagnostic, not the full benchmark.

```bash
conda activate wam
CACHE="$PWD/outputs/evaluation/robodojo/runtime_assets"
OMNI_KIT_ACCEPT_EULA=YES python -m lerobot.evaluation.eval_qwen3vl_mot_robodojo_delta eval \
  --checkpoint /data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_stack_bowl_2nodes_8gpu_decay_20k_high_noise_shift_5/step9k \
  --qwen3vl-dir "$CACHE/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$CACHE/pretrained/Cosmos3-Edge" \
  --stage1-text-dir "$CACHE/pretrained/siglip2-base-patch16-224" \
  --task stack_bowls --eval-num 3 --num-envs 1 \
  --gpu 0 --sim-gpu 1 --replan-steps 10 --seed 0 \
  --inference-steps 10 --save-predicted-video --port 19090 --timeout 7200 \
  --output-dir outputs/evaluation/robodojo/stack_bowls_delta_high_noise_shift5_step9k_sampling10_20261028_v1
```

The command above is the standard entry; this instrumented run uses the
existing session-local camera wrapper around that entry without modifying
the shared simulator.

The instrumented evaluation started successfully: the policy server loaded
12.585B parameters with the single joint-prediction task and ten denoising
steps, and returned a finite `(32, 20)` action chunk to the native simulator.
Settings match the step7k run except for checkpoint, output directory, and
port. The layout0 initial state matches step7k exactly; the initial head
camera transform is stable, and the saved actual head image shows the
expected table, three bowls, and robot framing. Evidence is retained in
`startup_validation.json` and `camera_audit/`. The detached run is still
in progress at this check; no success or score result is available yet.
Numerical, video, and passive-camera audits are scheduled automatically
after native evaluation finishes.

The run subsequently completed uninterrupted in **2993.22 seconds
(49.89 minutes)**. Compared with the matched step7k run:

| Metric | Shift5 step7k, sampling10 | Shift5 step9k, sampling10 |
|---|---|---|
| Full success | 0/3 | **1/3 (33.33%)** |
| Layout 0/1/2 success | no / no / no | no / no / **yes** |
| Layout 0/1/2 normalized scores | 0 / 0 / 0 | **0.15 / 0 / 1.0** |
| Native aggregate score (0--100) | 0.0 | **38.33** |
| Layout 0/1/2 control steps | 800 / 800 / 800 | 800 / 800 / **367** |
| Mean model inference per chunk | 6.21 s | 6.15 s |
| Mean prediction-video export per chunk | 5.04 s | 5.10 s |

Layout2 completes the stack; its final native head frame shows three nested
bowls. Layout0 receives partial credit but does not stack the third bowl.
Layout1 reaches the step limit without a completed stack. All layouts have
zero native unstable-episode flags. This is an improvement on these three
fixed layouts, not an estimate of full-benchmark success or statistical
evidence of a general improvement. The shorter run also reflects layout2
terminating early; shared-machine load prevents causal speed comparisons.

All 197 chunks contain finite numeric arrays, pass inverse-normalization
and delta-integration checks, and match their exported prediction actions
exactly. Inverse-normalization error is zero and integration errors are
below `1e-7`. Initial states match step7k exactly on all three layouts.
All nine actual videos decode at 640x480/25 FPS, with 801/801/368 frames
per layout/view. There are 591 prediction videos and 6,501 prediction PNGs.
All 42 passive snapshots have stable head/scene transforms and correct
camera bindings; wrist-to-EE transform errors remain below `3.2e-7`.
Source checkpoint file sizes and modification times remain unchanged.

Visual review covers all 197 conditioning head-image thumbnails, sampled
native head-video contact sheets, and final native views from all cameras.
The final-frame sheet is `diagnostics/native_final_frames.jpg`. Numerical,
camera, and visual-review results are retained in `final_adapter_audit.json`.
Tracking is not perfect despite the adapter checks: previous step9 target
versus next measured anchor has left median/max 0.26/7.63 mm and right
0.07/111.90 mm. The right maximum occurs in unsuccessful layout1 between
control steps680 and690; these are physical tracking discrepancies, not
demonstration prediction errors. Details are in `worst_target_tracking.json`.

#### High-noise shift5 step7k evaluation (2026-10-28)

The requested experiment is
`/data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_stack_bowl_2nodes_8gpu_decay_20k_high_noise_shift_5`.
Its `step7k` model and experiment-specific configurations were copied to
the local `runtime_assets` cache. All 19 files (25,200,777,614 bytes,
including 16 DCP shards) match source SHA256 checksums; source sizes and
modification times remained unchanged during verification. All 16,908
DCP storage references fit their shard files. Provenance is retained as
`checkpoint_provenance_step7k.json` in the experiment cache.

This checkpoint uses `shifted_uniform` training times with shift 5,
sigma range 0.001--0.999, and only the `joint_prediction` task. Its 2,560
model tensors match step16k keys and dtypes; the task embedding has one
row instead of three, matching the new saved configuration. Loading must
use this experiment's own configuration, not the older step16k config.
The evaluation uses ten sampling steps instead of the saved twenty,
without modifying the checkpoint configuration.

The matched protocol is `stack_bowls`, layouts 0/1/2, seed 0, one
environment, 32 predicted actions, ten executed actions per replan,
BF16 inference, and actual/predicted video export. It uses the same
session-local passive camera wrapper as the previous runs. This is a
three-layout diagnostic, not the full benchmark. Differences from the
previous step16k experiment include training tasks, training recipe, and
checkpoint step; the comparison does not isolate a noise-shift effect.

```bash
conda activate wam
CACHE="$PWD/outputs/evaluation/robodojo/runtime_assets"
OMNI_KIT_ACCEPT_EULA=YES python -m lerobot.evaluation.eval_qwen3vl_mot_robodojo_delta eval \
  --checkpoint "$CACHE/robodojo_stack_bowl_2nodes_8gpu_decay_20k_high_noise_shift_5/step7k" \
  --qwen3vl-dir "$CACHE/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$CACHE/pretrained/Cosmos3-Edge" \
  --stage1-text-dir "$CACHE/pretrained/siglip2-base-patch16-224" \
  --task stack_bowls --eval-num 3 --num-envs 1 \
  --gpu 0 --sim-gpu 1 --replan-steps 10 --seed 0 \
  --inference-steps 10 --save-predicted-video --port 19070 --timeout 7200 \
  --output-dir outputs/evaluation/robodojo/stack_bowls_delta_high_noise_shift5_step7k_sampling10_20261028_v1
```

Preparation verifies checkpoint integrity and protocol only; it does not
establish a task result. Logs and artifacts are retained in the output
directory above, with actual videos under `simulator/eval_result/` and
predicted videos under `policy/predictions/`.

The instrumented run started successfully. Strict model loading restored
all 2,560 tensors using the single-task configuration; the simulator
returned its first native observation and received a finite `(32, 20)`
physical-action chunk. The layout0 initial state matches the step16k
ten-step baseline exactly. After resolving relative/absolute asset paths,
the coordinator settings differ only in checkpoint, output directory, and
port. The initial head-camera snapshot and actual conditioning image show
the expected table/bowl framing. Evidence is saved in
`startup_validation.json`. Evaluation is still running at this startup
check; no task success result is available yet.

The run subsequently completed uninterrupted in **3619.45 seconds
(60.32 minutes)**. All three layouts reached the 800-control-step limit
with no native unstable-episode flags:

| Metric | Previous step16k, sampling10 | Shift5 step7k, sampling10 |
|---|---|---|
| Full success | 0/3 | 0/3 |
| Layout 0/1/2 normalized scores | 0 / 0 / 0 | 0 / 0 / 0 |
| Native aggregate score | 0.0 | 0.0 |
| Mean model inference per chunk | 7.16 s | 6.21 s |
| Mean prediction-video export per chunk | 5.96 s | 5.04 s |
| Wall time including startup | 68.43 min | 60.32 min |

These three layouts show no improvement in native success or partial
credit. Training tasks, training recipe, and checkpoint step differ;
this is not an isolated noise-shift experiment. Shared-machine load also
prevents attributing the wall-time difference to the model configuration.

All 240 saved chunks contain finite numeric arrays, pass inverse
normalization and cumulative delta-integration checks, and exactly match
their exported prediction action arrays. Maximum integration errors are
below `1e-7`; inverse-normalization error is zero. Initial states match the
previous step16k run exactly for all three layouts. Target-to-next-measured
position discrepancy has median/max 0.14/17.19 mm on the left and
0.33/30.47 mm on the right; these are tracking measurements, not errors
against demonstration actions.

All 240 conditioning head images and native head-video contact sheets
were reviewed: table/bowl framing remains normal, with reaching/contact
and bowl movement but no completed stack. All 51 passive snapshots retain
fixed head/scene transforms and correct render-camera bindings; wrist-to-EE
transform errors remain below `4e-7`.

The output retains nine actual camera videos (640x480, 25 FPS, 801 frames
each), 720 prediction videos, and 7920 prediction PNGs. See
`summary.json`, `final_adapter_audit.json`,
`camera_audit/camera_integrity_report.json`, and `diagnostics/` for the
native result, numerical checks, and reviewed contact sheets. Actual
videos are nested under `simulator/eval_result/`; predicted videos are
under `policy/predictions/`. This remains a three-layout diagnostic, not
the full benchmark.

##### Shift5 step7k state/action prediction errors

The state diagnostic excludes chunk index0: joint sampling overwrites it
with the current observation, so its zero error is not predictive accuracy.
Future states use indices 1--31; the first-ten-state prefix uses 1--10.
Actions use all 32 predictions or indices 0--9 for the executed prefix.
Only the 20 active dimensions enter MSE. State and action MSE use different
normalization statistics and must not be directly compared.

```bash
conda activate wam
CACHE="$PWD/outputs/evaluation/robodojo/runtime_assets"
CUDA_VISIBLE_DEVICES=0 python -m lerobot.evaluation.eval_qwen3vl_mot_robodojo_delta diagnose \
  --checkpoint "$CACHE/robodojo_stack_bowl_2nodes_8gpu_decay_20k_high_noise_shift_5/step7k" \
  --qwen3vl-dir "$CACHE/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$CACHE/pretrained/Cosmos3-Edge" \
  --stage1-text-dir "$CACHE/pretrained/siglip2-base-patch16-224" \
  --diagnostic-episodes 0 6 --diagnostic-frames 0 \
  --diagnostic-inference-steps 10 20 --replan-steps 10 --seed 0 \
  --output-dir outputs/evaluation/robodojo/stack_bowls_delta_high_noise_shift5_step7k_state_action_error_20261028_v1
```

This BF16 measurement uses the same four windows as the previous step16k
action diagnostic: episode/global-frame pairs `(6,2299)`, `(85,39103)`,
`(0,0)`, and `(6,1884)`. All have a complete, unpadded 32-frame window.
Both state and action training/deployment sampler outputs match exactly.
Saved arrays independently reproduce the reported MSEs. These are four
**training-data** windows, not held-out generalization or benchmark-wide
error estimates. Each window uses the same seed at 10 and 20 steps.

| Normalized MSE, equally averaged across windows | Sampling10 | Sampling20 |
|---|---|---|
| Future state, 31 frames | 0.070089 | 0.076964 |
| Future state, first 10 frames | 0.059804 | 0.062515 |
| Action, 32 steps | 0.194308 | 0.245919 |
| Action, first 10 steps | 0.133362 | 0.146457 |

Physical errors below use **ten sampling steps**. Position is mean
Euclidean error, not per-coordinate MAE; orientation is mean geodesic
rotation error. Action-target errors compare predicted and demonstration
delta sequences integrated from the same measured anchor, not predicted
states or actual simulator tracking.

| Compared quantity | Position L/R | Rotation L/R |
|---|---|---|
| Absolute future state, 31 frames | 27.64 / 44.78 mm | 6.67 / 8.19 deg |
| Absolute future state, first 10 frames | 28.28 / 39.57 mm | 6.52 / 6.43 deg |
| Integrated action targets, 32 steps | 9.03 / 22.33 mm | 2.04 / 4.18 deg |
| Integrated action targets, first 10 steps | 3.20 / 6.62 mm | 0.62 / 1.01 deg |
| Individual action increments, 32 steps | 1.03 / 2.60 mm | 0.27 / 0.43 deg |
| Individual action increments, first 10 steps | 0.91 / 2.14 mm | 0.18 / 0.37 deg |

Raw state gripper MAE is 0.0524/0.0648 over future states and
0.0601/0.1074 over the first ten. Integrated-action clipped-gripper MAE is
0.0057/0.0356 over 32 targets and 0.0024/0.0078 over the first ten.
Gripper values use the native 0--1 scale. Raw state gripper errors are
not clipped to hide out-of-range predictions.

For context, simply repeating the current state gives first-ten-state
position errors of 5.13/21.08 mm and rotation errors of 1.26/3.16 deg on
these windows, better than the model's short-term state forecast. Over
all 31 future frames, persistence gives 26.04/73.21 mm, so the model's
position improvement is not uniform across arms or horizons.

A separate check uses the completed native rollout, not demonstrations.
Its 237 within-episode chunk pairs compare predicted state index10 with
the next measured observation after executing action indices0--9:

| State forecast at +10 control steps (0.4 s) | Left | Right |
|---|---|---|
| Mean position error | 48.63 mm | 57.96 mm |
| 95th-percentile position error | 83.84 mm | 105.23 mm |
| Maximum position error | 102.33 mm | 152.57 mm |
| Mean rotation error | 19.86 deg | 19.17 deg |
| Mean raw gripper error | 0.1574 | 0.1844 |
| Hold-current baseline mean position error | 10.17 mm | 13.42 mm |
| Hold-current baseline mean rotation error | 2.13 deg | 3.32 deg |

Rollout-state normalized MSE is 0.431321. These numbers indicate weak
short-term state forecasting, including worse errors than persistence in
this rollout. They do not establish that the state head caused task
failure. Actual execution tracking and imitation-target accuracy remain
different measurements.

Raw label predictions, state targets, and per-window metrics are in
`diagnostics.json` and `sample_*/{actions.npz,metrics.json}` under the
diagnostic output. `aggregate_state_action_error.json` retains the
independently checked aggregates and persistence baseline.
`rollout_state_forecast_error.json` in the native evaluation output
retains aligned rollout-pair errors and their scope.

##### Why a small training `state_pred` loss does not match sampling error

The trainer's `state_pred` display maps to
`state_prediction_reconstruction_loss`, not `state_flow_loss`.
For embedding-space flow, the reconstruction metric is computed from
`decode(z_sigma - sigma * predicted_velocity)` against normalized real
states, where `z_sigma = (1-sigma)*encode(real_states) + sigma*noise`.
The other target modalities in joint training also contain correspondingly
corrupted real futures. This is the intended denoising objective, not a
pure-noise rollout or evidence of an unintended future-condition leak.

Deployment instead initializes all future streams from noise and
repeatedly feeds its own intermediate predictions through the model.
One-step reconstruction accuracy on training corruption is therefore not
the same quantity as multi-step free-running forecast accuracy.
Both metrics use the same 31 future frames and 20 active dimensions here:
each probed training sample reports exactly 620 valid state-reconstruction
elements. The loss is not diluted by padded dimensions or the clamped
current state. The saved reconstruction weight 5 scales its contribution
to the total optimization loss, not the reported `state_pred` value.

A fixed-sigma joint-training-style forward probe on the same four
step7k windows gives:

| Evaluation input/procedure | Mean normalized state reconstruction MSE |
|---|---|
| Real future corrupted at sigma 0.1, one forward | 0.052600 |
| Real future corrupted at sigma 0.5, one forward | 0.005537 |
| Real future corrupted at sigma 0.75, one forward | 0.006116 |
| Real future corrupted at sigma 0.9, one forward | 0.007191 |
| Real future corrupted at sigma 0.99, one forward | 0.011451 |
| Pure-noise endpoint sigma 1.0, one forward | 0.011584 |
| Free-running 10-step integration | 0.070089 |
| Free-running 20-step integration | 0.076964 |

These are BF16 probes, with fixed per-window noise seeds; the one-forward
and sampler procedures do not use identical noise tensors or trajectories.
They demonstrate a loss/forecast gap but do not quantitatively isolate its
causes or reproduce the user's full-training average of 0.004.

There is a specific warning sign in this checkpoint: its low-noise
sigma0.1 reconstruction error is much worse than its mid-noise error.
Under the saved shifted-uniform distribution with shift5 and sigma range
0.001--0.999, only 2.15% of draws are below sigma0.1 and 16.67% below
sigma0.5. The linear inference schedule must nevertheless traverse this
low-noise region. Undertraining or distribution mismatch there may worsen
sampling, but these probes alone do not prove it causes the rollout
failure or establish which noise schedule would fix it.

Finally, 0.004 is standardized MSE over mixed state coordinates, not
meters, millimeters, or pose error: its square root is approximately
0.063 standardized units. It does not imply 0.004-meter position error.
Free-running state validation, physical pose errors, and per-sigma
denoising metrics should be tracked separately from the training average.
Full probe values and valid-element counts are retained in
`conditional_state_loss_sigma_probe.json` under the state/action
diagnostic output above.

## All non-random tasks: full-task shift5 step16k

The sequential suite entry point is
`python -m lerobot.evaluation.eval_qwen3vl_mot_robodojo_suite eval`.
For the installed native RoboDojo checkout this selects **42 tasks**, excluding
all 12 `*_random` definitions, and evaluates native layouts **0, 1, 2** for
each task: **126 episodes**. Bowls and bottles are rerun from scratch; older
individual results are not mixed into this sweep.

The checkpoint is
`/data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_2nodes_8gpu_high_noise_shift_5_decay_25k/step16k`.
The protocol remains seed0, one environment, sampling10, prediction32,
execution10, and BF16, with actual three-camera videos and predicted videos.
Each task runs in its own evaluator process with policy GPU0 and simulator
GPU1; RNG is continuous across that task's three layouts. Original native
task horizons are retained, from 200 to 1900 controls, rather than imposing
the bowls task's 800-control cap. The maximum scheduled total is 98,400
controls. This is a long, sequential shared-machine run, not a short smoke
test or the official 50-episode-per-task benchmark.

Run from the repository root using the existing environments and cached
pretrained assets:

```bash
unset ROBODOJO_STOP_AFTER_FIRST_FAILURE
export PYTHONPATH="$PWD" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export OMNI_KIT_ACCEPT_EULA=YES HF_HUB_OFFLINE=1
/home/jianf/miniforge3/envs/wam/bin/python -u \
  -m lerobot.evaluation.eval_qwen3vl_mot_robodojo_suite eval \
  --checkpoint /data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_2nodes_8gpu_high_noise_shift_5_decay_25k/step16k \
  --dataset-root /data2/lizuolei/data/lerobot_data/simulation_data/RoboDojo/data/RoboDojo_ee_lerobot_v30_video \
  --qwen3vl-dir "$PWD/outputs/evaluation/robodojo/runtime_assets/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$PWD/outputs/evaluation/robodojo/runtime_assets/pretrained/Cosmos3-Edge" \
  --stage1-text-dir "$PWD/outputs/evaluation/robodojo/runtime_assets/pretrained/siglip2-base-patch16-224" \
  --checkpoint-provenance "$PWD/outputs/evaluation/robodojo/runtime_assets/robodojo_2nodes_8gpu_high_noise_shift_5_decay_25k/checkpoint_provenance_step16k.json" \
  --camera-wrapper /home/jianf/lzl_project/.copilot/session-state/1341fac2-578e-4665-a582-dc35f01c3d5b/files/inspect_robodojo_camera_run.py \
  --gpu 0 --sim-gpu 1 --port 19188 --seed 0 \
  --inference-steps 10 --replan-steps 10 \
  --save-predicted-video --timeout 14400 \
  --output-dir "$PWD/outputs/evaluation/robodojo/nonrandom42_alltask_shift5_decay25k_step16k_sampling10_20261104_v1"
```

`--camera-wrapper` points to the existing passive camera instrumentation;
it can be omitted for an uninstrumented run. `--tasks TASK ...` selects an
explicit non-random subset, but is deliberately omitted for this sweep.
The output directory must be fresh. To resume an interrupted sweep, repeat
the **identical command** with `--resume`, using the same wrapper, output
directory, code, task/layout sources, checkpoint, and dataset.

The output root contains:

| Artifact | Meaning |
|---|---|
| `manifest.json` | Exact protocol, checkpoint/data provenance, task horizons/layout hashes, and runtime code hashes |
| `progress.json` | Durable task states, attempts, completed-episode denominator, successes and aggregate score |
| `results.csv` | One row per task with success count out of three, native score, output path and any error |
| `tasks/TASK/attempt_001/` | Independent native evaluator output, policy chunks, predictions, and actual videos |
| `tasks/TASK/attempt_001.log` | Preserved evaluator startup/runtime log |
| `tasks/TASK/attempt_001/suite_audit.json` | Episode, chunk integration/export, camera, tracking and video audits |
| `tasks/TASK/attempt_001/native_final_frames.jpg` | Exact final frames of all nine native videos for visual review |

Infrastructure errors are recorded separately and excluded from the
completed-episode success denominator, **not reported as policy 0/3**.
The suite continues to other tasks, but exits incomplete if any task lacks
three valid episodes. Resume skips audited completed tasks (including
genuine policy failures), retains earlier error attempts, and starts a
new attempt only for incomplete tasks. There are no automatic retries or
success-based result selection. The final sweep is complete only with
42 valid task results / 126 evaluated episodes.

Full checkpoint/data SHA256 checks run at both ends; source size/mtime,
small metadata hashes, task/layout hashes, and runtime code hashes are
rechecked between tasks. Source changes or less than 10 GiB free disk
stop the sweep without deleting files. A coordinator lock prevents
duplicate runs. Actual video frame counts must equal controls+1 at
640x480/25FPS; predicted exports must match every logged chunk.
Automatic audits do not claim human visual review:
`visual_review_pending` remains true.

Preparation validation: **56 focused tests passed**, with Ruff lint and
format checks passing. The generic audit also passed on the completed
real bottles run: 202 chunks, 43 passive camera snapshots, nine actual
videos, and independent integration errors below 1.1e-7.

Startup was verified for the output directory in the command above:
checkpoint/data full-hash preflight passed, the persisted manifest has
42 non-random tasks / 126 episodes and native caps200--1900 (98,400
maximum controls), and the first task `align_blocks` advanced from
control0 to control10. Its first delta chunk is finite with shape32x20,
the task-specific alignment instruction and all three conditioning
images are saved, and the policy server is responsive on port19188.
The sweep is running detached as `nonrandom42-alltask16k-eval`; coordinator
log is
`/home/jianf/lzl_project/.copilot/session-state/1341fac2-578e-4665-a582-dc35f01c3d5b/files/nonrandom42_alltask16k_suite.log`.
No full-sweep success rate is available at startup; use the live
`progress.json` and `results.csv` rather than extrapolating old task results.

The sweep was **paused at the user's request on 2026-11-05**. At pause,
11 tasks had completed with **0/33 successes**; two tasks remained errors:
`fill_egg_holder` encountered unstable scene settling, and `fold_clothes`
failed garment rest-mesh initialization. Neither error is counted as 0/3.
`imitate_sorting_sequence` was interrupted mid-episode and is not included
in the valid-episode denominator. All existing attempts, results and
videos are retained; `progress.json` has status `interrupted`, and a later
explicit resume can retain the completed tasks.

The coordinator exited and port19188 is closed. Its evaluator did not
finish graceful child cleanup within the timeout; the remaining owned
policy/simulator processes were stopped explicitly, with the simulator
requiring forced termination. GPU memory release was verified, and
unrelated GPU jobs were left running. No automatic restart is scheduled.

## Full-task shift5 step18k: bowls and bottles

The requested step18k comparison is limited to **`stack_bowls` and
`put_bottles_into_dustbin`, three native layouts each**, not a restart of
the paused 42-task step16k sweep. The checkpoint is the same experiment's
`robodojo_2nodes_8gpu_high_noise_shift_5_decay_25k/step18k`.
Keep seed0, one environment, BF16, sampling10, prediction32, execution10,
actual/predicted videos, and native horizons800/700 respectively.
For reference, the completed individual step16k runs gave bowls0/3
(scores0/0/0) and bottles1/3 (scores0.25/1/0.25); these are comparison
baselines, not part of the new six-episode denominator.

Preparation waits for all 16 DCP shards and `.metadata` to be complete
and stable, hashes the checkpoint and dataset, checks metadata tensor
keys/shapes/dtypes and storage byte ranges against evaluated step16k,
and verifies the two tasks' layouts/assets. It produces
`runtime_assets/robodojo_2nodes_8gpu_high_noise_shift_5_decay_25k/checkpoint_provenance_step18k.json`
and `two_tasks_step18k_preflight_20261105.json`.
No model weights are copied and no training is performed.

From the repository root, use the same environment exports as the full
suite above:

```bash
unset ROBODOJO_STOP_AFTER_FIRST_FAILURE
export PYTHONPATH="$PWD" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export OMNI_KIT_ACCEPT_EULA=YES HF_HUB_OFFLINE=1
/home/jianf/miniforge3/envs/wam/bin/python -u -c \
  'import runpy, signal; signal.signal(signal.SIGINT, signal.default_int_handler); runpy.run_module("lerobot.evaluation.eval_qwen3vl_mot_robodojo_suite", run_name="__main__")' \
  eval \
  --tasks stack_bowls put_bottles_into_dustbin \
  --checkpoint /data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_2nodes_8gpu_high_noise_shift_5_decay_25k/step18k \
  --dataset-root /data2/lizuolei/data/lerobot_data/simulation_data/RoboDojo/data/RoboDojo_ee_lerobot_v30_video \
  --qwen3vl-dir "$PWD/outputs/evaluation/robodojo/runtime_assets/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$PWD/outputs/evaluation/robodojo/runtime_assets/pretrained/Cosmos3-Edge" \
  --stage1-text-dir "$PWD/outputs/evaluation/robodojo/runtime_assets/pretrained/siglip2-base-patch16-224" \
  --checkpoint-provenance "$PWD/outputs/evaluation/robodojo/runtime_assets/robodojo_2nodes_8gpu_high_noise_shift_5_decay_25k/checkpoint_provenance_step18k.json" \
  --camera-wrapper /home/jianf/lzl_project/.copilot/session-state/1341fac2-578e-4665-a582-dc35f01c3d5b/files/inspect_robodojo_camera_run.py \
  --gpu 0 --sim-gpu 1 --port 19189 --seed 0 \
  --inference-steps 10 --replan-steps 10 \
  --save-predicted-video --timeout 14400 \
  --output-dir "$PWD/outputs/evaluation/robodojo/two_tasks_alltask_shift5_decay25k_step18k_sampling10_20261105_v1"
```

The entry prefix restores SIGINT handling before starting the suite,
including for its evaluator children, rather than inheriting an ignored
interrupt from a detached shell. This changes neither policy sampling
nor native task behavior. Interrupt inheritance was checked with an
actual child Python process before launch. The existing suite retains
its separate per-task logs, attempts, audits, final-frame sheets, videos,
`progress.json`, and `results.csv`; infrastructure errors remain excluded
from the valid-episode success denominator.

Preflight passed: 19 checkpoint/config files, 25,200,777,551 bytes,
2560 tensors and 16,908 valid storage references. All 16 shard hashes
differ from evaluated step16k; saved configs and all five full-task
dataset file hashes are unchanged. The earlier complete dataset delta
audit is therefore reused rather than rerunning the 1,856,102-row
derivation. Both tasks' layouts/assets and original horizons passed.

Startup verified in
`two_tasks_alltask_shift5_decay25k_step18k_sampling10_20261105_v1`:
the persisted manifest selects exactly these two tasks / six episodes,
full-source preflight passed, and `stack_bowls` advanced control0 to10
with the correct instruction, finite32x20 delta actions and all three
conditioning camera images. Policy port19189 is responsive. Actual
coordinator/evaluator/policy/simulator Python processes were checked not
to ignore SIGINT; the previous step16k full sweep remains paused.
Detached shell is `alltask18k-two-eval` (coordinator PID1685270).
Coordinator log is
`/home/jianf/lzl_project/.copilot/session-state/1341fac2-578e-4665-a582-dc35f01c3d5b/files/alltask18k_two_suite.log`.
Success counts are pending; live results are in the new output root's
`progress.json` and `results.csv`.

### Completed step18k results

The two-task run exited0 with **six valid episodes, zero infrastructure
errors and 0/6 successes**. All three native layouts ran to the original
task horizons; there were no retries, substituted layouts or RNG restarts
within a task.

| Task | Step16k successes | Step18k successes | Step18k scores, layouts0/1/2 | Step18k aggregate score /100 |
|---|---:|---:|---|---:|
| `stack_bowls` | 0/3 | 0/3 | 0 / 0.15 / 0 | 5 |
| `put_bottles_into_dustbin` | 1/3 | 0/3 | 0.4 / 0.4 / 0.25 | 35 |

Compared with the matched individual step16k runs, bowls gains a small
partial score but still never completes a stack; bottles loses the
previous layout1 success. These three-layout observations do not show
an improvement at step18k and are too small to establish a reliable
overall checkpoint ranking.

Bowls used controls800/800/800, 240 chunks and 3315.84 seconds;
bottles used700/700/700, 210 chunks and 3106.23 seconds. These per-task
times include evaluator startup but exclude suite-level hash preflight.
All 450 chunks passed numerical integration/export audits, with maximum
integration-component errors below1.1e-7. The completed coordinator
verified full checkpoint/data SHA256 at the end; native asset, layout
and runtime code hashes and completed-result artifacts were independently
rechecked afterward. All 96 passive camera observations passed.

Artifacts total **1350 predicted-camera MP4s, 14,850 prediction/conditioning
PNGs and 18 actual-camera MP4s**. All nine bowls native videos have801
frames; all nine bottles native videos have701 frames, at640x480/25FPS.
Per-task actual videos are under
`tasks/TASK/attempt_001/simulator/eval_result/RoboDojo/TASK/Qwen3VLMoT/arx_x5/0_action_type=ee,simulator=isaacsim6/attempt_001/`;
predictions are under `tasks/TASK/attempt_001/policy/predictions/`.

Both `native_final_frames.jpg` sheets were visually reviewed: all18 exact
final camera frames, not the complete rollout videos or all conditioning
images. Bowls layout1 has overlapping bowls near the left gripper, with
the third still separate; no layout completes the three-bowl stack.
Bottles remain outside the bin in every final layout. The review scope
and matched step16k comparison are recorded in `completion_review.json`;
the generic audit's broader `visual_review_pending` flag is not cleared
to imply complete video review.

Physical target tracking remains distinct from numerical integration:
maximum sampled translation discrepancies were left/right0.899/5.111mm
for bowls and6.539/7.744mm for bottles. Correct export and these sampled
tracking measurements do not prove accurate contact, grasping or task
completion, nor do the final frames establish the failure mechanism.
Policy port19189 is closed, evaluation GPU memory is released, unrelated
jobs remain running, and the previous step16k full sweep stays paused.

## Full-task shift5 step20k: bowls and bottles

The step20k comparison uses the same experiment and the same two tasks as
step18k: **`stack_bowls` and `put_bottles_into_dustbin`, three native
layouts each**. It does not resume the paused step16k full sweep.
Keep seed0, one environment, BF16, sampling10, prediction32, execution10,
continuous RNG across each task's layouts, passive camera audits, and
actual/predicted videos. Original native horizons remain800/700.

Preparation checks a complete, stable 16-shard DCP checkpoint and metadata,
all storage byte ranges and tensor keys/shapes/dtypes, and hashes every
checkpoint/config and full-task dataset file. Compare the new shard
hashes against evaluated step18k and require unchanged saved configs and
dataset statistics before making a matched comparison. Task layouts,
assets and the measured-state delta contract are checked as before.
Evidence is saved in
`runtime_assets/robodojo_2nodes_8gpu_high_noise_shift_5_decay_25k/checkpoint_provenance_step20k.json`
and `two_tasks_step20k_preflight_20261106.json`.

Run from the repository root:

```bash
unset ROBODOJO_STOP_AFTER_FIRST_FAILURE
export PYTHONPATH="$PWD" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export OMNI_KIT_ACCEPT_EULA=YES HF_HUB_OFFLINE=1
/home/jianf/miniforge3/envs/wam/bin/python -u -c \
  'import runpy, signal; signal.signal(signal.SIGINT, signal.default_int_handler); runpy.run_module("lerobot.evaluation.eval_qwen3vl_mot_robodojo_suite", run_name="__main__")' \
  eval \
  --tasks stack_bowls put_bottles_into_dustbin \
  --checkpoint /data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_2nodes_8gpu_high_noise_shift_5_decay_25k/step20k \
  --dataset-root /data2/lizuolei/data/lerobot_data/simulation_data/RoboDojo/data/RoboDojo_ee_lerobot_v30_video \
  --qwen3vl-dir "$PWD/outputs/evaluation/robodojo/runtime_assets/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$PWD/outputs/evaluation/robodojo/runtime_assets/pretrained/Cosmos3-Edge" \
  --stage1-text-dir "$PWD/outputs/evaluation/robodojo/runtime_assets/pretrained/siglip2-base-patch16-224" \
  --checkpoint-provenance "$PWD/outputs/evaluation/robodojo/runtime_assets/robodojo_2nodes_8gpu_high_noise_shift_5_decay_25k/checkpoint_provenance_step20k.json" \
  --camera-wrapper /home/jianf/lzl_project/.copilot/session-state/1341fac2-578e-4665-a582-dc35f01c3d5b/files/inspect_robodojo_camera_run.py \
  --gpu 0 --sim-gpu 1 --port 19190 --seed 0 \
  --inference-steps 10 --replan-steps 10 \
  --save-predicted-video --timeout 14400 \
  --output-dir "$PWD/outputs/evaluation/robodojo/two_tasks_alltask_shift5_decay25k_step20k_sampling10_20261106_v1"
```

The safe interrupt entry prefix and existing audit/error semantics are
unchanged from step18k. Use a fresh output directory; native videos,
predictions, task audits and final-frame sheets remain under each
`tasks/TASK/attempt_001/`. Follow `progress.json` and `results.csv` in the
new root. Do not mix the earlier step16k/18k outcomes into this run's
six-episode denominator.

Preflight passed: 19 checkpoint/config files, 25,200,777,551 bytes,
2560 tensors and16,908 valid storage references. All16 new shard hashes
differ from evaluated step18k; saved configs and all five dataset file
hashes are unchanged. The two-task native layout/config/asset fingerprints
and evaluation protocol exactly match the step18k preflight. Existing
full-dataset delta derivation evidence is reused based on matching hashes.
Preparation helpers passed Ruff lint and format checks. No inference,
training or simulator source code was changed.

Startup verified in
`two_tasks_alltask_shift5_decay25k_step20k_sampling10_20261106_v1`:
the persisted manifest selects exactly these two tasks / six episodes,
full-source preflight passed, and `stack_bowls` advanced control0 to10
with the correct instruction, finite32x20 delta actions and all three
conditioning camera images. Policy port19190 is responsive.
The actual saved settings differ from step18k only in checkpoint,
provenance path, output directory and port; task specifications and all
pinned evaluation/model runtime code hashes are identical.
Actual owned Python processes were checked not to ignore SIGINT; the
previous step16k full sweep remains paused.
Detached shell is `alltask20k-two-eval` (coordinator PID2563904).
Coordinator log is
`/home/jianf/lzl_project/.copilot/session-state/1341fac2-578e-4665-a582-dc35f01c3d5b/files/alltask20k_two_suite.log`.
Success counts are pending; live results are in the new output root's
`progress.json` and `results.csv`.

### Completed step20k results

The two-task run exited0 with **six valid episodes, zero infrastructure
errors and 1/6 successes**. Bowls remains0/3; bottles succeeds on layout1,
restoring the layout that succeeded at step16k but failed at step18k.
There were no retries, substituted layouts or RNG restarts within a task.

| Task | Step16k successes | Step18k successes | Step20k successes | Step20k scores, layouts0/1/2 |
|---|---:|---:|---:|---|
| `stack_bowls` | 0/3 | 0/3 | 0/3 | 0 / 0 / 0 |
| `put_bottles_into_dustbin` | 1/3 | 0/3 | 1/3 | 0.4 / 1 / 0.25 |

Step20k aggregate native scores are0/100 for bowls and55/100 for bottles
(versus50/100 at step16k and35/100 at step18k for bottles). This is an
improvement over step18k on these three bottles layouts, not evidence of
a statistically reliable overall checkpoint ranking. Bowls loses the
small layout1 partial score seen at step18k and still does not stack.

Bowls used controls800/800/800, 240 chunks and3494.71 seconds.
Bottles used700/456/700, 186 chunks and2951.20 seconds; layout1 stopped
successfully after456 controls. These per-task times include evaluator
startup but exclude suite-level hash preflight. The shorter successful
episode ends mid-prefix, so its native video length is457 frames and
its final replan is not required to execute a complete ten-action prefix.

All426 chunks passed numerical integration/export audits, with maximum
integration-component errors below1.2e-7. The coordinator verified full
checkpoint/data SHA256 at the end; native asset, layout and runtime code
hashes and completed-result artifacts were independently rechecked
afterward. All91 passive camera observations passed.
Artifacts total **1278 predicted-camera MP4s, 14,058 prediction/conditioning
PNGs and18 actual-camera MP4s**. Bowls videos have801 frames each;
bottles videos have701/457/701 frames per layout, each with three
camera views at640x480/25FPS.

Both `native_final_frames.jpg` sheets were visually reviewed: all18 exact
final camera frames, not the complete rollout videos or all conditioning
images. Bowls remain separate in all final layouts. The successful bottles
layout1 has a cleared tabletop; layouts0 and2 still have bottles outside
the bin. Review scope and exact16k/18k/20k native score comparisons are in
`completion_review.json`. The broader `visual_review_pending` audit flag
remains true rather than implying complete video review.

Maximum sampled left/right target tracking discrepancies were5.299/11.768mm
for bowls and4.507/3.986mm for bottles. Numerical correctness does not
guarantee physical tracking, and these checks do not establish the failure
mechanism. The successful head video is
`tasks/put_bottles_into_dustbin/attempt_001/simulator/eval_result/RoboDojo/put_bottles_into_dustbin/Qwen3VLMoT/arx_x5/0_action_type=ee,simulator=isaacsim6/attempt_001/episode_0000001_cam_head_success.mp4`.
Policy port19190 is closed and evaluation GPU memory is released; unrelated
jobs remain running and the previous step16k full sweep stays paused.

## Full-task shift5 step25k: bowls and bottles

The step25k comparison keeps **`stack_bowls` and
`put_bottles_into_dustbin`, three native layouts each**, from the same
`robodojo_2nodes_8gpu_high_noise_shift_5_decay_25k` experiment. Keep seed0,
one environment, BF16, sampling10, prediction32, execution10, original
native horizons800/700, continuous RNG within each task, and actual/
predicted videos with passive camera audits. The paused step16k full
sweep is not resumed.

Reuse the existing checkpoint preflight with step25k and evaluated
step20k as reference: check a complete stable16-shard DCP upload,
metadata tensor keys/shapes/dtypes and storage byte ranges, full-source/
dataset SHA256, the measured-state delta contract and native assets.
Evidence is saved in
`runtime_assets/robodojo_2nodes_8gpu_high_noise_shift_5_decay_25k/checkpoint_provenance_step25k.json`
and `two_tasks_step25k_preflight_20261108.json`.

Run from the repository root:

```bash
unset ROBODOJO_STOP_AFTER_FIRST_FAILURE
export PYTHONPATH="$PWD" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export OMNI_KIT_ACCEPT_EULA=YES HF_HUB_OFFLINE=1
CACHE="$PWD/outputs/evaluation/robodojo/runtime_assets"
/home/jianf/miniforge3/envs/wam/bin/python -u -c \
  'import runpy, signal; signal.signal(signal.SIGINT, signal.default_int_handler); runpy.run_module("lerobot.evaluation.eval_qwen3vl_mot_robodojo_suite", run_name="__main__")' \
  eval --tasks stack_bowls put_bottles_into_dustbin \
  --checkpoint /data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_2nodes_8gpu_high_noise_shift_5_decay_25k/step25k \
  --dataset-root /data2/lizuolei/data/lerobot_data/simulation_data/RoboDojo/data/RoboDojo_ee_lerobot_v30_video \
  --qwen3vl-dir "$CACHE/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$CACHE/pretrained/Cosmos3-Edge" \
  --stage1-text-dir "$CACHE/pretrained/siglip2-base-patch16-224" \
  --checkpoint-provenance "$CACHE/robodojo_2nodes_8gpu_high_noise_shift_5_decay_25k/checkpoint_provenance_step25k.json" \
  --camera-wrapper /home/jianf/lzl_project/.copilot/session-state/1341fac2-578e-4665-a582-dc35f01c3d5b/files/inspect_robodojo_camera_run.py \
  --gpu 0 --sim-gpu 1 --port 19191 --seed 0 \
  --inference-steps 10 --replan-steps 10 \
  --save-predicted-video --timeout 14400 \
  --output-dir "$PWD/outputs/evaluation/robodojo/two_tasks_alltask_shift5_decay25k_step25k_sampling10_20261108_v1"
```

Use a fresh output directory and preserve the same safe interrupt,
per-task attempt, audit and infrastructure-error semantics as step20k.
The new root's `progress.json` and `results.csv` contain live outcomes;
videos and final-frame sheets are under `tasks/TASK/attempt_001/`.
Earlier checkpoints are comparison baselines only, not part of this
run's six-episode denominator. No training or inference/simulator code
changes are needed.

Preflight passed: 19 checkpoint/config files, 25,200,777,551 bytes,
2560 tensors and16,908 valid storage references. All16 shard hashes
differ from evaluated step20k; saved configs and all five dataset file
hashes are unchanged. The native task/layout/asset fingerprints and
evaluation protocol exactly match the step20k preflight. The prior
full-dataset delta derivation audit is reused based on identical hashes.

Startup verified in
`two_tasks_alltask_shift5_decay25k_step25k_sampling10_20261108_v1`:
the persisted manifest selects exactly these two tasks / six episodes,
full-source preflight passed, and `stack_bowls` advanced control0 to10
with the correct instruction, finite32x20 delta actions and all three
conditioning camera images. Policy port19191 is responsive.
Saved settings differ from step20k only in checkpoint, provenance path,
output directory and port; native task specifications and pinned
evaluation/model runtime code hashes are identical. Actual owned Python
processes were checked not to ignore SIGINT; the step16k full sweep
remains paused.
Detached shell is `alltask25k-two-eval` (coordinator PID3930741).
Coordinator log is
`/home/jianf/lzl_project/.copilot/session-state/1341fac2-578e-4665-a582-dc35f01c3d5b/files/alltask25k_two_suite.log`.
Success counts are pending; live outcomes are in the new root's
`progress.json` and `results.csv`.

### Completed step25k results

The two-task run exited0 with **six valid episodes, zero infrastructure
errors and1/6 successes**. Success counts and aggregate task scores match
step20k on these layouts; no improvement is established by this small
three-layout-per-task comparison.

| Task | Step16k | Step18k | Step20k | Step25k | Step25k scores, layouts0/1/2 |
|---|---:|---:|---:|---:|---|
| `stack_bowls` | 0/3 | 0/3 | 0/3 | 0/3 | 0 / 0 / 0 |
| `put_bottles_into_dustbin` | 1/3 | 0/3 | 1/3 | 1/3 | 0.25 / 1 / 0.4 |

Native aggregate scores are0/100 for bowls and55/100 for bottles.
Bottles again succeeds on layout1, at497 controls versus456 at step20k;
layout0 loses partial score0.4 to0.25 while layout2 gains0.25 to0.4.
This does not establish a general checkpoint ranking.

Bowls used controls800/800/800, 240 chunks and2411.94 seconds;
bottles used700/497/700, 190 chunks and2092.80 seconds. Per-task times
include startup but exclude suite-level hash preflight. GPU sharing
conditions differed from earlier runs, so wall-time differences must
not be interpreted as checkpoint inference-speed improvements.
There were no retries, substituted layouts or RNG restarts within a task.

All430 chunks passed integration/export audits, with numerical component
errors below1.1e-7. The coordinator verified full checkpoint/data SHA256
at the end; native asset/layout/runtime hashes and completed artifacts
were independently rechecked afterward. The initial head-camera and
static-scene transforms also match step20k on all six layouts.
All91 passive camera snapshots passed. Artifacts total **1290 predicted
MP4s, 14,190 prediction/conditioning PNGs and18 actual camera MP4s**:
bowls videos have801 frames each, and bottles have701/498/701 frames per
layout, three views each at640x480/25FPS.

Both final-frame sheets were visually reviewed:18 exact final camera
frames, not complete videos or all conditioning images. No bowls layout
completes a three-bowl stack. Bottles layout1 clears the tabletop; layouts0
and2 retain bottles outside the bin. Scope and exact checkpoint comparisons
are retained in `completion_review.json`; the broader audit flag
`visual_review_pending` remains true.
Maximum sampled left/right target tracking discrepancies are22.089/6.122mm
for bowls and3.681/3.121mm for bottles. Passing numerical checks does not
guarantee physical contact/grasping accuracy or establish why a task failed.

The successful head video is
`tasks/put_bottles_into_dustbin/attempt_001/simulator/eval_result/RoboDojo/put_bottles_into_dustbin/Qwen3VLMoT/arx_x5/0_action_type=ee,simulator=isaacsim6/attempt_001/episode_0000001_cam_head_success.mp4`.
Policy port19191 is closed, owned evaluation Python/GPU processes exited,
and unrelated GPU jobs remain running. The step16k full sweep stays paused.

## Step25k: five additional non-random tasks

The user selected the following five tasks, evaluated afresh in this order
with native layouts0/1/2 each: **15 episodes**, using the same full-task
step25k checkpoint and protocol as the completed bowls/bottles run.

| Task | Original native control limit per episode |
|---|---:|
| `stack_blocks` | 550 |
| `general_pickup` | 200 |
| `insert_tubes` | 500 |
| `hang_mugs` | 800 |
| `pack_objects_into_box` | 1300 |

The maximum planned total is10,050 controls; the long packing task is
not truncated to800. Keep seed0, one environment, BF16, sampling10,
prediction32, execution10, continuous RNG within each task, passive
camera audits and actual/predicted videos. No random task is selected,
the step16k full sweep remains paused, and old results are not mixed into
this new denominator.

Preparation passed using the existing verified25k provenance: checkpoint/
dataset sizes, mtimes, small metadata hashes and the delta contract match
the just-completed run; evaluation/model runtime hashes are unchanged.
The coordinator will repeat full-source SHA256 before the first task and
after the last. All15 layout files, original task horizons and140 native
asset/config files passed preflight. `general_pickup` cluttered objects
resolve under `Assets/Object/RoboDojo/Clutter/`, following the native
layout loader rather than treating their physical `Rigid` group as the
asset folder. No native asset or simulator source was changed.
Evidence is
`runtime_assets/robodojo_2nodes_8gpu_high_noise_shift_5_decay_25k/five_tasks_step25k_preflight_20261108.json`.
The preparation helper passed Ruff lint and was formatted.

Run from the repository root:

```bash
unset ROBODOJO_STOP_AFTER_FIRST_FAILURE
export PYTHONPATH="$PWD" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export OMNI_KIT_ACCEPT_EULA=YES HF_HUB_OFFLINE=1
CACHE="$PWD/outputs/evaluation/robodojo/runtime_assets"
/home/jianf/miniforge3/envs/wam/bin/python -u -c \
  'import runpy, signal; signal.signal(signal.SIGINT, signal.default_int_handler); runpy.run_module("lerobot.evaluation.eval_qwen3vl_mot_robodojo_suite", run_name="__main__")' \
  eval --tasks stack_blocks general_pickup insert_tubes hang_mugs pack_objects_into_box \
  --checkpoint /data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_2nodes_8gpu_high_noise_shift_5_decay_25k/step25k \
  --dataset-root /data2/lizuolei/data/lerobot_data/simulation_data/RoboDojo/data/RoboDojo_ee_lerobot_v30_video \
  --qwen3vl-dir "$CACHE/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$CACHE/pretrained/Cosmos3-Edge" \
  --stage1-text-dir "$CACHE/pretrained/siglip2-base-patch16-224" \
  --checkpoint-provenance "$CACHE/robodojo_2nodes_8gpu_high_noise_shift_5_decay_25k/checkpoint_provenance_step25k.json" \
  --camera-wrapper /home/jianf/lzl_project/.copilot/session-state/1341fac2-578e-4665-a582-dc35f01c3d5b/files/inspect_robodojo_camera_run.py \
  --gpu 0 --sim-gpu 1 --port 19192 --seed 0 \
  --inference-steps 10 --replan-steps 10 \
  --save-predicted-video --timeout 14400 \
  --output-dir "$PWD/outputs/evaluation/robodojo/five_tasks_alltask_shift5_decay25k_step25k_sampling10_20261108_v1"
```

Reuse the existing per-task attempt/audit semantics. Infrastructure errors
are separate from genuine0/3 policy outcomes; completed tasks are not
retried for a better score. Live results are in the new root's
`progress.json` and `results.csv`; each task saves its logs, predictions,
native videos, `suite_audit.json` and exact-final-frame sheet.

Startup verified in
`five_tasks_alltask_shift5_decay25k_step25k_sampling10_20261108_v1`:
the manifest selects exactly the five tasks above /15 episodes, with
native horizons550/200/500/800/1300. Full-source SHA256 preflight passed.
`stack_blocks` advanced control0 to10 with instruction
"Stack the three blocks with different textures.", finite32x20 delta
actions and all three conditioning images; policy port19192 is responsive.
Checkpoint/provenance and pinned runtime hashes are identical to the
completed25k bowls/bottles run. Actual saved settings differ only in
task selection, output directory and port.
Owned Python processes were checked not to ignore SIGINT; the step16k
full sweep remains paused.
Detached shell is `25k-five-tasks-eval` (coordinator PID4053433).
Coordinator log is
`/home/jianf/lzl_project/.copilot/session-state/1341fac2-578e-4665-a582-dc35f01c3d5b/files/25k_five_tasks_suite.log`.
No five-task success rate is available at startup; follow the new root's
`progress.json` and `results.csv`.

### Completed five-task step25k results

The selected five-task run exited0 with **15 valid episodes, zero
infrastructure errors and0/15 successes**. Every native episode score
is0, and every episode ran to its original task horizon. There were no
retries, substituted layouts or RNG restarts within a task.

| Task | Successes | Scores, layouts0/1/2 | Controls per layout |
|---|---:|---|---|
| `stack_blocks` | 0/3 | 0 / 0 / 0 | 550 / 550 / 550 |
| `general_pickup` | 0/3 | 0 / 0 / 0 | 200 / 200 / 200 |
| `insert_tubes` | 0/3 | 0 / 0 / 0 | 500 / 500 / 500 |
| `hang_mugs` | 0/3 | 0 / 0 / 0 | 800 / 800 / 800 |
| `pack_objects_into_box` | 0/3 | 0 / 0 / 0 | 1300 / 1300 / 1300 |

All10,050 controls and1005 chunks were retained. Per-task wall times were
2157.92, 837.95, 1977.89, 3119.02 and4858.51 seconds in the order above;
the total is12,951.28 seconds (215.85 minutes), including per-task startup
but excluding suite-level hash preflight.

Full checkpoint/data SHA256 passed at completion. Native task/layout,
all140 asset/config and evaluation/model runtime hashes were rechecked
afterward, as were completed native results and artifact counts.
All1005 chunk integration/export audits passed with numerical component
errors below1.1e-7; all216 passive camera snapshots passed.
There are **3015 predicted MP4s, 33,165 prediction/conditioning PNGs and
45 actual camera MP4s**. Native videos have551/201/501/801/1301 frames for
the five tasks respectively, three cameras per layout, at640x480/25FPS.

Native instructions were checked for all15 layouts. General pickup
correctly requests the mint green scissors, lavender plastic shovel and
conch shell for layouts0/1/2 rather than leaving a target placeholder.
All45 exact final camera frames in the five sheets were visually
reviewed, not complete videos or all conditioning images. Stacking,
insertion and hanging remain incomplete; packing retains objects outside
the box. Final pickup frames alone do not establish the full lift history.
Scope and per-task outcomes are saved in `completion_review.json`;
the broader `visual_review_pending` audit flag remains true.

**Physical tracking caveat:** packing has a sampled left-arm discrepancy
of327.598mm on layout1, controls1270->1280. Its previous ten-action
prefix commands only about4.2mm cumulative XYZ displacement, but the
next measured position is32.8cm from its target endpoint. Exact measured/
target positions and the two chunk paths are retained in the completion
review. The cause has not been established; numerical integration/export
parity does not guarantee physical tracking, and this outlier must not
be silently treated as a32cm requested translation or proof of a specific
model/controller failure mechanism. Other sampled maximum left/right
translation discrepancies in mm are16.633/10.344 (stacking),
2.525/0.157 (pickup), 2.739/8.454 (insertion), 7.429/10.754 (hanging)
and327.598/15.302 (packing).

Policy port19192 is closed and evaluation GPU processes exited; unrelated
jobs remain running. The step16k full sweep stays paused. These15 episodes
are a selected-task diagnostic, not the complete RoboDojo benchmark.

## No-Stage1 physical-MLP baseline: step21k, two tasks (2026-11-11)

Evaluate the user-supplied experiment
`/data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_2nodes_8gpu_high_noise_shift_5_decay_25k_base`.
The uploaded checkpoint is **step21k**, not step25k. The selected scope is
`stack_bowls` and `put_bottles_into_dustbin`, native layouts0/1/2 each:
six episodes, original horizons800/700, at most4500 controls.
The interrupted step16k full sweep remains paused.

Use the exact matching branch
`baseline/qwen3vl-mot-no-stage1-physical-mlp`, verified remote revision
`8a2fce0e6f32b3edd73cef01f33851de35b57504`, in a clean detached
worktree under
`outputs/evaluation/robodojo/runtime_assets/runtime_sources/no_stage1_physical_mlp_8a2fce0e`.
The main worktree stays on `qwen3vl_mot_pretrained_generation`; neither
its architecture nor the separate PI05 worktree is replaced.
This branch's existing loader handles a null Stage1 config correctly.
Do not fabricate a Stage1 config, discard architecture fields, relax DCP
key/shape checks or restore these weights with the staged-model loader.

Preflight passed for **35 source files:32 DCP shards, metadata and two
configs;23,353,964,787 bytes,1357 tensors and30,664 storage references**.
All source SHA256 hashes were recorded with stable sizes/mtimes, and
every storage range fits its shard. The12 physical-encoder tensor names
and shapes match the branch's actual `PhysicalMLP` state dictionary.
Full-model exact-key/shape verification and strict weight restoration
remain mandatory at actual policy startup, not inferred from metadata.
Weights are read in place without a full checkpoint copy.

The full-task delta-action contract and all five previously audited
dataset metadata/Parquet hashes match the staged experiment, including
stats SHA256
`94d978e4177fa8db22a1a48daedcffda28c8d24a325210a488d4c9006c1778e1`.
Native task sources/configs, all six layouts and39 asset/config files
match the completed staged25k two-task preflight. Evidence is under
`runtime_assets/robodojo_2nodes_8gpu_high_noise_shift_5_decay_25k_base/`:
`checkpoint_provenance_step21k.json` and
`two_tasks_step21k_preflight_20261111.json`.

This is **not a controlled single-variable Stage1 ablation**. In addition
to step21k versus25k, the baseline uses trainable state/action MLPs,
raw physical flow rather than embedding flow, no latent-action prediction
or action/state reconstruction losses, `understanding_lr_scale=1.0`
rather than0.01, and a32000-step rather than25000-step decay schedule.
Saved training uses batch16, accumulation1, seed1000, AdamW8bit at1e-4
and no FP8. No Stage1 teacher is constructed, but Qwen/Cosmos still use
pretrained assets: this is not training every network from scratch.
The directory name is not authoritative for the schedule.

Matching-branch regression tests passed: **124 tests**, covering MLP
initialization/training/sampling/self-contained restore, architecture
guards, the Stage1-free evaluation loader, strict DCP restoration and
native RoboDojo evaluation. The session preparation helper passes Ruff.
Keep the previous evaluation protocol: BF16, seed0, one environment,
sampling10 (saved20), prediction32/execution10, continuous RNG within a
task and independent policy initialization for the next task. Save native
and predicted videos, with passive camera and integration/export audits.

After creating the detached worktree, launch from it, using absolute
paths for shared cached assets and main-worktree output:

```bash
MAIN=/home/jianf/lzl_project/gcr_latent_action
CACHE="$MAIN/outputs/evaluation/robodojo/runtime_assets"
CODE="$CACHE/runtime_sources/no_stage1_physical_mlp_8a2fce0e"
cd "$CODE"
unset ROBODOJO_STOP_AFTER_FIRST_FAILURE
export PYTHONPATH="$CODE" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export OMNI_KIT_ACCEPT_EULA=YES HF_HUB_OFFLINE=1
/home/jianf/miniforge3/envs/wam/bin/python -u -c \
  'import runpy, signal; signal.signal(signal.SIGINT, signal.default_int_handler); runpy.run_module("lerobot.evaluation.eval_qwen3vl_mot_robodojo_suite", run_name="__main__")' \
  eval --tasks stack_bowls put_bottles_into_dustbin \
  --checkpoint /data2/lizuolei/weights/qwen3vl_mot_weights/robodojo_2nodes_8gpu_high_noise_shift_5_decay_25k_base/step21k \
  --dataset-root /data2/lizuolei/data/lerobot_data/simulation_data/RoboDojo/data/RoboDojo_ee_lerobot_v30_video \
  --qwen3vl-dir "$CACHE/pretrained/Qwen3-VL-4B-Instruct" \
  --cosmos3-dir "$CACHE/pretrained/Cosmos3-Edge" \
  --robodojo-root /data2/guoqi/robodojo_pi05_repro \
  --assets /data2/guoqi/robodojo_pi05_repro/Assets \
  --sim-python /home/jianf/miniforge3/envs/RoboDojo6/bin/python \
  --checkpoint-provenance "$CACHE/robodojo_2nodes_8gpu_high_noise_shift_5_decay_25k_base/checkpoint_provenance_step21k.json" \
  --camera-wrapper /home/jianf/lzl_project/.copilot/session-state/1341fac2-578e-4665-a582-dc35f01c3d5b/files/inspect_robodojo_camera_run.py \
  --gpu 0 --sim-gpu 1 --port 19193 --seed 0 \
  --inference-steps 10 --replan-steps 10 \
  --save-predicted-video --timeout 14400 \
  --output-dir "$MAIN/outputs/evaluation/robodojo/two_tasks_baseline_mlp_step21k_sampling10_20261111_v1"
```

No Stage1 text asset is needed by this MLP loader. The coordinator pins
the isolated model/evaluation sources, including `modeling_physical.py`,
and repeats full checkpoint/data SHA256 before and after the suite.
Restore default SIGINT handling before starting detached Python, then
check the actual owned Python children do not ignore it.
Other GPU jobs are present; do not stop them or infer causal speed
comparisons from shared-machine wall time. Live outcomes will be in the
new root's `progress.json` and `results.csv`; infrastructure errors remain
separate from genuine policy failures. Actual startup and results are
pending at this preparation checkpoint.

### Verified physical-MLP startup

The new two-task suite is running in
`two_tasks_baseline_mlp_step21k_sampling10_20261111_v1`.
Full-source hash preflight passed. The matching branch constructed the
MLP baseline with **no Stage1 teacher**, completed full-model strict
key/shape checks and DCP restoration, then reported11.650B parameters,
BF16 and ten denoising steps. No missing or unexpected weights were
silently skipped.

The native bowls rollout advanced control0 to10 with finite32x20 delta
actions and instruction "Stack the three bowls together." Its initial
measured state matches staged25k layout0 exactly. All19 pinned runtime
files and native task/layout hashes pass. Saved per-task settings differ
from staged25k only in checkpoint/output/port and the unused Stage1-text
argument; this argument is not read by the MLP loader.
All three first-chunk conditioning PNGs were visually checked: ordinary
wooden tabletop, three white bowls and expected head/left-wrist/right-wrist
camera roles. This does not constitute a complete-video review.
Matched predicted videos and passive camera observations are saved.

Policy port19193 completes a valid WebSocket handshake. The four owned
Python processes do not ignore SIGINT; other GPU jobs remain running.
Startup evidence is `startup_validation.json` in the suite root.
Detached shell is `baseline21k-two-eval`, coordinator PID2023270.
Coordinator log is
`/home/jianf/lzl_project/.copilot/session-state/1341fac2-578e-4665-a582-dc35f01c3d5b/files/baseline21k_two_tasks_suite.log`.
Preparation commit: `e666598`. Both tasks' final outcomes and full
completion audits remain pending; do not report a startup success as a
successful task episode.
