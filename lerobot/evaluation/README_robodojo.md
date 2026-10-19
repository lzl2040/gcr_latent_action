# Qwen3-VL MoT on RoboDojo

## Runtime and checkpoint contract

Use branch `qwen3vl_mot_pretrained_generation`. The policy runs in `wam`;
Isaac Sim runs in a separate simulator environment. Do not install Isaac Sim
into the policy environment or change the shared RoboDojo/Pi_05 installation.

This integration targets RoboDojo's XPolicyLab WebSocket interface
(`update_obs`, `get_action`, `reset`, and their batched variants), not a
LIBERO/Gym interface. Upstream reference:
<https://github.com/robodojo-benchmark/RoboDojo>.

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
directory, and the checkpoint being investigated:

```bash
CUDA_VISIBLE_DEVICES=1 python -m lerobot.evaluation.eval_qwen3vl_mot_robodojo diagnose \
  --checkpoint outputs/evaluation/robodojo/runtime_assets/checkpoint/step8k \
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
