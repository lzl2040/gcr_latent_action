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
floor of 1e-3. The local copy under
`/data2/lizuolei/data/lerobot_data/simulation_data` lacks these statistics and
must **not** be substituted. Dataset `info.json` may retain its original 16D
quaternion features even though converted parquet and statistics contain the
20D ort6d fields.

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
