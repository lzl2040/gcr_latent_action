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

The quaternion/ort6d mapping was also compared against a real converted
training parquet row: maximum absolute errors were below 8e-8 for state
and action. The original 16D fields and the converted 20D fields agree
with wxyz input and interleaved first-two matrix columns.

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
