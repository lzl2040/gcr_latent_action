# Qwen3-VL/Cosmos LIBERO rollout

From the repository root:

```bash
conda activate libero_lerobot_qwen3vl

# Already installed in this environment; these commands reproduce the setup.
python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r lerobot/evaluation/requirements_libero_qwen3vl.txt
```

Evaluate all 10 tasks in one suite, with 50 fixed benchmark initial states per task:

```bash
python -m lerobot.evaluation.eval_qwen3vl_mot_libero \
  --checkpoint /Data/lzl/qwen3vl_mot_weights/ft_on_libero/1014_v0/step12k \
  --policy-config /Data/lzl/qwen3vl_mot_weights/ft_on_libero/1014_v0/config.json \
  --dataset-root /Data/lerobot_data_ort6d/v30 \
  --libero-root /home/v-wangxiaofa/lzl/LIBERO \
  --task-suite libero_goal \
  --trials-per-task 50 \
  --replan-steps 8
```

Other supported suites are `libero_spatial`, `libero_object`, and `libero_10`.
Run the command once for each suite; normalization uses that suite's original
`*_no_noops_lerobot/meta/stats.json`, not pooled statistics or quantiles.
Use `--task-ids 0 1` to select tasks. Standard control horizons are spatial 220,
object 280, goal 300, and 10-suite 520 steps, after 10 settling steps.

For a short plumbing check, add `--task-ids 0 --trials-per-task 1 --max-steps 16`.
A shortened horizon is **not a benchmark success-rate evaluation**.
The default flow integration count comes from the saved configuration (20 for
this checkpoint); `--inference-steps` explicitly overrides it for ablations.
`--replan-steps` executes the first N consecutive actions of each predicted chunk
before replanning and cannot exceed the saved `n_action_steps` (32 here).

Use `CUDA_VISIBLE_DEVICES` to choose the policy GPU. For example, to use physical
GPU 3 for both inference and EGL rendering:

```bash
CUDA_VISIBLE_DEVICES=3 python -m lerobot.evaluation.eval_qwen3vl_mot_libero \
  --checkpoint /Data/lzl/qwen3vl_mot_weights/ft_on_libero/1014_v0/step12k \
  --task-suite libero_goal \
  --render-gpu-device-id 3
```

`--device` defaults to logical `cuda:0`; `--render-gpu-device-id` is a **physical
EGL device index**, not the remapped CUDA index. The runner creates its own LIBERO
configuration inside the output directory and does not modify `~/.libero` or the
external LIBERO repository.

## Checkpoint and embodiment contract

- Loads the four `step12k/model/*.distcp` shards into one inference model,
  validating every key and shape. No optimizer, Stage 1 checkpoint, FSDP process
  group, or four-GPU launch is required. The saved `config.json` is discovered in
  the experiment directory or selected explicitly.
- Preserves the saved task embedding order, model size, physical flow space,
  and RGB views. This checkpoint is Nano, primary-camera-only, and uses
  **joint prediction** for action inference because no action-only task was
  trained. Video, action, and state are jointly sampled; video decoding is not
  required for robot controls.
- The local assets default to `/Data/lzl/huggingface/Qwen3-VL-4B-Instruct`,
  `/Data/lzl/huggingface/Cosmos3-Edge` (VAE), and
  `/Data/lzl/huggingface/siglip2-base-patch16-224` (Stage 1 text).
  Override with `--qwen3vl-dir`, `--cosmos3-dir`, and `--stage1-text-dir`.
  Paths do not silently fall back to a different backbone.
- Reproduces `lerobot_convert/version_convert_final/v20_to_v21.py`: the raw
  three rotation numbers are encoded with `Rotation.from_euler("xyz", ...)`,
  and the first two matrix **columns** are flattened in row-major order.
  Identity is `[1,0,0,1,0,0]`, not the PyTorch3D layout. For state, the reference
  LIBERO quaternion-to-axis-angle step happens **before** this Euler encoding.
  This seemingly redundant conversion is necessary to match the existing
  training data; directly encoding the observation quaternion would differ.
- State retains the second gripper finger, z-score normalizes its first 10
  canonical dimensions, and masks the other 30. Standard deviations have the
  training floor `1e-3`. Action predictions are inverse-normalized, rotation
  columns are orthonormalized, and Euler decoding recovers the original three
  OSC control numbers. Degenerate or nonfinite predictions fail explicitly.
- Gripper follows `run_pi05_eval.py`: dataset `0=close, 1=open` becomes native
  `-sign(2*x-1)`, where `-1=open, +1=close` (the exact midpoint gives 0).
  Translation and rotation control values are clipped to the native `[-1,1]`
  limits with a warning, not to empirical dataset extrema.
- Both camera images rotate 180 degrees as in the reference evaluation.
  There is no random RGB augmentation. Saved multi-view configurations also
  receive wrist RGB; unavailable secondary views are explicitly masked.

## Outputs

Each invocation creates a fresh directory under `outputs/evaluation/libero/`,
or the **new, nonexistent** directory specified by `--output-dir`:

- `results.json`: settings, completed episodes, per-task and overall success
  rates, horizons, seeds, initial-state indices, and timing.
- `evaluation.log`: checkpoint restoration and episode results.
- `task_XX_trial_YYY_actions.npy`: the executed native 7D action sequence.
- `task_XX_trial_YYY_success.mp4` / `_failure.mp4`: primary-camera rollout at
  source FPS (20 here); disable with `--no-save-video`.

Success comes from `env.check_success()`, not merely an environment `done`
flag. Timeouts count as failures. Episodes use the original benchmark language
and distinct fixed initial states, with no success-seeking retries.
Results persist after every episode, and environments close even when
inference or video writing fails. An interrupted run reports only completed
episodes; `evaluation_complete` stays false until the requested task/trial count
is reached. Completion of a shortened run still does not make it a full benchmark.
Inference is synchronous in simulated time, not a real-time control service.

Isolated evaluator regression tests (without the repository's unrelated
hardware-test fixtures):

```bash
python -m pytest --noconftest -q tests/test_libero_qwen3vl_evaluation.py
```
