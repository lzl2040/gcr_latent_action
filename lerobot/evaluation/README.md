# Qwen3-VL/Cosmos LIBERO rollout

For the dual-arm RoboDojo benchmark, see [README_robodojo.md](README_robodojo.md).

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
  --replan-steps 10 \
  --output-dir /Data/lzl/libero_eval/step12k_goal
```

Other supported suites are `libero_spatial`, `libero_object`, and `libero_10`.
Run the command once for each suite; normalization uses that suite's original
`*_no_noops_lerobot/meta/stats.json`, not pooled statistics or quantiles.
Use `--task-ids 0 1` to select tasks. Standard control horizons are spatial 220,
object 280, goal 300, and 10-suite 520 steps, after 10 settling steps.
Use `--stop-on-success` for a smoke evaluation that saves the first successful
episode and stops without running further tasks or trials. Early termination is
not a complete benchmark evaluation.

For a short plumbing check, add `--task-ids 0 --trials-per-task 1 --max-steps 16`.
A shortened horizon is **not a benchmark success-rate evaluation**.
`--seed` initializes Python, NumPy and Torch (CPU/CUDA) RNGs once per run, before
loading the policy. Tasks, episode resets and replans do not reseed them; LIBERO
uses the ambient NumPy RNG. Each task still uses the first fixed benchmark
initial states in trial order. Changing task order, selecting a subset of tasks,
or changing earlier rollout lengths can change later sampled actions because
the RNG stream advances continuously. Results identify this protocol with
`sampling_rng: "run_seeded_continuous"`; older episode-reseeded results use a
different protocol.
The default flow integration count comes from the saved configuration (20 for
this checkpoint); `--inference-steps` explicitly overrides it for ablations.
`--replan-steps` defaults to 10 and executes the first N consecutive actions of
each predicted chunk before replanning. It cannot exceed the saved
`n_action_steps` (32 here).

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

## Cluster installation and optional BF16/FP8 validation

Run setup **inside the cluster node/container**, with the repository mounted
and `--libero-root` pointing to your downloaded
node-local LIBERO source. It does not submit any cluster jobs. Installation requires
Conda, git, package-index access, and GitHub access if LIBERO is not already present.
Pretrained assets download from Hugging Face to node-local disk by default; you do
not need to mount the training machine's pretrained-weight or dataset directories.
Use the same existing Conda environment **`lerobot`** for installation and execution,
matching `conda run --name lerobot bash train.sh` in your cluster job. Configure
Python **3.10, 3.11 or 3.12** in that environment beforehand. The script does not
create another environment, switch to a private prefix, or upgrade Python.

```bash
conda run --name lerobot \
  bash lerobot/evaluation/scripts/setup_libero_qwen3vl.sh \
  --libero-root /scratch/LIBERO \
  --model-cache-dir /scratch/libero_qwen3vl_models
```

**Setup does not load the policy, read a fine-tuned checkpoint, run CUDA/EGL
smoke checks, or run BF16/FP8 model inference.** It installs/checks dependencies,
prepares LIBERO, downloads or reuses pretrained assets, and exports normalization
metadata. No checkpoint, FP8 recipe, GPU index or verification output is required.
The script prints all prepared model paths and the `--dataset-root` path for
formal evaluation; in this example it is
`/scratch/libero_qwen3vl_models/dataset_metadata`.

Use `--skip-install` to reuse installed dependencies and the existing LIBERO
source while preparing resources. The old `--verify-only` is a deprecated alias
for `--skip-install`, with a warning; it no longer runs verification.
There is no required `--` separator. Old checkpoint/precision/testing arguments
are rejected before installation rather than silently ignored; use the standalone
verifier below when those checks are wanted. The trailing `\` in examples is
Bash line continuation, not part of an option.

Default pretrained downloads use fixed revisions of `Qwen/Qwen3-VL-4B-Instruct`,
`nvidia/Cosmos3-Edge` (**VAE only**, not its large original generation transformer),
and `google/siglip2-base-patch16-224`. The cache defaults to
`MODEL_LOCAL_CACHE` or `/tmp/libero_qwen3vl_models`; choose a node-local SSD/scratch
directory, not Blob/NFS or a RAM-backed mount. Downloads have a local file lock,
disk-space checks and completeness validation, and subsequent runs reuse complete
cached assets. Use Hugging Face's normal environment-based authentication or
`hf auth login` if a repository requires access; do not put credentials in scripts.
Explicit `--qwen3vl-dir`, `--cosmos3-dir`, and `--stage1-text-dir` paths override
individual downloads; invalid explicit paths fail instead of silently downloading
replacements. `--offline` prohibits all pretrained downloads, but does not make
dependency installation or LIBERO cloning offline; combine it with `--skip-install`
to reuse an already installed environment. Only resource preparation accesses the
Hub; optional verification's model subprocesses run offline using resolved paths.

The repository includes the exact per-suite 10D state/action mean/std and 20 Hz
metadata from the existing LIBERO ort6d v3 evaluation protocol in
`lerobot/evaluation/assets/libero_ort6d_normalization.json`, with original metadata
SHA256s. Setup exports all bundled suites under
`MODEL_CACHE/dataset_metadata/*_no_noops_lerobot/meta/{info,stats}.json`,
and repeated setup refreshes these small managed files atomically. The standalone
verifier exports only the selected suite's metadata under its output directory,
so no original training dataset is required. This snapshot is
for the current LIBERO-converted training data, not arbitrary retraining datasets:
pass `--dataset-root /path/to/your/dataset/root` to the evaluator or standalone
verifier (not setup) to supply different suite-specific `meta/info.json` /
`meta/stats.json`. `verification.json` records the chosen source.

The installer uses `CONDA_PREFIX/bin/python` from the selected `lerobot` environment
for dependency installation and resource preparation; the standalone verifier
uses its own selected interpreter for both model subprocesses. Running setup
outside `lerobot` fails before installation. User-site imports and pip's
user-install fallback are disabled, so
an unwritable environment fails rather than installing into `~/.local`.
There is no `--env-prefix` option. Installation updates dependencies
inside `lerobot` to the pinned evaluation stack, including TorchAO 0.15.0; use
`--skip-install` if dependencies have already been installed by your AMLT setup.
The shared environment is reconciled to NumPy 1.26.4, Zarr `>=2.17,<3`,
Numcodecs `>=0.10,<0.16`, and Rerun SDK `>=0.21,<0.23`. Zarr 3 and Rerun SDK
0.39 require NumPy 2; Numcodecs 0.16 removed Blosc helpers used by Zarr 2.
The selected Rerun versions also preserve the existing dataset visualization API.
Existing incompatible versions are downgraded during
normal installation. If an earlier setup stopped at this dependency conflict,
rerun it **without `--skip-install`** so the updated requirements can repair it;
`pip check` still rejects any remaining dependency conflicts.
After selecting PyTorch, installation also probes any preinstalled `flash-attn`.
Only an undefined-symbol binary import failure in the selected environment
triggers removal; importable packages and packages outside that environment are
preserved. Unrelated import errors stop setup.
The model uses native PyTorch SDPA and does not require the external extension.
The installer then imports Diffusers' `AutoencoderKLWan` before any model-weight
downloads/loading, so optional-extension failures are reported early rather than
hidden in a later model subprocess. `--skip-install` does not perform these repairs.
The installer reuses your local LIBERO.
If the source is absent, it clones official LIBERO at
`8f1084e3132a39270c3a13ebe37270a43ece2a01`. It does **not** install LIBERO's obsolete
requirements or the entire LeRobot hardware/training stack. Existing LIBERO trees
are never patched/reset; a different revision requires an explicit `--libero-ref`
full SHA. Existing archive-extracted sources are also accepted, with an explicit
warning that their revision is unverified. The evaluator handles official NumPy-pickled initial states explicitly
under PyTorch 2.9. Use only trusted LIBERO assets.

The default wheels are PyTorch 2.9.1 / torchvision 0.24.1 **CUDA 12.8**.
The host NVIDIA driver must support that CUDA runtime; wheels do not install or
upgrade the driver. Use `--torch-index-url https://download.pytorch.org/whl/cu130`
only when the driver supports CUDA 13.0. EGL additionally needs the host NVIDIA
graphics libraries exposed to the container, e.g.
`NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics` **when launching the container**;
setting it after container creation cannot mount missing driver libraries.
There is no silent software-rendering fallback.

### Optional precision/environment validation

Run this separately only when model or GPU environment validation is needed:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --name lerobot \
  python -u -m lerobot.evaluation.scripts.verify_libero_qwen3vl \
  --libero-root /scratch/LIBERO \
  --checkpoint /mnt/weights/1014_v1_8gpu/step40k \
  --fp8-training-config /mnt/weights/1014_v1_8gpu/train_config.json \
  --model-cache-dir /scratch/libero_qwen3vl_models \
  --render-gpu-device-id 0 \
  --output-dir /mnt/results/libero_precision_smoke
```

Replace `/mnt/...` with actual cluster mounts. The fine-tuned checkpoint must
retain its adjacent saved `config.json`, or supply `--policy-config`; native FP8
also needs the same run's saved `train_config.json`. These are never downloaded
or replaced. Every model tensor is restored strictly from this fine-tuned DCP
checkpoint; preparing base assets does not switch evaluation to a base model.
The standalone verifier never installs packages.

You can reuse the verifier's `--output-dir`: an existing directory receives a
fresh `run_<UTC timestamp>_<unique suffix>` subdirectory. Old results are never
overwritten or resumed. The printed path and `verification.json`'s `output_dir`
identify the actual run directory; `requested_output_dir` records the original
path. All logs, environment configuration, normalization and precision outputs
remain inside that run directory.

By default the verifier checks CUDA and real NVIDIA EGL rendering for both
cameras, then runs BF16 and native FP8 sequentially in separate processes. Both
use the same checkpoint, task 0 / fixed initial state 0, seed, saved flow-step count,
and one 32-step prediction, executing only its first 10 actions. First-plan input
and CPU/CUDA RNG hashes must match. The FP8 run uses the saved training recipe,
scope and dimension threshold, preserves BF16 master weights, and must exhibit
actual `scaled_mm` operator calls; counting converted layers alone is insufficient.
TorchAO 0.15's rowwise tensor subclass needs view tracking, so FP8 uses
`eval()` + `no_grad()` with inference mode disabled; BF16 retains its original
inference-mode behavior. Neither mode builds backward graphs or changes the recipe.

`verification.json` records environment details, failures, native-FP8 verification
status and action differences. Each precision directory preserves its rollout,
actions, first-plan probes and evaluator settings; subprocess logs are next to the
report. These are **environment/precision smoke checks, not benchmark success rates
or a speed benchmark**. No numerical-difference threshold is interpreted as policy
accuracy. Full accuracy comparisons require the normal multi-task, multi-trial
rollout protocol with `--precision bf16` versus
`--precision fp8 --fp8-training-config ...`.

Native FP8 requires the training implementation's compute capability **>= 9.0**,
e.g. H100/H200/B200. A6000 cannot validate it; an explicitly requested unsupported
FP8 run fails rather than falling back to BF16 or emulation. For BF16-only checking,
add `--precisions bf16` to the standalone verifier command. Missing pretrained
assets can still download unless `--offline` is set. To check only CUDA/EGL
without loading a checkpoint, use:

```bash
conda run --name lerobot \
  python -m lerobot.evaluation.scripts.verify_libero_qwen3vl \
  --libero-root /path/to/LIBERO \
  --environment-only --precisions bf16 --render-gpu-device-id 0 \
  --output-dir /mnt/results/libero_egl_smoke
```

**Local validation limitation:** the development machine has A6000 GPUs. BF16
model/EGL validation and CPU FP8-emulation tests do not establish native FP8
execution; that part must pass on the cluster's supported GPU.

## Checkpoint and embodiment contract

- Loads the four `step12k/model/*.distcp` shards into one inference model,
  validating every key and shape. No optimizer, Stage 1 checkpoint, FSDP process
  group, or four-GPU launch is required. The saved `config.json` is discovered in
  the experiment directory or selected explicitly.
- Preserves the saved task embedding order, model size, physical flow space,
  and RGB views. This checkpoint is Nano, primary-camera-only, and uses
  **joint prediction** for action inference because no action-only task was
  trained. With future-video export enabled (default), the runner uses
  `generate_validation` once per replan: the saved video and executed actions
  come from the same joint sample, not from a second video-generation pass.
  Ambient run-seeded RNG is preserved, so exporting a video does not
  reseed each replan or change this checkpoint's sampled actions.
  A separate frozen VAE decoder is loaded once, without changing the strict
  inference checkpoint's encoder-only model layout.
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

Each invocation creates a timestamped directory under `outputs/evaluation/libero/`,
or uses the directory specified by `--output-dir`. **An existing output directory
is cleared and overwritten**, including old results, logs, actions, videos and
generated configuration; this starts over rather than resuming. Use a different
directory to preserve an earlier run. Filesystem roots, top-level directories,
mount points, home/current/repository directories, directories containing
evaluation inputs, and symlink output directories are rejected before cleanup.
The cluster setup/verifier still uses a fresh `run_*` subdirectory on repeated
verification; this overwrite behavior applies to the formal evaluator.

- `results.json`: settings, completed episodes, per-task and overall success
  rates, horizons, the run seed/RNG mode, initial-state indices, and timing.
- `evaluation.log`: checkpoint restoration and episode results.
- `task_XX_trial_YYY_actions.npy`: the executed native 7D action sequence.
- `task_XX_trial_YYY_success.mp4` / `_failure.mp4`: primary-camera rollout at
  source FPS (20 here); disable with `--no-save-video`, or save only failures
  with `--save-failed-videos-only`. These two options are mutually exclusive.
- `predictions/task_XX_trial_YYY/replan_NNNN_step_SSSS/primary.mp4`: the predicted
  video at **every** replan, with separate `wrist.mp4` for saved multi-view
  models when that camera is available. Missing views are skipped.
- Each replan directory also contains `action_chunk.npy` (the complete predicted
  native action chunk) and `metadata.json` (replan index, rollout control step,
  run seed/RNG mode, per-view paths, and source-frame timestamps). These are
  referenced by `episodes[*].prediction_videos` in `results.json`.

Future-video saving is on by default. `--no-save-predicted-video` disables only
future decoding/export and calls `sample_canonical_action` instead (this
checkpoint still requires joint sampling to produce its actions);
`--no-save-video` independently disables only the **actual** rollout video.
For 20 trials per goal task without future decoding, keeping only failed
rollout videos, use `--trials-per-task 20 --no-save-predicted-video
--save-failed-videos-only` and do not pass `--stop-on-success`.
`--output-dir` controls all outputs, including future videos. Reusing it
overwrites the previous evaluation, including artifacts for tasks or trials
not included in the new run.

This checkpoint predicts 9 sparse video frames over its 32-action window:
one current conditioning frame and 8 future frames, spanning 1.55 seconds at
20 Hz. Predicted MP4 playback therefore uses approximately **5.16 FPS**, not
20 FPS; the exact rounded source-frame timestamps are saved in metadata.
The full action chunk and corresponding future clip are saved even though
only its first `--replan-steps` actions are executed before replanning.
Decoding/export adds overhead; `inference_seconds` records model inference
separately from `prediction_export_seconds`, while `planning_seconds` includes
both.

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
python -m pytest --noconftest -q tests/test_libero_qwen3vl_evaluation.py tests/test_libero_cluster_setup.py
```
