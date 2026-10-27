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
budget and batch size explicitly, e.g. `--steps 30000 --batch-size 4`.
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
