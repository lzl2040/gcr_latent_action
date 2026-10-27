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
