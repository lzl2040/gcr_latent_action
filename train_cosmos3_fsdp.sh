#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_DIR="${MODEL_DIR:-/Data/lzl/huggingface/Cosmos3-Nano-Policy-LIBERO}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/cosmos3_nano_libero}"
WEIGHT_RESUME="${WEIGHT_RESUME:-false}"
LOAD_ACTION_MODULES="${LOAD_ACTION_MODULES:-true}"
for value in "$WEIGHT_RESUME" "$LOAD_ACTION_MODULES"; do
    case "$value" in
        true|false) ;;
        *) echo "WEIGHT_RESUME and LOAD_ACTION_MODULES must be true or false." >&2; exit 1 ;;
    esac
done
if [[ "$WEIGHT_RESUME" == true && "$LOAD_ACTION_MODULES" != true ]]; then
    echo "Resume must reload trained action modules; use LOAD_ACTION_MODULES=true." >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export LEROBOT_VIDEO_DECODER_CACHE_SIZE="${LEROBOT_VIDEO_DECODER_CACHE_SIZE:-64}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ -n "${FFMPEG_LIB_DIR:-}" ]]; then
    export LD_LIBRARY_PATH="$FFMPEG_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#devices[@]}}"
MASTER_PORT="${MASTER_PORT:-$("$PYTHON_BIN" -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')}"

"$PYTHON_BIN" -m torch.distributed.run \
    --nproc_per_node="$NPROC_PER_NODE" \
    --nnodes="${NNODES:-1}" \
    --node_rank="${NODE_RANK:-0}" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="$MASTER_PORT" \
    --module lerobot.scripts.fsdp_train_contrast \
    --policy.type="${POLICY_TYPE:-cosmos3_nano}" \
    --policy.pretrained_path="$MODEL_DIR" \
    --policy.dtype=bfloat16 \
    --policy.load_action_modules="$LOAD_ACTION_MODULES" \
    --policy.gradient_checkpointing=true \
    --policy.fps=20 \
    --policy.view_size='[256,256]' \
    --policy.training_action_layout="${TRAINING_ACTION_LAYOUT:-libero_euler_interleaved}" \
    --policy.optimizer_lr="${LEARNING_RATE:-5e-5}" \
    --policy.scheduler_warmup_steps="${WARMUP_STEPS:-500}" \
    --policy.scheduler_decay_steps="${DECAY_STEPS:-16000}" \
    --stage=finetune \
    --task_type=train_stage2 \
    --data_mix="${DATA_MIX:-libero}" \
    --dataset.repo_id=local_libero \
    --dataset.parent_dir_v21="${DATA_ROOT_V21:-/Data/lerobot_data_ort6d}" \
    --dataset.parent_dir_v30="${DATA_ROOT_V30:-/Data/lerobot_data_ort6d/v30}" \
    --dataset.image_transforms.enable=false \
    --dataset.image_transforms.img_size=256 \
    --dataset.video_backend=torchcodec \
    --batch_size="${BATCH_SIZE:-2}" \
    --num_workers="${NUM_WORKERS:-4}" \
    --gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-4}" \
    --steps="${STEPS:-5000}" \
    --save_freq="${SAVE_FREQ:-500}" \
    --log_freq="${LOG_FREQ:-1}" \
    --eval_freq=0 \
    --generation_eval.enabled=false \
    --fsdp.fp8=false \
    --weight_resume="$WEIGHT_RESUME" \
    --output_dir="$OUTPUT_DIR" \
    --device=cuda \
    --seed="${SEED:-1000}" \
    --wandb.enable=false \
    "$@"
