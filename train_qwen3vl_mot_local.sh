#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   conda activate lerobot_v2
#   STAGE1_CHECKPOINT=/path/to/exported_robo_contrast \
#     CUDA_VISIBLE_DEVICES=0,1 bash train_qwen3vl_mot_local.sh
#
# For a fresh run, the stage-one path must contain config.json and model.safetensors.
# A stage-two resume reads the embedded stage-one architecture from its saved config.
WEIGHT_RESUME="${WEIGHT_RESUME:-false}"
if [[ "${WEIGHT_RESUME}" != "true" ]]; then
    : "${STAGE1_CHECKPOINT:?Set STAGE1_CHECKPOINT for a fresh stage-two run}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-qwen3vl_mot}"
TRAINING_STAGE="${TRAINING_STAGE:-auto}"
MAX_EPOCHS="${MAX_EPOCHS:-}"
if [[ "$TRAINING_STAGE" != "auto" && "$TRAINING_STAGE" != "pretrain" && "$TRAINING_STAGE" != "finetune" ]]; then
    echo "Error: TRAINING_STAGE must be auto, pretrain or finetune" >&2
    exit 1
fi
EPOCH_ARGS=()
if [[ -n "$MAX_EPOCHS" ]]; then
    if [[ ! "$MAX_EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: MAX_EPOCHS must be a positive integer, got $MAX_EPOCHS" >&2
        exit 1
    fi
    EPOCH_ARGS+=(--max_epochs="${MAX_EPOCHS}")
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
QWEN3VL_DIR="${QWEN3VL_DIR:-/Data/lzl/huggingface/Qwen3-VL-4B-Instruct}"
COSMOS3_DIR="${COSMOS3_DIR:-/Data/lzl/huggingface/Cosmos3-Edge}"
MODEL_AUTO_DOWNLOAD="${MODEL_AUTO_DOWNLOAD:-true}"
MODEL_LOCAL_CACHE="${MODEL_LOCAL_CACHE:-${TMPDIR:-/tmp}/qwen3vl_mot_models}"
if [[ "$MODEL_AUTO_DOWNLOAD" != "true" && "$MODEL_AUTO_DOWNLOAD" != "false" ]]; then
    echo "Error: MODEL_AUTO_DOWNLOAD must be true or false" >&2
    exit 1
fi
SAMPLE_POOL_ROOT="${SAMPLE_POOL_ROOT:-$(dirname "${OUTPUT_DIR}")/_sample_pools}"
SAMPLE_POOL_CACHE_DIR="${SAMPLE_POOL_CACHE_DIR:-${TMPDIR:-/tmp}/robo_contrast_sample_pools}"
STAGE2_TASK_NAMES="${STAGE2_TASK_NAMES:-}"
STAGE2_TASK_WEIGHTS="${STAGE2_TASK_WEIGHTS:-}"
RGB_VIEWS="${RGB_VIEWS:-}"
RGB_AUG_ENABLED="${RGB_AUG_ENABLED:-false}"
if [[ "$RGB_AUG_ENABLED" != "true" && "$RGB_AUG_ENABLED" != "false" ]]; then
    echo "Error: RGB_AUG_ENABLED must be true or false" >&2
    exit 1
fi
PHYSICAL_FLOW_SPACE="${PHYSICAL_FLOW_SPACE:-}"
ACTION_PRED_RECON_WEIGHT="${ACTION_PRED_RECON_WEIGHT:-}"
STATE_PRED_RECON_WEIGHT="${STATE_PRED_RECON_WEIGHT:-}"
TIME_SAMPLING_DISTRIBUTION="${TIME_SAMPLING_DISTRIBUTION:-}"
TIME_SAMPLING_SHIFT="${TIME_SAMPLING_SHIFT:-}"
TIME_SAMPLING_BETA_ALPHA="${TIME_SAMPLING_BETA_ALPHA:-}"
TIME_SAMPLING_BETA_BETA="${TIME_SAMPLING_BETA_BETA:-}"
TIME_SAMPLING_BETA_PROBABILITY="${TIME_SAMPLING_BETA_PROBABILITY:-}"
GENERATION_BACKBONE="${GENERATION_BACKBONE:-}"
GENERATION_PRETRAINED_DIR="${GENERATION_PRETRAINED_DIR:-}"
GENERATION_TUNING_MODE="${GENERATION_TUNING_MODE:-}"
GENERATION_ARGS=()
if [[ "$WEIGHT_RESUME" != "true" ]]; then
    GENERATION_BACKBONE="${GENERATION_BACKBONE:-edge}"
fi
if [[ -n "$GENERATION_BACKBONE" ]]; then
    case "$GENERATION_BACKBONE" in
        random|edge|nano) ;;
        *) echo "Error: GENERATION_BACKBONE must be random, edge or nano" >&2; exit 1 ;;
    esac
    GENERATION_ARGS+=(--policy.generation_backbone="${GENERATION_BACKBONE}")
    if [[ "$WEIGHT_RESUME" != "true" && "$GENERATION_BACKBONE" != "random" ]]; then
        _cosmos_name=Cosmos3-Edge
        if [[ "$GENERATION_BACKBONE" == "nano" ]]; then
            _cosmos_name=Cosmos3-Nano
        fi
        GENERATION_PRETRAINED_DIR="${GENERATION_PRETRAINED_DIR:-/Data/lzl/huggingface/${_cosmos_name}}"
    fi
fi
if [[ -n "$GENERATION_TUNING_MODE" ]]; then
    if [[ "$GENERATION_TUNING_MODE" != "full" && "$GENERATION_TUNING_MODE" != "frozen" ]]; then
        echo "Error: GENERATION_TUNING_MODE must be full or frozen" >&2
        exit 1
    fi
    GENERATION_ARGS+=(--policy.generation_tuning_mode="${GENERATION_TUNING_MODE}")
fi
PHYSICAL_FLOW_ARGS=()
if [[ "$WEIGHT_RESUME" != "true" ]]; then
    PHYSICAL_FLOW_SPACE="${PHYSICAL_FLOW_SPACE:-embedding}"
fi
if [[ -n "$PHYSICAL_FLOW_SPACE" ]]; then
    if [[ "$PHYSICAL_FLOW_SPACE" != "raw" && "$PHYSICAL_FLOW_SPACE" != "embedding" ]]; then
        echo "Error: PHYSICAL_FLOW_SPACE must be raw or embedding" >&2
        exit 1
    fi
    PHYSICAL_FLOW_ARGS+=(--policy.physical_flow_space="${PHYSICAL_FLOW_SPACE}")
fi
if [[ "$WEIGHT_RESUME" != "true" && "$PHYSICAL_FLOW_SPACE" == "embedding" ]]; then
    ACTION_PRED_RECON_WEIGHT="${ACTION_PRED_RECON_WEIGHT:-0.1}"
    STATE_PRED_RECON_WEIGHT="${STATE_PRED_RECON_WEIGHT:-0.1}"
fi
if [[ -n "$ACTION_PRED_RECON_WEIGHT" ]]; then
    PHYSICAL_FLOW_ARGS+=(
        --policy.action_prediction_reconstruction_loss_weight="${ACTION_PRED_RECON_WEIGHT}"
    )
fi
if [[ -n "$STATE_PRED_RECON_WEIGHT" ]]; then
    PHYSICAL_FLOW_ARGS+=(
        --policy.state_prediction_reconstruction_loss_weight="${STATE_PRED_RECON_WEIGHT}"
    )
fi
if [[ -n "$TIME_SAMPLING_DISTRIBUTION" ]]; then
    case "$TIME_SAMPLING_DISTRIBUTION" in
        uniform|beta|mixed|shifted_uniform) ;;
        *) echo "Error: TIME_SAMPLING_DISTRIBUTION must be uniform, beta, mixed or shifted_uniform" >&2; exit 1 ;;
    esac
    PHYSICAL_FLOW_ARGS+=(--policy.time_sampling_distribution="${TIME_SAMPLING_DISTRIBUTION}")
fi
if [[ -n "$TIME_SAMPLING_SHIFT" ]]; then
    PHYSICAL_FLOW_ARGS+=(--policy.time_sampling_shift="${TIME_SAMPLING_SHIFT}")
fi
if [[ -n "$TIME_SAMPLING_BETA_ALPHA" ]]; then
    PHYSICAL_FLOW_ARGS+=(--policy.time_sampling_beta_alpha="${TIME_SAMPLING_BETA_ALPHA}")
fi
if [[ -n "$TIME_SAMPLING_BETA_BETA" ]]; then
    PHYSICAL_FLOW_ARGS+=(--policy.time_sampling_beta_beta="${TIME_SAMPLING_BETA_BETA}")
fi
if [[ -n "$TIME_SAMPLING_BETA_PROBABILITY" ]]; then
    PHYSICAL_FLOW_ARGS+=(--policy.time_sampling_beta_probability="${TIME_SAMPLING_BETA_PROBABILITY}")
fi

if [[ -n "$STAGE2_TASK_NAMES" || -n "$STAGE2_TASK_WEIGHTS" ]]; then
    if [[ -z "$STAGE2_TASK_NAMES" || -z "$STAGE2_TASK_WEIGHTS" ]]; then
        echo "Error: STAGE2_TASK_NAMES and STAGE2_TASK_WEIGHTS must be set together" >&2
        exit 1
    fi
fi

STAGE2_TASK_ARGS=()
RGB_VIEW_ARGS=()
if [[ -n "$RGB_VIEWS" ]]; then
    RGB_VIEWS_JSON="[\"${RGB_VIEWS//,/\",\"}\"]"
    RGB_VIEW_ARGS+=(--policy.rgb_views="${RGB_VIEWS_JSON}")
fi
if [[ -n "$STAGE2_TASK_NAMES" ]]; then
    STAGE2_TASK_NAMES_JSON="[\"${STAGE2_TASK_NAMES//,/\",\"}\"]"
    STAGE2_TASK_WEIGHTS_JSON="[${STAGE2_TASK_WEIGHTS}]"
    STAGE2_TASK_ARGS+=(
        --policy.task_names="${STAGE2_TASK_NAMES_JSON}"
        --policy.task_weights="${STAGE2_TASK_WEIGHTS_JSON}"
    )
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LEROBOT_VIDEO_DECODER_CACHE_SIZE="${LEROBOT_VIDEO_DECODER_CACHE_SIZE:-256}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_CACHE_ARGS=()
if [[ "$WEIGHT_RESUME" != "true" ]]; then
    MODEL_CACHE_ARGS+=(--need-generation)
fi
RESOLVED_MODEL_PATHS="$("${PYTHON_BIN}" -m lerobot.common.utils.stage2_model_cache \
    --qwen-dir="$QWEN3VL_DIR" --cosmos-dir="$COSMOS3_DIR" \
    --generation-dir="$GENERATION_PRETRAINED_DIR" \
    --generation-backbone="$GENERATION_BACKBONE" \
    --cache-dir="$MODEL_LOCAL_CACHE" --auto-download="$MODEL_AUTO_DOWNLOAD" \
    "${MODEL_CACHE_ARGS[@]}")"
mapfile -t _model_paths <<< "$RESOLVED_MODEL_PATHS"
QWEN3VL_DIR="${_model_paths[0]}"
COSMOS3_DIR="${_model_paths[1]}"
GENERATION_PRETRAINED_DIR="${_model_paths[2]:-}"
if [[ -n "$GENERATION_PRETRAINED_DIR" ]]; then
    GENERATION_ARGS+=(--policy.generation_pretrained_dir="${GENERATION_PRETRAINED_DIR}")
fi

MASTER_PORT="${MASTER_PORT:-$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")}"
STAGE1_CONFIG_ARGS=()
STAGE1_CHECKPOINT_ARGS=()
if [[ -n "${STAGE1_CHECKPOINT:-}" ]]; then
    STAGE1_CHECKPOINT_ARGS+=(--policy.stage1_checkpoint="${STAGE1_CHECKPOINT}")
fi
if [[ -n "${STAGE1_CONFIG:-}" ]]; then
    STAGE1_CONFIG_ARGS+=(--policy.stage1_config="${STAGE1_CONFIG}")
fi

deepspeed --master_port="${MASTER_PORT}" lerobot/scripts/dps_train_contrast.py \
    --deepspeed="${DEEPSPEED_CONFIG:-./ds_zero2_qwen3vl_mot.json}" \
    --policy.type="qwen3vl_mot" \
    --stage="${TRAINING_STAGE}" \
    "${EPOCH_ARGS[@]}" \
    --policy.qwen3vl_dir="${QWEN3VL_DIR}" \
    --policy.cosmos3_dir="${COSMOS3_DIR}" \
    "${GENERATION_ARGS[@]}" \
    --policy.understanding_tuning_mode="${UNDERSTANDING_TUNING_MODE:-lora}" \
    --policy.understanding_lora_rank="${UNDERSTANDING_LORA_RANK:-16}" \
    --policy.understanding_lora_alpha="${UNDERSTANDING_LORA_ALPHA:-16}" \
    --policy.understanding_text_lora_layers="${UNDERSTANDING_TEXT_LORA_LAYERS:-0}" \
    --policy.understanding_vision_lora_layers="${UNDERSTANDING_VISION_LORA_LAYERS:-4}" \
    --policy.understanding_lr_scale="${UNDERSTANDING_LR_SCALE:-0.1}" \
    --policy.physical_tuning_mode="${PHYSICAL_TUNING_MODE:-frozen}" \
    "${PHYSICAL_FLOW_ARGS[@]}" \
    --policy.window_mode="frames" \
    --policy.chunk_size=32 \
    --policy.n_action_steps=32 \
    --policy.world_video_frames="${WORLD_VIDEO_FRAMES:-9}" \
    "${RGB_VIEW_ARGS[@]}" \
    --policy.tactile_generation_target="${TACTILE_GENERATION_TARGET:-spatial_patches}" \
    --policy.generation_gradient_checkpointing=true \
    --policy.understanding_gradient_checkpointing=true \
    --policy.optimizer_lr="${LEARNING_RATE:-1e-4}" \
    --policy.scheduler_warmup_steps=500 \
    --policy.scheduler_plateau_steps=2000 \
    --policy.scheduler_decay_steps=100000 \
    --policy.scheduler_decay_lr=1.5e-6 \
    "${STAGE2_TASK_ARGS[@]}" \
    --dataset.repo_id="whatever" \
    --dataset.image_transforms.enable="${RGB_AUG_ENABLED}" \
    --dataset.image_transforms.img_size=256 \
    --dataset.wrist_image_transforms.enable=false \
    --dataset.wrist_image_transforms.is_primary=false \
    --dataset.parent_dir_v21="${DATA_ROOT_V21:-/Data/lerobot_data_ort6d}" \
    --dataset.parent_dir_v30="${DATA_ROOT_V30:-/Data/lerobot_data_ort6d/v30}" \
    --dataset.parent_dir_extra="${DATA_ROOT_EXTRA:-}" \
    --dataset.video_backend="torchcodec" \
    --dataset.dataset_size_one_epoch="${SAMPLES_PER_EPOCH:-100000}" \
    --dataset.sample_pool_enabled="${SAMPLE_POOL_ENABLED:-true}" \
    --dataset.sample_pool_root="${SAMPLE_POOL_ROOT}" \
    --dataset.sample_pool_cache_dir="${SAMPLE_POOL_CACHE_DIR}" \
    --dataset.sample_pool_keep_all_below_fps="${SAMPLE_POOL_KEEP_ALL_BELOW_FPS:-10}" \
    --dataset.sample_pool_target_hz="${SAMPLE_POOL_TARGET_HZ:-5}" \
    --data_mix="${DATA_MIX:-debug_research_data}" \
    --num_workers="${NUM_WORKERS:-8}" \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${JOB_NAME:-qwen3vl_mot}" \
    --weight_resume="${WEIGHT_RESUME}" \
    --save_freq="${SAVE_FREQ:-2000}" \
    --log_freq="${LOG_FREQ:-20}" \
    --eval_freq=0 \
    --steps="${STEPS:-600000}" \
    --task_type="train_stage2" \
    --wandb.enable="${WANDB_ENABLE:-false}" \
    "${STAGE1_CHECKPOINT_ARGS[@]}" \
    "${STAGE1_CONFIG_ARGS[@]}" \
    "$@"
