#!/usr/bin/env bash
set -euo pipefail

# Run the current ACE latent-action branch on one local EgoDex split while a
# network-bound primary job is preparing data. This process never decodes video.
#
# Examples:
#   LOCAL_DATA_ROOT=/local/lerobot/v30 GPU=0 \
#     bash train_ace_latent_background.sh start
#   bash train_ace_latent_background.sh status
#   bash train_ace_latent_background.sh stop
#
# LOCAL_DATA_ROOT must contain the relative path registered for ego_dex_split1
# in vla2root.json (currently EgoDex/EgoDex_part1).

cd "$(dirname "$0")" || exit 1

ACTION="${1:-start}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DATASET_SIZE_ONE_EPOCH="${DATASET_SIZE_ONE_EPOCH:-100000}"
LOG_FREQ="${LOG_FREQ:-20}"
MAX_RUNTIME="${MAX_RUNTIME:-6h}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/Data/lerobot_data_ort6d/v30}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-/mnt/wangxiaofa/pt_weights}"
VISION_MODEL="${VISION_MODEL:-${WEIGHTS_ROOT}/siglip2-base-patch16-224}"
PRETRAINED_PATH="${PRETRAINED_PATH:-}"
JOB_NAME="${JOB_NAME:-ace_latent_ego_dex_split1_background}"
RUN_DIR="${RUN_DIR:-${TMPDIR:-/tmp}/ace_latent_background}"
PID_FILE="${PID_FILE:-${RUN_DIR}/${JOB_NAME}.pid}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/${JOB_NAME}.log}"
DS_CONFIG="${RUN_DIR}/${JOB_NAME}_deepspeed.json"
DRY_RUN="${DRY_RUN:-false}"

mkdir -p "$RUN_DIR"

die() {
    echo "error: $*" >&2
    exit 1
}

require_positive_int() {
    [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer, got: $2"
}

read_pid() {
    [[ -f "$PID_FILE" ]] || return 1
    local pid
    pid="$(tr -d '[:space:]' < "$PID_FILE")"
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
    printf '%s' "$pid"
}

is_our_process() {
    local pid="$1"
    [[ -r "/proc/${pid}/cmdline" ]] || return 1
    tr '\0' ' ' < "/proc/${pid}/cmdline" | grep -q "lerobot/scripts/dps_train_ace.py"
}

stop_process() {
    local pid="$1"
    is_our_process "$pid" || die "PID ${pid} is not the ACE background process; refusing to stop it"

    local pgid
    pgid="$(ps -o pgid= -p "$pid" | tr -d '[:space:]')"
    if [[ "$pgid" == "$pid" ]]; then
        kill -TERM -- "-${pid}"
    else
        kill -TERM "$pid"
    fi

    for _ in $(seq 1 30); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
        die "process ${pid} did not stop after 30 seconds"
    fi
    rm -f "$PID_FILE"
    echo "stopped ACE latent-action background process ${pid}"
}

case "$ACTION" in
    status)
        if pid="$(read_pid)" && kill -0 "$pid" 2>/dev/null && is_our_process "$pid"; then
            echo "running: pid=${pid} gpu=${GPU} log=${LOG_FILE}"
        else
            echo "not running"
            rm -f "$PID_FILE"
        fi
        exit 0
        ;;
    stop)
        if pid="$(read_pid)" && kill -0 "$pid" 2>/dev/null; then
            stop_process "$pid"
        else
            rm -f "$PID_FILE"
            echo "ACE latent-action background process is not running"
        fi
        exit 0
        ;;
    start) ;;
    *)
        die "usage: bash train_ace_latent_background.sh {start|stop|status}"
        ;;
esac

if pid="$(read_pid)" && kill -0 "$pid" 2>/dev/null; then
    if is_our_process "$pid"; then
        die "ACE latent-action background process is already running with PID ${pid}"
    fi
    die "PID file points to another live process (${pid}); remove ${PID_FILE} after checking it"
fi
rm -f "$PID_FILE"

require_positive_int "BATCH_SIZE" "$BATCH_SIZE"
[[ "$GPU" =~ ^[0-9]+$ ]] || die "GPU must be one physical GPU index, got: ${GPU}"
[[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || die "NUM_WORKERS must be a non-negative integer, got: ${NUM_WORKERS}"
require_positive_int "DATASET_SIZE_ONE_EPOCH" "$DATASET_SIZE_ONE_EPOCH"
require_positive_int "LOG_FREQ" "$LOG_FREQ"
command -v python >/dev/null || die "python is not available; activate the training environment first"
command -v setsid >/dev/null || die "setsid is required for safe background process management"
command -v timeout >/dev/null || die "timeout is required to enforce MAX_RUNTIME"
python -c "import deepspeed; from torchcodec.decoders import VideoDecoder" \
    || die "the active Python environment cannot import deepspeed/torchcodec"

DATASET_RELATIVE_PATH="$(
    python - <<'PY'
import json

with open("vla2root.json") as f:
    print(json.load(f)["ego_dex_split1"])
PY
)"
DATASET_PATH="${LOCAL_DATA_ROOT%/}/${DATASET_RELATIVE_PATH}"
[[ -f "${DATASET_PATH}/meta/info.json" ]] \
    || die "local ego_dex_split1 not found at ${DATASET_PATH}; set LOCAL_DATA_ROOT"
[[ -d "$VISION_MODEL" ]] \
    || die "SigLIP2 weights not found at ${VISION_MODEL}; set VISION_MODEL or WEIGHTS_ROOT"
if [[ -n "$PRETRAINED_PATH" && ! -e "$PRETRAINED_PATH" ]]; then
    die "PRETRAINED_PATH does not exist: ${PRETRAINED_PATH}"
fi

python - "$BATCH_SIZE" "$DS_CONFIG" <<'PY'
import json
import sys

batch_size, output = sys.argv[1:]
config = {
    "bf16": {"enabled": True},
    "fp16": {"enabled": False},
    "train_micro_batch_size_per_gpu": int(batch_size),
    "gradient_accumulation_steps": 1,
    "gradient_clipping": 1.0,
    "zero_optimization": {
        "stage": 2,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_scatter": True,
    },
}
with open(output, "w") as f:
    json.dump(config, f, indent=4)
PY

MASTER_PORT="${MASTER_PORT:-$(
    python - <<'PY'
import socket

with socket.socket() as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PY
)}"

CMD=(
    python
    -m
    deepspeed.launcher.runner
    --master_port="$MASTER_PORT"
    lerobot/scripts/dps_train_ace.py
    --deepspeed="$DS_CONFIG"
    --policy.type="robo_clip"
    --policy.vision_model_name="$VISION_MODEL"
    --policy.frozen_ace=true
    --policy.train_latent_action_only=true
    --policy.chunk_size=32
    --policy.n_action_steps=16
    --policy.group_size=4
    --policy.max_action_dim=50
    --policy.max_state_dim=50
    --policy.pretrained_path="$PRETRAINED_PATH"
    --is_ft=false
    --save_checkpoint=false
    --save_freq=1000000000
    --dataset.repo_id="whatever"
    --dataset.image_transforms.enable=false
    --dataset.wrist_image_transforms.enable=false
    --dataset.wrist_image_transforms.is_primary=false
    --dataset.processor="$VISION_MODEL"
    --dataset.parent_dir_v21="$LOCAL_DATA_ROOT"
    --dataset.parent_dir_v30="$LOCAL_DATA_ROOT"
    --dataset.video_backend="torchcodec"
    --dataset.sample_ratio=1
    --dataset.dataset_size_one_epoch="$DATASET_SIZE_ONE_EPOCH"
    --data_mix="ego_dex_split1"
    --batch_size="$BATCH_SIZE"
    --gradient_accumulation_steps=1
    --num_workers="$NUM_WORKERS"
    --output_dir="$RUN_DIR/output"
    --log_dir="$RUN_DIR"
    --steps=1000000000
    --log_freq="$LOG_FREQ"
    --eval_freq=0
    --policy.scheduler_warmup_steps=10
    --policy.scheduler_decay_steps=25000
    --policy.scheduler_platform_steps=20000
    --policy.optimizer_lr=1e-4
    --wandb.enable=false
    --job_name="$JOB_NAME"
    --weight_resume=false
    --resume=false
    --task_type="train_action_decoder"
)

if [[ "$DRY_RUN" == "true" ]]; then
    printf 'dataset=%q\n' "$DATASET_PATH"
    printf 'command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi

echo "starting ACE latent-action background training"
echo "dataset=${DATASET_PATH}"
echo "gpu=${GPU} batch=${BATCH_SIZE} workers=${NUM_WORKERS} max_runtime=${MAX_RUNTIME}"
echo "log=${LOG_FILE}"

nohup setsid timeout --signal=TERM --kill-after=30s "$MAX_RUNTIME" \
    env \
        CUDA_VISIBLE_DEVICES="$GPU" \
        HF_HUB_OFFLINE=1 \
        TRANSFORMERS_OFFLINE=1 \
        WANDB_MODE=disabled \
        TOKENIZERS_PARALLELISM=false \
        OMP_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 \
        PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}" \
        "${CMD[@]}" \
    > "$LOG_FILE" 2>&1 < /dev/null &

pid=$!
printf '%s\n' "$pid" > "$PID_FILE"
sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "ACE latent-action background process exited during startup:" >&2
    tail -n 50 "$LOG_FILE" >&2 || true
    exit 1
fi

echo "started: pid=${pid}"
echo "stop with: bash train_ace_latent_background.sh stop"
