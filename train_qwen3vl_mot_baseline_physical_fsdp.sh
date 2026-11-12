#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export JOB_NAME="${JOB_NAME:-qwen3vl_mot_baseline_physical_libero}"
export WEIGHT_RESUME="${WEIGHT_RESUME:-false}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT:-/mnt/wangxiaofa/qwen3vl_mot_exp}/${JOB_NAME}}"
export OUTPUT_DIR

"${PYTHON_BIN:-python}" - "$OUTPUT_DIR" "$WEIGHT_RESUME" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
resume = sys.argv[2]
if resume not in ("true", "false"):
    raise SystemExit("WEIGHT_RESUME must be true or false.")
if path.exists() and any(path.iterdir()):
    config = path / "config.json"
    if resume != "true":
        raise SystemExit(f"Refusing to overwrite nonempty output directory: {path}")
    if not config.is_file() or json.loads(config.read_text()).get("stage1_physical_only") is not True:
        raise SystemExit(f"Refusing to resume a different model variant: {path}")
PY

export DATA_MIX="${DATA_MIX:-libero}"
export GENERATION_BACKBONE="${GENERATION_BACKBONE:-nano}"
export STAGE2_TASK_NAMES="${STAGE2_TASK_NAMES:-joint_prediction}"
export STAGE2_TASK_WEIGHTS="${STAGE2_TASK_WEIGHTS:-1.0}"
export RGB_VIEWS="${RGB_VIEWS:-primary,wrist}"
export RGB_AUG_ENABLED="${RGB_AUG_ENABLED:-true}"
export UNDERSTANDING_TUNING_MODE="${UNDERSTANDING_TUNING_MODE:-full}"
export UNDERSTANDING_LR_SCALE="${UNDERSTANDING_LR_SCALE:-1.0}"
export TIME_SAMPLING_DISTRIBUTION="${TIME_SAMPLING_DISTRIBUTION:-shifted_uniform}"
export TIME_SAMPLING_SHIFT="${TIME_SAMPLING_SHIFT:-5.0}"
export WARMUP_STEPS="${WARMUP_STEPS:-1000}"
export PLATEAU_STEPS="${PLATEAU_STEPS:-100}"
export DECAY_STEPS="${DECAY_STEPS:-20000}"
export FP8_ENABLED="${FP8_ENABLED:-false}"
export PHYSICAL_FLOW_SPACE=raw
export PHYSICAL_TUNING_MODE=full
export ACTION_PRED_RECON_WEIGHT=0
export STATE_PRED_RECON_WEIGHT=0
export LATENT_ACTION_PREDICTION_ENABLED=true
export UNDERSTANDING_DUAL_VISION=false

exec bash "$ROOT/train_qwen3vl_mot_fsdp.sh" \
    --policy.stage1_physical_only=true \
    --policy.require_stage1_vision_transfer=false \
    --policy.require_stage1_text_transfer=false \
    --policy.latent_action_loss_weight=0 \
    --policy.use_tactile_conditioning=false \
    --policy.physical_lr_scale="${PHYSICAL_LR_SCALE:-0.3}" \
    "$@"
