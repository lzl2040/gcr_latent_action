#!/usr/bin/env bash
set -euo pipefail

# Independent action/state MLPs; inherit only the frozen Stage-1 tactile image codec.
export PHYSICAL_ENCODER_TYPE="${PHYSICAL_ENCODER_TYPE:-mlp_tactile}"
if [[ "$PHYSICAL_ENCODER_TYPE" != "mlp_tactile" ]]; then
    echo "This launcher requires PHYSICAL_ENCODER_TYPE=mlp_tactile." >&2
    exit 1
fi
export TACTILE_GENERATION_TARGET="${TACTILE_GENERATION_TARGET:-spatial_patches}"
export TACTILE_SEQUENCE_PREDICTION="${TACTILE_SEQUENCE_PREDICTION:-true}"
# Empty preserves checkpoint settings on resume; a fresh run uses four frames.
export TACTILE_FRAMES="${TACTILE_FRAMES:-}"
export GENERATION_BACKBONE="${GENERATION_BACKBONE:-nano}"
export JOB_NAME="${JOB_NAME:-qwen3vl_mot_baseline_all_modality_sequence}"
if [[ -z "${STAGE2_TASK_NAMES:-}" && -z "${STAGE2_TASK_WEIGHTS:-}" ]]; then
    export STAGE2_TASK_NAMES=all_modality_prediction
    export STAGE2_TASK_WEIGHTS=1.0
fi

exec bash "$(dirname "$0")/train_qwen3vl_mot_fsdp.sh" "$@"
