#!/usr/bin/env bash
set -euo pipefail

# Independent MLP baseline + causal visual history; the original launcher is unchanged.
export PHYSICAL_ENCODER_TYPE="${PHYSICAL_ENCODER_TYPE:-mlp}"
if [[ "$PHYSICAL_ENCODER_TYPE" != "mlp" ]]; then
    echo "This launcher requires PHYSICAL_ENCODER_TYPE=mlp." >&2
    exit 1
fi
export GENERATION_BACKBONE="${GENERATION_BACKBONE:-nano}"
export JOB_NAME="${JOB_NAME:-qwen3vl_mot_baseline_history}"
if [[ -z "${STAGE2_TASK_NAMES:-}" && -z "${STAGE2_TASK_WEIGHTS:-}" ]]; then
    export STAGE2_TASK_NAMES=forward_dynamics,inverse_dynamics,joint_prediction
    export STAGE2_TASK_WEIGHTS=0.15,0.15,0.7
fi

exec bash "$(dirname "$0")/train_qwen3vl_mot_fsdp.sh" \
    --policy.visual_history_offsets='[-4,-2]' \
    --policy.visual_history_include_first=true \
    "$@"
