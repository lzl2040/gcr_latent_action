#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LIBERO_ROOT=""
LIBERO_REF="8f1084e3132a39270c3a13ebe37270a43ece2a01"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
VERIFY_ONLY=false

usage() {
    cat <<'EOF'
Install LIBERO/Qwen3-VL dependencies into the existing Conda environment lerobot.
Run with: conda run --name lerobot bash setup_libero_qwen3vl.sh ...
No cluster jobs are submitted, no extra environments or Python interpreters are created,
and only fine-tuned weights/configuration need to be mounted on the node.
Pretrained Qwen, Cosmos VAE and SigLIP assets download to node-local cache by default.

Usage: conda run --name lerobot bash setup_libero_qwen3vl.sh [installer options] -- [verifier options]
  --libero-root PATH      Existing LIBERO checkout, or new clone destination
                         (default: CONDA_PREFIX/libero-source)
  --libero-ref SHA        Required immutable LIBERO revision (default: tested upstream SHA)
  --torch-index-url URL   PyTorch CUDA wheel index (default: cu128; use cu130 for newer drivers)
  --verify-only          Skip dependency installation/source cloning; use the same lerobot environment
  --help                 Show this help

Verifier options include --checkpoint, --model-cache-dir, --offline, --dataset-root, --qwen3vl-dir,
--cosmos3-dir, --stage1-text-dir, --fp8-training-config, --output-dir,
--device, --render-gpu-device-id, --precisions bf16 fp8, and --environment-only.
The default verification compares BF16 and native FP8 using the saved FP8 recipe.
Use --precisions bf16 on GPUs without native FP8 support.

Requires an existing lerobot environment with Python 3.10-3.12, git for cloning, and host NVIDIA EGL drivers
inside the container. The GPU container must expose graphics capability as well
as compute/utility (NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics).
Use --offline after -- to prohibit downloads, including with --verify-only.
Bundled normalization avoids needing the original training dataset on the node.
No sudo, host-driver changes, fine-tuned weight downloads, or shared ~/.libero writes occur.
EOF
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
while (($#)); do
    case "$1" in
        --libero-root|--libero-ref|--torch-index-url)
            (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || die "Missing value for $1"
            case "$1" in
                --libero-root) LIBERO_ROOT="$2" ;;
                --libero-ref) LIBERO_REF="$2" ;;
                --torch-index-url) TORCH_INDEX_URL="$2" ;;
            esac
            shift 2 ;;
        --verify-only) VERIFY_ONLY=true; shift ;;
        --help|-h) usage; exit 0 ;;
        --) shift; break ;;
        *) die "Unknown installer option: $1 (verifier options belong after --)" ;;
    esac
done
[[ "$LIBERO_REF" =~ ^[0-9a-f]{40}$ ]] || die "--libero-ref must be a full commit SHA."
[[ "${CONDA_DEFAULT_ENV:-}" = lerobot && -n "${CONDA_PREFIX:-}" ]] \
    || die "Run this script inside lerobot: conda run --name lerobot bash $0 ..."
PYTHON="${CONDA_PREFIX}/bin/python"
[[ "$CONDA_PREFIX" = /* && -f "$CONDA_PREFIX/conda-meta/history" && -x "$PYTHON" ]] \
    || die "The active lerobot prefix is not a valid Conda environment: $CONDA_PREFIX"
LIBERO_ROOT="${LIBERO_ROOT:-${CONDA_PREFIX}/libero-source}"
export PYTHONNOUSERSITE=1
"$PYTHON" - <<'PY'
import os
import sys
from pathlib import Path

if not (3, 10) <= sys.version_info[:2] < (3, 13):
    raise SystemExit("lerobot requires Python 3.10-3.12; configure its Python version in your AMLT environment.")
if Path(sys.prefix).resolve() != Path(os.environ["CONDA_PREFIX"]).resolve():
    raise SystemExit("Selected Python does not belong to the active lerobot Conda environment.")
print(f"Using existing lerobot environment: {sys.prefix} (Python {sys.version.split()[0]})", flush=True)
PY

if [[ "$VERIFY_ONLY" = false ]]; then
    command -v git >/dev/null || die "git is required."
    "$PYTHON" -m pip install --no-user --disable-pip-version-check torch==2.9.1 torchvision==0.24.1 \
        --index-url "$TORCH_INDEX_URL"
    "$PYTHON" -m pip install --no-user --disable-pip-version-check \
        -r "$REPO_ROOT/lerobot/evaluation/requirements_libero_qwen3vl_fp8.txt"
    "$PYTHON" -m pip check
    if [[ ! -d "$LIBERO_ROOT/libero/libero" ]]; then
        git clone --quiet --no-checkout https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_ROOT"
        git -C "$LIBERO_ROOT" checkout --quiet --detach "$LIBERO_REF"
    fi
fi
[[ -x "$PYTHON" ]] || die "Environment Python not found: $PYTHON"
for ASSET_DIR in assets bddl_files init_files; do
    [[ -d "$LIBERO_ROOT/libero/libero/$ASSET_DIR" ]] || die "Incomplete LIBERO source: missing $ASSET_DIR"
done
if [[ -e "$LIBERO_ROOT/.git" ]]; then
    [[ "$(git -C "$LIBERO_ROOT" rev-parse HEAD)" = "$LIBERO_REF" ]] \
        || die "LIBERO revision differs from --libero-ref; refusing to checkout/reset an existing repository."
else
    printf 'Using existing LIBERO source without Git metadata; revision is unverified: %s\n' "$LIBERO_ROOT" >&2
fi

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
cd "$REPO_ROOT"
exec "$PYTHON" -m lerobot.evaluation.scripts.verify_libero_qwen3vl \
    --libero-root "$LIBERO_ROOT" "$@"
