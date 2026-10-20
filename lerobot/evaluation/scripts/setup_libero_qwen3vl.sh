#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_PREFIX="${HOME}/.conda/envs/libero_qwen3vl_cluster"
LIBERO_ROOT=""
LIBERO_REF="8f1084e3132a39270c3a13ebe37270a43ece2a01"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
VERIFY_ONLY=false

usage() {
    cat <<'EOF'
Install an isolated Python 3.12 LIBERO/Qwen3-VL environment and verify it.
No cluster jobs are submitted, existing environments are not activated globally,
and model weights/statistics must already be mounted on the node.

Usage: bash setup_libero_qwen3vl.sh [installer options] -- [verifier options]
  --env-prefix PATH       Isolated Conda prefix (default: ~/.conda/envs/libero_qwen3vl_cluster)
  --libero-root PATH      Existing LIBERO checkout, or new clone destination
                         (default: ENV_PREFIX/libero-source)
  --libero-ref SHA        Required immutable LIBERO revision (default: tested upstream SHA)
  --torch-index-url URL   PyTorch CUDA wheel index (default: cu128; use cu130 for newer drivers)
  --verify-only          Do not install or modify dependencies/source; verify existing prefix
  --help                 Show this help

Verifier options include --checkpoint, --dataset-root, --qwen3vl-dir,
--cosmos3-dir, --stage1-text-dir, --fp8-training-config, --output-dir,
--device, --render-gpu-device-id, --precisions bf16 fp8, and --environment-only.
The default verification compares BF16 and native FP8 using the saved FP8 recipe.
Use --precisions bf16 on GPUs without native FP8 support.

Requires Conda for installation, git for cloning, and host NVIDIA EGL drivers
inside the container. The GPU container must expose graphics capability as well
as compute/utility (NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics).
No sudo, host-driver changes, model downloads, or shared ~/.libero writes occur.
EOF
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
while (($#)); do
    case "$1" in
        --env-prefix|--libero-root|--libero-ref|--torch-index-url)
            (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || die "Missing value for $1"
            case "$1" in
                --env-prefix) ENV_PREFIX="$2" ;;
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
[[ "$ENV_PREFIX" = /* && "$ENV_PREFIX" != / && "$ENV_PREFIX" != "$HOME" ]] \
    || die "--env-prefix must be an absolute, dedicated environment directory."
[[ "$LIBERO_REF" =~ ^[0-9a-f]{40}$ ]] || die "--libero-ref must be a full commit SHA."
LIBERO_ROOT="${LIBERO_ROOT:-${ENV_PREFIX}/libero-source}"
PYTHON="${ENV_PREFIX}/bin/python"

if [[ "$VERIFY_ONLY" = false ]]; then
    CONDA="${CONDA_EXE:-$(command -v conda || true)}"
    [[ -x "$CONDA" ]] || die "Conda is required; install Miniconda or expose CONDA_EXE first."
    command -v git >/dev/null || die "git is required."
    if [[ ! -e "$ENV_PREFIX" ]]; then
        "$CONDA" create --yes --quiet --prefix "$ENV_PREFIX" --override-channels -c conda-forge \
            python=3.12 pip
    fi
    [[ -f "$ENV_PREFIX/conda-meta/history" && -x "$PYTHON" ]] \
        || die "Existing prefix is not a Conda environment; refusing to modify it."
    "$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Expected Python 3.12"'
    "$PYTHON" -m pip install --disable-pip-version-check torch==2.9.1 torchvision==0.24.1 \
        --index-url "$TORCH_INDEX_URL"
    "$PYTHON" -m pip install --disable-pip-version-check \
        -r "$REPO_ROOT/lerobot/evaluation/requirements_libero_qwen3vl_fp8.txt"
    "$PYTHON" -m pip check
    if [[ ! -e "$LIBERO_ROOT/.git" ]]; then
        git clone --quiet --no-checkout https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_ROOT"
        git -C "$LIBERO_ROOT" checkout --quiet --detach "$LIBERO_REF"
    fi
fi
[[ -x "$PYTHON" ]] || die "Environment Python not found: $PYTHON"
[[ -e "$LIBERO_ROOT/.git" ]] || die "LIBERO checkout not found: $LIBERO_ROOT"
[[ "$(git -C "$LIBERO_ROOT" rev-parse HEAD)" = "$LIBERO_REF" ]] \
    || die "LIBERO revision differs from --libero-ref; refusing to checkout/reset an existing repository."

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$REPO_ROOT"
exec "$PYTHON" -m lerobot.evaluation.scripts.verify_libero_qwen3vl \
    --libero-root "$LIBERO_ROOT" "$@"
