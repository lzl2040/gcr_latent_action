#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LIBERO_ROOT=""
LIBERO_REF="8f1084e3132a39270c3a13ebe37270a43ece2a01"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
SKIP_INSTALL=false
RESOURCE_ARGS=()

usage() {
    cat <<'EOF'
Install LIBERO/Qwen3-VL dependencies into the existing Conda environment lerobot.
Run with: conda run --name lerobot bash setup_libero_qwen3vl.sh ...
No cluster jobs are submitted, no extra environments or Python interpreters are created,
and no fine-tuned checkpoint is needed during setup.
Pretrained Qwen, Cosmos VAE and SigLIP assets download to node-local cache by default.
Setup prepares resources only; it never loads the policy or runs model tests.

Usage: conda run --name lerobot bash setup_libero_qwen3vl.sh [options]
  --libero-root PATH      Existing LIBERO checkout, or new clone destination
                         (default: CONDA_PREFIX/libero-source)
  --libero-ref SHA        Required immutable LIBERO revision (default: tested upstream SHA)
  --torch-index-url URL   PyTorch CUDA wheel index (default: cu128; use cu130 for newer drivers)
  --skip-install         Skip dependency installation/source cloning; prepare resources only
  --verify-only          Deprecated alias for --skip-install; does not run verification
  --model-cache-dir PATH Node-local model/metadata cache (default: MODEL_LOCAL_CACHE or /tmp/libero_qwen3vl_models)
  --qwen3vl-dir PATH     Use an explicit complete Qwen model instead of downloading it
  --cosmos3-dir PATH     Use an explicit complete Cosmos VAE instead of downloading it
  --stage1-text-dir PATH Use an explicit complete SigLIP text model instead of downloading it
  --offline              Prohibit pretrained downloads; require complete existing assets
  --help                 Show this help

All bundled suite normalization metadata is exported to MODEL_CACHE/dataset_metadata.
Run optional model/CUDA/EGL verification separately with:
  python -m lerobot.evaluation.scripts.verify_libero_qwen3vl --help
Checkpoint, precision, device and verification-output options do not belong to setup.
Installation removes optional flash-attn only if its binary has undefined-symbol import errors.

Requires an existing lerobot environment with Python 3.10-3.12, git for cloning, and host NVIDIA EGL drivers
for later evaluation. The GPU container must expose graphics capability as well
as compute/utility (NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics).
Use --offline to prohibit pretrained downloads, including with --skip-install;
dependency installation/source cloning still require their normal network access.
Bundled normalization avoids needing the original training dataset on the node.
No sudo, host-driver changes, fine-tuned weight downloads, or shared ~/.libero writes occur.
EOF
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
while (($#)); do
    case "$1" in
        --libero-root|--libero-ref|--torch-index-url|--model-cache-dir|--qwen3vl-dir|--cosmos3-dir|--stage1-text-dir)
            (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || die "Missing value for $1"
            case "$1" in
                --libero-root) LIBERO_ROOT="$2" ;;
                --libero-ref) LIBERO_REF="$2" ;;
                --torch-index-url) TORCH_INDEX_URL="$2" ;;
                *) RESOURCE_ARGS+=("$1" "$2") ;;
            esac
            shift 2 ;;
        --skip-install) SKIP_INSTALL=true; shift ;;
        --verify-only)
            printf 'WARNING: --verify-only is deprecated; use --skip-install. Setup no longer runs model tests.\n' >&2
            SKIP_INSTALL=true; shift ;;
        --offline) RESOURCE_ARGS+=("$1"); shift ;;
        --help|-h) usage; exit 0 ;;
        --) shift ;;
        *) die "Unknown installer option: $1. Model testing is separate: python -m lerobot.evaluation.scripts.verify_libero_qwen3vl --help" ;;
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

if [[ "$SKIP_INSTALL" = false ]]; then
    command -v git >/dev/null || die "git is required."
    "$PYTHON" -m pip install --no-user --disable-pip-version-check torch==2.9.1 torchvision==0.24.1 \
        --index-url "$TORCH_INDEX_URL"
    "$PYTHON" -m pip install --no-user --disable-pip-version-check \
        -r "$REPO_ROOT/lerobot/evaluation/requirements_libero_qwen3vl_fp8.txt"
    "$PYTHON" - <<'RUNTIME_PY'
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

try:
    flash_version = version("flash-attn")
except PackageNotFoundError:
    print("No pip-managed flash-attn package installed.", flush=True)
else:
    probe = subprocess.run(
        [sys.executable, "-c", "import torch; from flash_attn import flash_attn_func, flash_attn_varlen_func"],
        capture_output=True, text=True,
    )
    if probe.returncode:
        if not (
            probe.returncode > 0
            and "ImportError:" in probe.stderr
            and "flash_attn" in probe.stderr
            and "undefined symbol:" in probe.stderr
        ):
            raise RuntimeError(
                f"flash-attn import failed with exit code {probe.returncode}; "
                f"refusing automatic removal:\n{probe.stdout}\n{probe.stderr}"
            )
        package_root = Path(distribution("flash-attn").locate_file("")).resolve()
        if not package_root.is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError(
                f"flash-attn is outside the selected environment ({package_root}); "
                "refusing automatic removal."
            )
        print(
            f"Removing incompatible optional flash-attn {flash_version}; "
            "this evaluation uses native PyTorch SDPA.\n" + probe.stderr,
            file=sys.stderr, flush=True,
        )
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "--yes", "flash-attn"], check=True)
    else:
        print(f"Keeping importable flash-attn {flash_version}.", flush=True)

from diffusers import AutoencoderKLWan

print(f"Diffusers runtime import verified: {AutoencoderKLWan.__name__}.", flush=True)
RUNTIME_PY
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
"$PYTHON" - "${RESOURCE_ARGS[@]}" <<'ASSETS_PY'
import argparse
import logging
import os
from pathlib import Path
from types import SimpleNamespace

from lerobot.common.utils.stage2_model_cache import resolve_evaluation_model_paths
from lerobot.evaluation.scripts.verify_libero_qwen3vl import SUITE_DATASETS, prepare_dataset_metadata

parser = argparse.ArgumentParser(description="Prepare evaluation assets without loading or testing the policy.")
parser.add_argument(
    "--model-cache-dir", type=Path,
    default=Path(os.environ.get("MODEL_LOCAL_CACHE", "/tmp/libero_qwen3vl_models")),
)
parser.add_argument("--qwen3vl-dir", type=Path)
parser.add_argument("--cosmos3-dir", type=Path)
parser.add_argument("--stage1-text-dir", type=Path)
parser.add_argument("--offline", action="store_true")
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format="evaluation setup: %(message)s")
cache = args.model_cache_dir.expanduser().resolve()
paths = resolve_evaluation_model_paths(
    args.qwen3vl_dir, args.cosmos3_dir, args.stage1_text_dir,
    cache, auto_download=not args.offline,
)
for suite in SUITE_DATASETS:
    prepare_dataset_metadata(SimpleNamespace(dataset_root=None, task_suite=suite), cache, exist_ok=True)
for name, path in zip(("Qwen", "Cosmos VAE", "SigLIP text"), paths, strict=True):
    print(f"{name}: {path}", flush=True)
print(f"Dataset metadata (--dataset-root): {cache / 'dataset_metadata'}", flush=True)
print("Setup complete. No policy loading, CUDA/EGL smoke, or BF16/FP8 model tests were run.", flush=True)
ASSETS_PY
