#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-/home/v-wangxiaofa/anaconda3/envs/libero_lerobot_qwen3vl/bin/python}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/home/v-wangxiaofa/lzl/LIBERO_PLUS}"
SOURCE_REF="4976dc30028e805ff8094b55501d532c48fec182"

if [[ "${1:-}" = --help || "${1:-}" = -h ]]; then
    cat <<'EOF'
Install pinned LIBERO-Plus into the existing LIBERO evaluation environment.
Usage: bash lerobot/evaluation/scripts/setup_libero_plus.sh
Optional environment overrides: PYTHON, LIBERO_PLUS_ROOT.
Downloads the official 6.4 GB asset archive and verifies its SHA256 before extraction.
Preserves existing Torch/NumPy/simulator versions and never writes ~/.libero/config.yaml.
Does not download training demonstrations, load a policy, or start a benchmark.
Run verify_libero_plus.py separately for CPU rendering and perturbation smoke checks.
EOF
    exit 0
fi
[[ $# = 0 && -x "$PYTHON" ]] || { printf 'Invalid arguments or missing existing Python: %s\n' "$PYTHON" >&2; exit 1; }

if [[ ! -e "$LIBERO_PLUS_ROOT" ]]; then
    git clone --depth 1 https://github.com/sylvestf/LIBERO-plus.git "$LIBERO_PLUS_ROOT"
    if [[ "$(git -C "$LIBERO_PLUS_ROOT" rev-parse HEAD)" != "$SOURCE_REF" ]]; then
        git -C "$LIBERO_PLUS_ROOT" fetch --depth 1 origin "$SOURCE_REF"
        git -C "$LIBERO_PLUS_ROOT" checkout --detach "$SOURCE_REF"
    fi
fi
[[ "$(git -C "$LIBERO_PLUS_ROOT" rev-parse HEAD)" = "$SOURCE_REF" ]] \
    || { printf 'Existing checkout differs from the pinned revision; refusing to reset it.\n' >&2; exit 1; }
git -C "$LIBERO_PLUS_ROOT" diff --quiet
git -C "$LIBERO_PLUS_ROOT" diff --cached --quiet
STATE_DIR="$LIBERO_PLUS_ROOT/.setup"
mkdir -p "$STATE_DIR"
export PYTHONNOUSERSITE=1
if [[ ! -f "$STATE_DIR/environment-before.txt" ]]; then
    "$PYTHON" -m pip freeze --all > "$STATE_DIR/environment-before.txt"
fi
"$PYTHON" - "$STATE_DIR" <<'PY'
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

state = Path(sys.argv[1])
protected = [
    "torch", "torchvision", "transformers", "diffusers", "numpy", "scipy", "Pillow",
    "opencv-python", "opencv-python-headless", "mujoco", "robosuite", "bddl",
]
versions = {name: version(name) for name in protected}
path = state / "protected-runtime.json"
if path.exists():
    assert json.loads(path.read_text()) == versions, "Existing evaluation runtime changed."
else:
    path.write_text(json.dumps(versions, indent=2) + "\n")
(state / "constraints.txt").write_text("".join(f"{name}=={value}\n" for name, value in versions.items()))
config = Path.home() / ".libero/config.yaml"
snapshot = state / "original-config-sha256.txt"
current = hashlib.sha256(config.read_bytes()).hexdigest() if config.exists() else "absent"
if snapshot.exists():
    assert snapshot.read_text().strip() == current, "Shared LIBERO config changed."
else:
    snapshot.write_text(current + "\n")
print("Using existing evaluation Python:", sys.executable, flush=True)
PY
"$PYTHON" -m pip install --no-user --disable-pip-version-check \
    --constraint "$STATE_DIR/constraints.txt" -r "$REPO_ROOT/lerobot/evaluation/requirements_libero_plus.txt"
# Compatibility mode exposes upstream's nested namespace package with modern setuptools.
"$PYTHON" -m pip install --no-user --disable-pip-version-check --no-deps --no-build-isolation \
    --config-settings editable_mode=compat --editable "$LIBERO_PLUS_ROOT"

export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=30
"$PYTHON" - "$LIBERO_PLUS_ROOT" "$SOURCE_REF" <<'PY'
import hashlib
import json
import shutil
import stat
import sys
import zipfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download

root = Path(sys.argv[1]).resolve()
state = root / ".setup"
revision = "dd2bd61b7d9a6fef1abc52d606e983b41886a149"
expected = "96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf"
archive = Path(hf_hub_download(
    repo_id="Sylvest/LIBERO-plus", repo_type="dataset", filename="assets.zip",
    revision=revision, local_dir=root / ".downloads", token=False,
))
with archive.open("rb") as handle:
    actual = hashlib.file_digest(handle, "sha256").hexdigest()
assert actual == expected, "Asset archive SHA256 mismatch."
destination = root / "libero/libero"
marker = state / "assets.json"
if marker.exists():
    assets = json.loads(marker.read_text())
    assert assets["sha256"] == expected and assets["revision"] == revision
else:
    with zipfile.ZipFile(archive) as source:
        entries = source.infolist()
        first_parts = Path(entries[0].filename).parts
        prefix = first_parts[:first_parts.index("assets")]
        mapped = []
        for item in entries:
            parts = Path(item.filename).parts
            assert parts[:len(prefix)] == prefix, f"Inconsistent archive prefix: {item.filename}"
            relative = Path(*parts[len(prefix):])
            assert relative.parts[0] == "assets" and ".." not in relative.parts
            target = (destination / relative).resolve()
            assert target.is_relative_to(destination / "assets"), f"Unsafe ZIP path: {item.filename}"
            assert not stat.S_ISLNK(item.external_attr >> 16), f"ZIP symlink: {item.filename}"
            mapped.append((item, target))
        size = sum(item.file_size for item in entries)
        assert shutil.disk_usage(destination).free >= size + 5 * 1024**3, "Insufficient extraction disk space."
        print(f"Extracting {len(entries)} entries, {size / 1024**3:.2f} GiB.", flush=True)
        # The official archive embeds its build-machine path before assets/.
        for item, target in mapped:
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(item) as incoming, target.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, length=1024**2)
    assets = {
        "repo_id": "Sylvest/LIBERO-plus", "revision": revision, "filename": archive.name,
        "sha256": actual, "archive_bytes": archive.stat().st_size,
        "entries": len(entries), "uncompressed_bytes": size,
        "stripped_archive_prefix": "/".join(prefix),
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }
    marker.write_text(json.dumps(assets, indent=2) + "\n")
for name in ("articulated_objects", "new_objects", "scenes", "textures"):
    assert (destination / "assets" / name).is_dir(), f"Missing extracted assets/{name}"
datasets = state / "datasets"
datasets.mkdir(exist_ok=True)
paths = {
    "benchmark_root": str(destination), "bddl_files": str(destination / "bddl_files"),
    "init_states": str(destination / "init_files"), "assets": str(destination / "assets"),
    "datasets": str(datasets),
}
config_dir = state / "config"
config_dir.mkdir(exist_ok=True)
config = config_dir / "config.yaml"
if config.exists():
    assert yaml.safe_load(config.read_text()) == paths, "Refusing to overwrite an unrelated local config."
else:
    config.write_text(yaml.safe_dump(paths))
protected = json.loads((state / "protected-runtime.json").read_text())
assert {name: version(name) for name in protected} == protected
original = Path.home() / ".libero/config.yaml"
original_hash = hashlib.sha256(original.read_bytes()).hexdigest() if original.exists() else "absent"
assert (state / "original-config-sha256.txt").read_text().strip() == original_hash
report = {
    "installed_at": datetime.now(timezone.utc).isoformat(), "python": sys.executable,
    "source": str(root), "source_revision": sys.argv[2], "assets": assets,
    "config_dir": str(config_dir), "protected_runtime_unchanged": protected,
    "additional_packages": {
        name: version(name) for name in ("libero", "Wand", "scikit-image", "lazy-loader", "tifffile", "fonttools")
    },
    "shared_libero_config_unchanged": True,
}
(state / "installation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
print(f"Use LIBERO_CONFIG_PATH={config_dir} and PYTHONPATH={root} for standalone Plus tools.", flush=True)
print("For trusted upstream NumPy initial-state files with Torch2.6+, scope TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 to Plus processes.", flush=True)
PY
"$PYTHON" -m pip freeze --all > "$STATE_DIR/environment-after.txt"
