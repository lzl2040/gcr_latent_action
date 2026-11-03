#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/v-wangxiaofa/anaconda3/envs/libero_lerobot_qwen3vl/bin/python}"
LIBERO_PRO_ROOT="${LIBERO_PRO_ROOT:-/home/v-wangxiaofa/lzl/LIBERO_PRO}"
SOURCE_REF="eafdb809426b13153aa1e4c42d6601844217dfec"

if [[ "${1:-}" = --help || "${1:-}" = -h ]]; then
    cat <<'EOF'
Install pinned LIBERO-Pro into the existing LIBERO evaluation environment.
Usage: bash lerobot/evaluation/scripts/setup_libero_pro.sh
Optional environment overrides: PYTHON, LIBERO_PRO_ROOT.
Preserves all existing simulator/model dependencies and ~/.libero/config.yaml.
Downloads the classic official object/position/language/task BDDL and initial states.
Environment replacement has no prepackaged initial states in this dataset revision.
Does not enable the newer, incompatible seven-case dataset extension.
Does not download demonstrations, generate benchmark initial states, or load a policy.
Run verify_libero_pro.py separately for CPU rendering and perturbation smoke checks.
EOF
    exit 0
fi
[[ $# = 0 && -x "$PYTHON" ]] || { printf 'Invalid arguments or missing existing Python: %s\n' "$PYTHON" >&2; exit 1; }

if [[ ! -e "$LIBERO_PRO_ROOT" ]]; then
    git clone --depth 1 https://github.com/Zxy-MLlab/LIBERO-PRO.git "$LIBERO_PRO_ROOT"
    if [[ "$(git -C "$LIBERO_PRO_ROOT" rev-parse HEAD)" != "$SOURCE_REF" ]]; then
        git -C "$LIBERO_PRO_ROOT" fetch --depth 1 origin "$SOURCE_REF"
        git -C "$LIBERO_PRO_ROOT" checkout --detach "$SOURCE_REF"
    fi
fi
[[ "$(git -C "$LIBERO_PRO_ROOT" rev-parse HEAD)" = "$SOURCE_REF" ]] \
    || { printf 'Existing checkout differs from the pinned revision; refusing to reset it.\n' >&2; exit 1; }
git -C "$LIBERO_PRO_ROOT" diff --quiet
git -C "$LIBERO_PRO_ROOT" diff --cached --quiet
STATE_DIR="$LIBERO_PRO_ROOT/.setup"
mkdir -p "$STATE_DIR"
export PYTHONNOUSERSITE=1
if [[ ! -f "$STATE_DIR/environment-before.txt" ]]; then
    "$PYTHON" -m pip freeze --all > "$STATE_DIR/environment-before.txt"
fi
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=30
"$PYTHON" - "$LIBERO_PRO_ROOT" "$SOURCE_REF" <<'PY'
import hashlib
import json
import shutil
import sys
from collections import Counter
from importlib.metadata import distribution, version
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download

root = Path(sys.argv[1]).resolve()
state = root / ".setup"
protected = [
    "torch", "torchvision", "transformers", "diffusers", "numpy", "scipy", "Pillow",
    "opencv-python", "opencv-python-headless", "mujoco", "robosuite", "bddl",
]
versions = {name: version(name) for name in protected}
snapshot = state / "protected-runtime.json"
if snapshot.exists():
    assert json.loads(snapshot.read_text()) == versions, "Existing evaluation runtime changed."
else:
    snapshot.write_text(json.dumps(versions, indent=2) + "\n")
(state / "constraints.txt").write_text("".join(f"{name}=={value}\n" for name, value in versions.items()))
config = Path.home() / ".libero/config.yaml"
original_hash = hashlib.sha256(config.read_bytes()).hexdigest() if config.exists() else "absent"
snapshot = state / "original-config-sha256.txt"
if snapshot.exists():
    assert snapshot.read_text().strip() == original_hash, "Shared LIBERO config changed."
else:
    snapshot.write_text(original_hash + "\n")
previous = state / "previous-libero-installation.json"
if not previous.exists():
    previous.write_text(distribution("libero").read_text("direct_url.json") or "{}")

repo = "zhouxueyang/LIBERO-Pro"
revision = "bcaa82b85c591fc04151499ba61bfdba4bf55eb0"
api = HfApi(token=False)
files = [
    entry for entry in api.dataset_info(repo, revision=revision, files_metadata=True).siblings
    if entry.rfilename.startswith(("bddl_files/", "init_files/"))
]
assert len(files) == 320, "Unexpected classic dataset file count."
download = Path(snapshot_download(
    repo_id=repo, repo_type="dataset", revision=revision, token=False,
    local_dir=root / ".downloads", allow_patterns=["bddl_files/**", "init_files/**"], max_workers=8,
))
benchmark = root / "libero/libero"
records = []
for entry in files:
    relative = Path(entry.rfilename)
    assert not relative.is_absolute() and ".." not in relative.parts
    assert relative.suffix in (".bddl", ".pruned_init")
    incoming = download / relative
    data = incoming.read_bytes()
    assert len(data) == entry.size, f"Size mismatch: {relative}"
    digest = hashlib.sha256(data).hexdigest()
    if entry.lfs:
        assert digest == entry.lfs.sha256, f"LFS SHA256 mismatch: {relative}"
    else:
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        assert blob == entry.blob_id, f"Git blob checksum mismatch: {relative}"
    target = benchmark / relative
    assert target.resolve().is_relative_to(benchmark), f"Unsafe destination: {relative}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest, f"Refusing to overwrite: {target}"
    else:
        shutil.copyfile(incoming, target)
    records.append({"path": entry.rfilename, "bytes": len(data), "sha256": digest})
counts = dict(Counter("/".join(record["path"].split("/")[:2]) for record in records))
assert len(counts) == 32 and set(counts.values()) == {10}
assets = {
    "repo_id": repo, "revision": revision, "files": records, "folder_counts": counts,
    "total_bytes": sum(record["bytes"] for record in records),
    "prepackaged_environment_initial_states": False,
    "seven_case_extension_enabled": False,
}
(state / "assets.json").write_text(json.dumps(assets, indent=2) + "\n")
datasets = state / "datasets"
datasets.mkdir(exist_ok=True)
paths = {
    "benchmark_root": str(benchmark), "bddl_files": str(benchmark / "bddl_files"),
    "init_states": str(benchmark / "init_files"), "assets": str(benchmark / "assets"),
    "datasets": str(datasets),
}
assert all(Path(value).is_dir() for value in paths.values()), "Missing simulator resources."
config_dir = state / "config"
config_dir.mkdir(exist_ok=True)
local_config = config_dir / "config.yaml"
if local_config.exists():
    assert yaml.safe_load(local_config.read_text()) == paths, "Refusing to overwrite an unrelated local config."
else:
    local_config.write_text(yaml.safe_dump(paths))
print(f"Verified and installed {len(records)} official evaluation files ({assets['total_bytes']} bytes).", flush=True)
PY
# Compatibility mode exposes the nested libero namespace without changing any runtime dependencies.
"$PYTHON" -m pip install --no-user --disable-pip-version-check --no-deps --no-build-isolation \
    --constraint "$STATE_DIR/constraints.txt" --config-settings editable_mode=compat --editable "$LIBERO_PRO_ROOT"
"$PYTHON" -m pip freeze --all > "$STATE_DIR/environment-after.txt"
env -u PYTHONPATH LIBERO_CONFIG_PATH="$STATE_DIR/config" "$PYTHON" - "$LIBERO_PRO_ROOT" "$SOURCE_REF" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import libero.libero as libero

root = Path(sys.argv[1]).resolve()
state = root / ".setup"
assert Path(libero.__file__).resolve().is_relative_to(root), "Editable import selected another checkout."
protected = json.loads((state / "protected-runtime.json").read_text())
assert {name: version(name) for name in protected} == protected
original = Path.home() / ".libero/config.yaml"
digest = hashlib.sha256(original.read_bytes()).hexdigest() if original.exists() else "absent"
assert (state / "original-config-sha256.txt").read_text().strip() == digest
before = set((state / "environment-before.txt").read_text().splitlines())
after = set((state / "environment-after.txt").read_text().splitlines())
changed = before.symmetric_difference(after)
assert all("libero" in line for line in changed), f"Unrelated package changes: {changed}"
report = {
    "installed_at": datetime.now(timezone.utc).isoformat(), "python": sys.executable,
    "source": str(root), "source_revision": sys.argv[2], "libero_module": libero.__file__,
    "config_dir": str(state / "config"), "protected_runtime_unchanged": protected,
    "shared_libero_config_unchanged": True, "environment_changes": sorted(changed),
    "assets_manifest_sha256": hashlib.sha256((state / "assets.json").read_bytes()).hexdigest(),
    "policy_loaded": False, "benchmark_started": False,
}
(state / "installation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
PY
"$PYTHON" -m pip check
