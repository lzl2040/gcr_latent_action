"""Resolve Stage 2 model files without writing to the mounted weight directories."""

from __future__ import annotations

import argparse
import fcntl
import fnmatch
import json
import logging
import os
import re
import shutil
from pathlib import Path

_MODELS = {
    "Qwen3-VL-4B-Instruct": ("Qwen/Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17"),
    "Cosmos3-Edge": ("nvidia/Cosmos3-Edge", "344d602b128d1bbdacb43b08d0a3626f46343e29"),
    "Cosmos3-Nano": ("nvidia/Cosmos3-Nano", "e59a53c25979a090fa8706c9acc0c254a6e89b92"),
    "siglip2-base-patch16-224": (
        "google/siglip2-base-patch16-224",
        "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
    ),
}
_PATTERNS = {
    "qwen": ["*.json", "*.txt", "*.jinja", "*.safetensors"],
    "vae": ["vae/*"],
    "generation": ["transformer/*", "vae/config.json"],
    "text": ["*.json", "*.txt", "*.model", "*.safetensors"],
}
_LOGGER = logging.getLogger(__name__)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _safetensor_keys(path: Path) -> set[str]:
    from safetensors import SafetensorError, safe_open

    if not path.is_file():
        raise FileNotFoundError(f"Missing safetensors file: {path}.")
    try:
        with safe_open(path, framework="np") as source:
            return set(source.keys())
    except SafetensorError as error:
        raise ValueError(f"Invalid or incomplete safetensors file {path}: {error}.") from error


def _check_weights(root: Path, basename: str) -> None:
    index = root / f"{basename}.safetensors.index.json"
    if not index.exists():
        _safetensor_keys(root / f"{basename}.safetensors")
        return
    weight_map = _read_json(index).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Missing weight_map in {index}.")
    if any(
        not isinstance(name, str) or not isinstance(filename, str) or Path(filename).name != filename
        for name, filename in weight_map.items()
    ):
        raise ValueError(f"Invalid tensor name or shard filename in {index}.")
    for filename in set(weight_map.values()):
        keys = _safetensor_keys(root / filename)
        missing = {name for name, shard in weight_map.items() if shard == filename} - keys
        if missing:
            raise ValueError(f"Shard {root / filename} is missing indexed tensors: {sorted(missing)}.")


def check_model_directory(root: Path, component: str) -> None:
    if component == "qwen":
        for filename in ("config.json", "preprocessor_config.json", "tokenizer_config.json"):
            _read_json(root / filename)
        tokenizer = root / "tokenizer.json"
        if tokenizer.stat().st_size == 0:
            raise ValueError(f"Empty tokenizer file: {tokenizer}.")
        if not (
            (root / "chat_template.json").is_file()
            or (root / "chat_template.jinja").is_file()
            or _read_json(root / "tokenizer_config.json").get("chat_template")
        ):
            raise FileNotFoundError(f"Missing Qwen chat template in {root}.")
        _check_weights(root, "model")
    elif component == "text":
        for filename in ("config.json", "tokenizer_config.json"):
            _read_json(root / filename)
        if not any(
            (root / filename).is_file() and (root / filename).stat().st_size > 0
            for filename in ("tokenizer.json", "tokenizer.model")
        ):
            raise FileNotFoundError(f"Missing SigLIP tokenizer in {root}.")
        _check_weights(root, "model")
    elif component in ("vae", "generation"):
        subdir = "vae" if component == "vae" else "transformer"
        _read_json(root / subdir / "config.json")
        _check_weights(root / subdir, "diffusion_pytorch_model")
        if component == "generation":
            _read_json(root / "vae" / "config.json")
    else:
        raise ValueError(f"Unknown Stage 2 model component: {component}.")


def _mount_type(path: Path) -> str:
    path = path.resolve()
    matches = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        mount, filesystem = line.split(" - ", 1)
        mountpoint = re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), mount.split()[4])
        root = Path(mountpoint)
        if path.is_relative_to(root):
            matches.append((len(root.parts), int(mount.split()[0]), filesystem.split()[0]))
    if not matches:
        raise RuntimeError(f"Cannot identify the filesystem for model cache {path}.")
    return max(matches)[2]


def _require_local_disk(path: Path) -> None:
    filesystem = _mount_type(path)
    if filesystem.startswith("fuse") or filesystem in {
        "nfs", "nfs4", "cifs", "smb3", "ceph", "afs", "9p", "lustre", "gpfs", "tmpfs", "ramfs",
    }:
        raise ValueError(
            f"MODEL_LOCAL_CACHE must use node-local disk, not {filesystem}: {path}. "
            "Choose a local SSD/scratch directory; downloads must not go to Blob or RAM."
        )


def _download_model(model: str, component: str, target: Path, cache: Path, *, force: bool) -> None:
    hub_cache, xet_cache = cache / "_hub", cache / "_xet"
    for path in (hub_cache, xet_cache):
        _require_local_disk(path)
        path.mkdir(parents=True, exist_ok=True)
    # This helper runs before torchrun, with no Hub imports before setting the Xet cache.
    os.environ["HF_XET_CACHE"] = str(xet_cache)
    from huggingface_hub import HfApi, snapshot_download

    repo, revision = _MODELS[model]
    patterns = _PATTERNS[component]
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    files = [item for item in (info.siblings or []) if any(fnmatch.fnmatch(item.rfilename, p) for p in patterns)]
    if not files or any(item.size is None for item in files):
        raise RuntimeError(f"Cannot determine download sizes for {repo} ({component}).")
    force = force or any(
        (target / item.rfilename).is_file() and (target / item.rfilename).stat().st_size != item.size
        for item in files
    )
    needed = sum(
        item.size for item in files
        if force or not (target / item.rfilename).is_file()
    )
    free = shutil.disk_usage(cache).free
    if needed > free:
        raise OSError(
            f"Insufficient node-local disk for {repo} ({component}): "
            f"need {needed / 2**30:.2f} GiB, available {free / 2**30:.2f} GiB at {cache}."
        )
    _LOGGER.info("Downloading %s (%s), revision %s, into %s", repo, component, revision, target)
    snapshot_download(
        repo_id=repo, revision=revision, allow_patterns=patterns,
        local_dir=str(target), cache_dir=str(hub_cache), force_download=force,
    )


def _resolve_model(
    requested: str, model: str, component: str, cache: Path, *, auto_download: bool
) -> str:
    source = Path(requested).expanduser()
    try:
        check_model_directory(source, component)
    except (FileNotFoundError, ValueError) as error:
        _LOGGER.warning("Incomplete %s model at %s: %s; using node-local cache.", component, source, error)
    else:
        _LOGGER.info("Using existing %s model: %s", component, source)
        return str(source)

    cache = cache.expanduser().resolve()
    _require_local_disk(cache)
    target = cache / model
    _require_local_disk(target)
    if not auto_download:
        try:
            check_model_directory(target, component)
        except (FileNotFoundError, ValueError) as error:
            raise FileNotFoundError(
                f"No complete {component} model at {source} or {target}: {error}. "
                "MODEL_AUTO_DOWNLOAD=false prohibits downloading replacement weights."
            ) from error
        _LOGGER.info("Using complete offline node-local %s model: %s", component, target)
        return str(target)
    cache.mkdir(parents=True, exist_ok=True)
    lock = cache / f"{model}.lock"
    if lock.is_symlink() or target.is_symlink():
        raise ValueError(f"Managed model cache paths must not be symlinks: {target}, {lock}.")
    with lock.open("a") as handle:
        _LOGGER.info("Waiting for node-local model lock: %s", lock)
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            check_model_directory(target, component)
        except (FileNotFoundError, ValueError) as error:
            force = isinstance(error, ValueError)
        else:
            _LOGGER.info("Using complete node-local %s model: %s", component, target)
            return str(target)
        target.mkdir(parents=True, exist_ok=True)
        if any(path.is_symlink() for path in target.rglob("*")):
            raise ValueError(f"Managed download directory contains symlinks: {target}.")
        _download_model(model, component, target, cache, force=force)
        check_model_directory(target, component)
        _LOGGER.info("Using complete node-local %s model: %s", component, target)
    return str(target)


def resolve_model_paths(
    qwen_dir: str, cosmos_dir: str, generation_dir: str, generation_backbone: str,
    cache: Path, *, need_generation: bool, auto_download: bool = True,
) -> tuple[str, str, str]:
    if generation_backbone not in ("", "random", "edge", "nano"):
        raise ValueError(f"Unknown generation backbone: {generation_backbone}.")
    if any("\n" in path or "\r" in path for path in (qwen_dir, cosmos_dir, generation_dir, str(cache))):
        raise ValueError("Model paths must not contain newlines.")
    qwen_dir = _resolve_model(qwen_dir, "Qwen3-VL-4B-Instruct", "qwen", cache, auto_download=auto_download)
    vae_model = "Cosmos3-Nano" if Path(cosmos_dir).name == "Cosmos3-Nano" else "Cosmos3-Edge"
    cosmos_dir = _resolve_model(cosmos_dir, vae_model, "vae", cache, auto_download=auto_download)
    if need_generation and generation_backbone in ("edge", "nano"):
        model = "Cosmos3-Edge" if generation_backbone == "edge" else "Cosmos3-Nano"
        generation_dir = _resolve_model(
            generation_dir, model, "generation", cache, auto_download=auto_download
        )
    return qwen_dir, cosmos_dir, generation_dir


def resolve_evaluation_model_paths(
    qwen_dir: Path | None,
    cosmos_dir: Path | None,
    text_dir: Path | None,
    cache: Path,
    *,
    auto_download: bool = True,
) -> tuple[str, str, str]:
    paths = []
    for requested, model, component in (
        (qwen_dir, "Qwen3-VL-4B-Instruct", "qwen"),
        (cosmos_dir, "Cosmos3-Edge", "vae"),
        (text_dir, "siglip2-base-patch16-224", "text"),
    ):
        if requested is not None:
            source = requested.expanduser().resolve()
            check_model_directory(source, component)
            paths.append(str(source))
        else:
            paths.append(
                _resolve_model(
                    str(cache / model),
                    model,
                    component,
                    cache,
                    auto_download=auto_download,
                )
            )
    return paths[0], paths[1], paths[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-dir", required=True)
    parser.add_argument("--cosmos-dir", required=True)
    parser.add_argument("--generation-dir", default="")
    parser.add_argument("--generation-backbone", default="")
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--need-generation", action="store_true")
    parser.add_argument("--auto-download", choices=("true", "false"), default="true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="model cache: %(message)s")
    paths = resolve_model_paths(
        args.qwen_dir, args.cosmos_dir, args.generation_dir, args.generation_backbone, args.cache_dir,
        need_generation=args.need_generation, auto_download=args.auto_download == "true",
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
