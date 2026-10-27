"""Prepare pinned PI05 base assets on node-local disk before torchrun."""

import argparse
import fcntl
import hashlib
import logging
import os
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

from lerobot.common.utils.stage2_model_cache import _require_local_disk

BASE_REPO = "lerobot/pi05_base"
BASE_REVISION = "22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0"
BASE_SHA256 = "0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f"
TOKENIZER_URL = "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
TOKENIZER_SHA256 = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
_LOGGER = logging.getLogger(__name__)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_file(path, expected_hash=None):
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty asset: {path}")
    if expected_hash is not None and _sha256(path) != expected_hash:
        raise ValueError(f"SHA256 mismatch for {path}")


def _check_weights(path, expected_hash=None):
    from safetensors import SafetensorError, safe_open

    _check_file(path, expected_hash)
    if path.suffix == ".safetensors":
        try:
            with safe_open(path, framework="np") as source:
                if not list(source.keys()):
                    raise ValueError(f"No tensors in {path}")
        except SafetensorError as error:
            raise ValueError(f"Invalid safetensors asset {path}: {error}") from error


def _check_tokenizer(path, expected_hash=None):
    if path.is_dir():
        _check_file(path / "tokenizer_config.json")
        if not any((path / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "spiece.model")):
            raise FileNotFoundError(f"No tokenizer vocabulary in {path}")
    else:
        _check_file(path, expected_hash)


def _available(path, check, expected_hash=None):
    try:
        check(path, expected_hash)
    except (FileNotFoundError, ValueError) as error:
        _LOGGER.warning("%s", error)
        return False
    return True


def _download_weights(target, cache):
    hub, xet = cache / "_hub", cache / "_xet"
    for path in (hub, xet):
        _require_local_disk(path)
        path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_XET_CACHE"] = str(xet)
    from huggingface_hub import snapshot_download

    _LOGGER.info("Downloading %s@%s to %s", BASE_REPO, BASE_REVISION, target.parent)
    snapshot_download(
        repo_id=BASE_REPO,
        revision=BASE_REVISION,
        allow_patterns=["model.safetensors"],
        local_dir=str(target.parent),
        cache_dir=str(hub),
        force_download=target.exists(),
    )


def _download_tokenizer(target):
    _LOGGER.info("Downloading official PaliGemma SentencePiece vocabulary to %s", target)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".tokenizer-", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with urlopen(TOKENIZER_URL, timeout=120) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        _check_file(temporary, TOKENIZER_SHA256)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_assets(pretrained_path, tokenizer_path, cache_dir, *, auto_download=True):
    source = Path(pretrained_path).expanduser().resolve()
    tokenizer_source = Path(tokenizer_path).expanduser().resolve()
    weights_source = (
        source
        if source.is_file() or source.suffix in (".safetensors", ".pt", ".pth")
        else source / "model.safetensors"
    )
    cache = Path(cache_dir).expanduser().resolve()
    target = cache / "pi05_base"
    weights_target = target / "model.safetensors"
    tokenizer_target = target / "paligemma_tokenizer.model"
    weights = (
        weights_source
        if weights_source != weights_target and _available(weights_source, _check_weights)
        else None
    )
    tokenizer = (
        tokenizer_source
        if tokenizer_source != tokenizer_target and _available(tokenizer_source, _check_tokenizer)
        else None
    )
    if weights is not None and tokenizer is not None:
        return str(weights), str(tokenizer)
    _require_local_disk(cache)
    _require_local_disk(target)
    if not auto_download and not cache.is_dir():
        raise FileNotFoundError("Missing PI05 assets; MODEL_AUTO_DOWNLOAD=false prohibits downloading them.")
    cache.mkdir(parents=True, exist_ok=True)
    lock = cache / ".pi05_base.lock"
    if lock.is_symlink() or target.is_symlink():
        raise ValueError(f"Managed PI05 cache paths must not be symlinks: {lock}, {target}")
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        need_weights = weights is None and not _available(weights_target, _check_weights, BASE_SHA256)
        need_tokenizer = tokenizer is None and not _available(
            tokenizer_target, _check_tokenizer, TOKENIZER_SHA256
        )
        if (need_weights or need_tokenizer) and not auto_download:
            raise FileNotFoundError(
                "Missing/invalid PI05 assets; MODEL_AUTO_DOWNLOAD=false prohibits downloading."
            )
        if need_weights or need_tokenizer:
            target.mkdir(parents=True, exist_ok=True)
            if any(path.is_symlink() for path in target.rglob("*")):
                raise ValueError(f"Managed PI05 download directory contains symlinks: {target}")
        if need_weights:
            _download_weights(weights_target, cache)
            _check_weights(weights_target, BASE_SHA256)
        if need_tokenizer:
            _download_tokenizer(tokenizer_target)
            _check_tokenizer(tokenizer_target, TOKENIZER_SHA256)
    return str(weights or weights_target), str(tokenizer or tokenizer_target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--auto-download", choices=("true", "false"), default="true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="PI05 assets: %(message)s")
    paths = (args.pretrained_path, args.tokenizer_path, args.cache_dir)
    if any("\n" in path or "\r" in path for path in paths):
        parser.error("Asset paths must not contain newlines.")
    for path in resolve_assets(*paths, auto_download=args.auto_download == "true"):
        print(path)


if __name__ == "__main__":
    main()
