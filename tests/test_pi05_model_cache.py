import hashlib
import io
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from safetensors.numpy import save_file

from lerobot.common.utils import pi05_model_cache as cache


@pytest.fixture
def downloads(tmp_path, monkeypatch):
    source = tmp_path / "source.safetensors"
    save_file({"weight": np.ones((2, 2), dtype=np.float32)}, source)
    model_bytes = source.read_bytes()
    tokenizer_bytes = b"fixture vocabulary"
    monkeypatch.setattr(cache, "BASE_SHA256", hashlib.sha256(model_bytes).hexdigest())
    monkeypatch.setattr(cache, "TOKENIZER_SHA256", hashlib.sha256(tokenizer_bytes).hexdigest())
    monkeypatch.setattr(cache, "_require_local_disk", Mock())

    def snapshot(**kwargs):
        destination = Path(kwargs["local_dir"]) / "model.safetensors"
        destination.write_bytes(model_bytes)
        return kwargs["local_dir"]

    import huggingface_hub

    download = Mock(side_effect=snapshot)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    tokenizer = Mock(side_effect=lambda *args, **kwargs: io.BytesIO(tokenizer_bytes))
    monkeypatch.setattr(cache, "urlopen", tokenizer)
    return download, tokenizer, source, tokenizer_bytes


def test_missing_assets_download_only_to_node_cache(tmp_path, downloads):
    download, tokenizer, _, _ = downloads
    mounted = tmp_path / "mounted" / "missing_base"
    local = tmp_path / "local"
    weights, vocabulary = cache.resolve_assets(mounted, mounted / "tokenizer.model", local)
    assert Path(weights) == local / "pi05_base/model.safetensors"
    assert Path(vocabulary) == local / "pi05_base/paligemma_tokenizer.model"
    assert not mounted.exists()
    kwargs = download.call_args.kwargs
    assert kwargs["repo_id"] == cache.BASE_REPO and kwargs["revision"] == cache.BASE_REVISION
    assert kwargs["allow_patterns"] == ["model.safetensors"]
    assert kwargs["local_dir"] == str(local / "pi05_base")
    assert kwargs["cache_dir"] == str(local / "_hub")
    assert kwargs["force_download"] is False
    tokenizer.assert_called_once_with(cache.TOKENIZER_URL, timeout=120)
    assert not list((local / "pi05_base").glob(".tokenizer-*"))


def test_valid_existing_assets_need_no_cache_or_network(tmp_path, downloads):
    download, tokenizer_download, source, tokenizer_bytes = downloads
    vocabulary = tmp_path / "existing.model"
    vocabulary.write_bytes(tokenizer_bytes)
    local = tmp_path / "unused cache"
    assert cache.resolve_assets(source, vocabulary, local) == (str(source), str(vocabulary))
    download.assert_not_called()
    tokenizer_download.assert_not_called()
    cache._require_local_disk.assert_not_called()
    assert not local.exists()


def test_completed_cache_is_reused_offline(tmp_path, downloads):
    download, tokenizer, _, _ = downloads
    local = tmp_path / "local"
    expected = cache.resolve_assets(tmp_path / "missing", tmp_path / "missing.model", local)
    download.reset_mock()
    tokenizer.reset_mock()
    assert (
        cache.resolve_assets(tmp_path / "missing", tmp_path / "missing.model", local, auto_download=False)
        == expected
    )
    download.assert_not_called()
    tokenizer.assert_not_called()


def test_auto_download_disabled_reports_missing_assets(tmp_path, downloads):
    download, tokenizer, _, _ = downloads
    with pytest.raises(FileNotFoundError, match="MODEL_AUTO_DOWNLOAD=false"):
        cache.resolve_assets(
            tmp_path / "missing", tmp_path / "missing.model", tmp_path / "local", auto_download=False
        )
    download.assert_not_called()
    tokenizer.assert_not_called()


def test_bad_cached_weights_are_redownloaded(tmp_path, downloads):
    download, _, _, _ = downloads
    local = tmp_path / "local"
    cache.resolve_assets(tmp_path / "missing", tmp_path / "missing.model", local)
    (local / "pi05_base/model.safetensors").write_bytes(b"truncated")
    cache.resolve_assets(tmp_path / "missing", tmp_path / "missing.model", local)
    assert download.call_count == 2 and download.call_args.kwargs["force_download"] is True


def test_download_checksum_failure_is_not_accepted(tmp_path, downloads, monkeypatch):
    monkeypatch.setattr(cache, "BASE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        cache.resolve_assets(tmp_path / "missing", tmp_path / "missing.model", tmp_path / "local")


def test_tokenizer_checksum_failure_cleans_temporary_file(tmp_path, downloads, monkeypatch):
    monkeypatch.setattr(cache, "TOKENIZER_SHA256", "0" * 64)
    local = tmp_path / "local"
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        cache.resolve_assets(tmp_path / "missing", tmp_path / "missing.model", local)
    assert not (local / "pi05_base/paligemma_tokenizer.model").exists()
    assert not list((local / "pi05_base").glob(".tokenizer-*"))


def test_download_errors_propagate(tmp_path, downloads):
    download, _, _, _ = downloads
    download.side_effect = ConnectionError("network unavailable")
    with pytest.raises(ConnectionError, match="network unavailable"):
        cache.resolve_assets(tmp_path / "missing", tmp_path / "missing.model", tmp_path / "local")


def test_rejects_nonlocal_download_cache(tmp_path, downloads, monkeypatch):
    download, _, _, _ = downloads
    monkeypatch.setattr(cache, "_require_local_disk", Mock(side_effect=ValueError("not node-local")))
    with pytest.raises(ValueError, match="node-local"):
        cache.resolve_assets(tmp_path / "missing", tmp_path / "missing.model", tmp_path / "mounted")
    download.assert_not_called()
