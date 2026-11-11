import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file

from lerobot.common.utils import stage2_model_cache as cache_module


def _write_model(root, component):
    root.mkdir(parents=True, exist_ok=True)
    if component == "cosmos3_policy":
        (root / "config.json").write_text(json.dumps({"type": "cosmos3_nano"}))
        save_file({"weight": np.ones(1, dtype=np.float32)}, str(root / "model.safetensors"))
        for name in ("policy_preprocessor", "policy_postprocessor"):
            (root / f"{name}.json").write_text(json.dumps({"steps": []}))
        tokenizer = root / "tokenizer"
        tokenizer.mkdir(exist_ok=True)
        (tokenizer / "tokenizer_config.json").write_text("{}")
        (tokenizer / "tokenizer.json").write_text("{}")
        (tokenizer / "chat_template.jinja").write_text("template")
        return
    if component == "text":
        for name in ("config", "tokenizer_config", "tokenizer"):
            (root / f"{name}.json").write_text("{}")
        save_file({"weight": np.ones(1, dtype=np.float32)}, str(root / "model.safetensors"))
        return
    if component == "qwen":
        for name in ("config", "preprocessor_config", "tokenizer_config", "tokenizer", "chat_template"):
            (root / f"{name}.json").write_text("{}")
        save_file({"weight": np.ones(1, dtype=np.float32)}, str(root / "model.safetensors"))
        return
    subdir = root / ("vae" if component == "vae" else "transformer")
    subdir.mkdir(exist_ok=True)
    (subdir / "config.json").write_text("{}")
    save_file({"weight": np.ones(1, dtype=np.float32)}, str(subdir / "diffusion_pytorch_model.safetensors"))
    index = subdir / "diffusion_pytorch_model.safetensors.index.json"
    if index.exists():
        index.write_text(json.dumps({"weight_map": {"weight": "diffusion_pytorch_model.safetensors"}}))
    if component == "generation":
        (root / "vae").mkdir(exist_ok=True)
        (root / "vae/config.json").write_text("{}")


@pytest.fixture
def downloads(monkeypatch):
    calls = []

    def download(model, component, target, cache, *, force):
        calls.append((model, component, target, force))
        _write_model(target, component)

    monkeypatch.setattr(cache_module, "_download_model", download)
    return calls


def _resolve(tmp_path, **kwargs):
    return cache_module.resolve_model_paths(
        str(tmp_path / "blob/Qwen3-VL-4B-Instruct"),
        str(tmp_path / "blob/Cosmos3-Edge"),
        str(tmp_path / "blob/Cosmos3-Nano"),
        "nano", tmp_path / "local",
        **kwargs,
    )


def test_complete_mounted_models_need_no_network_or_cache(tmp_path, downloads, monkeypatch):
    _write_model(tmp_path / "blob/Qwen3-VL-4B-Instruct", "qwen")
    _write_model(tmp_path / "blob/Cosmos3-Edge", "vae")
    _write_model(tmp_path / "blob/Cosmos3-Nano", "generation")
    monkeypatch.setattr(
        cache_module, "_require_local_disk",
        lambda _path: pytest.fail("Complete mounted models must not access the download cache"),
    )
    paths = _resolve(tmp_path, need_generation=True)
    assert paths[2] == str(tmp_path / "blob/Cosmos3-Nano")
    assert downloads == []
    assert not (tmp_path / "local").exists()


def test_native_cosmos3_cache_is_distinct_from_nvidia_generation_and_reused_offline(tmp_path, downloads):
    source = tmp_path / "mounted/Cosmos3-Nano"
    _write_model(source, "generation")
    cache = tmp_path / "local"
    _write_model(cache / "Cosmos3-Nano", "generation")
    resolved = cache_module.resolve_cosmos3_policy_path(str(source), cache)
    assert resolved == str(cache / "Cosmos3-Nano-LeRobot")
    assert [call[:2] for call in downloads] == [("Cosmos3-Nano-LeRobot", "cosmos3_policy")]
    assert not (source / "config.json").exists()
    assert cache_module.resolve_cosmos3_policy_path(str(source), cache, auto_download=False) == resolved
    assert len(downloads) == 1
    with pytest.raises(FileNotFoundError, match="prohibits downloading"):
        cache_module.resolve_cosmos3_policy_path(str(source), tmp_path / "offline", auto_download=False)
    assert not (tmp_path / "offline").exists()


def test_complete_native_cosmos3_mount_needs_no_network_or_cache(tmp_path, downloads, monkeypatch):
    source = tmp_path / "mounted"
    _write_model(source, "cosmos3_policy")
    monkeypatch.setattr(cache_module, "_require_local_disk", lambda _: pytest.fail("No cache needed"))
    assert cache_module.resolve_cosmos3_policy_path(str(source), tmp_path / "local") == str(source)
    assert not downloads
    assert not (tmp_path / "local").exists()


@pytest.mark.parametrize(
    "missing", ["tokenizer/chat_template.jinja", "model.safetensors", "normalizer.safetensors"]
)
def test_native_cosmos3_assets_must_be_complete(tmp_path, missing):
    _write_model(tmp_path, "cosmos3_policy")
    if missing == "normalizer.safetensors":
        (tmp_path / "policy_preprocessor.json").write_text(
            json.dumps(
                {
                    "steps": [{"registry_name": "normalizer_processor", "state_file": missing}],
                }
            )
        )
    else:
        (tmp_path / missing).unlink()
    with pytest.raises(FileNotFoundError):
        cache_module.check_model_directory(tmp_path, "cosmos3_policy")


def test_evaluation_defaults_download_only_base_qwen_vae_and_text_then_reuse(tmp_path, downloads):
    cache = tmp_path / "local"
    paths = cache_module.resolve_evaluation_model_paths(None, None, None, cache)
    assert [call[:2] for call in downloads] == [
        ("Qwen3-VL-4B-Instruct", "qwen"),
        ("Cosmos3-Edge", "vae"),
        ("siglip2-base-patch16-224", "text"),
    ]
    assert all(Path(path).is_relative_to(cache) for path in paths)
    assert cache_module.resolve_evaluation_model_paths(None, None, None, cache) == paths
    assert len(downloads) == 3
    assert not (cache / "Cosmos3-Edge/transformer").exists()


def test_evaluation_explicit_assets_are_strict_and_need_no_download_cache(tmp_path, downloads, monkeypatch):
    paths = [tmp_path / name for name in ("qwen", "vae", "text")]
    for path, component in zip(paths, ("qwen", "vae", "text"), strict=True):
        _write_model(path, component)
    monkeypatch.setattr(
        cache_module,
        "_require_local_disk",
        lambda _: pytest.fail("Explicit complete assets must not access a download cache"),
    )
    assert cache_module.resolve_evaluation_model_paths(*paths, tmp_path / "cache") == tuple(map(str, paths))
    assert not downloads
    (paths[0] / "config.json").unlink()
    with pytest.raises(FileNotFoundError):
        cache_module.resolve_evaluation_model_paths(*paths, tmp_path / "cache")
    assert not downloads


def test_evaluation_offline_requires_complete_cache_without_network(tmp_path, downloads):
    with pytest.raises(FileNotFoundError, match="prohibits downloading"):
        cache_module.resolve_evaluation_model_paths(None, None, None, tmp_path / "cache", auto_download=False)
    assert not downloads


def test_siglip_requires_tokenizer_and_complete_safetensors(tmp_path):
    _write_model(tmp_path, "text")
    cache_module.check_model_directory(tmp_path, "text")
    (tmp_path / "tokenizer.json").unlink()
    with pytest.raises(FileNotFoundError, match="tokenizer"):
        cache_module.check_model_directory(tmp_path, "text")
    (tmp_path / "tokenizer.model").write_bytes(b"tokenizer")
    cache_module.check_model_directory(tmp_path, "text")
    (tmp_path / "model.safetensors").write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="incomplete"):
        cache_module.check_model_directory(tmp_path, "text")


def test_missing_models_download_only_to_node_local_disk_and_are_reused(tmp_path, downloads):
    paths = _resolve(tmp_path, need_generation=True)
    assert [call[:2] for call in downloads] == [
        ("Qwen3-VL-4B-Instruct", "qwen"), ("Cosmos3-Edge", "vae"), ("Cosmos3-Nano", "generation"),
    ]
    assert all(Path(path).is_relative_to(tmp_path / "local") for path in paths)
    assert not (tmp_path / "blob").exists()
    assert _resolve(tmp_path, need_generation=True) == paths
    assert len(downloads) == 3


@pytest.mark.parametrize("backbone", ["", "random", "edge", "nano"])
def test_resume_never_downloads_original_generation_weights(tmp_path, downloads, backbone):
    paths = cache_module.resolve_model_paths(
        str(tmp_path / "qwen"), str(tmp_path / "vae"), "/missing/original/generator",
        backbone, tmp_path / "local", need_generation=False,
    )
    assert paths[2] == "/missing/original/generator"
    assert [call[1] for call in downloads] == ["qwen", "vae"]


def test_fresh_random_generation_does_not_download_cosmos_transformer(tmp_path, downloads):
    cache_module.resolve_model_paths(
        str(tmp_path / "qwen"), str(tmp_path / "vae"), "",
        "random", tmp_path / "local", need_generation=True,
    )
    assert [call[1] for call in downloads] == ["qwen", "vae"]


def test_missing_shard_in_mounted_model_falls_back_without_modifying_it(tmp_path, downloads):
    mounted = tmp_path / "blob/Qwen3-VL-4B-Instruct"
    _write_model(mounted, "qwen")
    original = (mounted / "model.safetensors").read_bytes()
    (mounted / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"weight": "model-00001.safetensors", "missing": "model-00002.safetensors"},
    }))
    save_file({"weight": np.ones(1)}, str(mounted / "model-00001.safetensors"))
    cache_module._resolve_model(mounted.as_posix(), "Qwen3-VL-4B-Instruct", "qwen", tmp_path / "local",
                                auto_download=True)
    assert downloads[0][1] == "qwen"
    assert (mounted / "model.safetensors").read_bytes() == original
    assert not (mounted / "model-00002.safetensors").exists()


@pytest.mark.parametrize("failure", ["truncated", "bad_config", "bad_index"])
def test_corrupted_local_cache_is_explicitly_repaired(tmp_path, downloads, failure):
    target = tmp_path / "local/Cosmos3-Nano"
    _write_model(target, "generation")
    if failure == "truncated":
        path = target / "transformer/diffusion_pytorch_model.safetensors"
        path.write_bytes(path.read_bytes()[:-1])
    elif failure == "bad_config":
        (target / "transformer/config.json").write_text("{broken")
    else:
        (target / "transformer/diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"weight": "../outside.safetensors"},
        }))

    cache_module._resolve_model("/missing", "Cosmos3-Nano", "generation", tmp_path / "local",
                                auto_download=True)
    assert downloads[-1][3] is True


def test_disabled_download_fails_without_touching_local_or_blob(tmp_path, downloads):
    with pytest.raises(FileNotFoundError, match="MODEL_AUTO_DOWNLOAD=false"):
        _resolve(tmp_path, need_generation=True, auto_download=False)
    assert not (tmp_path / "local").exists()
    assert not (tmp_path / "blob").exists()
    assert downloads == []


def test_disabled_download_can_reuse_complete_node_local_cache(tmp_path, downloads):
    paths = _resolve(tmp_path, need_generation=True)
    assert _resolve(tmp_path, need_generation=True, auto_download=False) == paths
    assert len(downloads) == 3


@pytest.mark.parametrize("filesystem", ["fuse.blobfuse2", "nfs4", "cifs", "tmpfs"])
def test_blob_network_and_ram_cache_roots_are_rejected(tmp_path, downloads, monkeypatch, filesystem):
    monkeypatch.setattr(cache_module, "_mount_type", lambda path: filesystem)
    with pytest.raises(ValueError, match="must use node-local disk"):
        _resolve(tmp_path, need_generation=True)
    assert downloads == []
    assert not (tmp_path / "local").exists()


def test_symlink_to_blob_cache_root_is_rejected(tmp_path, downloads, monkeypatch):
    blob = tmp_path / "blob"
    blob.mkdir()
    (tmp_path / "local").symlink_to(blob, target_is_directory=True)
    monkeypatch.setattr(
        cache_module, "_mount_type",
        lambda path: "fuse.blobfuse2" if path.resolve().is_relative_to(blob) else "ext4",
    )
    with pytest.raises(ValueError, match="must use node-local disk"):
        _resolve(tmp_path, need_generation=True)
    assert list(blob.iterdir()) == []
    assert downloads == []


def test_concurrent_launches_download_each_model_only_once(tmp_path, downloads, monkeypatch):
    def download(model, component, target, cache, *, force):
        time.sleep(0.03)
        downloads.append((model, component, target, force))
        _write_model(target, component)

    monkeypatch.setattr(cache_module, "_download_model", download)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_resolve, tmp_path, need_generation=True) for _ in range(2)]
        assert futures[0].result() == futures[1].result()
    assert len(downloads) == 3


def test_download_errors_are_not_hidden_and_no_blob_files_are_written(tmp_path, monkeypatch):
    def download(*args, **kwargs):
        raise ConnectionError("download unavailable")

    monkeypatch.setattr(cache_module, "_download_model", download)
    with pytest.raises(ConnectionError, match="download unavailable"):
        _resolve(tmp_path, need_generation=True)
    assert not (tmp_path / "blob").exists()


def test_incomplete_download_is_rejected_before_training(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "_download_model", lambda *args, **kwargs: None)
    with pytest.raises(FileNotFoundError):
        _resolve(tmp_path, need_generation=True)
    assert not (tmp_path / "blob").exists()


@pytest.mark.parametrize(
    "model,component,repo",
    [
        ("Cosmos3-Nano", "generation", "nvidia/Cosmos3-Nano"),
        ("Cosmos3-Nano-LeRobot", "cosmos3_policy", "lerobot/Cosmos3-Nano"),
    ],
)
def test_official_download_has_pinned_revision_and_node_local_hub_xet_paths(
    tmp_path, monkeypatch, model, component, repo
):
    import huggingface_hub

    files = [
        SimpleNamespace(rfilename="transformer/config.json", size=2),
        SimpleNamespace(rfilename="vae/config.json", size=2),
        SimpleNamespace(rfilename="audio/model.safetensors", size=100),
        SimpleNamespace(rfilename="model.safetensors", size=10),
        SimpleNamespace(rfilename="tokenizer/tokenizer.json", size=2),
    ]
    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda: SimpleNamespace(model_info=lambda *args, **kwargs: SimpleNamespace(siblings=files)),
    )
    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setenv("HF_HOME", "/mounted/blob/hf")
    monkeypatch.setenv("HF_XET_CACHE", "/mounted/blob/xet")
    cache_module._download_model(model, component, tmp_path / model, tmp_path, force=False)
    assert calls[0]["repo_id"] == repo
    assert calls[0]["revision"] == cache_module._MODELS[model][1]
    assert calls[0]["local_dir"] == str(tmp_path / model)
    assert calls[0]["cache_dir"] == str(tmp_path / "_hub")
    assert calls[0]["allow_patterns"] == cache_module._PATTERNS[component]
    assert os.environ["HF_XET_CACHE"] == str(tmp_path / "_xet")


def test_download_checks_disk_space_before_fetching_weights(tmp_path, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(
        model_info=lambda *args, **kwargs: SimpleNamespace(siblings=[
            SimpleNamespace(rfilename="vae/config.json", size=100),
        ]),
    ))
    monkeypatch.setattr(cache_module.shutil, "disk_usage", lambda path: SimpleNamespace(free=1))
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda **kwargs: pytest.fail("Insufficient space must stop"),
    )
    with pytest.raises(OSError, match="Insufficient node-local disk"):
        cache_module._download_model("Cosmos3-Edge", "vae", tmp_path / "model", tmp_path, force=False)


def test_complete_index_checks_all_shards_and_tensor_names(tmp_path):
    _write_model(tmp_path, "qwen")
    save_file({"second": np.ones(1)}, str(tmp_path / "second.safetensors"))
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {
        "weight": "model.safetensors", "second": "second.safetensors",
    }}))
    cache_module.check_model_directory(tmp_path, "qwen")
    index.write_text(json.dumps({"weight_map": {"not_in_header": "second.safetensors"}}))
    with pytest.raises(ValueError, match="missing indexed tensors"):
        cache_module.check_model_directory(tmp_path, "qwen")


@pytest.mark.parametrize("filename", ["../outside.safetensors", "/outside.safetensors", ["invalid"]])
def test_invalid_index_shard_paths_are_rejected(tmp_path, filename):
    _write_model(tmp_path, "qwen")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"weight": filename},
    }))
    with pytest.raises(ValueError, match="Invalid tensor name or shard filename"):
        cache_module.check_model_directory(tmp_path, "qwen")


@pytest.mark.parametrize("component", ["qwen", "vae"])
def test_missing_processor_or_vae_weights_trigger_local_fallback(tmp_path, downloads, component):
    mounted = tmp_path / "mounted"
    _write_model(mounted, component)
    if component == "qwen":
        (mounted / "preprocessor_config.json").unlink()
        model = "Qwen3-VL-4B-Instruct"
    else:
        (mounted / "vae/diffusion_pytorch_model.safetensors").unlink()
        model = "Cosmos3-Edge"
    resolved = cache_module._resolve_model(
        str(mounted), model, component, tmp_path / "local", auto_download=True,
    )
    assert resolved == str(tmp_path / "local" / model)
    assert downloads[0][1] == component


def test_edge_vae_and_generation_share_one_cache_directory(tmp_path, downloads):
    paths = cache_module.resolve_model_paths(
        str(tmp_path / "qwen"), str(tmp_path / "blob/Cosmos3-Edge"), str(tmp_path / "blob/Cosmos3-Edge"),
        "edge", tmp_path / "local", need_generation=True,
    )
    assert paths[1] == paths[2] == str(tmp_path / "local/Cosmos3-Edge")
    cache_module.check_model_directory(Path(paths[1]), "vae")
    cache_module.check_model_directory(Path(paths[2]), "generation")


def test_cli_emits_only_resolved_paths_on_stdout(tmp_path):
    qwen, cosmos = tmp_path / "qwen", tmp_path / "cosmos"
    _write_model(qwen, "qwen")
    _write_model(cosmos, "vae")
    result = subprocess.run(
        [sys.executable, "-m", "lerobot.common.utils.stage2_model_cache",
         f"--qwen-dir={qwen}", f"--cosmos-dir={cosmos}",
         f"--cache-dir={tmp_path / 'cache'}", "--auto-download=false"],
        text=True, capture_output=True, check=True, timeout=30,
    )
    assert result.stdout.splitlines() == [str(qwen), str(cosmos), ""]
    assert "Using existing" in result.stderr
    assert not (tmp_path / "cache").exists()


def test_cli_resolves_native_cosmos3_without_mot_assets(tmp_path):
    source = tmp_path / "native"
    _write_model(source, "cosmos3_policy")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lerobot.common.utils.stage2_model_cache",
            f"--cosmos3-policy-dir={source}",
            f"--cache-dir={tmp_path / 'cache'}",
            "--auto-download=false",
        ],
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    assert result.stdout.splitlines() == [str(source)]
    assert not (tmp_path / "cache").exists()


def _cosmos3_launcher_env(tmp_path, **overrides):
    env = dict(os.environ)
    for name in (
        "WEIGHTS_ROOT",
        "MODEL_DIR",
        "PARENT_DIR_V21",
        "PARENT_DIR_V30",
        "PARENT_DIR_EXTRA",
        "DATA_ROOT_V21",
        "DATA_ROOT_V30",
        "DATA_ROOT_EXTRA",
        "OUTPUT_ROOT",
        "OUTPUT_DIR",
        "LOG_DIR",
        "JOB_NAME",
        "LOAD_ACTION_MODULES",
        "FFMPEG_LIB_DIR",
    ):
        env.pop(name, None)
    env.update(
        {
            "PYTHON_BIN": sys.executable,
            "DRY_RUN": "true",
            "CHECK_PATHS": "false",
            "MODEL_AUTO_DOWNLOAD": "true",
            "MODEL_LOCAL_CACHE": str(tmp_path / "cache"),
            "WEIGHT_RESUME": "false",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "NNODES": "1",
            "NODE_RANK": "0",
            "NPROC_PER_NODE": "8",
            "MASTER_ADDR": "10.0.0.1",
            "MASTER_PORT": "29599",
            "DISTRIBUTED_TIMEOUT_MINUTES": "17",
        }
    )
    env.update(overrides)
    return env


def _cosmos3_launcher(env):
    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        ["bash", str(root / "train_cosmos3_fsdp.sh")],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )


@pytest.mark.parametrize("resume", ["false", "true"])
def test_cosmos3_cluster_defaults_use_base_weights_and_resume_action_modules(tmp_path, resume):
    result = _cosmos3_launcher(
        _cosmos3_launcher_env(
            tmp_path,
            NNODES="2",
            NODE_RANK="1",
            WEIGHT_RESUME=resume,
        )
    )
    assert result.returncode == 0, result.stderr
    for argument in (
        "--policy.pretrained_path=/mnt/wangxiaofa/pt_weights/Cosmos3-Nano",
        f"--policy.load_action_modules={resume}",
        "--dataset.parent_dir_v21=/mnt/wangxiaofa/robot_dataset/lerobot-format-v30-0710/",
        "--dataset.parent_dir_v30=/mnt/wangxiaofa/robot_dataset/lerobot-format-v30-0710/",
        "--dataset.parent_dir_extra=/mnt/wangxiaofa/robot_dataset/lerobot-format-v30/",
        "--output_dir=/mnt/wangxiaofa/qwen3vl_mot_exp/cosmos3_nano_fsdp",
        "--log_dir=/mnt/wangxiaofa/ace_logs",
        "--job_name=cosmos3_nano_fsdp",
        "--nnodes=2",
        "--node_rank=1",
        "--master_addr=10.0.0.1",
        "--master_port=29599",
        "--rdzv_conf=timeout=1020",
    ):
        assert argument in result.stdout
    assert "Policy-LIBERO" not in result.stdout
    assert not (tmp_path / "cache").exists()


def test_cosmos3_launcher_respects_dataset_aliases_extra_disable_and_output_overrides(tmp_path):
    import shlex

    env = _cosmos3_launcher_env(
        tmp_path,
        PARENT_DIR_V21="/primary data",
        DATA_ROOT_V21="/ignored",
        DATA_ROOT_V30="/legacy data",
        PARENT_DIR_EXTRA="",
        DATA_ROOT_EXTRA="/ignored",
        OUTPUT_ROOT=str(tmp_path / "outputs"),
        JOB_NAME="base_test",
        LOG_DIR=str(tmp_path / "logs"),
    )
    result = _cosmos3_launcher(env)
    assert result.returncode == 0, result.stderr
    args = shlex.split(result.stdout.split("command:", 1)[1])
    assert "--dataset.parent_dir_v21=/primary data" in args
    assert "--dataset.parent_dir_v30=/legacy data" in args
    assert "--dataset.parent_dir_extra=" in args
    assert f"--output_dir={tmp_path / 'outputs/base_test'}" in args
    assert f"--log_dir={tmp_path / 'logs'}" in args
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"NNODES": "2", "MASTER_PORT": ""}, "required for multi-node"),
        ({"NNODES": "2", "MASTER_ADDR": "127.0.0.1"}, "reachable from other nodes"),
        ({"NNODES": "2", "NODE_RANK": "2"}, "smaller than NNODES"),
        ({"WEIGHT_RESUME": "true", "LOAD_ACTION_MODULES": "false"}, "reload trained action modules"),
    ],
)
def test_cosmos3_launcher_rejects_bad_rendezvous_and_action_reset(tmp_path, changes, message):
    result = _cosmos3_launcher(_cosmos3_launcher_env(tmp_path, **changes))
    assert result.returncode != 0
    assert message in result.stderr
    assert not (tmp_path / "cache").exists()


def test_cosmos3_launcher_resolves_native_cache_before_workers(tmp_path):
    python = tmp_path / "python"
    calls = tmp_path / "calls.jsonl"
    python.write_text(
        f"#!{sys.executable}\nimport json, sys\n"
        f"with open({str(calls)!r}, 'a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1:3] == ['-m', 'lerobot.common.utils.stage2_model_cache']:\n"
        "    print('/node/Cosmos3-Nano-LeRobot')\n"
        "elif sys.argv[1:3] != ['-m', 'torch.distributed.run']:\n"
        "    raise SystemExit('unexpected invocation')\n"
    )
    python.chmod(0o755)
    result = _cosmos3_launcher(
        _cosmos3_launcher_env(
            tmp_path,
            PYTHON_BIN=str(python),
            DRY_RUN="false",
            MODEL_DIR="/mounted/Cosmos3-Nano",
        )
    )
    assert result.returncode == 0, result.stderr
    helper, workers = [json.loads(line) for line in calls.read_text().splitlines()]
    assert helper[:2] == ["-m", "lerobot.common.utils.stage2_model_cache"]
    assert "--cosmos3-policy-dir=/mounted/Cosmos3-Nano" in helper
    assert "--auto-download=true" in helper
    assert workers[:2] == ["-m", "torch.distributed.run"]
    assert "--policy.pretrained_path=/node/Cosmos3-Nano-LeRobot" in workers
    assert "--policy.load_action_modules=false" in workers


@pytest.mark.parametrize("launcher", ["fsdp", "local"])
@pytest.mark.parametrize("resume", [False, True])
def test_launchers_resolve_models_before_workers_and_skip_generator_on_resume(
    tmp_path, launcher, resume, fsdp_checkpoint_factory
):
    root = Path(__file__).resolve().parents[1]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    helper_args = tmp_path / "helper_args.json"
    python = bin_dir / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if 'lerobot.common.utils.fsdp_training' in sys.argv:\n"
        f"    os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n"
        "if 'lerobot.common.utils.stage2_model_cache' in sys.argv:\n"
        "    with open(os.environ['HELPER_ARGS'], 'w') as f: json.dump(sys.argv[1:], f)\n"
        "    print('/node/qwen')\n"
        "    print('/node/vae')\n"
        "    print('/node/nano' if '--need-generation' in sys.argv else '')\n"
    )
    python.chmod(0o755)
    for name in ("torchrun", "deepspeed"):
        worker = bin_dir / name
        worker.write_text(
            f"#!{sys.executable}\nimport json, sys\n"
            f"print(json.dumps(sys.argv[1:]))\n"
        )
        worker.chmod(0o755)
    run_dir = tmp_path / "run"
    if resume:
        fsdp_checkpoint_factory(run_dir)
    env = {
        **os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "PYTHON_BIN": str(python), "TORCHRUN_BIN": str(bin_dir / "torchrun"),
        "HELPER_ARGS": str(helper_args), "CHECK_PATHS": "false", "DRY_RUN": "false",
        "MODEL_AUTO_DOWNLOAD": "true", "MODEL_LOCAL_CACHE": str(tmp_path / "cache"),
        "WEIGHT_RESUME": str(resume).lower(), "OUTPUT_DIR": str(run_dir),
        "STAGE1_CHECKPOINT": "/mounted/stage1", "GENERATION_BACKBONE": "" if resume else "nano",
        "GENERATION_PRETRAINED_DIR": "", "GENERATION_TUNING_MODE": "",
        "QWEN3VL_DIR": "/mounted/qwen", "COSMOS3_DIR": "/mounted/cosmos",
        "MASTER_PORT": "29599", "NNODES": "1", "NPROC_PER_NODE": "1",
        "TRAINING_STAGE": "finetune",
    }
    result = subprocess.run(
        ["bash", str(root / f"train_qwen3vl_mot_{launcher}.sh")],
        cwd=root, env=env, text=True, capture_output=True, check=True, timeout=30,
    )
    args = json.loads(helper_args.read_text())
    assert ("--need-generation" in args) is not resume
    assert f"--cache-dir={tmp_path / 'cache'}" in args
    assert "--policy.qwen3vl_dir=/node/qwen" in result.stdout
    assert "--policy.cosmos3_dir=/node/vae" in result.stdout
    assert "--stage=finetune" in result.stdout
    if resume:
        assert "--policy.generation_pretrained_dir=" not in result.stdout
        assert "--policy.generation_backbone=" not in result.stdout
    else:
        assert "--policy.generation_pretrained_dir=/node/nano" in result.stdout
        assert "--policy.generation_backbone=nano" in result.stdout


@pytest.mark.parametrize("nodes", [1, 2])
def test_fsdp_dry_run_does_not_download_or_create_cache(tmp_path, nodes):
    root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ, "PYTHON_BIN": sys.executable, "DRY_RUN": "true", "CHECK_PATHS": "false",
        "MODEL_AUTO_DOWNLOAD": "true", "MODEL_LOCAL_CACHE": str(tmp_path / "cache"),
        "WEIGHT_RESUME": "false", "OUTPUT_DIR": str(tmp_path / "run"), "STAGE1_CHECKPOINT": "/missing",
        "QWEN3VL_DIR": "/missing/qwen", "COSMOS3_DIR": "/missing/vae", "GENERATION_BACKBONE": "nano",
        "GENERATION_PRETRAINED_DIR": "/missing/generator", "GENERATION_TUNING_MODE": "",
        "MASTER_PORT": "29599", "MASTER_ADDR": "10.0.0.1", "NNODES": str(nodes), "NPROC_PER_NODE": "1",
        "NODE_RANK": "0", "DISTRIBUTED_TIMEOUT_MINUTES": "17",
    }
    result = subprocess.run(
        ["bash", str(root / "train_qwen3vl_mot_fsdp.sh")],
        cwd=root, env=env, text=True, capture_output=True, check=True, timeout=30,
    )
    assert "--policy.qwen3vl_dir=/missing/qwen" in result.stdout
    if nodes > 1:
        assert "--rdzv_conf=timeout=1020" in result.stdout
    assert not (tmp_path / "cache").exists()
