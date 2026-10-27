import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import TensorDataset

from lerobot.scripts import train_pi05_robodojo_fsdp as training

SCRIPT = Path(__file__).resolve().parents[1] / "train_pi05_robodojo_fsdp.sh"


@pytest.fixture
def launch_env(tmp_path):
    data = tmp_path / "dataset with spaces"
    (data / "meta").mkdir(parents=True)
    for name in ("info.json", "stats.json"):
        (data / "meta" / name).write_text("{}")
    base = tmp_path / "base with spaces"
    base.mkdir()
    (base / "model.safetensors").touch()
    (base / "paligemma_tokenizer.model").touch()
    return {
        **os.environ,
        "PYTHON_BIN": sys.executable,
        "DATASET_ROOT": str(data),
        "PRETRAINED_PATH": str(base),
        "TOKENIZER_PATH": str(base / "paligemma_tokenizer.model"),
        "OUTPUT_DIR": str(tmp_path / "new run"),
        "LOG_DIR": str(tmp_path / "logs"),
        "NNODES": "1",
        "NODE_RANK": "0",
        "NPROC_PER_NODE": "2",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29517",
        "STEPS": "2",
        "BATCH_SIZE": "1",
        "GRAD_ACCUMULATION_STEPS": "1",
        "FSDP_WRAP_STRATEGY": "layer",
        "MASTER_DTYPE": "float32",
        "OPTIMIZER": "adamw",
        "ATTENTION_BACKEND": "eager",
        "CHUNK_SIZE": "32",
        "NUM_WORKERS": "0",
        "SAVE_FREQ": "1",
        "LR": "2.5e-5",
        "WARMUP_STEPS": "1",
        "SEED": "0",
        "VIDEO_BACKEND": "pyav",
        "WEIGHT_RESUME": "false",
        "EXPORT_FINAL": "true",
        "MAX_UPDATES_THIS_RUN": "",
        "CHECK_PATHS": "true",
        "DRY_RUN": "true",
        "MODEL_AUTO_DOWNLOAD": "false",
        "MODEL_LOCAL_CACHE": str(tmp_path / "node cache"),
    }


def launch(env):
    return subprocess.run(["bash", str(SCRIPT)], env=env, text=True, capture_output=True, check=False)


def test_launcher_dry_run_quotes_paths_and_has_no_side_effects(launch_env):
    result = launch(launch_env)
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout.split("Command: ", 1)[1])
    assert command[command.index("--dataset-root") + 1] == launch_env["DATASET_ROOT"]
    assert command[command.index("--tokenizer-path") + 1] == launch_env["TOKENIZER_PATH"]
    assert command[command.index("--nproc-per-node") + 1] == "2"
    assert "--export-final" in command
    assert command[command.index("--fsdp-wrap-strategy") + 1] == "layer"
    assert not Path(launch_env["OUTPUT_DIR"]).exists()
    assert not Path(launch_env["LOG_DIR"]).exists()


def test_launcher_multinode_resume_and_accumulation(launch_env):
    output = Path(launch_env["OUTPUT_DIR"])
    output.mkdir()
    (output / "latest_checkpoint").write_text("checkpoint_00000001")
    result = launch(
        {
            **launch_env,
            "NNODES": "2",
            "NODE_RANK": "1",
            "MASTER_ADDR": "training-node-0",
            "BATCH_SIZE": "2",
            "GRAD_ACCUMULATION_STEPS": "4",
            "WEIGHT_RESUME": "true",
            "MAX_UPDATES_THIS_RUN": "1",
            "FSDP_WRAP_STRATEGY": "root",
        }
    )
    assert result.returncode == 0, result.stderr
    assert "Effective batch size: 32" in result.stdout
    command = shlex.split(result.stdout.split("Command: ", 1)[1])
    assert "--resume" in command
    assert command[command.index("--node-rank") + 1] == "1"
    assert command[command.index("--max-updates-this-run") + 1] == "1"
    assert command[command.index("--fsdp-wrap-strategy") + 1] == "root"


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("STEPS", "0", "positive integer"),
        ("NODE_RANK", "1", "smaller than NNODES"),
        ("WEIGHT_RESUME", "yes", "true or false"),
        ("LR", "nan", "finite and positive"),
        ("PRETRAINED_PATH", "/definitely/missing/pi05", "Missing base weights"),
        ("WEIGHT_RESUME", "true", "No latest_checkpoint"),
        ("NNODES", "2", "reachable from other nodes"),
        ("FSDP_WRAP_STRATEGY", "decoder", "must be layer or root"),
        ("MASTER_DTYPE", "float16", "must be float32 or bfloat16"),
        ("OPTIMIZER", "sgd", "must be adamw or adamw8bit"),
        ("ATTENTION_BACKEND", "flash", "must be eager or sdpa"),
    ],
)
def test_launcher_rejects_invalid_settings(launch_env, key, value, message):
    result = launch({**launch_env, key: value})
    assert result.returncode != 0
    assert message in result.stderr


def test_launcher_propagates_training_failure_through_tee(launch_env, tmp_path):
    executable = tmp_path / "failed-python"
    executable.write_text(
        '#!/usr/bin/env bash\nif [[ "$1" == "-m" ]]; then echo "training failed" >&2; exit 23; fi\n'
    )
    executable.chmod(0o755)
    result = launch(
        {
            **launch_env,
            "PYTHON_BIN": str(executable),
            "DRY_RUN": "false",
            "CHECK_PATHS": "false",
        }
    )
    assert result.returncode == 23
    assert "training failed" in result.stdout
    assert len(list(Path(launch_env["LOG_DIR"]).glob("*.log"))) == 1


def test_distributed_sampler_and_resumed_epoch_order():
    dataset = TensorDataset(torch.arange(16))
    args = SimpleNamespace(seed=7, batch_size=2, num_workers=0)
    loader, sampler, generator = training.make_loader(dataset, args, 0, 2)
    full = list(training.epoch_iterator(loader, sampler, generator, 7, 0, 0))
    resumed = list(training.epoch_iterator(loader, sampler, generator, 7, 0, 2))
    assert len(full) == 4 and len(resumed) == 2
    for expected, actual in zip(full[2:], resumed, strict=True):
        torch.testing.assert_close(expected[0], actual[0])
    other_loader, other_sampler, other_generator = training.make_loader(dataset, args, 1, 2)
    other = list(training.epoch_iterator(other_loader, other_sampler, other_generator, 8, 0, 0))
    assert set(torch.cat([row[0] for row in full]).tolist()).isdisjoint(
        torch.cat([row[0] for row in other]).tolist()
    )
    with pytest.raises(ValueError, match="batch position"):
        training.epoch_iterator(loader, sampler, generator, 7, 0, 5)


def test_resume_rejects_changed_geometry(tmp_path):
    checkpoint = tmp_path / "checkpoint_00000001"
    checkpoint.mkdir()
    geometry = {"batch_size": 1, "steps": 2}
    (tmp_path / "latest_checkpoint").write_text(checkpoint.name)
    (checkpoint / "metadata.json").write_text(json.dumps({"training_geometry": geometry}))
    training.check_resume_geometry(tmp_path, geometry)
    with pytest.raises(ValueError, match="unchanged"):
        training.check_resume_geometry(tmp_path, {**geometry, "batch_size": 2})


def test_resume_accepts_identical_tokenizer_at_another_node_local_path(tmp_path):
    checkpoint = tmp_path / "checkpoint_00000001"
    checkpoint.mkdir()
    geometry = {"tokenizer_path": "/scratch/node0/tokenizer.model", "tokenizer_sha256": "same-content"}
    (tmp_path / "latest_checkpoint").write_text(checkpoint.name)
    (checkpoint / "metadata.json").write_text(json.dumps({"training_geometry": geometry}))
    current = {**geometry, "tokenizer_path": "/scratch/node1/tokenizer.model"}
    assert training.check_resume_geometry(tmp_path, current) == geometry
    with pytest.raises(ValueError, match="unchanged"):
        training.check_resume_geometry(tmp_path, {**current, "tokenizer_sha256": "different-content"})


@pytest.mark.parametrize("use_default_cache", [False, True])
def test_launcher_defaults_use_mounts_for_outputs_and_local_disk_for_assets(launch_env, use_default_cache):
    env = dict(launch_env)
    for key in ("OUTPUT_DIR", "OUTPUT_ROOT", "LOG_DIR", "PRETRAINED_PATH", "TOKENIZER_PATH"):
        env.pop(key, None)
    if use_default_cache:
        env.pop("MODEL_LOCAL_CACHE")
    env["TMPDIR"] = "/another/temporary/directory"
    expected_cache = "/scratch/pi05_models" if use_default_cache else env["MODEL_LOCAL_CACHE"]
    env.update(DRY_RUN="true", CHECK_PATHS="false", MODEL_AUTO_DOWNLOAD="true")
    result = launch(env)
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout.split("Command: ", 1)[1])
    assert command[command.index("--output-dir") + 1].startswith("/mnt/wangxiaofa/qwen3vl_mot_exp/")
    assert command[command.index("--pretrained-path") + 1] == f"{expected_cache}/pi05_base"
    assert (
        command[command.index("--tokenizer-path") + 1]
        == f"{expected_cache}/pi05_base/paligemma_tokenizer.model"
    )
    assert not Path(launch_env["MODEL_LOCAL_CACHE"]).exists()


def test_launcher_dry_run_allows_missing_downloadable_assets(launch_env):
    env = {
        **launch_env,
        "MODEL_AUTO_DOWNLOAD": "true",
        "PRETRAINED_PATH": "/missing/mounted/pi05_base",
        "TOKENIZER_PATH": "/missing/mounted/tokenizer.model",
    }
    result = launch(env)
    assert result.returncode == 0, result.stderr
    assert "auto_download=true" in result.stdout
    assert not Path(env["MODEL_LOCAL_CACHE"]).exists()


def test_root_wrapper_preserves_fp32_masters_and_normalization(monkeypatch):
    options = {}

    def fake_fsdp(policy, **kwargs):
        options.update(kwargs)
        return policy

    monkeypatch.setattr(training, "FullyShardedDataParallel", fake_fsdp)
    policy = torch.nn.Linear(2, 2).bfloat16()
    for name in ("normalize_inputs", "normalize_targets", "unnormalize_outputs"):
        normalizer = torch.nn.Linear(1, 1, bias=False).requires_grad_(False)
        setattr(policy, name, normalizer)
    wrapped = training.wrap_pi05(policy, torch.device("cpu"), strategy="root")
    assert wrapped.weight.dtype == torch.float32
    assert options["use_orig_params"] is True
    assert options["mixed_precision"].buffer_dtype is None
    assert options["mixed_precision"].param_dtype == torch.bfloat16
    assert options["mixed_precision"].reduce_dtype == torch.float32
    expected = {
        id(parameter)
        for normalizer in (policy.normalize_inputs, policy.normalize_targets, policy.unnormalize_outputs)
        for parameter in normalizer.parameters()
    }
    assert {id(parameter) for parameter in options["ignored_states"]} == expected
    assert all(parameter.dtype == torch.float32 for parameter in options["ignored_states"])


def test_launcher_memory_modes_and_root_rejection(launch_env):
    env = {
        **launch_env,
        "MASTER_DTYPE": "bfloat16",
        "OPTIMIZER": "adamw8bit",
        "ATTENTION_BACKEND": "sdpa",
        "BATCH_SIZE": "16",
    }
    result = launch(env)
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout.split("Command: ", 1)[1])
    for key, value in (
        ("--master-dtype", "bfloat16"),
        ("--optimizer", "adamw8bit"),
        ("--attention-backend", "sdpa"),
        ("--batch-size", "16"),
    ):
        assert command[command.index(key) + 1] == value
    result = launch({**env, "FSDP_WRAP_STRATEGY": "root"})
    assert result.returncode != 0 and "require FSDP_WRAP_STRATEGY=layer" in result.stderr


@pytest.mark.parametrize(
    ("key", "changed"),
    [("master_dtype", "bfloat16"), ("optimizer", "adamw8bit"), ("attention_backend", "sdpa")],
)
def test_resume_memory_modes_preserve_old_defaults_and_reject_changes(tmp_path, key, changed):
    checkpoint = tmp_path / "checkpoint_00000001"
    checkpoint.mkdir()
    (tmp_path / "latest_checkpoint").write_text(checkpoint.name)
    old = {"batch_size": 1}
    (checkpoint / "metadata.json").write_text(json.dumps({"training_geometry": old}))
    current = {**old, "master_dtype": "float32", "optimizer": "adamw", "attention_backend": "eager"}
    assert training.check_resume_geometry(tmp_path, current) == old
    with pytest.raises(ValueError, match="unchanged"):
        training.check_resume_geometry(tmp_path, {**current, key: changed})


def test_bf16_master_preserves_original_fp32_quantiles(monkeypatch):
    options = {}

    def fake_fsdp(policy, **kwargs):
        options.update(kwargs)
        return policy

    monkeypatch.setattr(training, "FullyShardedDataParallel", fake_fsdp)
    policy = torch.nn.Module()
    policy.model = torch.nn.Linear(2, 2)
    expected = torch.tensor([[0.123456789]], dtype=torch.float32)
    for name in ("normalize_inputs", "normalize_targets", "unnormalize_outputs"):
        normalizer = torch.nn.Linear(1, 1, bias=False).requires_grad_(False)
        normalizer.weight.copy_(expected)
        setattr(policy, name, normalizer)
    training.wrap_pi05(policy, torch.device("cpu"), strategy="root", master_dtype="bfloat16")
    assert policy.model.weight.dtype == torch.bfloat16
    for parameter in options["ignored_states"]:
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
        assert parameter.dtype == torch.float32
    assert options["mixed_precision"].reduce_dtype == torch.bfloat16


def test_8bit_optimizer_uses_pi05_hyperparameters(monkeypatch):
    settings = {}

    class FakeConfig:
        def __init__(self, **kwargs):
            settings.update(kwargs)

        def build(self, parameters):
            return list(parameters)

    monkeypatch.setattr(training, "AdamW8bitConfig", FakeConfig)
    config = training.PI05Config()
    model = torch.nn.Linear(2, 2)
    assert training.make_optimizer(config, model, "adamw8bit") == list(model.parameters())
    assert settings == {
        "lr": 2.5e-5,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.01,
        "grad_clip_norm": 1.0,
    }
