import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.common.utils.fsdp_training import FSDPTrainingConfig, convert_policy_to_fp8
from lerobot.evaluation.eval_qwen3vl_mot_libero import (
    build_parser,
    inference_fp8_config,
    load_libero_initial_states,
    policy_inference_context,
    sampling_fingerprint,
)
from lerobot.evaluation.scripts import verify_libero_qwen3vl as verification

SCRIPT = Path(__file__).resolve().parents[1] / "lerobot/evaluation/scripts/setup_libero_qwen3vl.sh"


def saved_config(tmp_path, **changes):
    config = asdict(FSDPTrainingConfig(fp8=True))
    config.update(changes)
    path = tmp_path / "train_config.json"
    path.write_text(json.dumps({"fsdp": config}))
    return path


def test_default_precision_is_bf16_without_torchao():
    args = build_parser().parse_args(["--checkpoint", "/model"])
    assert args.precision == "bf16"
    assert not args.verify_fp8_kernels and not args.capture_first_plan
    assert inference_fp8_config("bf16", None, torch.device("cpu")) is None


def test_cluster_defaults_need_no_training_machine_asset_or_dataset_paths():
    args = verification.build_parser().parse_args(
        [
            "--libero-root",
            "/local/LIBERO",
            "--checkpoint",
            "/cluster/step40k",
            "--render-gpu-device-id",
            "0",
            "--output-dir",
            "/local/result",
        ]
    )
    assert args.dataset_root is None
    assert (args.qwen3vl_dir, args.cosmos3_dir, args.stage1_text_dir) == (None, None, None)
    assert not args.offline


@pytest.mark.parametrize("suite", tuple(verification.SUITE_DATASETS))
def test_bundled_suite_metadata_is_valid_and_preserves_exact_snapshot(tmp_path, suite):
    args = SimpleNamespace(dataset_root=None, task_suite=suite)
    root, provenance = verification.prepare_dataset_metadata(args, tmp_path)
    snapshot = json.loads(verification.NORMALIZATION_SNAPSHOT.read_text())
    path = root / verification.SUITE_DATASETS[suite]
    assert json.loads((path / "meta/stats.json").read_text()) == snapshot["suites"][suite]["stats"]
    adapter = verification.LiberoAdapter.from_dataset(path)
    assert adapter.fps == 20
    assert provenance["source"] == "bundled_libero_ort6d_v30"
    assert len(provenance["snapshot_sha256"]) == 64
    chosen, explicit = verification.prepare_dataset_metadata(
        SimpleNamespace(dataset_root=root, task_suite=suite), tmp_path
    )
    assert chosen == root and explicit["source"] == "explicit_dataset_root"


def test_explicit_missing_normalization_never_falls_back_to_snapshot(tmp_path):
    args = SimpleNamespace(dataset_root=tmp_path / "missing", task_suite="libero_goal")
    with pytest.raises(FileNotFoundError):
        verification.prepare_dataset_metadata(args, tmp_path)


@pytest.mark.parametrize("offline", [False, True])
def test_verifier_resolves_default_assets_and_persists_download_failure(tmp_path, monkeypatch, offline):
    from lerobot.common.utils import stage2_model_cache

    calls = []

    def unavailable(*paths, **options):
        calls.append((paths, options))
        raise ConnectionError("Hub unavailable")

    monkeypatch.setattr(verification, "check_environment", lambda *args: {})
    monkeypatch.setattr(verification, "checkpoint_paths", lambda *args: None)
    monkeypatch.setattr(stage2_model_cache, "resolve_evaluation_model_paths", unavailable)
    output = tmp_path / "output"
    command = [
        "--libero-root",
        str(tmp_path),
        "--checkpoint",
        "/cluster/step40k",
        "--render-gpu-device-id",
        "0",
        "--precisions",
        "bf16",
        "--output-dir",
        str(output),
        "--model-cache-dir",
        str(tmp_path / "cache"),
    ]
    with pytest.raises(ConnectionError, match="Hub unavailable"):
        verification.main(command + (["--offline"] if offline else []))
    assert calls[0][0][:3] == (None, None, None)
    assert calls[0][1]["auto_download"] is not offline
    report = json.loads((output / "verification.json").read_text())
    assert report["status"] == "failed" and report["normalization"]["source"] == "bundled_libero_ort6d_v30"
    assert not (output / "bf16").exists()


def test_verifier_runs_only_fine_tuned_checkpoint_with_resolved_offline_assets(tmp_path, monkeypatch):
    from lerobot.common.utils import stage2_model_cache

    paths = tuple(str(tmp_path / key) for key in ("qwen", "vae", "text"))
    monkeypatch.setattr(verification, "check_environment", lambda *args: {})
    monkeypatch.setattr(verification, "checkpoint_paths", lambda *args: None)
    monkeypatch.setattr(stage2_model_cache, "resolve_evaluation_model_paths", lambda *args, **kwargs: paths)
    calls = []

    def stop(command, **options):
        calls.append((command, options))
        raise RuntimeError("inspected child command")

    monkeypatch.setattr(verification.subprocess, "run", stop)
    with pytest.raises(RuntimeError, match="inspected child command"):
        verification.main(
            [
                "--libero-root",
                str(tmp_path),
                "--checkpoint",
                "/cluster/step40k",
                "--render-gpu-device-id",
                "0",
                "--precisions",
                "bf16",
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
    command, options = calls[0]
    assert command[command.index("--checkpoint") + 1] == "/cluster/step40k"
    assert command[command.index("--qwen3vl-dir") + 1] == paths[0]
    assert command[command.index("--cosmos3-dir") + 1] == paths[1]
    assert command[command.index("--stage1-text-dir") + 1] == paths[2]
    assert options["env"]["HF_HUB_OFFLINE"] == options["env"]["TRANSFORMERS_OFFLINE"] == "1"


def test_fp8_rejects_missing_recipe_emulation_and_unsupported_hardware(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="saved train_config"):
        inference_fp8_config("fp8", None, torch.device("cpu"))
    path = saved_config(tmp_path, fp8_emulate=True)
    with pytest.raises(ValueError, match="not emulation"):
        inference_fp8_config("fp8", path, torch.device("cuda"))
    path = saved_config(tmp_path, fp8=False)
    with pytest.raises(ValueError, match="native FP8"):
        inference_fp8_config("fp8", path, torch.device("cuda"))
    path.write_text('{"fsdp": {"fp8": true}}')
    with pytest.raises(ValueError, match="explicit FP8 settings"):
        inference_fp8_config("fp8", path, torch.device("cuda"))
    path = saved_config(tmp_path)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (8, 6))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "A6000")
    with pytest.raises(RuntimeError, match="9.0 or newer"):
        inference_fp8_config("fp8", path, torch.device("cuda"))


def test_fp8_uses_saved_recipe_and_rejects_incompatible_torchao(tmp_path, monkeypatch):
    import importlib.metadata

    path = saved_config(tmp_path, fp8_recipe="tensorwise", fp8_scope="generation", fp8_min_features=256)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.15.0")
    config = inference_fp8_config("fp8", path, torch.device("cuda"))
    assert (config.fp8_recipe, config.fp8_scope, config.fp8_min_features) == ("tensorwise", "generation", 256)
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.16.0")
    with pytest.raises(RuntimeError, match="training-compatible"):
        inference_fp8_config("fp8", path, torch.device("cuda"))


@pytest.mark.parametrize("shape", [(2, 128), (1, 7, 128), (1, 1, 128)])
def test_torchao_eval_emulation_preserves_bf16_weights_and_uses_fp8_forward(shape):
    pytest.importorskip("torchao.float8")
    torch.manual_seed(123)
    policy = torch.nn.Module()
    policy.generation = torch.nn.Sequential(torch.nn.Linear(128, 128, dtype=torch.bfloat16))
    policy.requires_grad_(False).eval()
    weights = policy.generation[0].weight.detach().clone()
    inputs = torch.randn(*shape, dtype=torch.bfloat16)
    with policy_inference_context("bf16"):
        reference = policy.generation(inputs)
    names = convert_policy_to_fp8(policy, FSDPTrainingConfig(fp8=True, fp8_emulate=True), torch.device("cpu"))
    policy.eval()
    with torch.inference_mode(), policy_inference_context("fp8"):
        assert not torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        result = policy.generation(inputs)
    assert names == ("generation.0",)
    assert type(policy.generation[0]).__name__ == "Float8Linear"
    assert torch.equal(weights, policy.generation[0].weight)
    assert not policy.training and not policy.generation[0].training
    assert torch.isfinite(result).all() and result.dtype == torch.bfloat16
    assert not torch.equal(reference, result)


def test_initial_states_load_official_numpy_assets_without_global_torch_patch(tmp_path, monkeypatch):
    module = ModuleType("libero.libero")
    module.get_libero_path = lambda _: str(tmp_path)
    monkeypatch.setitem(sys.modules, "libero.libero", module)
    states = [np.arange(4, dtype=np.float64), np.ones(4)]
    torch.save(states, tmp_path / "init.pt")
    suite = SimpleNamespace(get_task=lambda _: SimpleNamespace(problem_folder="", init_states_file="init.pt"))
    result = load_libero_initial_states(suite, 0)
    np.testing.assert_array_equal(result, states)


def test_input_fingerprint_tracks_images_language_and_both_rng_states(monkeypatch):
    cuda_state = torch.tensor([1, 2], dtype=torch.uint8)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda _: cuda_state)
    batch = {"image": torch.zeros(2, dtype=torch.bfloat16), "task": ["open drawer"]}
    first = sampling_fingerprint(batch, torch.device("cuda"))
    assert first == sampling_fingerprint(batch, torch.device("cuda"))
    batch["task"] = ["close drawer"]
    assert first["batch_sha256"] != sampling_fingerprint(batch, torch.device("cuda"))["batch_sha256"]
    batch["task"] = ["open drawer"]
    batch["image"][0] = 1
    assert first["batch_sha256"] != sampling_fingerprint(batch, torch.device("cuda"))["batch_sha256"]
    cuda_state[0] = 3
    assert first["cuda_rng_sha256"] != sampling_fingerprint(batch, torch.device("cuda"))["cuda_rng_sha256"]
    torch.rand(1)
    assert first["cpu_rng_sha256"] != sampling_fingerprint(batch, torch.device("cuda"))["cpu_rng_sha256"]


def comparison_fixture(tmp_path):
    for precision in ("bf16", "fp8"):
        root = tmp_path / precision
        probes = root / "first_plans"
        probes.mkdir(parents=True)
        result = {
            "evaluation_complete": True,
            "episodes_completed": 1,
            "episodes": [
                {
                    "policy_calls": 1,
                    "steps": 10,
                    "actions_path": str(root / "executed.npy"),
                }
            ],
            "settings": {"precision": precision, "fp8_kernel_evidence": {"scaled_mm_calls": 10}},
        }
        (root / "results.json").write_text(json.dumps(result))
        (probes / "task_00_trial_000.json").write_text(
            json.dumps(dict.fromkeys(("batch_sha256", "cpu_rng_sha256", "cuda_rng_sha256"), "a" * 64))
        )
        offset = 0.25 if precision == "fp8" else 0.0
        np.save(root / "executed.npy", np.full((10, 7), offset, dtype=np.float32))
        np.savez(
            probes / "task_00_trial_000.npz",
            normalized_actions=np.full((1, 32, 40), offset, dtype=np.float32),
            action_mask=np.array([[1] * 10 + [0] * 30]),
            native_actions=np.full((32, 7), offset, dtype=np.float32),
        )


def test_comparison_measures_full_first_chunk_and_rejects_mismatched_inputs(tmp_path):
    comparison_fixture(tmp_path)
    report = verification.compare_results(tmp_path, 0)
    assert report["initial_inputs_and_rng_identical"]
    assert report["normalized_valid_action_rmse"] == 0.25
    assert report["native_gripper_disagreements"] == 32
    probe = tmp_path / "fp8/first_plans/task_00_trial_000.json"
    fingerprint = json.loads(probe.read_text())
    fingerprint["batch_sha256"] = "b" * 64
    probe.write_text(json.dumps(fingerprint))
    with pytest.raises(RuntimeError, match="not a controlled comparison"):
        verification.compare_results(tmp_path, 0)


@pytest.mark.parametrize(
    "fault",
    ["incomplete", "no_kernel", "replanning", "nonfinite", "bad_shape", "wrong_prefix", "missing_rng"],
)
def test_comparison_rejects_invalid_runs(tmp_path, fault):
    comparison_fixture(tmp_path)
    result_path = tmp_path / "fp8/results.json"
    result = json.loads(result_path.read_text())
    if fault == "incomplete":
        result["evaluation_complete"] = False
    elif fault == "no_kernel":
        result["settings"]["fp8_kernel_evidence"]["scaled_mm_calls"] = 0
    elif fault == "replanning":
        result["episodes"][0]["policy_calls"] = 2
    elif fault == "wrong_prefix":
        np.save(tmp_path / "fp8/executed.npy", np.ones((10, 7), dtype=np.float32))
    elif fault == "missing_rng":
        (tmp_path / "fp8/first_plans/task_00_trial_000.json").write_text("{}")
    else:
        path = tmp_path / "fp8/first_plans/task_00_trial_000.npz"
        with np.load(path) as data:
            fields = dict(data)
        if fault == "bad_shape":
            fields["normalized_actions"] = fields["normalized_actions"][:, :30]
        else:
            fields["normalized_actions"][0, 0, 0] = np.nan
        np.savez(path, **fields)
    result_path.write_text(json.dumps(result))
    with pytest.raises(RuntimeError):
        verification.compare_results(tmp_path, 0)


def test_verifier_persists_failures_and_never_overwrites(tmp_path, monkeypatch):
    output = tmp_path / "failed"

    def fail(*args):
        raise RuntimeError("EGL unavailable")

    monkeypatch.setattr(verification, "check_environment", fail)
    command = [
        "--libero-root",
        str(tmp_path),
        "--output-dir",
        str(output),
        "--render-gpu-device-id",
        "0",
        "--environment-only",
        "--precisions",
        "bf16",
    ]
    with pytest.raises(RuntimeError, match="EGL unavailable"):
        verification.main(command)
    report = json.loads((output / "verification.json").read_text())
    assert report["status"] == "failed" and not report["native_fp8_model_validated"]
    assert "EGL unavailable" in report["error"]
    with pytest.raises(FileExistsError):
        verification.main(command)


def test_installer_shell_syntax_help_and_explicit_unknown_option_failure():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    result = subprocess.run(["bash", str(SCRIPT), "--help"], capture_output=True, text=True, check=True)
    assert "--verify-only" in result.stdout and "No cluster jobs are submitted" in result.stdout
    result = subprocess.run(["bash", str(SCRIPT), "--unknown"], capture_output=True, text=True)
    assert result.returncode != 0 and "Unknown installer option" in result.stderr


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--torch-index-url"], "Missing value"),
        (["--libero-root", "--verify-only"], "Missing value"),
        (["--env-prefix", "/"], "Unknown installer option"),
        (["--libero-ref", "main"], "full commit SHA"),
    ],
)
def test_installer_invalid_arguments_fail_before_installation(arguments, message):
    result = subprocess.run(["bash", str(SCRIPT), *arguments], capture_output=True, text=True)
    assert result.returncode != 0 and message in result.stderr


@pytest.fixture
def selected_lerobot(tmp_path, monkeypatch):
    prefix = tmp_path / "lerobot"
    python = prefix / "bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['SETUP_CALL_LOG'], 'a') as log:\n"
        "    log.write(json.dumps({'python': sys.argv[0], 'args': sys.argv[1:], "
        "'offline': os.environ.get('HF_HUB_OFFLINE'), "
        "'user_site': os.environ.get('PYTHONNOUSERSITE')}) + '\\n')\n"
    )
    python.chmod(0o755)
    (prefix / "conda-meta").mkdir()
    (prefix / "conda-meta/history").touch()
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "lerobot")
    monkeypatch.setenv("CONDA_PREFIX", str(prefix))
    monkeypatch.setenv("CONDA_EXE", str(tmp_path / "must-not-create-another-environment"))
    calls = tmp_path / "calls.jsonl"
    monkeypatch.setenv("SETUP_CALL_LOG", str(calls))
    root = tmp_path / "LIBERO"
    for directory in ("assets", "bddl_files", "init_files"):
        (root / "libero/libero" / directory).mkdir(parents=True)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    return prefix, root, calls


def test_installer_reuses_extracted_libero_and_keeps_preparation_online(selected_lerobot):
    prefix, root, calls = selected_lerobot
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--verify-only",
            "--libero-root",
            str(root),
            "--",
            "--environment-only",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "revision is unverified" in result.stderr
    records = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["args"] == ["-"]
    assert records[1]["args"][:2] == ["-m", "lerobot.evaluation.scripts.verify_libero_qwen3vl"]
    assert all(record["offline"] is None for record in records)
    assert all(record["user_site"] == "1" for record in records)
    assert all(record["python"] == str(prefix / "bin/python") for record in records)
    assert not (root / ".git").exists()


def test_installer_installs_and_verifies_with_only_existing_lerobot_python(selected_lerobot):
    prefix, root, calls = selected_lerobot
    subprocess.run(["bash", str(SCRIPT), "--libero-root", str(root), "--", "--environment-only"], check=True)
    records = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(records) == 5
    assert records[0]["args"] == ["-"]
    assert records[1]["args"][:3] == ["-m", "pip", "install"]
    assert "torch==2.9.1" in records[1]["args"]
    assert records[2]["args"][:3] == ["-m", "pip", "install"]
    assert all("--no-user" in record["args"] for record in records[1:3])
    assert records[3]["args"] == ["-m", "pip", "check"]
    assert records[4]["args"][:2] == ["-m", "lerobot.evaluation.scripts.verify_libero_qwen3vl"]
    assert all(record["python"] == str(prefix / "bin/python") for record in records)


@pytest.mark.parametrize("name", [None, "base", "other"])
def test_installer_rejects_other_environments_before_installation(selected_lerobot, monkeypatch, name):
    _, root, calls = selected_lerobot
    if name is None:
        monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)
    else:
        monkeypatch.setenv("CONDA_DEFAULT_ENV", name)
    result = subprocess.run(["bash", str(SCRIPT), "--libero-root", str(root)], capture_output=True, text=True)
    assert result.returncode != 0 and "conda run --name lerobot" in result.stderr
    assert not calls.exists()


@pytest.mark.parametrize(
    "invalid", ["missing-prefix", "relative-prefix", "missing-history", "missing-python"]
)
def test_installer_rejects_invalid_active_conda_prefix_before_python(selected_lerobot, monkeypatch, invalid):
    prefix, root, calls = selected_lerobot
    if invalid == "missing-prefix":
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
    elif invalid == "relative-prefix":
        monkeypatch.setenv("CONDA_PREFIX", "lerobot")
    elif invalid == "missing-history":
        (prefix / "conda-meta/history").unlink()
    else:
        (prefix / "bin/python").unlink()
    result = subprocess.run(["bash", str(SCRIPT), "--libero-root", str(root)], capture_output=True, text=True)
    message = "conda run --name lerobot" if invalid == "missing-prefix" else "not a valid Conda environment"
    assert result.returncode != 0 and message in result.stderr
    assert not calls.exists()


@pytest.mark.parametrize(
    ("version", "supported"),
    [((3, 9), False), ((3, 10), True), ((3, 11), True), ((3, 12), True), ((3, 13), False)],
)
def test_installer_checks_current_python_without_replacing_interpreter(version, supported):
    probe = SCRIPT.read_text().split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    result = subprocess.run(
        [sys.executable, "-c", f"import sys; sys.version_info = {version!r}\n{probe}"],
        env=dict(os.environ, CONDA_PREFIX=sys.prefix),
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) is supported
    if not supported:
        assert "Python 3.10-3.12" in result.stderr


def test_installer_rejects_python_from_a_different_conda_prefix(tmp_path):
    probe = SCRIPT.read_text().split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    result = subprocess.run(
        [sys.executable, "-c", probe],
        env=dict(os.environ, CONDA_PREFIX=str(tmp_path / "wrong")),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "does not belong" in result.stderr
