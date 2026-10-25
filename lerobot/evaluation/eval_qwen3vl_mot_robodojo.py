"""Offline inference, WebSocket serving and isolated RoboDojo closed-loop evaluation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

POLICY_NAME = "Qwen3VLMoT"
REPRO_ROOT = Path("/data2/guoqi/robodojo_pi05_repro")
EXPERIMENT = Path("/data2/lizuolei/weights/qwen3vl_mot_weights/1016_ft_on_robodojo")
PRETRAINED = Path("/home/jianf/blob_data/pt_weights")
DATASET = Path(
    "/home/jianf/blob_data/robot_dataset/lerobot-format-v30/RoboDojo/data/RoboDojo_ee_lerobot_v30_video"
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(precision="bf16", fp8_training_config=None)
    parser.add_argument("mode", choices=("offline", "serve", "eval", "diagnose"))
    parser.add_argument("--checkpoint", type=Path, default=EXPERIMENT / "step4k")
    parser.add_argument("--policy-config", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=DATASET)
    parser.add_argument(
        "--action-mode",
        choices=("absolute", "delta"),
        default="absolute",
        help="Delta mode cumulatively integrates measured-state increments from each current observation.",
    )
    parser.add_argument("--qwen3vl-dir", type=Path, default=PRETRAINED / "Qwen3-VL-4B-Instruct")
    parser.add_argument("--cosmos3-dir", type=Path, default=PRETRAINED / "Cosmos3-Edge")
    parser.add_argument("--stage1-text-dir", type=Path, default=PRETRAINED / "siglip2-base-patch16-224")
    parser.add_argument(
        "--device", default="cuda:0", help="Logical policy device after CUDA_VISIBLE_DEVICES."
    )
    parser.add_argument("--gpu", type=int, default=0, help="Physical policy GPU for eval mode.")
    parser.add_argument("--sim-gpu", type=int, default=0, help="Physical simulator GPU for eval mode.")
    parser.add_argument("--inference-steps", type=int)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument(
        "--save-predicted-video",
        action="store_true",
        help="Export matching joint-sample future RGB and PNG frames.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--training-config", type=Path, help="Defaults to train_config.json beside the saved policy config."
    )
    parser.add_argument("--diagnostic-episodes", type=int, nargs="+", default=[3000, 3001, 3002])
    parser.add_argument("--diagnostic-frames", type=int, nargs="+", default=[0, 64])
    parser.add_argument("--diagnostic-inference-steps", type=int, nargs="+", default=[20, 50])
    parser.add_argument(
        "--diagnostic-fp8", action="store_true", help="Reapply the saved training FP8 recipe."
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--robodojo-root", type=Path, default=REPRO_ROOT)
    parser.add_argument("--assets", type=Path, default=REPRO_ROOT / "Assets")
    parser.add_argument("--episode", type=Path)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19030)
    parser.add_argument("--task", default="stack_bowls")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--eval-num", default="1", help="Positive episode count, or native.")
    parser.add_argument("--sim-python", type=Path, default=Path(sys.prefix).parent / "RoboDojo6/bin/python")
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument(
        "--timeout", type=float, default=3600, help="Simulator wall-clock timeout in seconds."
    )
    return parser


def add_xpolicylab(root: Path) -> None:
    for path in (root, root / "XPolicyLab"):
        if not path.is_dir():
            raise FileNotFoundError(f"Missing RoboDojo/XPolicyLab directory: {path}")
    sys.path[:0] = [str(root.resolve()), str((root / "XPolicyLab").resolve())]


def create_workspace(root: Path, assets: Path, workspace: Path) -> None:
    """Use local package paths so env.global_configs resolves assets and outputs here."""
    workspace.mkdir(parents=True, exist_ok=False)
    # env must be a real directory: global_configs builds ROOT_DIR as env/..,
    # and the OS would traverse a directory symlink before resolving that "..".
    environment = workspace / "env"
    environment.mkdir()
    if not (root / "env").is_dir():
        raise FileNotFoundError(f"Missing RoboDojo environment package: {root / 'env'}")
    for source in (root / "env").iterdir():
        if source.name != "__pycache__":
            (environment / source.name).symlink_to(source.resolve(), target_is_directory=source.is_dir())
    for name in ("env_cfg", "task", "src", "utils", "third_party"):
        source = root / name
        if not source.is_dir():
            raise FileNotFoundError(f"Missing RoboDojo simulator component: {source}")
        (workspace / name).symlink_to(source.resolve(), target_is_directory=True)
    if not assets.is_dir():
        raise FileNotFoundError(f"Missing simulation assets: {assets}")
    (workspace / "Assets").symlink_to(assets.resolve(), target_is_directory=True)
    policy_root = workspace / "XPolicyLab"
    policy_root.mkdir()
    for source in (root / "XPolicyLab").iterdir():
        if source.name not in ("policy", "__pycache__", ".git"):
            (policy_root / source.name).symlink_to(source.resolve(), target_is_directory=source.is_dir())
    policies = policy_root / "policy"
    policies.mkdir()
    (policies / "__init__.py").write_text("")
    integration = policies / POLICY_NAME
    integration.mkdir()
    (integration / "__init__.py").write_text("")
    (integration / "deploy.py").symlink_to(Path(__file__).with_name("robodojo_deploy.py").resolve())
    (integration / "deploy.yml").write_text("eval_batch: true\naction_type: ee\n")


def validate_training_action_mode(args) -> dict:
    from lerobot.evaluation.eval_qwen3vl_mot_libero import checkpoint_paths
    from lerobot.evaluation.robodojo_diagnostics import matching_training_config

    _, policy_path = checkpoint_paths(args.checkpoint, args.policy_config)
    training_path = args.training_config or policy_path.with_name("train_config.json")
    if not training_path.is_file():
        if args.action_mode == "delta" or args.training_config is not None:
            raise FileNotFoundError(f"Missing training configuration: {training_path}")
        logger.warning("No saved training configuration; absolute action semantics cannot be cross-checked.")
        return {"action_mode": args.action_mode, "training_configuration_verified": False}
    training_path = matching_training_config(args)
    saved = json.loads(training_path.read_text())
    expected = {"absolute": "robodojo", "delta": "delta_robodojo_stack_bowl"}[args.action_mode]
    if saved["data_mix"] != expected or saved["stage"] != "finetune":
        raise ValueError(
            f"Action mode {args.action_mode!r} requires finetuning data_mix={expected!r}; "
            f"checkpoint declares {saved['data_mix']!r}/{saved['stage']!r}."
        )
    stats_path = args.dataset_root / "meta/stats.json"
    return {
        "action_mode": args.action_mode,
        "training_configuration_verified": True,
        "training_config": str(training_path),
        "data_mix": saved["data_mix"],
        "dataset_root": str(args.dataset_root.resolve()),
        "stats_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
        "normalization": "per-feature mean/std with std floor 1e-3",
        "integration_anchor": "current measured observation at each replan"
        if args.action_mode == "delta"
        else None,
    }


def load_model(args):
    import numpy as np
    import torch

    from lerobot.evaluation.eval_qwen3vl_mot_libero import PredictionVideoWriter, load_policy
    from lerobot.evaluation.robodojo_qwen3vl import RoboDojoAdapter, RoboDojoModel

    contract = validate_training_action_mode(args)
    adapter = RoboDojoAdapter.from_dataset(args.dataset_root, action_mode=args.action_mode)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Qwen3-VL MoT evaluation requires a CUDA GPU.")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    policy = load_policy(args)
    model = RoboDojoModel(
        policy,
        adapter,
        device=args.device,
        replan_steps=args.replan_steps,
        output_dir=args.output_dir / "chunks",
        prediction_writer=(
            PredictionVideoWriter(policy.config, torch.device(args.device), adapter.fps)
            if args.save_predicted_video
            else None
        ),
        seed=args.seed,
    )
    model.reset()
    return model


def recorded_observation(root: Path, episode: Path, frame: int) -> dict:
    import h5py

    add_xpolicylab(root)
    from XPolicyLab.utils.process_data import decode_image_bit

    state_keys = {
        "left_ee_pose": "left_ee_poses",
        "right_ee_pose": "right_ee_poses",
        "left_ee_joint_state": "left_ee_joint_states",
        "right_ee_joint_state": "right_ee_joint_states",
    }
    with h5py.File(episode, "r") as data:
        return {
            "env_idx": 0,
            "instruction": data["instruction"].asstr()[()],
            "state": {key: data["state"][source][frame] for key, source in state_keys.items()},
            "vision": {
                name: {"color": decode_image_bit(camera["colors"][frame])}
                for name, camera in data["vision"].items()
            },
        }


def offline(args) -> None:
    import numpy as np

    episode = args.episode or (
        args.robodojo_root / "data/RoboDojo" / args.task / "arx_x5/data/episode_0000000.hdf5"
    )
    observation = recorded_observation(args.robodojo_root, episode, args.frame)
    model = load_model(args)
    model.update_obs(observation)
    started = time.perf_counter()
    for _ in range(args.repeat):
        model.get_action()
    summary = {
        "status": "passed",
        "mode": "recorded_observation_only",
        "benchmark_evaluation": False,
        "checkpoint": str(args.checkpoint.resolve()),
        "episode": str(episode.resolve()),
        "frame": args.frame,
        "repeat": args.repeat,
        "replan_steps": args.replan_steps,
        "instruction": observation["instruction"],
        "image_shapes": {
            key: list(np.asarray(value["color"]).shape) for key, value in observation["vision"].items()
        },
        "seconds": time.perf_counter() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("Recorded-observation inference passed: %s", args.output_dir)


def serve(args) -> None:
    add_xpolicylab(args.robodojo_root)
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig

    model = load_model(args)
    server = PolicyServer(model, PolicyServerConfig(host=args.host, port=args.port, ws_ping_timeout_s=600))
    asyncio.run(server.serve_forever())


def wait_for_server(process, args) -> None:
    from websockets.sync.client import connect

    add_xpolicylab(args.robodojo_root)
    from client_server.ws.model_client import WsModelClient

    deadline = time.monotonic() + args.startup_timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Policy server exited with {process.returncode}; see policy.log.")
        try:
            connection = socket.create_connection((args.host, args.port), timeout=1)
        except (ConnectionRefusedError, TimeoutError):
            time.sleep(1)
            continue
        # Complete a valid handshake on the probe socket instead of causing a
        # spurious "invalid HTTP request" traceback in the server log.
        with connect(f"ws://{args.host}:{args.port}", sock=connection, open_timeout=30):
            pass
        client = WsModelClient(
            url=f"ws://{args.host}:{args.port}",
            evaluation_id="qwen3vl-readiness",
            trial_id="readiness",
            max_connect_attempts=1,
            request_timeout_s=30,
        )
        client.close()
        return
    raise TimeoutError("Policy startup timed out; see policy.log.")


def validate_result(workspace: Path, task: str, expected: int) -> tuple[Path, dict]:
    paths = list((workspace / "eval_result/RoboDojo" / task / POLICY_NAME).glob("**/_result.json"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one native RoboDojo result, found {paths}.")
    result = json.loads(paths[0].read_text())
    details = result["details"]
    if result["eval_time"] != expected or len(details) != expected:
        raise RuntimeError(f"Expected {expected} completed episodes, got {result}.")
    layouts = [item["layout_id"] for item in details.values()]
    if sorted(layouts) != list(range(expected)):
        raise RuntimeError(f"Evaluation skipped, duplicated or replaced layouts: {layouts}.")
    if any(not isinstance(item["success"], bool) for item in details.values()):
        raise RuntimeError("Native episode success labels must be Boolean.")
    successes = sum(item["success"] for item in details.values())
    if not math.isclose(result["success_rate"], successes / expected, abs_tol=1e-8):
        raise RuntimeError("Native success rate does not match the episode success labels.")
    return paths[0], result


def evaluate(args) -> None:
    import yaml

    if not args.sim_python.is_file():
        raise FileNotFoundError(f"Simulator Python is missing: {args.sim_python}")
    task_config = args.robodojo_root / "task/RoboDojo/config" / f"{args.task}.yml"
    if not task_config.is_file():
        raise FileNotFoundError(f"Unknown task configuration: {task_config}")
    registry = yaml.safe_load((task_config.parent / "_task.yml").read_text())
    native = registry["tasks"].get(args.task, {}).get("eval_nums", registry["common"]["eval_nums"])
    expected = native if args.eval_num == "native" else min(int(args.eval_num), native)
    workspace = args.output_dir / "simulator"
    create_workspace(args.robodojo_root, args.assets, workspace)
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((args.host, args.port))
    repo_root = Path(__file__).resolve().parents[2]
    policy_env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        PYTHONPATH=str(repo_root),
        OMP_NUM_THREADS="8",
        OPENBLAS_NUM_THREADS="8",
        PYTHONNOUSERSITE="1",
    )
    policy_command = [
        sys.executable,
        "-u",
        "-m",
        "lerobot.evaluation.eval_qwen3vl_mot_robodojo",
        "serve",
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--dataset-root",
        str(args.dataset_root.resolve()),
        "--action-mode",
        args.action_mode,
        "--qwen3vl-dir",
        str(args.qwen3vl_dir.resolve()),
        "--cosmos3-dir",
        str(args.cosmos3_dir.resolve()),
        "--stage1-text-dir",
        str(args.stage1_text_dir.resolve()),
        "--robodojo-root",
        str(args.robodojo_root.resolve()),
        "--output-dir",
        str(args.output_dir / "policy"),
        "--device",
        args.device,
        "--replan-steps",
        str(args.replan_steps),
        "--seed",
        str(args.seed),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.policy_config:
        policy_command.extend(("--policy-config", str(args.policy_config.resolve())))
    if args.training_config:
        policy_command.extend(("--training-config", str(args.training_config.resolve())))
    if args.inference_steps is not None:
        policy_command.extend(("--inference-steps", str(args.inference_steps)))
    if args.save_predicted_video:
        policy_command.append("--save-predicted-video")
    run_id = args.output_dir.name
    sim_env = dict(
        os.environ,
        PATH=os.pathsep.join((str(args.sim_python.absolute().parent), os.environ.get("PATH", ""))),
        CUDA_VISIBLE_DEVICES=str(args.sim_gpu),
        PYTHONPATH=f"{workspace}:{workspace / 'XPolicyLab'}",
        PYTHONDONTWRITEBYTECODE="1",
        ROBODOJO_RUN_ID=run_id,
        EVAL_NUM=args.eval_num,
        OMP_NUM_THREADS="8",
        OPENBLAS_NUM_THREADS="8",
        PYTHONNOUSERSITE="1",
    )
    for env in (policy_env, sim_env):
        for key in ("CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH"):
            env.pop(key, None)
    # Isaac Sim EULA acceptance must be supplied by the person running the command.
    if sim_env.get("OMNI_KIT_ACCEPT_EULA") != "YES":
        raise RuntimeError("Review the Isaac Sim EULA and set OMNI_KIT_ACCEPT_EULA=YES before eval.")
    deprecated = subprocess.check_output(
        [
            str(args.sim_python),
            "-c",
            "from importlib.metadata import distribution; "
            'print(distribution("isaacsim").locate_file("isaacsim/extsDeprecated"))',
        ],
        text=True,
        env=sim_env,
    ).strip()
    kit_args = (
        "--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera "
        "--/renderer/multiGpu/enabled=false "
        f"--/renderer/activeGpu={args.sim_gpu} --/physics/cudaDevice=0 "
        "--/plugins/carb.tasking.plugin/threadCount=8 "
        f"--/log/file={args.output_dir / 'kit.log'} "
        f"--ext-folder {deprecated} --enable isaacsim.core.api"
    )
    sim_command = [
        str(args.sim_python),
        "-u",
        str(workspace / "src/eval_client/main.py"),
        "--task_name",
        args.task,
        "--env_cfg_type",
        "arx_x5",
        "--num_envs",
        str(args.num_envs),
        "--device_id",
        str(args.sim_gpu),
        "--enable_cameras",
        "--headless",
        "--policy_name",
        POLICY_NAME,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--protocol",
        "ws",
        "--seed",
        str(args.seed),
        "--additional_info",
        "action_type=ee,simulator=isaacsim6",
        "--profile_eval",
        "--fail_fast",
        "--ws_ping_timeout_s",
        "600",
        "--ws_request_timeout_s",
        "600",
        "--kit_args",
        kit_args,
    ]
    metadata = {
        "status": "running",
        "evaluation_complete": False,
        "task": args.task,
        "expected_episodes": expected,
        "seed": args.seed,
        "policy_gpu": args.gpu,
        "simulator_gpu": args.sim_gpu,
        "replan_steps": args.replan_steps,
        "save_predicted_video": args.save_predicted_video,
        "checkpoint": str(args.checkpoint.resolve()),
        "action_mode": args.action_mode,
        "dataset_root": str(args.dataset_root.resolve()),
        "assets": str(args.assets.resolve()),
        "simulator": "Isaac Sim 6 (local compatibility runtime)",
        "policy_command": policy_command,
        "simulator_command": sim_command,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(metadata, indent=2) + "\n")
    processes = []
    started = time.perf_counter()
    try:
        with (
            (args.output_dir / "policy.log").open("w") as policy_log,
            (args.output_dir / "simulator.log").open("w") as sim_log,
        ):
            process = subprocess.Popen(
                policy_command,
                cwd=repo_root,
                env=policy_env,
                stdout=policy_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(process)
            wait_for_server(process, args)
            logger.info(
                "Policy ready; launching %s, %d episode(s). Logs: %s", args.task, expected, args.output_dir
            )
            simulator = subprocess.Popen(
                sim_command,
                cwd=workspace,
                env=sim_env,
                stdout=sim_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(simulator)
            code = simulator.wait(timeout=args.timeout)
            if code:
                raise RuntimeError(f"Simulator exited with {code}; see simulator.log.")
        path, result = validate_result(workspace, args.task, expected)
        metadata.update(status="completed", evaluation_complete=True, result_path=str(path), result=result)
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        metadata["wall_seconds"] = time.perf_counter() - started
        summary_path.write_text(json.dumps(metadata, indent=2) + "\n")
    logger.info("Completed native RoboDojo evaluation: %s", summary_path)


def main(parser: argparse.ArgumentParser | None = None) -> None:
    parser = build_parser() if parser is None else parser
    args = parser.parse_args()
    for name in ("replan_steps", "repeat", "num_envs", "startup_timeout", "timeout"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.inference_steps is not None and args.inference_steps <= 0:
        parser.error("--inference-steps must be positive.")
    if min(args.frame, args.gpu, args.sim_gpu, args.seed) < 0:
        parser.error("Frame, GPU indices and seed must be nonnegative.")
    if not args.task.replace("_", "").isalnum():
        parser.error("--task must be an alphanumeric task configuration name.")
    if args.eval_num != "native" and (not args.eval_num.isdigit() or int(args.eval_num) <= 0):
        parser.error("--eval-num must be a positive integer or native.")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1,65535].")
    args.output_dir = (
        args.output_dir or Path("outputs/evaluation/robodojo") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.output_dir / "evaluation.log")],
    )
    (args.output_dir / "settings.json").write_text(
        json.dumps(
            {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            indent=2,
        )
        + "\n"
    )
    if args.mode == "diagnose":
        from lerobot.evaluation.robodojo_diagnostics import diagnose

        diagnose(args)
    else:
        {"offline": offline, "serve": serve, "eval": evaluate}[args.mode](args)


if __name__ == "__main__":
    main()
