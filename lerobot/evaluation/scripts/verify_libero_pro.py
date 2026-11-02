"""CPU-only LIBERO-Pro resource/rendering checks, not a model benchmark."""

import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
PERTURBATIONS = {"object": "object", "position": "swap", "language": "lan", "task": "task"}


def trusted_initial_states(suite, index):
    key = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
    previous = os.environ.get(key)
    os.environ[key] = "1"
    try:
        return suite.get_task_init_states(index)
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero-root", type=Path, default=Path("/home/v-wangxiaofa/lzl/LIBERO_PRO"))
    args = parser.parse_args()
    root = args.libero_root.resolve()
    state = root / ".setup"
    if not (state / "config/config.yaml").is_file():
        parser.error("Run setup_libero_pro.sh first; the isolated Pro configuration is missing.")
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        MUJOCO_GL="osmesa",
        PYOPENGL_PLATFORM="osmesa",
        LIBERO_CONFIG_PATH=str(state / "config"),
    )
    sys.path.insert(0, str(root))
    import libero.libero as libero
    import numpy as np
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.envs.bddl_utils import get_problem_info
    from perturbation import BDDLParser, EnvironmentReplacePerturbator

    assert Path(libero.__file__).resolve().is_relative_to(root)
    suites, counts, cases, initial_states_report = {}, {}, [], []
    with contextlib.redirect_stdout(io.StringIO()):
        for base in SUITES:
            for suffix in ("", *PERTURBATIONS.values(), "env"):
                name = base + (f"_{suffix}" if suffix else "")
                suites[name] = benchmark.get_benchmark_dict()[name](task_order_index=0)
                assert suites[name].n_tasks == 10, name
                counts[name] = suites[name].n_tasks
    for base in SUITES:
        for category, suffix in PERTURBATIONS.items():
            name = f"{base}_{suffix}"
            suite = suites[name]
            for index in range(suite.n_tasks):
                task = suite.get_task(index)
                bddl = Path(suite.get_task_bddl_file_path(index))
                assert bddl.is_file(), bddl
                states = np.asarray(trusted_initial_states(suite, index))
                assert states.ndim == 2 and len(states) == 50 and np.isfinite(states).all(), (name, task.name)
                initial_states_report.append(
                    {
                        "suite": name,
                        "task": task.name,
                        "shape": list(states.shape),
                        "bddl_language": get_problem_info(str(bddl))["language_instruction"],
                    }
                )
            cases.append(
                (name, category, Path(suite.get_task_bddl_file_path(0)), trusted_initial_states(suite, 0)[0])
            )
        suite = suites[base]
        cases.append(
            (base, "original", Path(suite.get_task_bddl_file_path(0)), trusted_initial_states(suite, 0)[0])
        )
        # Upstream provides no fixed environment-replacement initial states; only smoke-test fresh resets.
        index = 2 if base == "libero_10" else 0
        task = suite.get_task(index)
        bddl = Path(suite.get_task_bddl_file_path(index))
        perturbator = EnvironmentReplacePerturbator(
            BDDLParser(bddl.read_text()), str(root / "libero_ood/ood_environment.yaml")
        )
        generated = state / "environment_smoke_bddl" / base / bddl.name
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_text(perturbator.perturb(task_suite_name=base, task_name=task.name, seed=0))
        cases.append((f"{base}_env", "environment_fresh_reset_smoke", generated, None))
    results = []
    for name, category, bddl, initial_state in cases:
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=256,
            camera_widths=256,
            render_gpu_device_id=-1,
        )
        try:
            env.seed(0)
            observation = env.reset()
            if initial_state is not None:
                observation = env.set_init_state(initial_state)
            for _ in range(3):
                observation, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            for key in ("agentview_image", "robot0_eye_in_hand_image"):
                image = observation[key]
                assert image.shape == (256, 256, 3) and image.dtype == np.uint8
                assert image.max() > image.min(), f"Blank {name}/{key}"
            assert np.isfinite(observation["robot0_eef_pos"]).all()
            result = {
                "suite": name,
                "category": category,
                "bddl_file": str(bddl),
                "initial_state_source": "official_index_0"
                if initial_state is not None
                else "fresh_reset_not_benchmark",
                "passed": True,
            }
            results.append(result)
            print(json.dumps(result), flush=True)
        finally:
            env.close()
    report = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "source": str(root),
        "libero_module": libero.__file__,
        "renderer": "osmesa",
        "gpu_used": False,
        "suite_task_counts": counts,
        "official_initial_state_files": initial_states_report,
        "environment_smokes": results,
        "policy_loaded": False,
        "benchmark_scores": False,
        "environment_benchmark_initial_states_available": False,
    }
    report_path = state / "verification.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        f"PASS: {len(initial_states_report)} official 50-state files; {len(results)} CPU environment cases.",
        flush=True,
    )
    print("REPORT_SHA256", hashlib.sha256(report_path.read_bytes()).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
