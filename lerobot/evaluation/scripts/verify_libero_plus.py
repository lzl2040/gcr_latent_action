"""CPU-only environment smoke checks, not policy evaluation or benchmark scores."""

import argparse
import contextlib
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero-root", type=Path, default=Path("/home/v-wangxiaofa/lzl/LIBERO_PLUS"))
    args = parser.parse_args()
    root = args.libero_root.resolve()
    state = root / ".setup"
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        MUJOCO_GL="osmesa",
        PYOPENGL_PLATFORM="osmesa",
        LIBERO_CONFIG_PATH=str(state / "config"),
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
    )
    sys.path.insert(0, str(root))
    import libero.libero as libero
    import numpy as np
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv, env_wrapper
    from PIL import Image

    assert Path(libero.__file__).resolve().is_relative_to(root)
    classification = json.loads((root / "libero/libero/benchmark/task_classification.json").read_text())
    suites = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for name in ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"):
            suites[name] = benchmark.get_benchmark_dict()[name](task_order_index=0)
    assert sum(suites[name].n_tasks for name in classification) == 10030
    cases = []
    seen = set()
    spatial_indices = {task.name: index for index, task in enumerate(suites["libero_spatial"].tasks)}
    for entry in classification["libero_spatial"]:
        category = entry["category"]
        if category not in seen:
            seen.add(category)
            cases.append(("libero_spatial", spatial_indices[entry["name"]], category))
    assert len(seen) == 7
    for name in ("libero_object", "libero_goal", "libero_10", "libero_90"):
        cases.append((name, 0, "suite_smoke"))
    results = []
    for name, index, category in cases:
        suite = suites[name]
        task = suite.get_task(index)
        np.random.seed(0)
        # These are trusted, revision-pinned upstream NumPy pickles, not model weights.
        initial_states = (
            suite.get_task_init_states_ori(index)
            if name == "libero_90"
            else suite.get_task_init_states(index)
        )
        assert len(initial_states) > 0
        env = OffScreenRenderEnv(
            bddl_file_name=str(root / "libero/libero/bddl_files" / task.problem_folder / task.bddl_file),
            camera_heights=256,
            camera_widths=256,
            render_gpu_device_id=-1,
        )
        try:
            env.reset()
            observation = env.set_init_state(initial_states[0])
            for _ in range(3):
                observation, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            for key in ("agentview_image", "robot0_eye_in_hand_image"):
                image = observation[key]
                assert image.shape == (256, 256, 3) and image.dtype == np.uint8
                assert image.max() > image.min(), f"Blank camera: {name}/{key}"
            assert np.isfinite(observation["robot0_eef_pos"]).all()
            result = {
                "suite": name,
                "task_id": index,
                "task": task.name,
                "category": category,
                "passed": True,
            }
            results.append(result)
            print(json.dumps(result), flush=True)
        finally:
            env.close()
    image = Image.fromarray(np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8))
    corruptions = ("motion_blur", "gaussian_blur", "zoom_blur", "fog", "glass_blur")
    for name in corruptions:
        corrupted = np.asarray(getattr(env_wrapper, name)(image, severity=1))
        assert corrupted.shape == (256, 256, 3) and np.isfinite(corrupted).all(), name
    report = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "source": str(root),
        "libero_module": libero.__file__,
        "renderer": "osmesa",
        "gpu_used": False,
        "suite_task_counts": {name: suite.n_tasks for name, suite in suites.items()},
        "category_smokes": sorted(seen),
        "environment_smokes": results,
        "image_corruption_smokes": list(corruptions),
        "policy_loaded": False,
        "benchmark_scores": False,
    }
    (state / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
