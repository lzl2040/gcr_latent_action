import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from scripts import compare_libero_pro_closed_loop as common
from scripts import export_libero_pro_prediction_gallery as gallery


def test_dense_gallery_preserves_all_original_frames_and_links_only_future_predictions(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    case = {"case_id": "case", "title": "Move the relocated bowl"}
    expected = {}
    for name, size, steps in [("ours", 256, [0, 10, 20, 30, 40]), ("cosmos3", 160, [0, 16])]:
        directory = tmp_path / name
        directory.mkdir()
        case[f"{name}_directory"] = str(directory)
        initial = np.zeros((256, 256, 3), dtype=np.uint8)
        for view in common.CAMERAS:
            Image.fromarray(initial).save(directory / f"initial_{view}.png")
        (directory / "all_joint_plans.pt").write_bytes(b"opaque-joint-plan-fixture")
        record = {
            "case_id": "case",
            "task": {
                "base_suite": "libero_goal",
                "suite": "libero_goal_swap",
                "category": "position",
                "task_id": 3,
                "instruction": case["title"],
            },
            "perturbation": {"summary": "Bowl and cheese placements swapped", "label": "Position"},
            "initial_state_sha256": "same",
            "trial": 3,
            "success": name == "ours",
            "steps": steps[-1] + 10,
            "outcome_from_environment": True,
            "snapshots": [],
        }
        offsets = [0, 4, 8, 12, 16, 19, 23, 27, 31] if name == "ours" else list(range(17))
        for step in steps:
            prediction = directory / f"prediction_step_{step:04d}"
            prediction.mkdir()
            frames = np.stack(
                [np.full((size, size, 3), step + index + 10, dtype=np.uint8) for index in range(len(offsets))]
            )
            np.savez_compressed(prediction / "frames.npz", primary=frames, wrist=frames + 1)
            condition = directory / f"condition_step_{step:04d}.npz"
            np.savez_compressed(condition, current_rgb=initial)
            record["snapshots"].append(
                {
                    "control_step": step,
                    "decoded_source_offsets": offsets,
                    "display_frame_index": offsets.index(16),
                    "forecast_start_s": step / 20,
                    "forecast_end_s": (step + 16) / 20,
                    "jointly_sampled_with_executed_action_prefix": True,
                    "executed_action_prefix_length": 10,
                    "prediction_directory": str(prediction),
                    "prediction_sha256": common.digest(prediction / "frames.npz"),
                    "condition_file": str(condition),
                    "condition_sha256": common.digest(condition),
                }
            )
            expected[name, step] = frames
        common.write_json(directory / "result.json", record)
    selection = tmp_path / "selection.json"
    common.write_json(selection, {"cases": [case]})
    output = tmp_path / "export"
    savefig = plt.Figure.savefig

    def checked_savefig(figure, *args, **kwargs):
        figure.canvas.draw()
        texts = list(figure.texts)
        for axis in figure.axes:
            texts.extend([axis.title, *axis.texts])
        bounds = [text.get_window_extent(figure.canvas.get_renderer()) for text in texts if text.get_text()]
        for index, box in enumerate(bounds):
            assert not any(box.overlaps(other) for other in bounds[index + 1 :])
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", checked_savefig)
    gallery.export_gallery(
        SimpleNamespace(selection=selection, output_dir=output, include_other_outcomes=False)
    )
    metadata = json.loads((output / "cases_metadata.json").read_text())["cases"][0]
    archive = output / metadata["relative_directory"]
    assert len(metadata["tiles"]) == 14
    for tile in metadata["tiles"]:
        frames = expected[tile["model"], tile["control_step"]]
        with np.load(archive / tile["source_prediction_directory"] / "frames.npz") as data:
            np.testing.assert_array_equal(data["primary"], frames)
        raw = frames[tile["source_frame_index"]] + (tile["view"] == "wrist")
        expected_display = np.asarray(Image.fromarray(raw).resize((160, 160), Image.Resampling.LANCZOS))
        np.testing.assert_array_equal(
            np.asarray(Image.open(archive / tile["display_file"])), expected_display
        )
        assert tile["source_offset_steps"] == 16
        assert tile["origin"] == "joint_model_video_latents_decoded_by_vae"
        assert common.digest(archive / tile["source_condition_file"]) == tile["condition_sha256"]
    assert (archive / "execution_audit/ours/all_joint_plans.pt").is_file()
    assert not list((archive / "display").rglob("*initial*"))
    assert (output / "all_cases_primary.pdf").is_file()
    assert (output / "overview_position_01_wrist.png").is_file()
    assert (output / "index.html").is_file()
    with pytest.raises(FileExistsError):
        gallery.export_gallery(
            SimpleNamespace(selection=selection, output_dir=output, include_other_outcomes=False)
        )
