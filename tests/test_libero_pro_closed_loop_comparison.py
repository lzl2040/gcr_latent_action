import copy
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from scripts import compare_libero_pro_closed_loop as comparison


def snapshots(offsets):
    return [
        {
            "control_step": step,
            "decoded_source_offsets": offsets,
            "display_frame_index": offsets.index(16),
            "forecast_start_s": step / 20,
            "forecast_end_s": (step + 16) / 20,
            "jointly_sampled_with_executed_action_prefix": True,
            "executed_action_prefix_length": 10,
        }
        for step in (0, 40, 80)
    ]


def test_three_replans_never_duplicates_an_observation():
    assert comparison.three_replans(3) == [0, 1, 2]
    assert comparison.three_replans(14) == [0, 6, 13]
    with pytest.raises(ValueError, match="distinct"):
        comparison.three_replans(2)


@pytest.mark.parametrize("offset", [0, 3, 32])
def test_future_selection_rejects_conditioning_or_interpolated_frames(offset):
    with pytest.raises(ValueError, match="strictly future"):
        comparison.forecast_index([0, 4, 8, 12, 16, 19, 23, 27, 31], offset)


@pytest.mark.parametrize(
    "field,value",
    [
        ("success", True),
        ("trial", 8),
        ("initial_state_sha256", "different"),
        ("outcome_from_environment", False),
    ],
)
def test_pair_requires_same_initial_state_and_actual_measured_failure(field, value):
    ours = {
        "case_id": "same",
        "task": {"task_id": 1},
        "trial": 0,
        "initial_state_sha256": "same",
        "success": True,
        "outcome_from_environment": True,
    }
    native = {**ours, "success": False}
    comparison.check_pair(ours, native)
    native[field] = value
    with pytest.raises(ValueError):
        comparison.check_pair(ours, native)


@pytest.mark.parametrize("change", ["duplicate", "current_frame", "wrong_time", "not_executed"])
def test_forecast_validation_rejects_misleading_replanned_futures(change):
    record = {"snapshots": snapshots([0, 4, 8, 12, 16, 19, 23, 27, 31])}
    comparison.check_snapshots(record)
    if change == "duplicate":
        record["snapshots"][1] = copy.deepcopy(record["snapshots"][0])
    elif change == "current_frame":
        record["snapshots"][2]["display_frame_index"] = 0
    elif change == "wrong_time":
        record["snapshots"][2]["forecast_end_s"] = 100
    else:
        record["snapshots"][2]["jointly_sampled_with_executed_action_prefix"] = False
    with pytest.raises(ValueError):
        comparison.check_snapshots(record)


def test_four_row_export_keeps_originals_and_only_downsamples_ours(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    cases = []
    initial = np.zeros((256, 256, 3), dtype=np.uint8)
    expected = {}
    for task in range(4):
        paths = {}
        for name, size in [("ours", 256), ("cosmos3", 160)]:
            directory = tmp_path / name / f"case{task}"
            directory.mkdir(parents=True)
            offsets = [0, 4, 8, 12, 16, 19, 23, 27, 31] if name == "ours" else list(range(17))
            record = {
                "case_id": f"case{task}",
                "task": {"base_suite": "libero_goal", "task_id": task},
                "trial": 0,
                "initial_state_sha256": "shared",
                "success": name == "ours",
                "outcome_from_environment": True,
                "snapshots": snapshots(offsets),
            }
            for view in comparison.CAMERAS:
                Image.fromarray(initial).save(directory / f"initial_{view}.png")
            Image.fromarray(initial).save(directory / "actual_final_primary.png")
            (directory / "actual_rollout.mp4").write_bytes(b"opaque-archived-rollout-fixture")
            grid = np.indices((size, size)).sum(0).astype(np.uint8)
            frames = np.stack(
                [np.stack((grid, np.full_like(grid, i * 10), grid), -1) for i in range(len(offsets))]
            )
            for i, snapshot in enumerate(record["snapshots"]):
                prediction = directory / f"prediction{i}"
                prediction.mkdir()
                np.savez_compressed(prediction / "frames.npz", primary=frames, wrist=frames)
                np.savez_compressed(prediction / "condition.npz", current_rgb=initial)
                snapshot.update(
                    prediction_directory=str(prediction),
                    prediction_sha256=comparison.digest(prediction / "frames.npz"),
                    condition_file=str(prediction / "condition.npz"),
                    condition_sha256=comparison.digest(prediction / "condition.npz"),
                )
            expected[task, name] = frames[offsets.index(16)]
            comparison.write_json(directory / "result.json", record)
            paths[f"{'cosmos3' if name == 'cosmos3' else 'ours'}_directory"] = str(directory)
        cases.append({"case_id": f"case{task}", "title": f"Task {task}", **paths})
    selection = tmp_path / "selection.json"
    comparison.write_json(selection, {"cases": cases})
    output = tmp_path / "figure"
    savefig = plt.Figure.savefig

    def save_without_overlapping_labels(figure, *args, **kwargs):
        figure.canvas.draw()
        bounds = [text.get_window_extent(figure.canvas.get_renderer()) for text in figure.texts]
        for index, box in enumerate(bounds):
            assert not any(box.overlaps(other) for other in bounds[index + 1 :])
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", save_without_overlapping_labels)
    comparison.figure(SimpleNamespace(selection=selection, output_dir=output))
    for task in range(4):
        for name in ("ours", "cosmos3"):
            raw = expected[task, name]
            np.testing.assert_array_equal(
                np.asarray(Image.open(output / f"case{task}/{name}_primary_03_original.png")), raw
            )
            display = np.asarray(Image.open(output / f"case{task}/{name}_primary_03_display.png"))
            reference = (
                raw
                if name == "cosmos3"
                else np.asarray(Image.fromarray(raw).resize((160, 160), Image.Resampling.LANCZOS))
            )
            np.testing.assert_array_equal(display, reference)
    assert (output / "comparison_primary.pdf").is_file()
    assert (output / "comparison_wrist.pdf").is_file()
    with pytest.raises(FileExistsError):
        comparison.figure(SimpleNamespace(selection=selection, output_dir=output))
