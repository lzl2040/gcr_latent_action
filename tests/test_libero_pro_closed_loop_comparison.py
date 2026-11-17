import copy
import csv
import json
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


@pytest.mark.parametrize(
    "count,maximum,expected",
    [(1, 9, [0]), (2, 9, [0, 1]), (3, 9, [0, 1, 2]), (14, 3, [0, 6, 13]), (14, 5, [0, 3, 6, 9, 13])],
)
def test_dense_replans_include_endpoints_without_duplicating_short_rollouts(count, maximum, expected):
    assert comparison.prediction_replans(count, maximum) == expected


def test_dense_forecast_validation_keeps_the_executed_action_check():
    record = {"snapshots": snapshots(list(range(17)))[:2]}
    comparison.check_snapshots(record, expected_count=None)
    with pytest.raises(ValueError, match="expected count"):
        comparison.check_snapshots(record)
    record["snapshots"][1]["jointly_sampled_with_executed_action_prefix"] = False
    with pytest.raises(ValueError, match="correspondence"):
        comparison.check_snapshots(record, expected_count=None)


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
        ("perturbation", {"category": "task"}),
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


def test_screening_other_outcomes_requires_explicit_opt_in_and_keeps_identity_checks():
    ours = {
        "case_id": "same",
        "task": {"task_id": 1},
        "trial": 0,
        "initial_state_sha256": "same",
        "success": False,
        "outcome_from_environment": True,
    }
    native = {**ours, "success": True}
    with pytest.raises(ValueError, match="Ours success"):
        comparison.check_pair(ours, native)
    comparison.check_pair(ours, native, require_advantage=False)
    native["trial"] = 1
    with pytest.raises(ValueError, match="trial"):
        comparison.check_pair(ours, native, require_advantage=False)


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


@pytest.mark.parametrize("category", ["environment", "object", "position"])
def test_perturbation_metadata_describes_actual_bddl_changes(tmp_path, category):
    root = tmp_path / "libero"
    original = root / "libero/libero/bddl_files/libero_goal/example.bddl"
    variant = tmp_path / "variant/example.bddl"
    original.parent.mkdir(parents=True)
    variant.parent.mkdir()
    before = (
        "(define (problem LIBERO_Tabletop_Manipulation) "
        "(:language place the bowl) "
        "(:objects bowl_1 bowl_2 - bowl plate_1 - plate) "
        "(:init (On bowl_1 table_left_region) (On plate_1 table_right_region)) "
        "(:goal (And (On bowl_1 plate_1))))"
    )
    after = (
        before.replace("LIBERO_Tabletop_Manipulation", "LIBERO_Living_Room_Tabletop_Manipulation")
        if category == "environment"
        else (
            before.replace("table_left_region", "table_middle_region")
            if category == "position"
            else before.replace("bowl", "bigger_bowl").replace("place the bigger_bowl", "place the bowl")
        )
    )
    original.write_text(before)
    variant.write_text(after)
    row = {
        "base_suite": "libero_goal",
        "suite": f"libero_goal_{'env' if category == 'environment' else 'object'}",
        "category": category,
        "bddl_file": str(variant),
        "bddl_sha256": comparison.digest(variant),
        "initial_state_source": "fixed_test_states",
        "original_environment": "main_table",
        "target_environment": "living_room_table",
    }
    result = comparison.describe_perturbation(row, root)
    assert result["suite"] == row["suite"]
    assert result["category"] == category
    assert result["environment_changed"] == (category == "environment")
    assert result["original_objects"]["bowl"] == ["bowl_1", "bowl_2"]
    assert result["variant_instruction"] == result["original_instruction"]
    if category == "object":
        assert result["objects_removed"] == {"bowl": ["bowl_1", "bowl_2"]}
        assert result["objects_added"] == {"bigger_bowl": ["bigger_bowl_1", "bigger_bowl_2"]}
        assert result["original_goal"] != result["variant_goal"]
    elif category == "environment":
        assert result["summary"] == "main_table -> living_room_table"
    else:
        assert result["placement_changes"] == {
            "bowl_1": {"original_region": "table_left_region", "variant_region": "table_middle_region"}
        }
        assert result["original_objects"] == result["variant_objects"]
        assert result["original_goal"] == result["variant_goal"]
    variant.write_text(after + "\n")
    with pytest.raises(ValueError, match="BDDL changed"):
        comparison.describe_perturbation(row, root)


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
                "task": {
                    "base_suite": "libero_goal",
                    "suite": "libero_goal_env" if task < 2 else "libero_goal_object",
                    "category": "environment" if task < 2 else "object",
                    "task_id": task,
                },
                "perturbation": {
                    "label": "Env" if task < 2 else "Object",
                    "summary": "main_table -> living_room_table" if task < 2 else "bowl -> bigger_bowl",
                },
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
                np.savez_compressed(prediction / "frames.npz", primary=frames, wrist=frames[:, :, ::-1])
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
        reference = tmp_path / "references" / f"case{task}"
        reference.mkdir(parents=True)
        for view in comparison.CAMERAS:
            Image.fromarray(initial).save(reference / f"ordinary_{view}.png")
            Image.fromarray(initial).save(reference / f"pro_{view}.png")
        comparison.write_json(reference / "ordinary_reset.json", {"reference_only": True})
        cases.append(
            {
                "case_id": f"case{task}",
                "title": f"Task {task}",
                "reference_directory": str(reference),
                "display_view": "wrist" if task == 1 else "primary",
                **paths,
            }
        )
    selection = tmp_path / "selection.json"
    comparison.write_json(selection, {"cases": cases})
    output = tmp_path / "figure"
    savefig = plt.Figure.savefig

    def save_without_overlapping_labels(figure, *args, **kwargs):
        figure.canvas.draw()
        texts = list(figure.texts)
        for axis in figure.axes:
            texts.extend([axis.title, *axis.texts])
        bounds = [text.get_window_extent(figure.canvas.get_renderer()) for text in texts if text.get_text()]
        for index, box in enumerate(bounds):
            assert not any(box.overlaps(other) for other in bounds[index + 1 :])
        if str(args[0]).endswith("comparison_selected_views.pdf"):
            expected_wrist = np.asarray(
                Image.fromarray(expected[1, "ours"][:, ::-1]).resize((160, 160), Image.Resampling.LANCZOS)
            )
            np.testing.assert_array_equal(figure.axes[11].images[0].get_array(), expected_wrist)
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
    assert (output / "comparison_selected_views.pdf").is_file()
    assert (output / "perturbation_reference.pdf").is_file()
    metadata = json.loads((output / "cases_metadata.json").read_text())["cases"]
    assert [row["perturbation_type"] for row in metadata] == ["Env", "Env", "Object", "Object"]
    with (output / "cases_metadata.csv").open(newline="") as stream:
        assert [row["suite"] for row in csv.DictReader(stream)] == [row["suite"] for row in metadata]
    assert json.loads((output / "case0/metadata.json").read_text())["suite"] == "libero_goal_env"
    assert json.loads((output / "case1/metadata.json").read_text())["display_view"] == "wrist"
    np.testing.assert_array_equal(
        np.asarray(Image.open(output / "case0/ordinary_reference_primary.png")), initial
    )
    with pytest.raises(FileExistsError):
        comparison.figure(SimpleNamespace(selection=selection, output_dir=output))
