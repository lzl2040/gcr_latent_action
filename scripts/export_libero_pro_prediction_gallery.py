"""Archive dense, source-linked LIBERO-PRO predictions for local visual screening."""

import argparse
import csv
import html
import json
import shutil
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image

from scripts import compare_libero_pro_closed_loop as common


def read_pair(case, require_advantage):
    records = {
        name: json.loads((Path(case[f"{name}_directory"]) / "result.json").read_text())
        for name in ("ours", "cosmos3")
    }
    common.check_pair(records["ours"], records["cosmos3"], require_advantage=require_advantage)
    if case["case_id"] != records["ours"]["case_id"]:
        raise ValueError("Gallery case identity does not match the measured rollout.")
    for record in records.values():
        common.check_snapshots(record, expected_count=None)
        for snapshot in record["snapshots"]:
            prediction = Path(snapshot["prediction_directory"]) / "frames.npz"
            if common.digest(prediction) != snapshot["prediction_sha256"]:
                raise ValueError("A decoded prediction changed.")
            if common.digest(snapshot["condition_file"]) != snapshot["condition_sha256"]:
                raise ValueError("A conditioning observation changed.")
    for view in common.CAMERAS:
        initial = [
            np.asarray(Image.open(Path(case[f"{name}_directory"]) / f"initial_{view}.png"))
            for name in records
        ]
        np.testing.assert_array_equal(*initial)
        if "reference_directory" in case:
            np.testing.assert_array_equal(
                initial[0],
                np.asarray(Image.open(Path(case["reference_directory"]) / f"pro_{view}.png")),
            )
    return records


def check_task_diversity(pairs, allow_repeated_base_tasks):
    tasks = [pair["ours"]["task"] for pair in pairs]
    bases = [(task["base_suite"], task["task_id"]) for task in tasks]
    variants = [(*base, task["category"]) for base, task in zip(bases, tasks, strict=True)]
    if len(set(bases)) != len(bases) and not allow_repeated_base_tasks:
        raise ValueError("Use different base tasks, not multiple variants of one task.")
    if len(set(variants)) != len(variants):
        raise ValueError("Repeated base tasks must have different perturbation categories.")
    return {
        "different_base_tasks": len(set(bases)),
        "different_task_category_variants": len(set(variants)),
        "allow_repeated_base_tasks": allow_repeated_base_tasks,
        "repeated_base_tasks": [
            {"base_suite": suite, "task_id": task_id}
            for suite, task_id in sorted(set(bases))
            if bases.count((suite, task_id)) > 1
        ],
    }


def archive_case(case, records, directory, *, repeated_base_task=False):
    directory.mkdir()
    ours = records["ours"]
    details = ours["perturbation"]
    metadata = {
        "case_id": case["case_id"],
        "base_suite": ours["task"]["base_suite"],
        "suite": ours["task"]["suite"],
        "task_id": ours["task"]["task_id"],
        "base_task_reused_with_different_category": repeated_base_task,
        "trial": ours["trial"],
        "sampling_seed": ours.get("seed"),
        "category": ours["task"]["category"],
        "instruction": ours["task"]["instruction"],
        "summary": details["summary"],
        "ours_success": ours["success"],
        "cosmos3_success": records["cosmos3"]["success"],
        "ours_steps": ours["steps"],
        "cosmos3_steps": records["cosmos3"]["steps"],
        "ours_success_cosmos3_failure": ours["success"] and not records["cosmos3"]["success"],
        "display_view": case.get("display_view", "primary"),
        "perturbation": details,
        "sources": {name: case[f"{name}_directory"] for name in records},
        "tiles": [],
    }
    for name, record in records.items():
        source = Path(case[f"{name}_directory"])
        audit = directory / "execution_audit" / name
        audit.mkdir(parents=True)
        generated = directory / "generated_originals" / name
        generated.mkdir(parents=True)
        for item in source.iterdir():
            if item.name.startswith("prediction_step_"):
                continue
            destination = audit / item.name
            if item.is_dir():
                shutil.copytree(item, destination)
            else:
                shutil.copy2(item, destination)
        for snapshot in record["snapshots"]:
            prediction = Path(snapshot["prediction_directory"])
            copied = generated / prediction.name
            shutil.copytree(prediction, copied)
            with np.load(copied / "frames.npz", allow_pickle=False) as data:
                for view in common.CAMERAS:
                    frames = data[view]
                    size = 160 if name == "cosmos3" else 256
                    if frames.shape != (len(snapshot["decoded_source_offsets"]), size, size, 3):
                        raise ValueError(
                            "Require the original decoded resolution and complete source frames."
                        )
                    raw = frames[snapshot["display_frame_index"]]
                    display = (
                        raw
                        if name == "cosmos3"
                        else np.asarray(Image.fromarray(raw).resize((160, 160), Image.Resampling.LANCZOS))
                    )
                    path = directory / "display" / name / view / f"step_{snapshot['control_step']:04d}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(display).save(path)
                    metadata["tiles"].append(
                        {
                            "model": name,
                            "view": view,
                            "control_step": snapshot["control_step"],
                            "condition_s": snapshot["forecast_start_s"],
                            "forecast_s": snapshot["forecast_end_s"],
                            "source_offset_steps": 16,
                            "source_frame_index": snapshot["display_frame_index"],
                            "source_prediction_directory": str(copied.relative_to(directory)),
                            "source_prediction_sha256": snapshot["prediction_sha256"],
                            "source_condition_file": str(
                                (audit / Path(snapshot["condition_file"]).name).relative_to(directory)
                            ),
                            "condition_sha256": snapshot["condition_sha256"],
                            "display_file": str(path.relative_to(directory)),
                            "display_sha256": common.digest(path),
                            "origin": "joint_model_video_latents_decoded_by_vae",
                        }
                    )
    inputs = directory / "inputs"
    inputs.mkdir()
    for view in common.CAMERAS:
        shutil.copy2(Path(case["ours_directory"]) / f"initial_{view}.png", inputs / f"initial_{view}.png")
    if "reference_directory" in case:
        shutil.copytree(case["reference_directory"], directory / "environment_references")
    common.write_json(directory / "metadata.json", metadata)
    return metadata


def model_tiles(metadata, name, view):
    return [tile for tile in metadata["tiles"] if tile["model"] == name and tile["view"] == view]


def timeline(metadata, directory, view):
    import matplotlib.pyplot as plt

    columns = max(4, 1 + max(len(model_tiles(metadata, name, view)) for name in ("cosmos3", "ours")))
    fig, axes = plt.subplots(2, columns, figsize=(1.65 * columns + 1, 5.3), squeeze=False)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.20, top=0.74, hspace=0.30, wspace=0.055)
    for row, name in enumerate(("cosmos3", "ours")):
        initial = np.asarray(Image.open(directory / "inputs" / f"initial_{view}.png"))
        images = [initial] + [
            np.asarray(Image.open(directory / tile["display_file"]))
            for tile in model_tiles(metadata, name, view)
        ]
        labels = ["Input (environment)"] + [
            f"{tile['condition_s']:.1f} -> {tile['forecast_s']:.1f} s"
            for tile in model_tiles(metadata, name, view)
        ]
        for column, axis in enumerate(axes[row]):
            axis.set_axis_off()
            if column < len(images):
                axis.imshow(images[column], interpolation="nearest")
                axis.set_title(labels[column], fontsize=8, pad=3)
        outcome = "success" if metadata[f"{name}_success"] else "failure"
        axes[row, 0].text(
            -0.10,
            0.5,
            f"{'Ours' if name == 'ours' else 'Cosmos3'}\n{outcome}\n{metadata[f'{name}_steps']} steps",
            transform=axes[row, 0].transAxes,
            va="center",
            ha="right",
            fontsize=9,
            color="#237441" if metadata[f"{name}_success"] else "#a63726",
        )
    reuse = (
        " | base task reused across perturbations"
        if metadata["base_task_reused_with_different_category"]
        else ""
    )
    fig.text(0.075, 0.955, f"{metadata['case_id']} | {view}{reuse}", fontsize=11, weight="bold")
    fig.text(
        0.075,
        0.895,
        textwrap.fill(
            f"{metadata['suite']} | {metadata['category']} | {metadata['summary']}",
            width=int((fig.get_figwidth() - 1.6) * 13),
        ),
        fontsize=9,
    )
    fig.text(
        0.075,
        0.80,
        textwrap.fill(metadata["instruction"], width=int((fig.get_figwidth() - 1.6) * 13)),
        fontsize=10,
    )
    fig.text(
        0.075,
        0.075,
        "Future tiles are VAE-decoded joint model predictions (+0.8 s), not environment execution frames.\n"
        "Each row uses its own closed-loop observations and times; columns do not match absolute control time.\n"
        "Environment-verified outcomes; hand-selected MLP-baseline system comparison, not a Stage-I grounding ablation.",
        fontsize=8,
    )
    return fig


def overview(rows, root, view, title):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), 7, figsize=(14, 3.2 * len(rows) + 1.4), squeeze=False)
    height_points = fig.get_figheight() * 72
    fig.subplots_adjust(
        left=0.025,
        right=0.995,
        top=0.86 - 48 / height_points,
        bottom=0.10,
        hspace=0.95,
        wspace=0.04,
    )
    fig.text(0.04, 0.97, f"{title} | {view}", weight="bold", fontsize=12)
    fig.text(
        0.04,
        0.92,
        "Input (environment)       Cosmos3: generated futures                     Ours: generated futures",
        fontsize=11,
    )
    for index, row in enumerate(rows):
        directory = root / row["relative_directory"]
        images = [np.asarray(Image.open(directory / "inputs" / f"initial_{view}.png"))]
        labels = ["Input"]
        for name in ("cosmos3", "ours"):
            tiles = model_tiles(row, name, view)
            selected = common.three_replans(len(tiles)) if len(tiles) >= 3 else list(range(len(tiles)))
            for selected_index in selected:
                tile = tiles[selected_index]
                images.append(np.asarray(Image.open(directory / tile["display_file"])))
                labels.append(f"{tile['condition_s']:.1f} -> {tile['forecast_s']:.1f} s")
            for _ in range(3 - len(selected)):
                images.append(None)
                labels.append("No additional replan")
        for axis, image, label in zip(axes[index], images, labels, strict=True):
            axis.set_axis_off()
            if image is not None:
                axis.imshow(image, interpolation="nearest")
            axis.set_title(label, fontsize=8, pad=3)
        top = axes[index, 0].get_position().y1
        fig.text(0.025, top + 48 / height_points, row["case_id"], weight="bold", fontsize=9)
        fig.text(0.025, top + 33 / height_points, textwrap.shorten(row["instruction"], width=125), fontsize=9)
        outcomes = " | ".join(
            f"{'Ours' if name == 'ours' else 'Cosmos3'}: {'success' if row[f'{name}_success'] else 'failure'}"
            for name in ("cosmos3", "ours")
        )
        reuse = (
            " | base task reused across perturbations"
            if row.get("base_task_reused_with_different_category")
            else ""
        )
        fig.text(
            0.025,
            top + 19 / height_points,
            f"{row['suite']} | {outcomes}{reuse}",
            fontsize=8,
            color="#35485c",
        )
    fig.text(
        0.025,
        0.025,
        "Future tiles are generated +0.8 s forecasts from different closed-loop observations, not simulator frames.\n"
        "Outcomes are newly environment-verified. Hand-selected system comparison; not a Stage-I causal ablation.",
        fontsize=8,
    )
    return fig


def export_gallery(args):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    selection = json.loads(args.selection.read_text())
    if not selection["cases"]:
        raise ValueError("Select at least one measured comparison.")
    pairs = [read_pair(case, not args.include_other_outcomes) for case in selection["cases"]]
    diversity = check_task_diversity(pairs, args.allow_repeated_base_tasks)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "cases").mkdir()
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "font.size": 9})
    rows = []
    with (
        PdfPages(args.output_dir / "all_cases_primary.pdf") as primary_pdf,
        PdfPages(args.output_dir / "all_cases_wrist.pdf") as wrist_pdf,
    ):
        for index, (case, pair) in enumerate(zip(selection["cases"], pairs, strict=True), 1):
            relative = Path("cases") / f"{index:02d}_{case['case_id']}"
            directory = args.output_dir / relative
            row = archive_case(
                case,
                pair,
                directory,
                repeated_base_task={
                    "base_suite": pair["ours"]["task"]["base_suite"],
                    "task_id": pair["ours"]["task"]["task_id"],
                }
                in diversity["repeated_base_tasks"],
            )
            row["relative_directory"] = str(relative)
            rows.append(row)
            for view, pdf in (("primary", primary_pdf), ("wrist", wrist_pdf)):
                fig = timeline(row, directory, view)
                fig.savefig(directory / f"comparison_{view}.png", dpi=160)
                fig.savefig(directory / f"comparison_{view}.pdf")
                pdf.savefig(fig)
                plt.close(fig)
    for category in ("position", "environment", "object"):
        category_rows = [row for row in rows if row["category"] == category]
        for offset in range(0, len(category_rows), 4):
            for view in common.CAMERAS:
                fig = overview(
                    category_rows[offset : offset + 4],
                    args.output_dir,
                    view,
                    f"{category.title()} variations | page {offset // 4 + 1}",
                )
                name = f"overview_{category}_{offset // 4 + 1:02d}_{view}"
                fig.savefig(args.output_dir / f"{name}.png", dpi=160)
                fig.savefig(args.output_dir / f"{name}.pdf")
                plt.close(fig)
    common.write_json(args.output_dir / "selection.json", selection)
    common.write_json(args.output_dir / "cases_metadata.json", {"task_diversity": diversity, "cases": rows})
    columns = [key for key in rows[0] if key not in ("tiles", "perturbation", "sources")]
    with (args.output_dir / "cases_metadata.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    links = []
    for row in rows:
        relative = html.escape(row["relative_directory"], quote=True)
        reuse = (
            " | Base task reused with a different perturbation category"
            if row["base_task_reused_with_different_category"]
            else ""
        )
        links.append(
            f"<h2>{html.escape(row['case_id'])}</h2><p>{html.escape(row['instruction'])}</p>"
            f"<p>{html.escape(row['suite'])} | Ours: {row['ours_success']} | Cosmos3: {row['cosmos3_success']}{reuse}</p>"
            f'<a href="{relative}/comparison_primary.pdf">Primary PDF</a> | '
            f'<a href="{relative}/comparison_wrist.pdf">Wrist PDF</a> | '
            f'<a href="{relative}/metadata.json">Metadata and source frame links</a>'
            f'<p><a href="{relative}/comparison_primary.png"><img style="width:100%" '
            f'src="{relative}/comparison_primary.png"></a></p>'
        )
    (args.output_dir / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>LIBERO-PRO generated prediction screening</title>"
        "<body style='max-width:1700px;margin:20px auto;font:15px sans-serif'>"
        "<h1>LIBERO-PRO generated prediction screening</h1>"
        f"<p>{len(rows)} comparisons across {diversity['different_base_tasks']} different base tasks. "
        "Any repeated base task uses a different perturbation category and is explicitly labeled.</p>"
        "<p>Only future tiles are model generated. Initial inputs, independent perturbation references and "
        "execution audits are environment renders. Each future is a +0.8 s forecast from its own replanning "
        "observation. This is a hand-selected system comparison, not a Stage-I grounding ablation.</p>"
        + "\n".join(links)
        + "</body>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-other-outcomes", action="store_true")
    parser.add_argument(
        "--allow-repeated-base-tasks",
        action="store_true",
        help="Allow multiple perturbation categories of a base task, never repeated trials of the same category.",
    )
    export_gallery(parser.parse_args())


if __name__ == "__main__":
    main()
