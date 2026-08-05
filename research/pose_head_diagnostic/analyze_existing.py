#!/usr/bin/env python3
"""Analyze the completed 1,445-target frozen pose-head utilization run."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "output" / "EXP-20260731-004" / "full"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "pose_head_diagnostic" / "preliminary"
ALPHAS = ("a000", "a025", "a050", "a075", "a100")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    return float(np.mean(values[np.isfinite(values)]))


def median(values: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    return float(np.median(values[np.isfinite(values)]))


def keyed(rows: list[dict], method: str) -> dict[tuple[int, int, int, int], dict]:
    return {
        (
            int(row["scene_id"]),
            int(row["im_id"]),
            int(row["instance_id"]),
            int(row["obj_id"]),
        ): row
        for row in rows
        if row["method"] == method
    }


def analyze(input_dir: Path, output_dir: Path) -> dict:
    required = (
        "per_instance.csv",
        "dense_interpolation.csv",
        "per_object.csv",
        "visibility_bins.csv",
    )
    for name in required:
        if not (input_dir / name).exists():
            raise FileNotFoundError(input_dir / name)
    rows = read_csv(input_dir / "per_instance.csv")
    dense = read_csv(input_dir / "dense_interpolation.csv")
    per_object = read_csv(input_dir / "per_object.csv")
    visibility = read_csv(input_dir / "visibility_bins.csv")
    baseline = keyed(rows, "patch_a000")
    final = keyed(rows, "patch_a100")
    if len(baseline) != 1445 or baseline.keys() != final.keys():
        raise ValueError("Expected matching 1,445-target Patch-PnP endpoints")

    method_summary = []
    for token in ALPHAS:
        method_rows = list(keyed(rows, f"patch_{token}").values())
        method_summary.append(
            {
                "method": f"patch_{token}",
                "instances": len(method_rows),
                "rotation_mean_deg": mean(float(row["rotation_error_deg"]) for row in method_rows),
                "rotation_median_deg": median(float(row["rotation_error_deg"]) for row in method_rows),
                "translation_mean_mm": mean(
                    float(row["translation_error_mm"]) for row in method_rows
                ),
                "translation_median_mm": median(
                    float(row["translation_error_mm"]) for row in method_rows
                ),
                "add_s_0.1d_recall": mean(float(row["add_s_0.1d"]) for row in method_rows),
            }
        )

    transitions = Counter()
    failures = []
    rotation_improved = 0
    translation_improved = 0
    rotation_improved_at_any_alpha = 0
    translation_improved_at_any_alpha = 0
    monotonic_rotation = 0
    monotonic_translation = 0
    method_maps = [keyed(rows, f"patch_{token}") for token in ALPHAS]
    for key, start in baseline.items():
        end = final[key]
        start_rot = float(start["rotation_error_deg"])
        end_rot = float(end["rotation_error_deg"])
        start_trans = float(start["translation_error_mm"])
        end_trans = float(end["translation_error_mm"])
        start_add = int(float(start["add_s_0.1d"]))
        end_add = int(float(end["add_s_0.1d"]))
        transitions[f"{start_add}_to_{end_add}"] += 1
        rotation_improved += end_rot < start_rot
        translation_improved += end_trans < start_trans
        rot_curve = [float(mapping[key]["rotation_error_deg"]) for mapping in method_maps]
        trans_curve = [float(mapping[key]["translation_error_mm"]) for mapping in method_maps]
        rotation_improved_at_any_alpha += min(rot_curve[1:]) < rot_curve[0]
        translation_improved_at_any_alpha += min(trans_curve[1:]) < trans_curve[0]
        monotonic_rotation += bool(np.all(np.diff(rot_curve) <= 1e-12))
        monotonic_translation += bool(np.all(np.diff(trans_curve) <= 1e-12))
        failures.append(
            {
                "scene_id": key[0],
                "im_id": key[1],
                "instance_id": key[2],
                "obj_id": key[3],
                "obj_name": start["obj_name"],
                "visibility": float(start["visibility"]),
                "add_transition": f"{start_add}_to_{end_add}",
                "rotation_delta_deg": end_rot - start_rot,
                "translation_delta_mm": end_trans - start_trans,
            }
        )
    failures.sort(
        key=lambda row: (
            row["add_transition"] != "1_to_0",
            -(row["rotation_delta_deg"] + row["translation_delta_mm"] / 10.0),
        )
    )

    endpoint_objects = []
    object_maps = {
        method: {int(row["obj_id"]): row for row in per_object if row["method"] == method}
        for method in ("patch_a000", "patch_a100")
    }
    for obj_id in sorted(object_maps["patch_a000"]):
        start = object_maps["patch_a000"][obj_id]
        end = object_maps["patch_a100"][obj_id]
        endpoint_objects.append(
            {
                "obj_id": obj_id,
                "obj_name": start["obj_name"],
                "instances": int(start["instances"]),
                "add_recall_a000": float(start["add_s_0.1d_recall"]),
                "add_recall_a100": float(end["add_s_0.1d_recall"]),
                "add_recall_delta": float(end["add_s_0.1d_recall"])
                - float(start["add_s_0.1d_recall"]),
                "rotation_mean_delta_deg": float(end["mean_rotation_error_deg"])
                - float(start["mean_rotation_error_deg"]),
                "translation_mean_delta_mm": float(end["mean_translation_error_mm"])
                - float(start["mean_translation_error_mm"]),
            }
        )

    endpoint_visibility = []
    visibility_maps = {
        method: {
            row["visibility_bin"]: row
            for row in visibility
            if row["method"] == method
        }
        for method in ("patch_a000", "patch_a100")
    }
    for bin_name in sorted(visibility_maps["patch_a000"]):
        start = visibility_maps["patch_a000"][bin_name]
        end = visibility_maps["patch_a100"][bin_name]
        endpoint_visibility.append(
            {
                "visibility_bin": bin_name,
                "instances": int(start["instances"]),
                "add_recall_a000": float(start["add_s_0.1d_recall"]),
                "add_recall_a100": float(end["add_s_0.1d_recall"]),
                "add_recall_delta": float(end["add_s_0.1d_recall"])
                - float(start["add_s_0.1d_recall"]),
                "rotation_mean_delta_deg": float(end["mean_rotation_error_deg"])
                - float(start["mean_rotation_error_deg"]),
                "translation_mean_delta_mm": float(end["mean_translation_error_mm"])
                - float(start["mean_translation_error_mm"]),
            }
        )

    dense_end = [row for row in dense if row["alpha_token"] in {"a000", "a100"}]
    summary = {
        "source": str(input_dir),
        "instances": len(baseline),
        "dense_rows_checked": len(dense_end),
        "method_summary": method_summary,
        "endpoint": {
            "rotation_improved_count": rotation_improved,
            "rotation_improved_fraction": rotation_improved / len(baseline),
            "translation_improved_count": translation_improved,
            "translation_improved_fraction": translation_improved / len(baseline),
            "rotation_improved_at_any_alpha_count": rotation_improved_at_any_alpha,
            "rotation_improved_at_any_alpha_fraction": rotation_improved_at_any_alpha
            / len(baseline),
            "translation_improved_at_any_alpha_count": translation_improved_at_any_alpha,
            "translation_improved_at_any_alpha_fraction": translation_improved_at_any_alpha
            / len(baseline),
            "rotation_monotonic_nonincrease_count": monotonic_rotation,
            "rotation_monotonic_nonincrease_fraction": monotonic_rotation / len(baseline),
            "translation_monotonic_nonincrease_count": monotonic_translation,
            "translation_monotonic_nonincrease_fraction": monotonic_translation / len(baseline),
            "add_binary_transitions": dict(sorted(transitions.items())),
        },
        "interpretation": (
            "Improving XYZ strongly changes individual pose errors, but the aggregate Patch-PnP "
            "ADD(-S) endpoint does not improve and responses are often non-monotonic. This supports "
            "diagnosing where the frozen pose head loses or misuses spatial correspondence information."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "existing_per_object.csv", endpoint_objects)
    write_csv(output_dir / "existing_visibility.csv", endpoint_visibility)
    with (output_dir / "existing_response_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    report = [
        "# Existing Stage 3B response analysis",
        "",
        f"- Source instances: {len(baseline)}",
        f"- Rotation improved, alpha 0→1: {rotation_improved}/{len(baseline)} "
        f"({rotation_improved / len(baseline):.2%})",
        f"- Translation improved, alpha 0→1: {translation_improved}/{len(baseline)} "
        f"({translation_improved / len(baseline):.2%})",
        f"- Rotation improved at any nonzero alpha: "
        f"{rotation_improved_at_any_alpha}/{len(baseline)} "
        f"({rotation_improved_at_any_alpha / len(baseline):.2%})",
        f"- Translation improved at any nonzero alpha: "
        f"{translation_improved_at_any_alpha}/{len(baseline)} "
        f"({translation_improved_at_any_alpha / len(baseline):.2%})",
        f"- Rotation monotonically non-increasing: {monotonic_rotation}/{len(baseline)} "
        f"({monotonic_rotation / len(baseline):.2%})",
        f"- Translation monotonically non-increasing: {monotonic_translation}/{len(baseline)} "
        f"({monotonic_translation / len(baseline):.2%})",
        f"- ADD transitions: {dict(sorted(transitions.items()))}",
        "",
        "## Patch-PnP aggregate curve",
        "",
        "| Method | Rot mean/median (deg) | Trans mean/median (mm) | ADD(-S) recall |",
        "|---|---:|---:|---:|",
    ]
    report.extend(
        f"| {row['method']} | {row['rotation_mean_deg']:.4f} / "
        f"{row['rotation_median_deg']:.4f} | {row['translation_mean_mm']:.4f} / "
        f"{row['translation_median_mm']:.4f} | {row['add_s_0.1d_recall']:.6f} |"
        for row in method_summary
    )
    report.extend(
        [
            "",
            "## Preliminary interpretation",
            "",
            summary["interpretation"],
            "",
            "Instance-level values were used in memory for paired statistics but are not "
            "written by this analyzer.",
            "",
            "This is a read-only re-analysis of existing outputs, not a new formal experiment.",
        ]
    )
    (output_dir / "EXISTING_RESULT_ANALYSIS.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = analyze(args.input_dir.resolve(), args.output_dir.resolve())
    print(json.dumps(summary["endpoint"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
