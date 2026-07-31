#!/usr/bin/env python3
"""Finalize a completed utilization run from its durable CSV/BOP artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.pose_aggregation.run_diagnostic import aggregate, write_csv
from research.pose_head_utilization.run_utilization_diagnostic import (
    BOP_NAMES,
    EXPECTED_LMO_TARGETS,
    METHODS,
    baseline_reproduction,
    sha256,
)
from research.pose_head_utilization.utilization_utils import utilization_decision


FLOAT_FIELDS = {
    "visibility",
    "inlier_ratio",
    "median_reprojection_error_px",
    "solver_time_ms",
    "rotation_error_deg",
    "translation_error_mm",
    "add_s_m",
    "add_s_0.1d",
}
INT_FIELDS = {
    "scene_id",
    "im_id",
    "instance_id",
    "obj_id",
    "num_points",
    "num_inliers",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_instance_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            row = dict(source)
            row["success"] = row["success"].strip().lower() == "true"
            for field in INT_FIELDS:
                row[field] = int(row[field])
            for field in FLOAT_FIELDS:
                row[field] = float(row[field])
            rows.append(row)
    return rows


def load_verified_bop_scores(output_dir: Path) -> dict[str, dict]:
    manifest_path = output_dir / "bop_eval" / "bop_result_sha256.json"
    with manifest_path.open(encoding="utf-8") as handle:
        expected_hashes = json.load(handle)
    actual_hashes = {
        filename: sha256(output_dir / "bop_results" / filename)
        for filename in BOP_NAMES.values()
    }
    if actual_hashes != expected_hashes:
        raise RuntimeError("BOP result hashes differ; refusing to reuse stale scores")

    scores = {}
    for method, filename in BOP_NAMES.items():
        score_path = output_dir / "bop_eval" / Path(filename).stem / "scores_bop19.json"
        with score_path.open(encoding="utf-8") as handle:
            scores[method] = json.load(handle)
    return scores


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    rows = read_instance_rows(output_dir / "per_instance.csv")
    expected_rows = EXPECTED_LMO_TARGETS * len(METHODS)
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} instance rows, found {len(rows)}")
    if {row["method"] for row in rows} != set(METHODS):
        raise RuntimeError("Method set in per_instance.csv does not match the protocol")

    per_method = aggregate(rows, ("method",))
    per_object = aggregate(rows, ("method", "obj_id", "obj_name"))
    bop_scores = load_verified_bop_scores(output_dir)
    for item in per_method:
        score = bop_scores[item["method"]]
        item["bop_ar"] = float(score["bop19_average_recall"])
        item["vsd_ar"] = float(score["bop19_average_recall_vsd"])
        item["mssd_ar"] = float(score["bop19_average_recall_mssd"])
        item["mspd_ar"] = float(score["bop19_average_recall_mspd"])

    conclusion = utilization_decision(per_method, per_object)
    reproduction = baseline_reproduction(per_method)
    write_csv(output_dir / "per_object.csv", per_object)
    with (output_dir / "utilization_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "methods": per_method,
                "conclusion": conclusion,
                "baseline_reproduction": reproduction,
                "finalized_from_existing_artifacts": True,
            },
            handle,
            indent=2,
            allow_nan=True,
        )
    with (output_dir / "protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "experiment_id": "EXP-20260731-004-gdrnpp-pose-head-utilization",
                "processed_targets": EXPECTED_LMO_TARGETS,
                "methods": list(METHODS),
                "bop_evaluation": "computed_and_hash_verified",
                "finalized_from_existing_artifacts": True,
                "recovery_note": (
                    "Inference and all BOP evaluations completed before the original "
                    "process hit a summary-only NumPy bool serialization error. "
                    "Timing and alpha-zero re-entry maxima were not durably retained."
                ),
            },
            handle,
            indent=2,
        )
    print(json.dumps({"conclusion": conclusion, "baseline_reproduction": reproduction}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
