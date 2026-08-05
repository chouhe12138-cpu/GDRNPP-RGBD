#!/usr/bin/env python3
"""Revalidate a computed full run after quality-gate calibration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from research.pose_head_diagnostic.diagnostic_utils import CONDITIONS
from research.pose_head_diagnostic.run_statistical_diagnostic import (
    EXPECTED_OFFICIAL_ADD_RECALL,
    EXPECTED_OFFICIAL_BOP_AR,
    EXPECTED_OFFICIAL_HASH,
    EXPERIMENT_ID,
    OFFICIAL_BOP_TOLERANCE,
    POSE_ROTATION_REENTRY_TOLERANCE_CUDA,
    POSE_TRANSLATION_REENTRY_TOLERANCE_CUDA,
    RAW_REENTRY_TOLERANCE_CUDA,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def output_hashes(output_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(output_dir)): sha256(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "hashes.sha256"
    }


def assess_full_quality_control(qc: dict, baseline_bop: float) -> dict:
    expected_nonfinite = int(qc["empty_support_targets"]) * len(CONDITIONS)
    unexpected_nonfinite = int(qc["nonfinite_scalar_count"]) - expected_nonfinite
    add_reproduced = (
        abs(float(qc["baseline_add_s_0.1d_recall"]) - EXPECTED_OFFICIAL_ADD_RECALL)
        <= 1e-12
    )
    bop_reproduced = (
        abs(float(baseline_bop) - EXPECTED_OFFICIAL_BOP_AR)
        <= OFFICIAL_BOP_TOLERANCE
    )
    passed = bool(
        int(qc["processed_targets"]) == 1445
        and all(int(count) == 1445 for count in qc["condition_counts"].values())
        and bool(qc["state_unchanged"])
        and unexpected_nonfinite == 0
        and float(qc["max_baseline_raw_rotation_abs_error"])
        <= RAW_REENTRY_TOLERANCE_CUDA
        and float(qc["max_baseline_raw_translation_abs_error"])
        <= RAW_REENTRY_TOLERANCE_CUDA
        and float(qc["max_baseline_rotation_abs_error"])
        <= POSE_ROTATION_REENTRY_TOLERANCE_CUDA
        and float(qc["max_baseline_translation_abs_error"])
        <= POSE_TRANSLATION_REENTRY_TOLERANCE_CUDA
        and add_reproduced
        and bop_reproduced
    )
    return {
        **qc,
        "expected_nonfinite_scalar_count": expected_nonfinite,
        "unexpected_nonfinite_scalar_count": unexpected_nonfinite,
        "raw_reentry_tolerance": RAW_REENTRY_TOLERANCE_CUDA,
        "rotation_reentry_tolerance": POSE_ROTATION_REENTRY_TOLERANCE_CUDA,
        "translation_reentry_tolerance": POSE_TRANSLATION_REENTRY_TOLERANCE_CUDA,
        "expected_official_add_s_0.1d_micro_recall": EXPECTED_OFFICIAL_ADD_RECALL,
        "expected_official_bop_ar": EXPECTED_OFFICIAL_BOP_AR,
        "official_bop_tolerance": OFFICIAL_BOP_TOLERANCE,
        "official_add_reproduced": add_reproduced,
        "official_bop_reproduced": bop_reproduced,
        "revalidated_from_complete_artifacts": True,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()

    protocol = read_json(output_dir / "protocol.json")
    qc = read_json(output_dir / "quality_control.json")
    run_state = read_json(output_dir / "run_state.json")
    bop_summary = read_json(output_dir / "bop_score_summary.json")
    recorded_hashes = read_json(output_dir / "hashes.sha256")

    expected_protocol = {
        "experiment_id": EXPERIMENT_ID,
        "mode": "full",
        "model_role": "official",
        "seed": 20260804,
        "checkpoint_sha256": EXPECTED_OFFICIAL_HASH,
        "precision": "FP32",
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise RuntimeError(
                f"Protocol mismatch for {key}: {protocol.get(key)!r} != {expected!r}"
            )
    if run_state.get("status") != "FAILED":
        raise RuntimeError("Revalidation requires the preserved FAILED run state")

    for relative_path, expected_hash in recorded_hashes.items():
        path = output_dir / relative_path
        if not path.is_file() or sha256(path) != expected_hash:
            raise RuntimeError(f"Precalibration hash mismatch: {relative_path}")

    scores = bop_summary["scores"]
    pose_hashes = bop_summary["pose_file_sha256"]
    if set(scores) != set(CONDITIONS):
        raise RuntimeError("BOP score conditions are incomplete")
    for condition in CONDITIONS:
        stem = condition.replace("_", "") + "_lmo-test"
        pose_name = stem + ".csv"
        pose_path = output_dir / "bop_results" / pose_name
        if sha256(pose_path) != pose_hashes[pose_name]:
            raise RuntimeError(f"Pose CSV hash mismatch: {pose_name}")
        score_path = output_dir / "bop_eval" / stem / "scores_bop19.json"
        score = float(read_json(score_path)["bop19_average_recall"])
        if abs(score - float(scores[condition])) > 1e-12:
            raise RuntimeError(f"BOP score mismatch: {condition}")

    with (output_dir / "overall_condition_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        baseline_add = next(
            float(row["mean"])
            for row in csv.DictReader(handle)
            if row["condition"] == "baseline" and row["metric"] == "add_s_0.1d"
        )
    if abs(baseline_add - float(qc["baseline_add_s_0.1d_recall"])) > 1e-12:
        raise RuntimeError("Baseline ADD summary does not match quality control")

    calibrated = assess_full_quality_control(qc, float(scores["baseline"]))
    if not calibrated["passed"]:
        raise RuntimeError("Calibrated quality control still fails")

    qc_backup = output_dir / "quality_control_precalibration.json"
    state_backup = output_dir / "run_state_precalibration.json"
    if qc_backup.exists() or state_backup.exists():
        raise RuntimeError("Precalibration backups already exist")
    write_json(qc_backup, qc)
    write_json(state_backup, run_state)
    write_json(output_dir / "quality_control.json", calibrated)
    write_json(
        output_dir / "run_state.json",
        {
            "status": "COMPLETE",
            "model_role": "official",
            "mode": "full",
            "revalidated_from_complete_artifacts": True,
        },
    )
    write_json(output_dir / "hashes.sha256", output_hashes(output_dir))
    print(json.dumps(calibrated, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
