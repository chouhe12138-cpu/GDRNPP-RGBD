"""Compare canonical normalized metrics using an experiment's frozen gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json
from .manifest import read_json, validate_experiment


def load_normalized_metrics(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema_version") != 1 or not isinstance(payload.get("metrics"), dict):
        raise ValueError(f"invalid normalized metric file: {path}")
    return payload


def compare_screening_metrics(
    experiment: dict[str, Any],
    baseline: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    validate_experiment(experiment)
    for field in ("dataset_id", "bbox_type"):
        if baseline.get(field) != result.get(field):
            raise ValueError(
                f"baseline/result {field} mismatch: {baseline.get(field)} != {result.get(field)}"
            )
    gate = experiment["protocol"].get("gate")
    if not isinstance(gate, dict):
        raise ValueError("experiment protocol has no screening gate")

    bop_id = "bop19_ar_macro"
    add_id = "add_s_0.1d_macro_object"
    baseline_bop = float(baseline["metrics"][bop_id]["value"])
    result_bop = float(result["metrics"][bop_id]["value"])
    baseline_add_record = baseline["metrics"][add_id]
    result_add_record = result["metrics"][add_id]
    baseline_add = float(baseline_add_record["value"])
    result_add = float(result_add_record["value"])
    baseline_objects = {
        str(key): float(value)
        for key, value in baseline_add_record.get("object_recalls", {}).items()
    }
    result_objects = {
        str(key): float(value)
        for key, value in result_add_record.get("object_recalls", {}).items()
    }
    if set(baseline_objects) != set(result_objects) or not baseline_objects:
        raise ValueError("baseline/result must contain the same non-empty object recall set")
    object_deltas = {
        object_id: result_objects[object_id] - baseline_objects[object_id]
        for object_id in sorted(baseline_objects, key=int)
    }
    nonnegative = sum(delta >= 0.0 for delta in object_deltas.values())
    bop_delta = result_bop - baseline_bop
    add_delta = result_add - baseline_add
    checks = {
        "bop19_ar_macro": bop_delta >= float(gate["minimum_bop19_ar_macro_delta"]),
        "add_s_0.1d_macro_object": add_delta
        >= float(gate["minimum_add_s_0.1d_macro_object_delta"]),
        "nonnegative_objects": nonnegative >= int(gate["minimum_nonnegative_objects"]),
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "experiment_id": experiment["experiment_id"],
        "status": "SCREEN_PASS" if passed else "SCREEN_FAIL",
        "dataset_id": result["dataset_id"],
        "bbox_type": result["bbox_type"],
        "baseline_checkpoint_id": baseline.get("checkpoint_id"),
        "result_checkpoint_id": result.get("checkpoint_id"),
        "metrics": {
            bop_id: {
                "baseline": baseline_bop,
                "result": result_bop,
                "delta": bop_delta,
                "delta_percentage_points": bop_delta * 100.0,
            },
            add_id: {
                "baseline": baseline_add,
                "result": result_add,
                "delta": add_delta,
                "delta_percentage_points": add_delta * 100.0,
            },
        },
        "object_add_s_deltas": object_deltas,
        "nonnegative_objects": nonnegative,
        "gate": gate,
        "gate_checks": checks,
    }


def write_screening_summary(output_path: Path, payload: dict[str, Any]) -> None:
    if output_path.exists():
        raise FileExistsError(f"summary already exists: {output_path}")
    atomic_write_json(output_path, payload)
