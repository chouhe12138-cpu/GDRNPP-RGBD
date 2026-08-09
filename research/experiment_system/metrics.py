"""Canonical metric definitions and strict BOP evaluator result indexing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json
from .manifest import sha256_file


@dataclass(frozen=True)
class MetricDefinition:
    metric_id: str
    display_name: str
    aggregation: str
    unit: str
    machine_scale: str = "fraction"


METRICS = {
    definition.metric_id: definition
    for definition in (
        MetricDefinition(
            "bop19_ar_macro",
            "BOP19 Average Recall",
            "official BOP error/threshold macro average",
            "ratio",
        ),
        MetricDefinition(
            "add_s_0.1d_macro_object",
            "ADD(-S)@0.1d macro-object recall",
            "mean recall across evaluated object classes",
            "ratio",
        ),
        MetricDefinition(
            "add_s_0.1d_micro_target",
            "ADD(-S)@0.1d micro-target recall",
            "successes divided by evaluated target instances",
            "ratio",
        ),
        MetricDefinition(
            "rotation_error_deg_mean",
            "Mean rotation error",
            "mean across target instances",
            "degree",
            "native",
        ),
        MetricDefinition(
            "translation_error_mm_mean",
            "Mean translation error",
            "mean across target instances",
            "millimetre",
            "native",
        ),
    )
}

LEGACY_ALIASES = {
    "bop_ar": "bop19_ar_macro",
    "add_s_0.1d": "add_s_0.1d_macro_object",
}


def metric_registry_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "metrics": {key: asdict(value) for key, value in sorted(METRICS.items())},
        "legacy_aliases": LEGACY_ALIASES,
    }


def _exactly_one(paths: list[Path], description: str) -> Path:
    unique = sorted(set(path.resolve() for path in paths))
    if len(unique) != 1:
        rendered = [str(path) for path in unique]
        raise RuntimeError(f"expected exactly one {description}, found {len(unique)}: {rendered}")
    return unique[0]


def discover_bop_scores(evaluation_root: Path) -> tuple[Path, Path]:
    """Find raw BOP and ADD scores once; later readers use the generated index."""

    evaluation_root = evaluation_root.resolve()
    bop_path = _exactly_one(
        list(evaluation_root.rglob("scores_bop19.json")),
        "scores_bop19.json",
    )
    result_root = bop_path.parent
    add_candidates: list[Path] = []
    for directory_pattern in ("error=ad_ntop=*", "error:ad_ntop:*"):
        for score_name in (
            "scores_th=0.100_min-visib=-1.000.json",
            "scores_th:0.100_min-visib:-1.000.json",
        ):
            add_candidates.extend(result_root.glob(f"{directory_pattern}/{score_name}"))
    add_path = _exactly_one(add_candidates, "ADD(-S)@0.1d score")
    return bop_path, add_path


def index_bop_evaluation(
    evaluation_root: Path,
    dataset_id: str,
    bbox_type: str,
    checkpoint_id: str,
    write: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if bbox_type not in {"gt", "det"}:
        raise ValueError("bbox_type must be 'gt' or 'det'")
    root = evaluation_root.resolve()
    bop_path, add_path = discover_bop_scores(root)
    bop = json.loads(bop_path.read_text(encoding="utf-8"))
    add = json.loads(add_path.read_text(encoding="utf-8"))
    bop_value = float(bop["bop19_average_recall"])
    if "mean_obj_recall" not in add:
        raise ValueError(
            "ADD(-S) score is missing mean_obj_recall; refusing to label "
            "target-level recall as macro-object recall"
        )
    add_macro_value = float(add["mean_obj_recall"])
    add_micro_value = float(add["recall"])
    for metric_id, value in (
        ("bop19_ar_macro", bop_value),
        ("add_s_0.1d_macro_object", add_macro_value),
        ("add_s_0.1d_micro_target", add_micro_value),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{metric_id} must be stored as a fraction in [0, 1], got {value}")

    index = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "bbox_type": bbox_type,
        "checkpoint_id": checkpoint_id,
        "raw_files": {
            "bop19": {
                "path": str(bop_path.relative_to(root)),
                "sha256": sha256_file(bop_path),
            },
            "add_s_0.1d": {
                "path": str(add_path.relative_to(root)),
                "sha256": sha256_file(add_path),
            },
        },
    }
    normalized = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "bbox_type": bbox_type,
        "checkpoint_id": checkpoint_id,
        "metrics": {
            "bop19_ar_macro": {
                "value": bop_value,
                "unit": "ratio",
                "source": "bop19",
            },
            "add_s_0.1d_macro_object": {
                "value": add_macro_value,
                "unit": "ratio",
                "source": "add_s_0.1d",
                "object_recalls": add.get("obj_recalls", {}),
            },
            "add_s_0.1d_micro_target": {
                "value": add_micro_value,
                "unit": "ratio",
                "source": "add_s_0.1d",
                "targets_count": add.get("targets_count"),
                "gt_count": add.get("gt_count"),
                "tp_count": add.get("tp_count"),
            },
        },
    }
    if write:
        atomic_write_json(root / "evaluation_index.json", index)
        atomic_write_json(root / "metrics.normalized.json", normalized)
    return index, normalized


def verify_indexed_evaluation(evaluation_root: Path) -> None:
    root = evaluation_root.resolve()
    index_path = root / "evaluation_index.json"
    metrics_path = root / "metrics.normalized.json"
    if not index_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError("evaluation index or normalized metrics is missing")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for record in index.get("raw_files", {}).values():
        raw_path = root / record["path"]
        if not raw_path.is_file() or sha256_file(raw_path) != record["sha256"]:
            raise RuntimeError(f"indexed evaluator result changed or is missing: {raw_path}")
