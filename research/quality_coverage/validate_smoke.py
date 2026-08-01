#!/usr/bin/env python3
"""Validate a completed Stage 3C-1 one-epoch smoke run."""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from pathlib import Path

import torch
from mmcv import Config

from core.gdrn_modeling.models.GDRN_double_mask import build_model_optimizer


CONFIG = Path(
    "configs/gdrn/lmo_pbr/convnext_stage3c1_quality_coverage_local_lmo.py"
)
QUALITY_PREFIX = "quality_coverage_net."
EXPECTED_OFFICIAL_TENSORS = 392
EXPECTED_NEW_TENSORS = 9
EXPECTED_FINAL_ITERATION = 2047


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    return {
        key.removeprefix("_module."): value
        for key, value in state.items()
    }


def metric_value(value: object) -> float:
    if isinstance(value, list):
        value = value[0]
    return float(value)


def load_metrics(path: Path) -> list[dict[str, object]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: validate_smoke.py OFFICIAL FINAL METRICS LOG"
        )
    official_path, final_path, metrics_path, log_path = map(Path, sys.argv[1:])

    records = load_metrics(metrics_path)
    if not records:
        raise RuntimeError("metrics.json is empty")
    final_iteration = int(records[-1]["iteration"])
    loss_values = [
        metric_value(record["total_loss"])
        for record in records
        if "total_loss" in record
    ]
    finite_losses = bool(loss_values) and all(math.isfinite(v) for v in loss_values)
    quarter = max(1, len(loss_values) // 4)
    first_median = statistics.median(loss_values[:quarter])
    last_median = statistics.median(loss_values[-quarter:])
    loss_change_percent = 100.0 * (last_median / first_median - 1.0)

    official = checkpoint_state(official_path)
    final = checkpoint_state(final_path)
    missing_official = sorted(set(official) - set(final))
    changed_official = sorted(
        key for key in official.keys() & final.keys()
        if not torch.equal(official[key], final[key])
    )
    new_tensors = sorted(set(final) - set(official))

    cfg = Config.fromfile(str(CONFIG))
    cfg.MODEL.DEVICE = "cpu"
    cfg.SOLVER.BASE_LR = float(cfg.SOLVER.OPTIMIZER_CFG.lr)
    model, _ = build_model_optimizer(cfg, is_test=True)
    incompatible = model.load_state_dict(final, strict=True)
    strict_reload = not incompatible.missing_keys and not incompatible.unexpected_keys

    branch_weights = (
        "quality_coverage_net.quality_net.3.weight",
        "quality_coverage_net.coverage_net.2.weight",
    )
    branch_nonzero = {
        key: key in final and bool(torch.count_nonzero(final[key]).item())
        for key in branch_weights
    }

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    peak_values = [int(value) for value in re.findall(r"max_mem:\s*(\d+)M", log_text)]

    checks = {
        "completed_2048_micro_batches": final_iteration == EXPECTED_FINAL_ITERATION,
        "finite_losses": finite_losses,
        "strict_checkpoint_reload": strict_reload,
        "official_tensor_count": len(official) == EXPECTED_OFFICIAL_TENSORS,
        "official_tensors_bit_identical": not missing_official and not changed_official,
        "new_tensor_count": len(new_tensors) == EXPECTED_NEW_TENSORS,
        "new_tensors_isolated": all(
            key.startswith(QUALITY_PREFIX) for key in new_tensors
        ),
        "quality_branch_updated": branch_nonzero[branch_weights[0]],
        "coverage_branch_updated": branch_nonzero[branch_weights[1]],
    }
    passed = all(checks.values())
    summary = {
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "final_iteration_zero_based": final_iteration,
        "metrics_records": len(records),
        "total_loss_first_quarter_median": first_median,
        "total_loss_last_quarter_median": last_median,
        "total_loss_change_percent": loss_change_percent,
        "peak_gpu_memory_mib": max(peak_values) if peak_values else None,
        "official_tensors": len(official),
        "changed_official_tensors": len(changed_official),
        "missing_official_tensors": len(missing_official),
        "new_quality_coverage_tensors": len(new_tensors),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
