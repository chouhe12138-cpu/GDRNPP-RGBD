#!/usr/bin/env python3
"""Summarize the Stage 3C-0 local pilot without treating loss as pose accuracy."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


LOSS_KEYS = ("total_loss", "loss_PM_R", "loss_centroid", "loss_z")


def scalar(record: dict, key: str) -> float:
    value = record[key]
    if isinstance(value, list):
        value = value[0]
    return float(value)


def summarize(metrics_path: Path) -> dict:
    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise RuntimeError(f"No metric records in {metrics_path}")
    for key in LOSS_KEYS:
        if not all(key in record and math.isfinite(scalar(record, key)) for record in records):
            raise RuntimeError(f"Missing or non-finite {key}")

    boundary = max(len(records) // 4, 1)
    first = records[:boundary]
    last = records[-boundary:]
    losses = {}
    for key in LOSS_KEYS:
        first_values = [scalar(record, key) for record in first]
        last_values = [scalar(record, key) for record in last]
        first_median = statistics.median(first_values)
        last_median = statistics.median(last_values)
        losses[key] = {
            "first_quarter_median": first_median,
            "last_quarter_median": last_median,
            "relative_change_percent": 100.0
            * (last_median - first_median)
            / first_median,
            "minimum_recorded": min(scalar(record, key) for record in records),
            "maximum_recorded": max(scalar(record, key) for record in records),
        }

    return {
        "status": "PASS",
        "scope": "training-pipeline feasibility only; not pose-accuracy evidence",
        "metrics_path": str(metrics_path.resolve()),
        "record_count": len(records),
        "first_iteration": int(records[0]["iteration"]),
        "last_iteration": int(records[-1]["iteration"]),
        "all_losses_finite": True,
        "losses": losses,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = summarize(args.metrics)
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
