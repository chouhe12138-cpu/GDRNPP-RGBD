#!/usr/bin/env python3
"""Apply the frozen Stage 3C-1 screening gate to completed LM-O evaluations."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from research.quality_coverage.plot_curves import load_epoch_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_output", type=Path)
    parser.add_argument("baseline_output", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_one(root: Path, suffix: str):
    paths = sorted(glob.glob(str(root / "**" / suffix), recursive=True))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one {suffix} below {root}, found {len(paths)}")
    return json.loads(Path(paths[0]).read_text(encoding="utf-8"))


def select_best(rows):
    best = None
    for row in rows:
        if best is None:
            best = row
            continue
        primary_gain = row["bop_ar"] - best["bop_ar"]
        if primary_gain > 0.001 or (
            abs(primary_gain) <= 0.001
            and float(row.get("add_s") or float("-inf"))
            > float(best.get("add_s") or float("-inf"))
        ):
            best = row
    return best


def main() -> int:
    args = parse_args()
    baseline_bop = load_one(args.baseline_output, "scores_bop19.json")
    baseline_add = load_one(
        args.baseline_output,
        "error=ad_ntop=*/scores_th=0.100_min-visib=-1.000.json",
    )
    epoch_rows = load_epoch_scores(args.training_output)
    if not epoch_rows:
        raise RuntimeError(f"No periodic LM-O evaluations found below {args.training_output}")
    best = select_best(epoch_rows)
    if best["add_s"] is None:
        raise RuntimeError("Best evaluation is missing ADD(-S)@0.1d")

    baseline_objects = baseline_add.get("obj_recalls", {})
    best_objects = best.get("object_add", {})
    object_ids = sorted(set(baseline_objects) | set(best_objects), key=int)
    object_deltas = {
        object_id: float(best_objects[object_id]) - float(baseline_objects[object_id])
        for object_id in object_ids
    }
    nonnegative_objects = sum(delta >= 0.0 for delta in object_deltas.values())
    bop_delta = float(best["bop_ar"]) - float(baseline_bop["bop19_average_recall"])
    add_delta = float(best["add_s"]) - float(baseline_add["recall"])
    passed = bop_delta >= 0.005 and add_delta >= 0.01 and nonnegative_objects >= 5

    summary = {
        "status": "C1_SCREEN_PASS" if passed else "C1_SCREEN_FAIL",
        "best_epoch": int(best["epoch"]),
        "baseline": {
            "bop_ar": float(baseline_bop["bop19_average_recall"]),
            "add_s_0.1d": float(baseline_add["recall"]),
        },
        "best": {
            "bop_ar": float(best["bop_ar"]),
            "add_s_0.1d": float(best["add_s"]),
        },
        "delta": {
            "bop_ar": bop_delta,
            "add_s_0.1d": add_delta,
        },
        "nonnegative_objects": nonnegative_objects,
        "object_add_s_deltas": object_deltas,
        "frozen_gate": {
            "minimum_bop_ar_delta": 0.005,
            "minimum_add_s_delta": 0.01,
            "minimum_nonnegative_objects": 5,
        },
    }
    output = args.output or args.training_output / "screening_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
