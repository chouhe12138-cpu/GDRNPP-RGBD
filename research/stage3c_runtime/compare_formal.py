#!/usr/bin/env python3
"""Compare fixed epoch-40 C2 directly against its matched B control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.stage3c_runtime.summarize_formal import load_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("b_output", type=Path)
    parser.add_argument("c2_output", type=Path)
    parser.add_argument("--epoch", type=int, default=40)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    b = load_result(args.b_output, args.epoch)
    c2 = load_result(args.c2_output, args.epoch)
    object_ids = sorted(set(b["objects"]) | set(c2["objects"]), key=int)
    object_deltas = {
        object_id: c2["objects"][object_id] - b["objects"][object_id]
        for object_id in object_ids
    }
    bop_delta = c2["bop_ar"] - b["bop_ar"]
    add_delta = c2["add_s_0.1d"] - b["add_s_0.1d"]
    nonnegative = sum(delta >= 0 for delta in object_deltas.values())
    passed = bop_delta >= 0.005 and add_delta >= 0.01 and nonnegative >= 5
    result = {
        "comparison": "C2_minus_B",
        "fixed_epoch": args.epoch,
        "B": b,
        "C2": c2,
        "delta": {"bop_ar": bop_delta, "add_s_0.1d": add_delta},
        "nonnegative_objects": nonnegative,
        "object_add_s_deltas": object_deltas,
        "status": "C2_ADDED_VALUE_PASS" if passed else "C2_ADDED_VALUE_FAIL",
    }
    output = args.output or args.c2_output.parent / "comparison_B_C2.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
