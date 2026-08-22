#!/usr/bin/env python3
"""Apply the preregistered EXP013 A/B/C metric gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXP012 = {"bop": 0.678800, "add_s": 0.494118, "ar_res": 0.491349, "ar_tes": 0.791926}


def load_metrics(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {"bop", "add_s", "per_object_delta"}
    missing = required - value.keys()
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    if len(value["per_object_delta"]) != 8:
        raise ValueError(
            f"{path} must contain exactly eight per-object ADD(-S) deltas versus EXP012"
        )
    return value


def versus_exp012(value: dict) -> dict:
    object_nonnegative = sum(
        float(score) >= 0 for score in value["per_object_delta"].values()
    )
    checks = {
        "bop_delta_ge_0.005": float(value["bop"]) - EXP012["bop"] >= 0.005,
        "add_s_delta_ge_0.010": float(value["add_s"]) - EXP012["add_s"] >= 0.010,
        "objects_nonnegative_ge_5_of_8": object_nonnegative >= 5,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "objects_nonnegative": object_nonnegative,
    }


def attention_effective(a: dict, b: dict) -> dict:
    bop_delta = float(b["bop"]) - float(a["bop"])
    if bop_delta > 0.001:
        effective, rule = True, "BOP_ABOVE_A_BY_MORE_THAN_0.001"
    elif bop_delta < -0.001:
        effective, rule = False, "BOP_BELOW_A_BY_MORE_THAN_0.001"
    else:
        effective = float(b["add_s"]) > float(a["add_s"])
        rule = "BOP_TIED_USE_ADD_S"
    return {
        "effective": effective,
        "rule": rule,
        "bop_delta": bop_delta,
        "add_s_delta": float(b["add_s"]) - float(a["add_s"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--c", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    a, b = load_metrics(args.a), load_metrics(args.b)
    a_gate, b_gate = versus_exp012(a), versus_exp012(b)
    b_effect = attention_effective(a, b)
    result = {
        "exp012_baseline": EXP012,
        "a_gate": a_gate,
        "b_gate": b_gate,
        "b_vs_a": b_effect,
        "authorize_c": bool(
            a_gate["passed"] and b_gate["passed"] and b_effect["effective"]
        ),
    }
    if args.c:
        c = load_metrics(args.c)
        if (
            "ar_res" not in c
            or "ar_tes" not in c
            or "ar_res" not in b
            or "ar_tes" not in b
        ):
            raise ValueError("C comparison requires ar_res and ar_tes in B and C")
        result["c_vs_b"] = {
            "rotation_improved": float(c["ar_res"]) > float(b["ar_res"]),
            "translation_drop_within_0.01": float(c["ar_tes"])
            >= float(b["ar_tes"]) - 0.01,
            "ar_res_delta": float(c["ar_res"]) - float(b["ar_res"]),
            "ar_tes_delta": float(c["ar_tes"]) - float(b["ar_tes"]),
        }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
