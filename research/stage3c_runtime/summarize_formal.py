#!/usr/bin/env python3
"""Summarize a fixed-epoch B or C2 formal result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from research.quality_coverage.plot_curves import find_add_score, find_score


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("B", "C2"))
    parser.add_argument("training_output", type=Path)
    parser.add_argument("baseline_output", type=Path)
    parser.add_argument("--epoch", type=int, default=40)
    return parser.parse_args()


def load_result(root: Path, epoch: int) -> dict[str, object]:
    eval_root = root / "evaluations" / f"epoch_{epoch:03d}" / "lmo_bop_test"
    bop = find_score(eval_root, "scores_bop19.json")
    add = find_add_score(eval_root)
    if bop is None or add is None:
        raise RuntimeError(f"Missing fixed epoch {epoch} BOP or ADD result below {eval_root}")
    return {
        "bop_ar": float(bop["bop19_average_recall"]),
        "add_s_0.1d": float(add["recall"]),
        "objects": {str(key): float(value) for key, value in add.get("obj_recalls", {}).items()},
    }


def load_baseline(root: Path) -> dict[str, object]:
    bop_paths = sorted(root.glob("**/scores_bop19.json"))
    if len(bop_paths) != 1:
        raise RuntimeError(f"Expected one baseline BOP result below {root}")
    bop = json.loads(bop_paths[0].read_text(encoding="utf-8"))
    add = find_add_score(root)
    if add is None:
        raise RuntimeError(f"Expected one baseline ADD result below {root}")
    return {
        "bop_ar": float(bop["bop19_average_recall"]),
        "add_s_0.1d": float(add["recall"]),
        "objects": {str(key): float(value) for key, value in add.get("obj_recalls", {}).items()},
    }


def main() -> int:
    args = parse_args()
    result = load_result(args.training_output, args.epoch)
    baseline = load_baseline(args.baseline_output)
    object_ids = sorted(set(baseline["objects"]) | set(result["objects"]), key=int)
    deltas = {
        object_id: result["objects"][object_id] - baseline["objects"][object_id]
        for object_id in object_ids
    }
    bop_delta = result["bop_ar"] - baseline["bop_ar"]
    add_delta = result["add_s_0.1d"] - baseline["add_s_0.1d"]
    nonnegative = sum(delta >= 0 for delta in deltas.values())
    passed = bop_delta >= 0.005 and add_delta >= 0.01 and nonnegative >= 5
    summary = {
        "role": args.role,
        "status": "SCREEN_PASS" if passed else "SCREEN_FAIL",
        "fixed_epoch": args.epoch,
        "baseline": baseline,
        "result": result,
        "delta": {
            "bop_ar": bop_delta,
            "add_s_0.1d": add_delta,
        },
        "nonnegative_objects": nonnegative,
        "object_add_s_deltas": deltas,
        "gate": {
            "minimum_bop_ar_delta": 0.005,
            "minimum_add_s_delta": 0.01,
            "minimum_nonnegative_objects": 5,
        },
    }
    output_dir = args.training_output / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["role", "epoch", "bop_ar", "add_s_0.1d", "nonnegative_objects", "status"])
        writer.writerow(
            [
                args.role,
                args.epoch,
                result["bop_ar"],
                result["add_s_0.1d"],
                nonnegative,
                summary["status"],
            ]
        )
    with (output_dir / "per_object.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["object_id", "baseline_add_s", "result_add_s", "delta"])
        for object_id in object_ids:
            writer.writerow(
                [
                    object_id,
                    baseline["objects"][object_id],
                    result["objects"][object_id],
                    deltas[object_id],
                ]
            )
    checkpoint_dir = args.training_output / "checkpoints"
    checkpoint_hashes = {
        path.name: sha256(path)
        for path in sorted(checkpoint_dir.glob("*.pth"))
    }
    (checkpoint_dir / "SHA256SUMS").write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in checkpoint_hashes.items()
        ),
        encoding="utf-8",
    )
    (checkpoint_dir / "checkpoint_index.json").write_text(
        json.dumps(
            {
                "fixed_epoch": args.epoch,
                "fixed_checkpoint": f"model_epoch_{args.epoch:03d}.pth",
                "sha256": checkpoint_hashes,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
