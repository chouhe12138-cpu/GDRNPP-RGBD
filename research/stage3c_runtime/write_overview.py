#!/usr/bin/env python3
"""Write the compact Stage 3C output index without moving legacy C1 files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


C1_LOG_SHA256 = "a7333b54f64aa9effd2f14677f047560727d489a9e5b0987a6bc70bdf7a5009a"
C1_CHECKPOINT_SHA256 = "d3ab7167f2fc5f6aab8d7e8444c5b816036bd64e38f647a26e994c8e91939aa6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
    )
    parser.add_argument(
        "--legacy-c1",
        type=Path,
        default=Path("output/EXP-20260731-006/quality_coverage_full"),
    )
    parser.add_argument(
        "--c1-checkpoint-sha256",
        default=C1_CHECKPOINT_SHA256,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stage_root = args.output_root / "stage3c"
    stage_root.mkdir(parents=True, exist_ok=True)
    c1_index = {
        "role": "C1_quality_coverage",
        "status": "C1_SCREEN_FAIL",
        "legacy_output": str(args.legacy_c1),
        "fixed_epoch": 40,
        "bop_ar": 0.6897416378316032,
        "add_s_0.1d": 0.5057,
        "nonnegative_objects": 4,
        "source_log_sha256": C1_LOG_SHA256,
        "checkpoint_sha256": args.c1_checkpoint_sha256,
        "legacy_output_preserved": True,
    }
    (stage_root / "C1_quality_coverage_index.json").write_text(
        json.dumps(c1_index, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    overview = {
        "stage": "3C",
        "formal_seed": 20260731,
        "experiments": {
            "A": {"status": "COMPLETE", "bop_ar": 0.6904152249134947, "add_s_0.1d": 0.5086},
            "C1": c1_index,
            "B": {"status": "RUNTIME_GATE_PENDING", "output": str(stage_root / "B_patch_pnp")},
            "C2": {"status": "RUNTIME_GATE_PENDING", "output": str(stage_root / "C2_joint")},
        },
    }
    (stage_root / "overview.json").write_text(
        json.dumps(overview, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(overview, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
