#!/usr/bin/env python3
"""Run the frozen EXP013 Three-Path alpha diagnostic on an E40 checkpoint."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VARIANT_CONFIGS = {
    "A": "configs/gdrn/lmo_pbr/research/exp013/a_xyz_residual/eval.py",
    "B": "configs/gdrn/lmo_pbr/research/exp013/b_geometry_attention/eval.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANT_CONFIGS), required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "audit80", "full"), default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PROJECT_ROOT / VARIANT_CONFIGS[args.variant]
    command = [
        sys.executable,
        "-m",
        "research.pose_head_diagnostic.run_statistical_diagnostic",
        "--mode",
        args.mode,
        "--model-role",
        "exp013",
        "--condition-set",
        "exp013_three_path",
        "--config-file",
        str(config),
        "--weights",
        str(args.weights),
        "--output-dir",
        str(args.output_dir),
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
        "--seed",
        "42",
    ]
    if args.mode == "full":
        command.append("--bop-eval")
    if args.resume:
        command.append("--resume")
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
