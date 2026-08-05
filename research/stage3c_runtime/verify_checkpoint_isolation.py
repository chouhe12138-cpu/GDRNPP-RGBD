#!/usr/bin/env python3
"""Verify that a B or C2 checkpoint changed only its allowed tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    return {key.removeprefix("_module."): value for key, value in state.items()}


def compare_states(role: str, official, trained) -> dict[str, object]:
    allowed = ("pnp_net.",) if role == "B" else ("pnp_net.", "quality_coverage_net.")
    shared = sorted(set(official) & set(trained))
    changed_allowed = []
    changed_frozen = []
    for name in shared:
        if torch.equal(official[name], trained[name]):
            continue
        if name.startswith(allowed):
            changed_allowed.append(name)
        else:
            changed_frozen.append(name)
    added = sorted(set(trained) - set(official))
    removed = sorted(set(official) - set(trained))
    return {
        "role": role,
        "changed_allowed": changed_allowed,
        "changed_frozen": changed_frozen,
        "added": added,
        "removed": removed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("B", "C2"))
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    official = load_state(args.official)
    trained = load_state(args.trained)
    result = compare_states(args.role, official, trained)
    if result["changed_frozen"] or result["removed"]:
        raise RuntimeError(f"Frozen checkpoint isolation failed: {result}")
    if not result["changed_allowed"]:
        raise RuntimeError("No allowed trainable tensor changed")
    if args.role == "B" and result["added"]:
        raise RuntimeError(f"B unexpectedly added tensors: {result['added']}")
    if args.role == "C2" and (
        not result["added"]
        or not all(name.startswith("quality_coverage_net.") for name in result["added"])
    ):
        raise RuntimeError(f"C2 added unexpected tensors: {result['added']}")
    result.update(
        {
            "status": "PASS",
            "official_sha256": sha256(args.official),
            "trained_sha256": sha256(args.trained),
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
