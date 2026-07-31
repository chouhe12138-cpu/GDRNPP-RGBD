#!/usr/bin/env python3
"""Verify that a Stage 3C-0 checkpoint changed only Patch-PnP tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", required=True, type=Path)
    parser.add_argument("--trained", required=True, type=Path)
    parser.add_argument("--allow-unchanged-pnp", action="store_true")
    return parser.parse_args()


def load_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path.resolve(), map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    normalized = {}
    for key, value in state.items():
        name = key.removeprefix("_module.").removeprefix("module.")
        if isinstance(value, torch.Tensor):
            normalized[name] = value
    return normalized


def compare_states(
    official: dict[str, torch.Tensor],
    trained: dict[str, torch.Tensor],
) -> dict[str, list[str]]:
    if official.keys() != trained.keys():
        missing = sorted(official.keys() - trained.keys())
        extra = sorted(trained.keys() - official.keys())
        raise RuntimeError(
            f"Checkpoint tensor sets differ; missing={missing[:5]}, extra={extra[:5]}"
        )

    changed_pnp = []
    changed_frozen = []
    for name in official:
        if torch.equal(official[name], trained[name]):
            continue
        if name.startswith("pnp_net."):
            changed_pnp.append(name)
        else:
            changed_frozen.append(name)
    return {
        "changed_pnp": changed_pnp,
        "changed_frozen": changed_frozen,
    }


def main() -> int:
    args = parse_args()
    official = load_state(args.official)
    trained = load_state(args.trained)
    result = compare_states(official, trained)
    changed_pnp = result["changed_pnp"]
    changed_frozen = result["changed_frozen"]
    if changed_frozen:
        raise RuntimeError(f"Frozen tensors changed: {changed_frozen[:10]}")
    if not changed_pnp and not args.allow_unchanged_pnp:
        raise RuntimeError("No Patch-PnP tensor changed")

    print(
        json.dumps(
            {
                "status": "PASS",
                "tensor_count": len(official),
                "changed_pnp_tensors": len(changed_pnp),
                "changed_frozen_tensors": len(changed_frozen),
                "first_changed_pnp_tensors": changed_pnp[:10],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
