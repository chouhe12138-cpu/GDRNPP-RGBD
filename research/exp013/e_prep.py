#!/usr/bin/env python3
"""Derive the pnp-stripped official checkpoint used by EXP013E.

EXP013E rebuilds the official ``ConvPnPNet`` pose head with random
initialization. The official checkpoint stores that head under the same
``pnp_net.*`` key prefix the rebuilt module uses, so loading the original
file would silently overwrite the fresh initialization. This tool derives a
copy with every ``pnp_net.*`` tensor removed, verifies the remaining tensors
are value-identical to the source, and writes the result atomically.

The output is a one-off derived artifact (not committed); the gate step
regenerates it deterministically from the SHA-256-verified official file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch

OFFICIAL_SHA256 = "bafa869d4e6c00410517ecb1add59f234ed1642e47fabcf3aa6e0e8a1b498a8c"
PNP_PREFIX = "pnp_net."


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strip_model_state(state: dict) -> tuple[dict, list[str]]:
    kept: dict = {}
    removed: list[str] = []
    for key, value in state.items():
        if key.removeprefix("_module.").startswith(PNP_PREFIX):
            removed.append(key)
        else:
            kept[key] = value
    return kept, removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official",
        type=Path,
        default=Path("pretrained_models/lmo_pbr/model_final_wo_optim.pth"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("pretrained_models/lmo_pbr/model_final_wo_optim_wo_pnp.pth"),
    )
    args = parser.parse_args()

    official_path = args.official.resolve()
    out_path = args.out.resolve()
    official_hash = sha256(official_path)
    if official_hash != OFFICIAL_SHA256:
        raise RuntimeError(f"Unexpected official checkpoint hash: {official_hash}")

    checkpoint = torch.load(official_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Official checkpoint is not a dict")
    wrapped = isinstance(checkpoint.get("model"), dict)
    state = checkpoint["model"] if wrapped else checkpoint
    kept, removed = strip_model_state(state)
    if not removed:
        raise RuntimeError("Official checkpoint contains no pnp tensors to strip")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    stripped_checkpoint = {**checkpoint, "model": kept} if wrapped else kept
    with tempfile.NamedTemporaryFile(
        dir=str(out_path.parent), prefix=".e_prep_", suffix=".pth", delete=False
    ) as handle:
        temp_path = Path(handle.name)
    try:
        torch.save(stripped_checkpoint, temp_path)
        os.replace(temp_path, out_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    reloaded = torch.load(out_path, map_location="cpu")
    reloaded_state = reloaded["model"] if wrapped else reloaded
    kept_reloaded, removed_reloaded = strip_model_state(reloaded_state)
    if removed_reloaded or len(kept_reloaded) != len(kept):
        raise RuntimeError("Stripped checkpoint failed post-write verification")
    for key, value in kept_reloaded.items():
        source_key = f"_module.{key}" if f"_module.{key}" in state else key
        if source_key not in state or not torch.equal(
            state[source_key].cpu(), value.cpu()
        ):
            raise RuntimeError(f"Stripped tensor diverges from source: {key}")

    print(
        json.dumps(
            {
                "status": "PASS",
                "official_checkpoint_sha256": official_hash,
                "stripped_checkpoint_sha256": sha256(out_path),
                "output": str(out_path),
                "kept_tensors": len(kept),
                "removed_pnp_tensors": len(removed),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def checkpoint_model_state(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    return {key.removeprefix("_module."): value for key, value in state.items()}


if __name__ == "__main__":
    raise SystemExit(main())
