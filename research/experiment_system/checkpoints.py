"""Checkpoint identity and integrity indexes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json, utc_now
from .manifest import read_json, sha256_file, validate_run_manifest


SELECTION_KINDS = {
    "fixed_final",
    "best_screening",
    "recent",
    "smoke",
    "invalid",
}


def checkpoint_index_path(run_dir: Path) -> Path:
    return run_dir / "checkpoints" / "checkpoint_index.json"


def load_checkpoint_index(run_dir: Path) -> dict[str, Any]:
    path = checkpoint_index_path(run_dir)
    if not path.exists():
        return {"schema_version": 1, "checkpoints": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("checkpoints"), list):
        raise ValueError(f"invalid checkpoint index: {path}")
    return payload


def record_checkpoint(
    run_dir: Path,
    checkpoint: Path,
    checkpoint_id: str,
    epoch: int,
    iteration: int,
    selection_kind: str,
    parent_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    if selection_kind not in SELECTION_KINDS:
        raise ValueError(f"invalid checkpoint selection kind: {selection_kind}")
    run_dir = run_dir.resolve()
    checkpoint = checkpoint.resolve()
    checkpoint_root = (run_dir / "checkpoints").resolve()
    if checkpoint_root not in checkpoint.parents:
        raise ValueError("checkpoint must be inside the run checkpoints directory")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    manifest = read_json(run_dir / "meta" / "run_manifest.json")
    validate_run_manifest(manifest)
    index = load_checkpoint_index(run_dir)
    if any(record["checkpoint_id"] == checkpoint_id for record in index["checkpoints"]):
        raise ValueError(f"duplicate checkpoint_id: {checkpoint_id}")
    record = {
        "checkpoint_id": checkpoint_id,
        "path": str(checkpoint.relative_to(run_dir)),
        "epoch": int(epoch),
        "iteration": int(iteration),
        "selection_kind": selection_kind,
        "sha256": sha256_file(checkpoint),
        "size_bytes": checkpoint.stat().st_size,
        "experiment_id": manifest["experiment_id"],
        "run_id": manifest["run_id"],
        "git_commit": manifest["source"]["git_commit"],
        "config_sha256": manifest["config"]["sha256"],
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "indexed_at": utc_now(),
    }
    index["checkpoints"].append(record)
    atomic_write_json(checkpoint_index_path(run_dir), index)
    return record


def verify_checkpoint_index(run_dir: Path) -> int:
    index = load_checkpoint_index(run_dir)
    seen: set[str] = set()
    for record in index["checkpoints"]:
        checkpoint_id = record["checkpoint_id"]
        if checkpoint_id in seen:
            raise ValueError(f"duplicate checkpoint_id: {checkpoint_id}")
        seen.add(checkpoint_id)
        path = run_dir / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint changed or is missing: {path}")
    return len(index["checkpoints"])
