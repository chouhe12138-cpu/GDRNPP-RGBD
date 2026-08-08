"""Non-overwriting artifact layout and run-state transitions."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARTIFACT_DIRS = (
    "meta",
    "train",
    "checkpoints",
    "evaluations",
    "diagnostics",
    "summary",
)

RUN_STATES = {
    "PREPARED",
    "RUNNING",
    "COMPLETE",
    "FAILED",
    "INVALID",
    "ARCHIVED",
}

ALLOWED_TRANSITIONS = {
    "PREPARED": {"RUNNING", "FAILED", "INVALID"},
    "RUNNING": {"COMPLETE", "FAILED", "INVALID"},
    "COMPLETE": {"ARCHIVED"},
    "FAILED": set(),
    "INVALID": set(),
    "ARCHIVED": set(),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON without exposing a partially-written metadata file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def create_run_directory(
    output_root: Path,
    experiment_id: str,
    run_id: str,
) -> Path:
    """Create the standard layout and refuse to reuse any existing run path."""

    run_dir = output_root / experiment_id / run_id
    if run_dir.exists():
        raise FileExistsError(f"run output already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    for name in ARTIFACT_DIRS:
        (run_dir / name).mkdir()
    return run_dir


def initialize_run_state(run_dir: Path) -> dict[str, Any]:
    state = {
        "schema_version": 1,
        "status": "PREPARED",
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }
    atomic_write_json(run_dir / "meta" / "run_state.json", state)
    append_event(run_dir, "PREPARED", "run directory created")
    return state


def read_run_state(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "meta" / "run_state.json"
    if not path.is_file():
        raise FileNotFoundError(f"run state is missing: {path}")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("status") not in RUN_STATES:
        raise ValueError(f"invalid run state: {state.get('status')}")
    return state


def append_event(
    run_dir: Path,
    event: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> None:
    record = {
        "at": utc_now(),
        "event": event,
        "message": message,
    }
    if details:
        record["details"] = details
    event_path = run_dir / "meta" / "events.jsonl"
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def transition_run_state(
    run_dir: Path,
    new_status: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if new_status not in RUN_STATES:
        raise ValueError(f"unknown run state: {new_status}")
    state = read_run_state(run_dir)
    old_status = state["status"]
    if new_status not in ALLOWED_TRANSITIONS[old_status]:
        raise ValueError(f"invalid run-state transition: {old_status} -> {new_status}")
    state["status"] = new_status
    state["updated_at"] = utc_now()
    state["message"] = message
    if details:
        state["details"] = details
    atomic_write_json(run_dir / "meta" / "run_state.json", state)
    append_event(run_dir, new_status, message, details)
    return state


def evaluation_directory(
    run_dir: Path,
    checkpoint_id: str,
    dataset_id: str,
) -> Path:
    return run_dir / "evaluations" / checkpoint_id / dataset_id


def diagnostic_directory(run_dir: Path, diagnostic_id: str, mode: str) -> Path:
    return run_dir / "diagnostics" / diagnostic_id / mode
