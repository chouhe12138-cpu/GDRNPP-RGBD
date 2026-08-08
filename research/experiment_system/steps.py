"""Record train/eval/diagnostic/summarize lifecycle without owning long-running jobs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import (
    append_event,
    atomic_write_json,
    read_run_state,
    transition_run_state,
    utc_now,
)


STEP_KINDS = {"train", "eval", "diagnostic", "summarize", "verify"}
STEP_STATES = {"PENDING", "RUNNING", "COMPLETE", "FAILED", "SKIPPED"}
STEP_TRANSITIONS = {
    "PENDING": {"RUNNING", "SKIPPED"},
    "RUNNING": {"COMPLETE", "FAILED"},
    "COMPLETE": set(),
    "FAILED": set(),
    "SKIPPED": set(),
}


def steps_path(run_dir: Path) -> Path:
    return run_dir / "meta" / "steps.json"


def load_steps(run_dir: Path) -> dict[str, Any]:
    path = steps_path(run_dir)
    if not path.exists():
        return {"schema_version": 1, "steps": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("steps"), list):
        raise ValueError(f"invalid step metadata: {path}")
    return payload


def register_step(
    run_dir: Path,
    step_id: str,
    kind: str,
    command: str,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
) -> dict[str, Any]:
    if kind not in STEP_KINDS:
        raise ValueError(f"invalid step kind: {kind}")
    payload = load_steps(run_dir)
    if any(step["step_id"] == step_id for step in payload["steps"]):
        raise ValueError(f"duplicate step_id: {step_id}")
    step = {
        "step_id": step_id,
        "kind": kind,
        "status": "PENDING",
        "command": command,
        "inputs": inputs or [],
        "outputs": outputs or [],
        "registered_at": utc_now(),
        "updated_at": utc_now(),
    }
    payload["steps"].append(step)
    atomic_write_json(steps_path(run_dir), payload)
    append_event(run_dir, "STEP_REGISTERED", f"registered {step_id}", {"kind": kind})
    return step


def transition_step(
    run_dir: Path,
    step_id: str,
    new_status: str,
    message: str,
    complete_run: bool = False,
) -> dict[str, Any]:
    if new_status not in STEP_STATES:
        raise ValueError(f"invalid step state: {new_status}")
    payload = load_steps(run_dir)
    matches = [step for step in payload["steps"] if step["step_id"] == step_id]
    if len(matches) != 1:
        raise KeyError(f"expected one step {step_id}, found {len(matches)}")
    step = matches[0]
    old_status = step["status"]
    if new_status not in STEP_TRANSITIONS[old_status]:
        raise ValueError(f"invalid step transition: {old_status} -> {new_status}")
    if complete_run and (new_status != "COMPLETE" or step["kind"] != "summarize"):
        raise ValueError("only a completed summarize step can complete a run")

    run_state = read_run_state(run_dir)["status"]
    if new_status == "RUNNING" and run_state == "PREPARED":
        transition_run_state(run_dir, "RUNNING", f"step {step_id} started")
    elif new_status == "RUNNING" and run_state != "RUNNING":
        raise ValueError(f"cannot start a step while run state is {run_state}")

    step["status"] = new_status
    step["updated_at"] = utc_now()
    step["message"] = message
    atomic_write_json(steps_path(run_dir), payload)
    append_event(
        run_dir,
        f"STEP_{new_status}",
        message,
        {"step_id": step_id, "kind": step["kind"]},
    )
    if new_status == "FAILED":
        transition_run_state(run_dir, "FAILED", f"step {step_id} failed: {message}")
    if complete_run:
        transition_run_state(run_dir, "COMPLETE", message)
    return step
