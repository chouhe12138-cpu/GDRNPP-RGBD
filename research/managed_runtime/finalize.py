#!/usr/bin/env python3
"""Finalize a successful fixed-Epoch-40 formal run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.experiment_system.artifacts import atomic_write_json, read_run_state, utc_now
from research.experiment_system.checkpoints import load_checkpoint_index, verify_checkpoint_index
from research.experiment_system.logs import load_or_write_warning_summary
from research.experiment_system.manifest import (
    manifest_environment_image_id,
    manifest_source_commit,
    read_json,
    sha256_file,
    validate_run_manifest,
)
from research.experiment_system.metrics import index_bop_evaluation, verify_indexed_evaluation
from research.experiment_system.steps import load_steps, register_step, transition_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def _register_completed_step(
    run_dir: Path,
    step_id: str,
    kind: str,
    command: str,
    inputs: list[str],
    outputs: list[str],
) -> None:
    existing = {step["step_id"]: step for step in load_steps(run_dir)["steps"]}
    if step_id in existing:
        if existing[step_id]["status"] != "COMPLETE":
            raise RuntimeError(f"step {step_id} already exists but is not complete")
        return
    register_step(run_dir, step_id, kind, command, inputs=inputs, outputs=outputs)
    transition_step(run_dir, step_id, "RUNNING", f"{step_id} started")
    transition_step(run_dir, step_id, "COMPLETE", f"{step_id} completed")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    state = read_run_state(run_dir)["status"]
    summary_path = run_dir / "summary/final_summary.json"
    if state == "COMPLETE" and summary_path.is_file():
        verify_checkpoint_index(run_dir)
        verify_indexed_evaluation(run_dir / "evaluations/epoch_040/lmo_bop_test")
        print(summary_path.read_text(encoding="utf-8"))
        return 0
    if state != "RUNNING":
        raise RuntimeError(f"formal run must be RUNNING before finalization, got {state}")

    launcher = read_json(run_dir / "meta/launcher_status.json")
    if launcher.get("exit_code") != 0 or launcher.get("mode") != "formal":
        raise RuntimeError(f"formal training is not complete: {launcher}")
    manifest = read_json(run_dir / "meta/run_manifest.json")
    validate_run_manifest(manifest)

    checkpoints = load_checkpoint_index(run_dir)["checkpoints"]
    fixed = [
        record
        for record in checkpoints
        if record["checkpoint_id"] == "epoch_040"
        and record["selection_kind"] == "fixed_final"
    ]
    if len(fixed) != 1:
        raise RuntimeError(f"expected one fixed Epoch-40 checkpoint, found {len(fixed)}")
    verify_checkpoint_index(run_dir)

    evaluation_root = run_dir / "evaluations/epoch_040/lmo_bop_test"
    index, normalized = index_bop_evaluation(
        evaluation_root,
        dataset_id="lmo_bop_test",
        bbox_type="gt",
        checkpoint_id="epoch_040",
        write=True,
    )
    _register_completed_step(
        run_dir,
        "eval-epoch-040",
        "eval",
        "index existing fixed Epoch-40 evaluator output",
        [fixed[0]["path"]],
        ["evaluations/epoch_040/lmo_bop_test/evaluation_index.json"],
    )
    warnings = load_or_write_warning_summary(run_dir)
    summary = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "experiment_id": manifest["experiment_id"],
        "run_id": manifest["run_id"],
        "seed": manifest["seed"],
        "git_commit": manifest_source_commit(manifest),
        "docker_image_id": manifest_environment_image_id(manifest),
        "fixed_checkpoint": fixed[0],
        "evaluation_index_sha256": sha256_file(
            evaluation_root / "evaluation_index.json"
        ),
        "indexed_raw_files": index["raw_files"],
        "metrics": normalized,
        "warnings": {
            "unique": warnings["unique_warnings"],
            "total_occurrences": warnings["total_warning_occurrences"],
        },
    }
    atomic_write_json(summary_path, summary)
    register_step(
        run_dir,
        "summarize",
        "summarize",
        "write fixed Epoch-40 formal summary",
        inputs=[
            "checkpoints/checkpoint_index.json",
            "evaluations/epoch_040/lmo_bop_test/metrics.normalized.json",
        ],
        outputs=["summary/final_summary.json", "summary/warnings.json"],
    )
    transition_step(run_dir, "summarize", "RUNNING", "formal summary started")
    transition_step(
        run_dir,
        "summarize",
        "COMPLETE",
        "fixed Epoch-40 formal run finalized",
        complete_run=True,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
