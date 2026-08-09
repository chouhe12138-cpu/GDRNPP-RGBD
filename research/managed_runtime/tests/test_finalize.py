import json
import sys

import torch

from research.experiment_system.artifacts import (
    create_run_directory,
    initialize_run_state,
    transition_run_state,
)
from research.experiment_system.checkpoints import record_checkpoint
from research.experiment_system.manifest import write_run_manifest
from research.managed_runtime.finalize import main


def test_finalize_indexes_fixed_epoch_40_and_completes_run(tmp_path, monkeypatch):
    experiment_id = "EXP-20260809-009-example"
    run_id = "RUN-20260810-010203-formal-s42-a01"
    run = create_run_directory(tmp_path, experiment_id, run_id)
    initialize_run_state(run)
    transition_run_state(run, "RUNNING", "training started")
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "mode": "formal",
        "seed": 42,
        "created_at": "2026-08-10T01:02:03+00:00",
        "source": {
            "git_commit": "a" * 40,
            "git_remote": "origin",
            "git_dirty": False,
        },
        "config": {"path": "config.py", "sha256": "b" * 64},
        "inputs": [],
        "execution": {
            "docker_image_id": "sha256:image",
            "docker_image_revision": "a" * 40,
            "path_profile_id": "test",
        },
    }
    write_run_manifest(run, manifest)
    (run / "meta/launcher_status.json").write_text(
        json.dumps({"mode": "formal", "exit_code": 0}), encoding="utf-8"
    )
    checkpoint = run / "checkpoints/model_epoch_040.pth"
    torch.save({"iteration": 99, "model": {}}, checkpoint)
    record_checkpoint(
        run,
        checkpoint,
        "epoch_040",
        40,
        99,
        "fixed_final",
    )
    result = run / "evaluations/epoch_040/lmo_bop_test/result"
    add = result / "error=ad_ntop=1"
    add.mkdir(parents=True)
    (result / "scores_bop19.json").write_text(
        json.dumps({"bop19_average_recall": 0.5}), encoding="utf-8"
    )
    (add / "scores_th=0.100_min-visib=-1.000.json").write_text(
        json.dumps(
            {
                "mean_obj_recall": 0.4,
                "recall": 0.3,
                "obj_recalls": {"1": 0.4},
            }
        ),
        encoding="utf-8",
    )
    (run / "train/console.log").write_text("formal complete\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["finalize", str(run)])

    assert main() == 0

    state = json.loads((run / "meta/run_state.json").read_text())
    summary = json.loads((run / "summary/final_summary.json").read_text())
    assert state["status"] == "COMPLETE"
    assert summary["seed"] == 42
    assert summary["fixed_checkpoint"]["checkpoint_id"] == "epoch_040"
    assert summary["metrics"]["metrics"]["bop19_ar_macro"]["value"] == 0.5
