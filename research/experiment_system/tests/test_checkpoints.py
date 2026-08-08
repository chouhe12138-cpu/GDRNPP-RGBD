import json

import pytest

from research.experiment_system.artifacts import create_run_directory, initialize_run_state
from research.experiment_system.checkpoints import record_checkpoint, verify_checkpoint_index


def make_run(tmp_path):
    run = create_run_directory(tmp_path, "EXP-20260808-008-example", "RUN-1")
    initialize_run_state(run)
    manifest = {
        "schema_version": 1,
        "experiment_id": "EXP-20260808-008-example",
        "run_id": "RUN-20260808-010203-smoke-s1-a01",
        "mode": "smoke",
        "seed": 1,
        "created_at": "2026-08-08T01:02:03+00:00",
        "source": {
            "git_commit": "a" * 40,
            "git_dirty": False,
        },
        "config": {"path": "config.py", "sha256": "b" * 64},
        "inputs": [],
        "execution": {"docker_image_id": None},
    }
    (run / "meta/run_manifest.json").write_text(json.dumps(manifest))
    return run


def test_checkpoint_index_binds_run_source_and_hash(tmp_path):
    run = make_run(tmp_path)
    checkpoint = run / "checkpoints/model_epoch_001.pth"
    checkpoint.write_bytes(b"checkpoint")
    record = record_checkpoint(
        run,
        checkpoint,
        "epoch_001",
        epoch=1,
        iteration=99,
        selection_kind="smoke",
        parent_checkpoint_sha256="c" * 64,
    )
    assert record["experiment_id"] == "EXP-20260808-008-example"
    assert record["git_commit"] == "a" * 40
    assert verify_checkpoint_index(run) == 1
    checkpoint.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed or is missing"):
        verify_checkpoint_index(run)
