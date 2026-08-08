from datetime import datetime, timezone

import pytest

from research.experiment_system.manifest import (
    build_run_manifest,
    make_run_id,
    validate_experiment,
    validate_run_manifest,
)


EXPERIMENT = {
    "schema_version": 1,
    "experiment_id": "EXP-20260808-008-example",
    "title": "示例",
    "stage": "future",
    "status": "PLANNED",
    "legacy_import": False,
    "protocol": {"metrics": []},
    "evidence": {},
}


def test_identity_validation_and_run_id():
    validate_experiment(EXPERIMENT)
    run_id = make_run_id(
        "smoke",
        7,
        now=datetime(2026, 8, 8, 1, 2, 3, tzinfo=timezone.utc),
    )
    assert run_id == "RUN-20260808-010203-smoke-s7-a01"


def test_formal_refuses_dirty_source(tmp_path, monkeypatch):
    config = tmp_path / "config.py"
    config.write_text("SEED = 1\n")
    monkeypatch.setattr(
        "research.experiment_system.manifest.collect_git_provenance",
        lambda _root: {
            "git_commit": "a" * 40,
            "git_remote": "origin",
            "git_dirty": True,
            "git_status_sha256": "b" * 64,
            "git_diff_sha256": "c" * 64,
        },
    )
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        build_run_manifest(
            EXPERIMENT,
            "RUN-20260808-010203-formal-s1-a01",
            "formal",
            1,
            tmp_path,
            config,
        )


def test_smoke_records_dirty_source(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    config = repo / "config.py"
    config.write_text("SEED = 1\n")
    monkeypatch.setattr(
        "research.experiment_system.manifest.collect_git_provenance",
        lambda _root: {
            "git_commit": "a" * 40,
            "git_remote": "origin",
            "git_dirty": True,
            "git_status_sha256": "b" * 64,
            "git_diff_sha256": "c" * 64,
        },
    )
    result = build_run_manifest(
        EXPERIMENT,
        "RUN-20260808-010203-smoke-s1-a01",
        "smoke",
        1,
        repo,
        config,
    )
    validate_run_manifest(result)
    assert result["source"]["git_dirty"]


def test_formal_requires_matching_image_revision(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    config = repo / "config.py"
    config.write_text("SEED = 1\n")
    monkeypatch.setattr(
        "research.experiment_system.manifest.collect_git_provenance",
        lambda _root: {
            "git_commit": "a" * 40,
            "git_remote": "origin",
            "git_dirty": False,
            "git_status_sha256": "b" * 64,
            "git_diff_sha256": "c" * 64,
        },
    )
    with pytest.raises(RuntimeError, match="revision does not match"):
        build_run_manifest(
            EXPERIMENT,
            "RUN-20260808-010203-formal-s1-a01",
            "formal",
            1,
            repo,
            config,
            image_id="sha256:image",
            image_revision="d" * 40,
        )
