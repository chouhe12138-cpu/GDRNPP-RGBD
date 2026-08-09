from datetime import datetime, timezone

import pytest

from research.experiment_system.manifest import (
    build_run_manifest,
    collect_git_provenance,
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


def source(clean: bool) -> dict:
    return {
        "source_git_commit": "a" * 40,
        "source_git_remote": "origin",
        "source_tree_clean": clean,
        "source_head_detached": True,
        "source_status_sha256": "b" * 64,
        "source_diff_sha256": "c" * 64,
        "untracked_files": [],
        "provenance_kind": "git_release_checkout",
    }


def binding(build_source: str = "d" * 40) -> dict:
    return {
        "schema_version": 1,
        "release": {
            "source_git_commit": "a" * 40,
            "source_tree_clean": True,
            "environment_contract_sha256": "e" * 64,
        },
        "environment": {
            "environment_image_id": "sha256:image",
            "environment_image_ref": "gdrnpp:env",
            "environment_build_source_commit": build_source,
            "environment_contract_sha256": "e" * 64,
            "native_artifact_manifest_sha256": "f" * 64,
            "native_abi": {"python_soabi": "cpython-310-x86_64-linux-gnu"},
        },
        "native_artifacts": [],
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
        lambda _root: source(False),
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
        lambda _root: source(False),
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
    assert not result["source"]["source_tree_clean"]


def test_formal_accepts_independent_source_and_environment_build_commits(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    repo.mkdir()
    config = repo / "config.py"
    config.write_text("SEED = 1\n")
    monkeypatch.setattr(
        "research.experiment_system.manifest.collect_git_provenance",
        lambda _root: source(True),
    )
    result = build_run_manifest(
        EXPERIMENT,
        "RUN-20260808-010203-formal-s1-a01",
        "formal",
        1,
        repo,
        config,
        environment_binding=binding(),
    )
    assert result["source"]["source_git_commit"] == "a" * 40
    assert result["environment"]["environment_build_source_commit"] == "d" * 40
    assert result["source"]["source_git_commit"] != result["environment"][
        "environment_build_source_commit"
    ]


def test_formal_requires_verified_environment_binding(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    config = repo / "config.py"
    config.write_text("SEED = 1\n")
    monkeypatch.setattr(
        "research.experiment_system.manifest.collect_git_provenance",
        lambda _root: source(True),
    )
    with pytest.raises(RuntimeError, match="verified environment binding"):
        build_run_manifest(
            EXPERIMENT,
            "RUN-20260808-010203-formal-s1-a01",
            "formal",
            1,
            repo,
            config,
        )


def test_formal_rejects_binding_for_different_source_commit(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    config = repo / "config.py"
    config.write_text("SEED = 1\n")
    monkeypatch.setattr(
        "research.experiment_system.manifest.collect_git_provenance",
        lambda _root: source(True),
    )
    wrong = binding()
    wrong["release"]["source_git_commit"] = "b" * 40
    with pytest.raises(RuntimeError, match="different source commit"):
        build_run_manifest(
            EXPERIMENT,
            "RUN-20260808-010203-formal-s1-a01",
            "formal",
            1,
            repo,
            config,
            environment_binding=wrong,
        )


def test_embedded_image_revision_is_not_accepted_as_source(tmp_path, monkeypatch):
    monkeypatch.setenv("GDRN_GIT_COMMIT", "a" * 40)
    with pytest.raises(RuntimeError, match="no .git metadata"):
        collect_git_provenance(tmp_path)
