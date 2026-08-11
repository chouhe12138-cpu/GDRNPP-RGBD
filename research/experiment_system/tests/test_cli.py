import argparse
import json
import subprocess

from research.experiment_system.cli import command_prepare, command_verify_run
from research.experiment_system.manifest import collect_source_snapshot, sha256_json


def test_prepare_creates_immutable_metadata_and_resolved_config(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    experiment_dir = repo / "research/experiments/EXP-20260808-008-example"
    experiment_dir.mkdir(parents=True)
    experiment = experiment_dir / "EXPERIMENT.json"
    experiment.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": "EXP-20260808-008-example",
                "title": "示例",
                "stage": "future",
                "status": "PLANNED",
                "legacy_import": False,
                "protocol": {"metrics": []},
                "evidence": {},
            }
        )
    )
    config = repo / "config.py"
    config.write_text(
        "EXPERIMENT_ID = 'EXP-20260808-008-example'\n"
        "SEED = 1\n"
        "OUTPUT_DIR = 'output/example'\n"
    )
    monkeypatch.setattr(
        "research.experiment_system.manifest.collect_git_provenance",
        lambda _root: {
            "source_git_commit": "a" * 40,
            "source_git_remote": "origin",
            "source_tree_clean": False,
            "source_head_detached": True,
            "source_status_sha256": "b" * 64,
            "source_diff_sha256": "c" * 64,
            "untracked_files": [],
            "provenance_kind": "git_release_checkout",
        },
    )
    output = tmp_path / "outputs"
    args = argparse.Namespace(
        repo_root=repo,
        experiment=experiment,
        config=config,
        mode="smoke",
        seed=1,
        attempt=1,
        run_id="RUN-20260808-010203-smoke-s1-a01",
        output_root=output,
        profile=None,
        catalog=None,
        asset_ids=None,
        environment_binding=None,
        environment_image_id=None,
        parent_run_id=None,
    )
    assert command_prepare(args) == 0
    run = output / "EXP-20260808-008-example/RUN-20260808-010203-smoke-s1-a01"
    manifest = json.loads((run / "meta/run_manifest.json").read_text())
    assert not manifest["source"]["source_tree_clean"]
    assert (run / "meta/config_snapshot.py").is_file()
    assert (run / "meta/resolved_config.py").is_file()
    assert command_verify_run(argparse.Namespace(run_dir=run)) == 0


def test_formal_prepare_uses_bound_snapshot_when_git_is_unavailable(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    experiment_dir = repo / "research/experiments/EXP-20260808-008-example"
    experiment_dir.mkdir(parents=True)
    experiment = experiment_dir / "EXPERIMENT.json"
    experiment.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": "EXP-20260808-008-example",
                "title": "示例",
                "stage": "future",
                "status": "AUTHORIZED",
                "legacy_import": False,
                "protocol": {"seed": 1, "metrics": []},
                "evidence": {},
            }
        )
    )
    config = repo / "config.py"
    config.write_text(
        "EXPERIMENT_ID = 'EXP-20260808-008-example'\n"
        "SEED = 1\n"
        "OUTPUT_DIR = 'output/example'\n"
    )
    (repo / ".gitignore").write_text(".local/\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "release",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    subprocess.run(["git", "-C", str(repo), "checkout", "--detach"], check=True)
    binding = {
        "schema_version": 2,
        "release": {
            "source_git_commit": commit,
            "source_git_remote": "",
            "source_tree_clean": True,
            "source_head_detached": True,
            "source_snapshot": collect_source_snapshot(repo, commit),
            "environment_contract_sha256": "e" * 64,
        },
        "environment": {
            "environment_image_id": "sha256:image",
            "environment_image_ref": "gdrnpp:env",
            "environment_build_source_commit": "d" * 40,
            "environment_contract_sha256": "e" * 64,
            "native_artifact_manifest_sha256": sha256_json({"artifacts": []}),
            "native_abi": {},
        },
        "native_artifacts": [],
    }
    binding_path = repo / ".local/environment_binding.json"
    binding_path.parent.mkdir()
    binding_path.write_text(json.dumps(binding))

    def no_git(*_args, **_kwargs):
        raise AssertionError("container-side formal preparation must not execute Git")

    monkeypatch.setattr("research.experiment_system.manifest._git", no_git)
    output = tmp_path / "outputs"
    args = argparse.Namespace(
        repo_root=repo,
        experiment=experiment,
        config=config,
        mode="formal",
        seed=1,
        attempt=1,
        run_id="RUN-20260808-010203-formal-s1-a01",
        output_root=output,
        profile=None,
        catalog=None,
        asset_ids=None,
        environment_binding=binding_path,
        environment_image_id="sha256:image",
        parent_run_id=None,
    )
    assert command_prepare(args) == 0
    run = output / "EXP-20260808-008-example/RUN-20260808-010203-formal-s1-a01"
    manifest = json.loads((run / "meta/run_manifest.json").read_text())
    assert manifest["source"]["source_git_commit"] == commit
    assert manifest["source"]["provenance_kind"] == (
        "verified_release_binding_snapshot"
    )
    assert manifest["environment"]["environment_image_id"] == "sha256:image"
