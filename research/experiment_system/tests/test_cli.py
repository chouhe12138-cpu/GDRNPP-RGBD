import argparse
import json

from research.experiment_system.cli import command_prepare, command_verify_run


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
