import json
import subprocess

import pytest

from research.experiment_system.environment import (
    environment_contract,
    verify_release_binding,
    verify_runtime_binding,
)
from research.experiment_system.manifest import (
    collect_source_snapshot,
    sha256_file,
    sha256_json,
)


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def commit(repo, message):
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
            "-m",
            message,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return git(repo, "rev-parse", "HEAD")


def make_contract_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "docker/l40/vendor").mkdir(parents=True)
    (repo / "core/csrc/flow/src").mkdir(parents=True)
    (repo / "configs").mkdir()
    (repo / ".gitignore").write_text("*.so\n.local/\n")
    (repo / ".dockerignore").write_text(".git\n")
    (repo / "docker/l40/Dockerfile").write_text("FROM example\n")
    (repo / "docker/l40/Dockerfile.dockerignore").write_text(".git\n")
    (repo / "docker/l40/requirements.lock").write_text("torch==test\n")
    (repo / "docker/l40/build_native.sh").write_text("build\n")
    (repo / "docker/l40/10_nvidia.json").write_text("{}\n")
    (repo / "docker/l40/managed_experiment.sh").write_text("launch-v1\n")
    (repo / "docker/l40/vendor/SHA256SUMS").write_text("vendor\n")
    (repo / "core/csrc/flow/setup.py").write_text("setup()\n")
    (repo / "core/csrc/flow/src/kernel.cu").write_text("kernel-v1\n")
    (repo / "core/model.py").parent.mkdir(exist_ok=True)
    (repo / "core/model.py").write_text("MODEL = 1\n")
    (repo / "configs/exp.py").write_text("SEED = 42\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    first = commit(repo, "initial")
    return repo, first


def test_contract_ignores_python_and_config_but_detects_native_change(tmp_path):
    repo, first = make_contract_repo(tmp_path)
    first_contract = environment_contract(repo, first)

    (repo / "core/model.py").write_text("MODEL = 2\n")
    (repo / "configs/exp.py").write_text("SEED = 7\n")
    (repo / "docker/l40/managed_experiment.sh").write_text("launch-v2\n")
    ordinary = commit(repo, "ordinary source")
    assert environment_contract(repo, ordinary)["sha256"] == first_contract["sha256"]

    (repo / "core/csrc/flow/src/kernel.cu").write_text("kernel-v2\n")
    native = commit(repo, "native change")
    assert environment_contract(repo, native)["sha256"] != first_contract["sha256"]


def test_binding_verifies_source_environment_and_artifact_independently(
    tmp_path, monkeypatch
):
    repo, commit_id = make_contract_repo(tmp_path)
    subprocess.run(["git", "-C", str(repo), "checkout", "--detach", commit_id], check=True)
    artifact = repo / "core/csrc/flow/flow_cuda.so"
    artifact.write_bytes(b"native")
    artifact_entry = {
        "path": "core/csrc/flow/flow_cuda.so",
        "kind": "file",
        "sha256": sha256_file(artifact),
        "size_bytes": artifact.stat().st_size,
    }
    contract = environment_contract(repo, commit_id)
    snapshot = collect_source_snapshot(repo, commit_id)
    binding = {
        "schema_version": 2,
        "release": {
            "source_git_commit": commit_id,
            "source_git_remote": "",
            "source_tree_clean": True,
            "source_head_detached": True,
            "source_status_sha256": "0" * 64,
            "source_diff_sha256": "0" * 64,
            "source_snapshot": snapshot,
            "environment_contract_sha256": contract["sha256"],
        },
        "environment": {
            "environment_image_ref": "gdrnpp:env",
            "environment_image_id": "sha256:image",
            "environment_build_source_commit": "f" * 40,
            "environment_contract_sha256": contract["sha256"],
            "native_artifact_manifest_sha256": sha256_json(
                {"artifacts": [artifact_entry]}
            ),
            "native_abi": {},
        },
        "native_artifacts": [artifact_entry],
    }
    binding_path = repo / ".local/environment_binding.json"
    binding_path.parent.mkdir()
    binding_path.write_text(json.dumps(binding))

    result = verify_release_binding(repo, binding_path, "sha256:image")
    assert result["source_git_commit"] == commit_id
    assert result["environment_build_source_commit"] == "f" * 40
    assert result["verification_mode"] == "host_git"

    def no_git(*_args, **_kwargs):
        raise AssertionError("runtime verification must not execute Git")

    monkeypatch.setattr(
        "research.experiment_system.environment.collect_git_provenance", no_git
    )
    monkeypatch.setattr(
        "research.experiment_system.environment.environment_contract", no_git
    )
    monkeypatch.setattr("research.experiment_system.manifest._git", no_git)
    runtime = verify_runtime_binding(repo, binding_path, "sha256:image")
    assert runtime["source_git_commit"] == commit_id
    assert runtime["source_snapshot_sha256"] == snapshot["sha256"]
    assert runtime["verification_mode"] == "runtime_snapshot"

    with pytest.raises(RuntimeError, match="image ID"):
        verify_runtime_binding(repo, binding_path, "sha256:other")

    artifact.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="artifacts changed|artifact paths or hashes"):
        verify_runtime_binding(repo, binding_path, "sha256:image")


def test_runtime_binding_detects_tracked_source_change_without_git(
    tmp_path, monkeypatch
):
    repo, commit_id = make_contract_repo(tmp_path)
    subprocess.run(["git", "-C", str(repo), "checkout", "--detach", commit_id], check=True)
    artifact = repo / "core/csrc/flow/flow_cuda.so"
    artifact.write_bytes(b"native")
    artifact_entry = {
        "path": "core/csrc/flow/flow_cuda.so",
        "kind": "file",
        "sha256": sha256_file(artifact),
        "size_bytes": artifact.stat().st_size,
    }
    contract = environment_contract(repo, commit_id)
    binding = {
        "schema_version": 2,
        "release": {
            "source_git_commit": commit_id,
            "source_tree_clean": True,
            "source_head_detached": True,
            "source_snapshot": collect_source_snapshot(repo, commit_id),
            "environment_contract_sha256": contract["sha256"],
        },
        "environment": {
            "environment_image_id": "sha256:image",
            "environment_build_source_commit": "f" * 40,
            "environment_contract_sha256": contract["sha256"],
            "native_artifact_manifest_sha256": sha256_json(
                {"artifacts": [artifact_entry]}
            ),
        },
        "native_artifacts": [artifact_entry],
    }
    binding_path = repo / ".local/environment_binding.json"
    binding_path.parent.mkdir()
    binding_path.write_text(json.dumps(binding))
    (repo / "core/model.py").write_text("MODEL = 9\n")

    monkeypatch.setattr(
        "research.experiment_system.environment.collect_git_provenance",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("runtime verification must not execute Git")
        ),
    )
    with pytest.raises(RuntimeError, match="tracked source snapshot mismatch"):
        verify_runtime_binding(repo, binding_path, "sha256:image")
