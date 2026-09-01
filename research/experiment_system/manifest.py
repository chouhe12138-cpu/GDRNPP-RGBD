"""Experiment and run manifest validation and source provenance."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json


EXPERIMENT_ID_RE = re.compile(r"^EXP-\d{8}-\d{3}-[a-z0-9][a-z0-9-]*$")
RUN_ID_RE = re.compile(
    r"^RUN-\d{8}-\d{6}-(smoke|audit|formal|diagnostic)-s\d+-a\d{2}$"
)
RUN_MODES = {"smoke", "audit", "formal", "diagnostic"}
EXPERIMENT_STATES = {
    "PLANNED",
    "AUTHORIZED",
    "RUNNING",
    "PAUSED",
    "COMPLETE",
    "FAILED",
    "TRIGGERED",
    "LEGACY",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(canonical)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def validate_experiment(payload: dict[str, Any], directory_name: str | None = None) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("experiment schema_version must be 1")
    experiment_id = _require_string(payload, "experiment_id")
    if not EXPERIMENT_ID_RE.fullmatch(experiment_id):
        raise ValueError(f"invalid experiment_id: {experiment_id}")
    if directory_name is not None and not directory_name.startswith(experiment_id):
        raise ValueError(
            f"experiment directory {directory_name!r} does not start with {experiment_id!r}"
        )
    _require_string(payload, "title")
    _require_string(payload, "stage")
    status = _require_string(payload, "status")
    if status not in EXPERIMENT_STATES:
        raise ValueError(f"invalid experiment status: {status}")
    if not isinstance(payload.get("legacy_import"), bool):
        raise ValueError("legacy_import must be a boolean")
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be an object")
    metrics = protocol.get("metrics", [])
    if not isinstance(metrics, list) or not all(isinstance(item, str) for item in metrics):
        raise ValueError("protocol.metrics must be a list of strings")
    evidence = payload.get("evidence", {})
    if not isinstance(evidence, dict):
        raise ValueError("evidence must be an object")


def validate_run_manifest(payload: dict[str, Any]) -> None:
    schema_version = payload.get("schema_version")
    if schema_version not in {1, 2}:
        raise ValueError("run manifest schema_version must be 1 or 2")
    experiment_id = _require_string(payload, "experiment_id")
    if not EXPERIMENT_ID_RE.fullmatch(experiment_id):
        raise ValueError(f"invalid experiment_id: {experiment_id}")
    run_id = _require_string(payload, "run_id")
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"invalid run_id: {run_id}")
    if payload.get("mode") not in RUN_MODES:
        raise ValueError(f"invalid run mode: {payload.get('mode')}")
    source = payload.get("source")
    if not isinstance(source, dict):
        raise ValueError("source must be an object")
    commit_key = "git_commit" if schema_version == 1 else "source_git_commit"
    clean_key = "git_dirty" if schema_version == 1 else "source_tree_clean"
    commit = source.get(commit_key)
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(f"source.{commit_key} must be a full lowercase commit SHA")
    if not isinstance(source.get(clean_key), bool):
        raise ValueError(f"source.{clean_key} must be a boolean")
    if schema_version == 2 and not isinstance(source.get("source_head_detached"), bool):
        raise ValueError("source.source_head_detached must be a boolean")
    if schema_version == 2 and payload["mode"] == "formal":
        if not source["source_tree_clean"]:
            raise ValueError("formal source.source_tree_clean must be true")
        if not source["source_head_detached"]:
            raise ValueError("formal source.source_head_detached must be true")
    config = payload.get("config")
    if not isinstance(config, dict) or not config.get("path") or not config.get("sha256"):
        raise ValueError("config.path and config.sha256 are required")
    if not isinstance(payload.get("inputs", []), list):
        raise ValueError("inputs must be a list")
    if schema_version == 2:
        environment = payload.get("environment")
        if not isinstance(environment, dict):
            raise ValueError("environment must be an object")
        if payload["mode"] == "formal":
            required = (
                "environment_image_id",
                "environment_build_source_commit",
                "environment_contract_sha256",
                "native_artifact_manifest_sha256",
                "environment_binding_sha256",
            )
            for key in required:
                if not isinstance(environment.get(key), str) or not environment[key]:
                    raise ValueError(f"environment.{key} is required for formal runs")
            if not environment["environment_image_id"].startswith("sha256:"):
                raise ValueError("environment.environment_image_id must be immutable")
            if not re.fullmatch(
                r"[0-9a-f]{40}", environment["environment_build_source_commit"]
            ):
                raise ValueError(
                    "environment.environment_build_source_commit must be a full commit"
                )
            for key in (
                "environment_contract_sha256",
                "native_artifact_manifest_sha256",
                "environment_binding_sha256",
            ):
                if not re.fullmatch(r"[0-9a-f]{64}", environment[key]):
                    raise ValueError(f"environment.{key} must be a SHA-256")
        if not isinstance(environment.get("native_artifacts", []), list):
            raise ValueError("environment.native_artifacts must be a list")


def manifest_source_commit(payload: dict[str, Any]) -> str:
    """Return source commit from either historical v1 or current v2 manifests."""

    source = payload["source"]
    if payload.get("schema_version") == 1:
        return source["git_commit"]
    return source["source_git_commit"]


def manifest_environment_image_id(payload: dict[str, Any]) -> str | None:
    """Return immutable environment image ID across manifest schema versions."""

    if payload.get("schema_version") == 1:
        return payload.get("execution", {}).get("docker_image_id")
    return payload.get("environment", {}).get("environment_image_id")


def make_run_id(mode: str, seed: int, attempt: int = 1, now: datetime | None = None) -> str:
    if mode not in RUN_MODES:
        raise ValueError(f"invalid run mode: {mode}")
    if seed < 0 or attempt < 1 or attempt > 99:
        raise ValueError("seed must be nonnegative and attempt must be in [1, 99]")
    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return f"RUN-{timestamp:%Y%m%d-%H%M%S}-{mode}-s{seed}-a{attempt:02d}"


def _git(repo_root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo_root), *args])


def collect_git_provenance(repo_root: Path) -> dict[str, Any]:
    if not (repo_root / ".git").exists():
        raise RuntimeError(
            "source checkout has no .git metadata; image-embedded revision is not "
            "accepted as source provenance"
        )
    commit = _git(repo_root, "rev-parse", "HEAD").decode().strip()
    remote = ""
    try:
        remote = _git(repo_root, "remote", "get-url", "origin").decode().strip()
        remote = re.sub(r"(https?://)[^/@]+@", r"\1", remote)
    except subprocess.CalledProcessError:
        pass
    status = _git(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    diff = _git(repo_root, "diff", "--binary", "HEAD")
    staged = _git(repo_root, "diff", "--binary", "--cached", "HEAD")
    untracked_output = _git(
        repo_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    untracked_files = []
    for raw_path in untracked_output.split(b"\0"):
        if not raw_path:
            continue
        relative = raw_path.decode("utf-8")
        path = repo_root / relative
        if path.is_file():
            untracked_files.append(
                {
                    "path": relative,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    untracked_material = json.dumps(
        untracked_files,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    provenance_material = status + b"\0" + diff + b"\0" + staged + b"\0" + untracked_material
    detached = subprocess.run(
        ["git", "-C", str(repo_root), "symbolic-ref", "-q", "HEAD"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode != 0
    return {
        "source_git_commit": commit,
        "source_git_remote": remote,
        "source_tree_clean": not bool(status.strip()),
        "source_head_detached": detached,
        "source_status_sha256": sha256_bytes(status),
        "source_diff_sha256": sha256_bytes(provenance_material),
        "untracked_files": untracked_files,
        "provenance_kind": "git_release_checkout",
    }


def _tracked_source_entry(repo_root: Path, relative: str) -> dict[str, Any]:
    relative_path = Path(relative)
    if (
        not relative
        or relative_path.is_absolute()
        or ".." in relative_path.parts
        or relative_path.as_posix() != relative
    ):
        raise ValueError(f"invalid tracked source path: {relative!r}")
    path = repo_root / relative_path
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"tracked source path is missing: {relative}") from exc
    if stat.S_ISLNK(info.st_mode):
        return {
            "path": relative,
            "kind": "symlink",
            "target": os.readlink(path),
        }
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"unsupported tracked source type: {relative}")
    return {
        "path": relative,
        "kind": "file",
        "sha256": sha256_file(path),
        "size_bytes": info.st_size,
        "executable": bool(info.st_mode & stat.S_IXUSR),
    }


def collect_source_snapshot(repo_root: Path, commit: str | None = None) -> dict[str, Any]:
    """Hash every tracked working-tree path for gitless runtime verification."""

    repo_root = repo_root.resolve()
    head_commit = _git(repo_root, "rev-parse", "HEAD").decode().strip()
    resolved_commit = commit or head_commit
    if not re.fullmatch(r"[0-9a-f]{40}", resolved_commit):
        raise ValueError("source snapshot requires a full lowercase Git commit")
    if resolved_commit != head_commit:
        raise RuntimeError("source snapshot commit does not match checked-out HEAD")
    raw_paths = _git(repo_root, "ls-files", "-z")
    paths = sorted(
        raw.decode("utf-8") for raw in raw_paths.split(b"\0") if raw
    )
    if not paths:
        raise RuntimeError("source snapshot selected no tracked files")
    entries = [_tracked_source_entry(repo_root, relative) for relative in paths]
    material = {
        "schema_version": 1,
        "source_git_commit": resolved_commit,
        "files": entries,
    }
    return {
        **material,
        "file_count": len(entries),
        "sha256": sha256_json(material),
    }


def verify_source_snapshot(repo_root: Path, snapshot: dict[str, Any]) -> dict[str, Any]:
    """Verify a host-created source snapshot without executing Git."""

    if snapshot.get("schema_version") != 1:
        raise ValueError("source snapshot schema_version must be 1")
    commit = snapshot.get("source_git_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("source snapshot has no valid source_git_commit")
    expected = snapshot.get("files")
    if not isinstance(expected, list) or not expected:
        raise ValueError("source snapshot files must be a non-empty list")
    paths = [entry.get("path") for entry in expected if isinstance(entry, dict)]
    if len(paths) != len(expected) or not all(isinstance(path, str) for path in paths):
        raise ValueError("source snapshot contains an invalid file entry")
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("source snapshot paths must be sorted and unique")
    if snapshot.get("file_count") != len(expected):
        raise ValueError("source snapshot file_count mismatch")
    material = {
        "schema_version": 1,
        "source_git_commit": commit,
        "files": expected,
    }
    expected_sha = sha256_json(material)
    if snapshot.get("sha256") != expected_sha:
        raise RuntimeError("source snapshot manifest hash mismatch")
    for expected_entry in expected:
        actual_entry = _tracked_source_entry(repo_root.resolve(), expected_entry["path"])
        if actual_entry != expected_entry:
            raise RuntimeError(
                f"tracked source snapshot mismatch: {expected_entry['path']}"
            )
    return {
        "source_git_commit": commit,
        "source_snapshot_sha256": expected_sha,
        "source_files_checked": len(expected),
    }


def collect_bound_source_provenance(
    repo_root: Path, environment_binding: dict[str, Any]
) -> dict[str, Any]:
    """Build run provenance from a verified v2 release binding, without Git."""

    if environment_binding.get("schema_version") != 2:
        raise ValueError("runtime provenance requires environment binding schema_version 2")
    release = environment_binding.get("release")
    if not isinstance(release, dict):
        raise ValueError("environment binding release must be an object")
    snapshot = release.get("source_snapshot")
    if not isinstance(snapshot, dict):
        raise ValueError("environment binding has no source snapshot")
    checked = verify_source_snapshot(repo_root, snapshot)
    commit = release.get("source_git_commit")
    if commit != checked["source_git_commit"]:
        raise RuntimeError("release commit does not match its source snapshot")
    if release.get("source_tree_clean") is not True:
        raise RuntimeError("release binding was not prepared from a clean source tree")
    if release.get("source_head_detached") is not True:
        raise RuntimeError("release binding was not prepared from detached HEAD")
    return {
        "source_git_commit": commit,
        "source_git_remote": release.get("source_git_remote", ""),
        "source_tree_clean": True,
        "source_head_detached": True,
        "source_status_sha256": release.get("source_status_sha256"),
        "source_diff_sha256": release.get("source_diff_sha256"),
        "source_snapshot_sha256": checked["source_snapshot_sha256"],
        "source_files_checked": checked["source_files_checked"],
        "untracked_files": [],
        "provenance_kind": "verified_release_binding_snapshot",
    }


def build_run_manifest(
    experiment: dict[str, Any],
    run_id: str,
    mode: str,
    seed: int,
    repo_root: Path,
    config_path: Path,
    inputs: list[dict[str, Any]] | None = None,
    environment_binding: dict[str, Any] | None = None,
    path_profile_id: str | None = None,
    parent_run_id: str | None = None,
) -> dict[str, Any]:
    validate_experiment(experiment)
    source = (
        collect_bound_source_provenance(repo_root, environment_binding)
        if environment_binding
        else collect_git_provenance(repo_root)
    )
    if mode == "formal" and not source["source_tree_clean"]:
        raise RuntimeError("formal runs require a clean Git worktree, including untracked files")
    if mode == "formal":
        if not source["source_head_detached"]:
            raise RuntimeError("formal runs require a detached release checkout")
        if not environment_binding:
            raise RuntimeError("formal runs require a verified environment binding")
        release = environment_binding.get("release", {})
        if release.get("source_git_commit") != source["source_git_commit"]:
            raise RuntimeError("environment binding belongs to a different source commit")
        if release.get("environment_contract_sha256") != environment_binding.get(
            "environment", {}
        ).get("environment_contract_sha256"):
            raise RuntimeError("release and environment contract identities differ")
    environment = (environment_binding or {}).get("environment", {})
    native_artifacts = (environment_binding or {}).get("native_artifacts", [])
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "experiment_id": experiment["experiment_id"],
        "experiment_protocol_sha256": sha256_json(experiment),
        "run_id": run_id,
        "mode": mode,
        "seed": int(seed),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "config": {
            "path": str(config_path.relative_to(repo_root.resolve())),
            "sha256": sha256_file(config_path),
        },
        "inputs": inputs or [],
        "environment": {
            "environment_image_id": environment.get("environment_image_id"),
            "environment_image_ref": environment.get("environment_image_ref"),
            "environment_build_source_commit": environment.get(
                "environment_build_source_commit"
            ),
            "environment_contract_sha256": environment.get(
                "environment_contract_sha256"
            ),
            "native_artifact_manifest_sha256": environment.get(
                "native_artifact_manifest_sha256"
            ),
            "native_abi": environment.get("native_abi"),
            "native_artifacts": native_artifacts,
            "environment_binding_sha256": (
                sha256_json(environment_binding) if environment_binding else None
            ),
        },
        "execution": {
            "path_profile_id": path_profile_id,
            "python_version": platform.python_version(),
            "platform": sys.platform,
        },
    }
    if parent_run_id:
        manifest["parent_run_id"] = parent_run_id
    validate_run_manifest(manifest)
    return manifest


def write_run_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    validate_run_manifest(manifest)
    path = run_dir / "meta" / "run_manifest.json"
    if path.exists():
        raise FileExistsError(f"run manifest already exists: {path}")
    atomic_write_json(path, manifest)
    return path
