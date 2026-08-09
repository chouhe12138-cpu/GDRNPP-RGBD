"""Experiment and run manifest validation and source provenance."""

from __future__ import annotations

import hashlib
import json
import os
import re
import platform
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
    if payload.get("schema_version") != 1:
        raise ValueError("run manifest schema_version must be 1")
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
    commit = source.get("git_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("source.git_commit must be a full lowercase commit SHA")
    if not isinstance(source.get("git_dirty"), bool):
        raise ValueError("source.git_dirty must be a boolean")
    config = payload.get("config")
    if not isinstance(config, dict) or not config.get("path") or not config.get("sha256"):
        raise ValueError("config.path and config.sha256 are required")
    if not isinstance(payload.get("inputs", []), list):
        raise ValueError("inputs must be a list")


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
        commit = os.environ.get("GDRN_GIT_COMMIT", "").strip()
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise RuntimeError(
                "repository has no .git metadata and no valid embedded GDRN_GIT_COMMIT"
            )
        return {
            "git_commit": commit,
            "git_remote": os.environ.get("GDRN_GIT_REMOTE", ""),
            "git_dirty": False,
            "git_status_sha256": sha256_bytes(b""),
            "git_diff_sha256": sha256_bytes(b""),
            "untracked_files": [],
            "provenance_kind": "embedded_docker_revision",
        }
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
    return {
        "git_commit": commit,
        "git_remote": remote,
        "git_dirty": bool(status.strip()),
        "git_status_sha256": sha256_bytes(status),
        "git_diff_sha256": sha256_bytes(provenance_material),
        "untracked_files": untracked_files,
        "provenance_kind": "git_worktree",
    }


def build_run_manifest(
    experiment: dict[str, Any],
    run_id: str,
    mode: str,
    seed: int,
    repo_root: Path,
    config_path: Path,
    inputs: list[dict[str, Any]] | None = None,
    image_id: str | None = None,
    image_revision: str | None = None,
    path_profile_id: str | None = None,
    parent_run_id: str | None = None,
) -> dict[str, Any]:
    validate_experiment(experiment)
    source = collect_git_provenance(repo_root)
    if mode == "formal" and source["git_dirty"]:
        raise RuntimeError("formal runs require a clean Git worktree, including untracked files")
    if mode == "formal":
        if not image_id or not image_revision:
            raise RuntimeError("formal runs require Docker image ID and image revision")
        if image_revision != source["git_commit"]:
            raise RuntimeError(
                "formal Docker image revision does not match the checked-out Git commit"
            )
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    manifest: dict[str, Any] = {
        "schema_version": 1,
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
        "execution": {
            "docker_image_id": image_id,
            "docker_image_revision": image_revision,
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
