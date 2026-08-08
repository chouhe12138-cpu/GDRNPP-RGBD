"""Repository hygiene checks for formal runs and Docker build contexts."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


REQUIRED_DOCKER_EXCLUDES = {
    ".git",
    ".local",
    "datasets",
    "pretrained_models",
    "output",
    "outputs",
    "logs",
    "secrets",
    ".env",
    "credentials*",
}

BLOCKED_STAGED_SUFFIXES = {
    ".pth",
    ".ckpt",
    ".pkl",
    ".npy",
    ".npz",
    ".engine",
    ".onnx",
    ".pem",
    ".key",
}

BLOCKED_PATH_PARTS = {
    ".local",
    "datasets",
    "output",
    "outputs",
    "logs",
    "secrets",
    "weights",
    "pretrained_models",
}


def validate_dockerignore(repo_root: Path) -> dict[str, Any]:
    path = repo_root / ".dockerignore"
    if not path.is_file():
        raise FileNotFoundError(".dockerignore is required because the Dockerfile uses COPY .")
    rules = {
        line.strip().rstrip("/")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#") and not line.startswith("!")
    }
    missing = sorted(REQUIRED_DOCKER_EXCLUDES - rules)
    if missing:
        raise RuntimeError(f".dockerignore is missing required exclusions: {missing}")
    if "!docker/l40/vendor/*.tar.gz" not in path.read_text(encoding="utf-8"):
        raise RuntimeError(".dockerignore must preserve checksum-pinned Docker vendor archives")
    return {"path": str(path), "required_exclusions": sorted(REQUIRED_DOCKER_EXCLUDES)}


def staged_paths(repo_root: Path) -> list[Path]:
    output = subprocess.check_output(
        ["git", "-C", str(repo_root), "diff", "--cached", "--name-only", "-z"]
    )
    return [Path(item.decode("utf-8")) for item in output.split(b"\0") if item]


def staged_path_violations(repo_root: Path, max_bytes: int = 10 * 1024 * 1024) -> list[str]:
    violations: list[str] = []
    for relative in staged_paths(repo_root):
        path = repo_root / relative
        lower_name = relative.name.lower()
        parts = set(relative.parts)
        if parts & BLOCKED_PATH_PARTS:
            violations.append(f"blocked path: {relative}")
        if relative.suffix.lower() in BLOCKED_STAGED_SUFFIXES or lower_name.startswith("credentials"):
            violations.append(f"blocked file type/name: {relative}")
        if path.is_file() and path.stat().st_size > max_bytes:
            allowed_vendor = (
                relative.parts[:3] == ("docker", "l40", "vendor")
                and relative.name.endswith((".tar.gz", ".tgz", ".whl"))
            )
            if not allowed_vendor:
                violations.append(f"large staged file ({path.stat().st_size} bytes): {relative}")
    return sorted(set(violations))


def audit_repository(repo_root: Path) -> dict[str, Any]:
    docker = validate_dockerignore(repo_root)
    violations = staged_path_violations(repo_root)
    if violations:
        raise RuntimeError("staged repository hygiene violations: " + "; ".join(violations))
    return {
        "status": "PASS",
        "dockerignore": docker,
        "staged_paths_checked": len(staged_paths(repo_root)),
    }
