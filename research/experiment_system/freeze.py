"""Verify the temporary B/C2 local reproduction-chain freeze."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .manifest import sha256_file


def verify_active_freeze(repo_root: Path, freeze_path: Path) -> dict[str, Any]:
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("files"), dict):
        raise ValueError(f"invalid active-run freeze file: {freeze_path}")
    checked = []
    for relative, expected in sorted(payload["files"].items()):
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"protected B/C2 file is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"protected B/C2 file changed: {relative}; expected {expected}, got {actual}"
            )
        checked.append(relative)
    return {
        "status": "PASS",
        "protected_files_checked": len(checked),
        "recorded_git_commit": payload.get("git_commit"),
        "server_last_checked": payload.get("server_observation", {}).get("last_checked"),
    }
