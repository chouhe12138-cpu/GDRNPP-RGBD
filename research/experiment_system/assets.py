"""Resolve logical research assets through ignored machine profiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .manifest import sha256_file


def load_asset_catalog(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("assets"), dict):
        raise ValueError(f"invalid asset catalog: {path}")
    return payload["assets"]


def load_path_profile(path: Path) -> tuple[str, dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"invalid path profile schema: {path}")
    profile_id = payload.get("profile_id")
    assets = payload.get("assets")
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError("path profile requires profile_id")
    if not isinstance(assets, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in assets.items()
    ):
        raise ValueError("path profile assets must map string IDs to string paths")
    return profile_id, assets


def validate_asset(asset_id: str, spec: dict[str, Any], resolved_path: Path) -> dict[str, Any]:
    kind = spec.get("kind")
    if kind not in {"file", "directory"}:
        raise ValueError(f"asset {asset_id} has invalid kind: {kind}")
    if kind == "file" and not resolved_path.is_file():
        raise FileNotFoundError(f"asset {asset_id} file is missing: {resolved_path}")
    if kind == "directory" and not resolved_path.is_dir():
        raise FileNotFoundError(f"asset {asset_id} directory is missing: {resolved_path}")

    result: dict[str, Any] = {
        "asset_id": asset_id,
        "kind": kind,
        "resolved_path": str(resolved_path.resolve()),
    }
    expected_sha = spec.get("sha256")
    if expected_sha:
        if kind != "file":
            raise ValueError(f"directory asset {asset_id} cannot use a whole-file SHA-256")
        actual_sha = sha256_file(resolved_path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"asset {asset_id} SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
            )
        result["sha256"] = actual_sha

    expected_size = spec.get("size_bytes")
    if expected_size is not None:
        actual_size = resolved_path.stat().st_size
        if actual_size != int(expected_size):
            raise RuntimeError(
                f"asset {asset_id} size mismatch: expected {expected_size}, got {actual_size}"
            )
        result["size_bytes"] = actual_size

    sentinels = spec.get("sentinels", [])
    if not isinstance(sentinels, list) or not all(isinstance(item, str) for item in sentinels):
        raise ValueError(f"asset {asset_id} sentinels must be a list of strings")
    missing = [item for item in sentinels if not (resolved_path / item).exists()]
    if missing:
        raise FileNotFoundError(f"asset {asset_id} is missing sentinels: {missing}")
    result["sentinels"] = sentinels
    return result


def resolve_assets(
    catalog_path: Path,
    profile_path: Path,
    required_ids: list[str] | None = None,
) -> dict[str, Any]:
    catalog = load_asset_catalog(catalog_path)
    profile_id, paths = load_path_profile(profile_path)
    selected = required_ids or sorted(catalog)
    unknown = sorted(set(selected) - set(catalog))
    if unknown:
        raise KeyError(f"unknown asset IDs: {unknown}")
    missing_paths = sorted(set(selected) - set(paths))
    if missing_paths:
        raise KeyError(f"path profile {profile_id} is missing assets: {missing_paths}")
    resolved = [
        validate_asset(asset_id, catalog[asset_id], Path(paths[asset_id]))
        for asset_id in selected
    ]
    return {
        "schema_version": 1,
        "profile_id": profile_id,
        "assets": resolved,
    }
