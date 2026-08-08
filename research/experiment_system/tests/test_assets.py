import json

import pytest

from research.experiment_system.assets import resolve_assets
from research.experiment_system.manifest import sha256_file


def test_asset_profile_resolves_files_and_directories(tmp_path):
    weight = tmp_path / "model.pth"
    weight.write_bytes(b"official")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "sentinel.json").write_text("{}")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "assets": {
                    "weight": {
                        "kind": "file",
                        "sha256": sha256_file(weight),
                        "size_bytes": weight.stat().st_size,
                        "sentinels": [],
                    },
                    "dataset": {
                        "kind": "directory",
                        "sentinels": ["sentinel.json"],
                    },
                },
            }
        )
    )
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile_id": "test",
                "assets": {"weight": str(weight), "dataset": str(dataset)},
            }
        )
    )
    result = resolve_assets(catalog, profile)
    assert result["profile_id"] == "test"
    assert {item["asset_id"] for item in result["assets"]} == {"weight", "dataset"}


def test_asset_profile_rejects_missing_asset(tmp_path):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"schema_version": 1, "assets": {"x": {"kind": "file"}}}))
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"schema_version": 1, "profile_id": "test", "assets": {}}))
    with pytest.raises(KeyError):
        resolve_assets(catalog, profile)
