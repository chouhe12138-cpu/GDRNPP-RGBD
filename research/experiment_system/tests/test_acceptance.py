import json
from pathlib import Path

import pytest

from research.experiment_system.acceptance import (
    evaluate_acceptance,
    render_acceptance_index,
    write_acceptance,
)
from research.experiment_system.manifest import sha256_file


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_spec(root: Path, experiment_id: str, expected: float = 0.5) -> Path:
    evidence = root / "evidence"
    evidence.mkdir()
    (evidence / "rows.csv").write_text("id,value\n1,0.5\n", encoding="utf-8")
    write_json(evidence / "summary.json", {"metric": expected})
    (evidence / "pose.csv").write_text("pose\n", encoding="utf-8")
    write_json(
        evidence / "hashes.json",
        {"pose.csv": sha256_file(evidence / "pose.csv")},
    )
    spec_path = (
        root / "research" / "experiments" / experiment_id / "ACCEPTANCE_SPEC.json"
    )
    write_json(
        spec_path,
        {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "scope": "unit test",
            "status_on_pass": "PASS",
            "required_paths": [{"path": "evidence/summary.json"}],
            "json_checks": [
                {
                    "name": "metric",
                    "path": "evidence/summary.json",
                    "pointer": ["metric"],
                    "expected": 0.5,
                    "tolerance": 0.001,
                }
            ],
            "csv_checks": [{"path": "evidence/rows.csv", "expected_rows": 1}],
            "hash_manifests": [{"path": "evidence/hashes.json", "base": "evidence"}],
        },
    )
    return spec_path


def test_acceptance_pass_and_non_overwriting_write(tmp_path):
    experiment_id = "EXP-20260808-999-unit-test"
    spec = make_spec(tmp_path, experiment_id)
    result = evaluate_acceptance(spec, tmp_path, "a" * 40)
    assert result["status"] == "PASS"
    assert {item["status"] for item in result["checks"]} == {"PASS"}

    write_acceptance(spec, result)
    assert (spec.parent / "ACCEPTANCE.json").is_file()
    assert (spec.parent / "ACCEPTANCE_CN.md").is_file()
    with pytest.raises(FileExistsError):
        write_acceptance(spec, result)


def test_acceptance_conflict_and_pending_external(tmp_path):
    experiment_id = "EXP-20260808-998-unit-test"
    spec = make_spec(tmp_path, experiment_id, expected=0.7)
    result = evaluate_acceptance(spec, tmp_path, "b" * 40)
    assert result["status"] == "CONFLICT"

    payload = json.loads(spec.read_text(encoding="utf-8"))
    payload["json_checks"] = []
    payload["required_paths"].append(
        {"path": "/server/result", "type": "directory", "external": True}
    )
    spec.write_text(json.dumps(payload), encoding="utf-8")
    result = evaluate_acceptance(spec, tmp_path, "b" * 40)
    assert result["status"] == "PENDING_EXTERNAL"
    assert "PENDING_EXTERNAL" in render_acceptance_index([result])
