import json
from pathlib import Path

from research.experiment_system.registry import (
    compare_generated_registry,
    load_experiment_registry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_checked_in_registry_matches_experiment_metadata():
    records = load_experiment_registry(PROJECT_ROOT / "research/experiments")
    assert len(records) == 9
    assert len({record["experiment_id"] for record in records}) == 9
    compare_generated_registry(
        PROJECT_ROOT / "research/experiments",
        PROJECT_ROOT / "research/experiment_index.json",
        PROJECT_ROOT / "research/EXPERIMENT_INDEX.md",
    )


def test_history_artifact_index_covers_every_registered_experiment():
    records = load_experiment_registry(PROJECT_ROOT / "research/experiments")
    artifact_index = json.loads(
        (PROJECT_ROOT / "research/artifact_index.json").read_text(encoding="utf-8")
    )
    indexed = [record["experiment_id"] for record in artifact_index["experiments"]]
    assert len(indexed) == len(set(indexed))
    assert set(indexed) == {record["experiment_id"] for record in records}
    assert all(record["artifacts"] for record in artifact_index["experiments"])
