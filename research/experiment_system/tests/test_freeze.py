from pathlib import Path

from research.experiment_system.freeze import verify_active_freeze


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_active_bc2_reproduction_chain_matches_freeze_record():
    result = verify_active_freeze(
        PROJECT_ROOT,
        PROJECT_ROOT / "research/active_run_freeze.json",
    )
    assert result["protected_files_checked"] == 10
