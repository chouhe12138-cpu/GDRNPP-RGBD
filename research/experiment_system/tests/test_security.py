from research.experiment_system.security import validate_dockerignore


def test_repository_dockerignore_excludes_machine_state_and_keeps_vendor():
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[3]
    result = validate_dockerignore(project_root)
    assert "output" in result["required_exclusions"]
    assert ".local" in result["required_exclusions"]
