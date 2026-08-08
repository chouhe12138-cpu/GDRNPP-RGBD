import json

import pytest

from research.experiment_system.artifacts import (
    create_run_directory,
    initialize_run_state,
    read_run_state,
    transition_run_state,
)


def test_non_overwriting_layout_and_state_machine(tmp_path):
    run = create_run_directory(tmp_path, "EXP-20260808-008-example", "RUN-1")
    assert (run / "meta").is_dir()
    assert (run / "diagnostics").is_dir()
    initialize_run_state(run)
    assert read_run_state(run)["status"] == "PREPARED"
    transition_run_state(run, "RUNNING", "started")
    transition_run_state(run, "COMPLETE", "finished")
    assert read_run_state(run)["status"] == "COMPLETE"
    events = [json.loads(line) for line in (run / "meta/events.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == ["PREPARED", "RUNNING", "COMPLETE"]
    with pytest.raises(FileExistsError):
        create_run_directory(tmp_path, "EXP-20260808-008-example", "RUN-1")
    with pytest.raises(ValueError):
        transition_run_state(run, "RUNNING", "cannot restart")
