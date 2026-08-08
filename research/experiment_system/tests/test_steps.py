import json

import pytest

from research.experiment_system.artifacts import (
    create_run_directory,
    initialize_run_state,
    read_run_state,
)
from research.experiment_system.steps import register_step, transition_step


def test_steps_record_detached_workflow_and_complete_after_summary(tmp_path):
    run = create_run_directory(tmp_path, "EXP-20260808-008-example", "RUN-1")
    initialize_run_state(run)
    register_step(run, "train", "train", "train.sh config.py")
    transition_step(run, "train", "RUNNING", "background job launched")
    transition_step(run, "train", "COMPLETE", "checkpoint written")
    register_step(run, "summary", "summarize", "python summarize.py")
    transition_step(run, "summary", "RUNNING", "summary started")
    transition_step(run, "summary", "COMPLETE", "result accepted", complete_run=True)
    assert read_run_state(run)["status"] == "COMPLETE"
    steps = json.loads((run / "meta/steps.json").read_text())["steps"]
    assert [step["status"] for step in steps] == ["COMPLETE", "COMPLETE"]


def test_failed_step_marks_run_failed(tmp_path):
    run = create_run_directory(tmp_path, "EXP-20260808-008-example", "RUN-1")
    initialize_run_state(run)
    register_step(run, "eval", "eval", "eval.sh")
    transition_step(run, "eval", "RUNNING", "started")
    transition_step(run, "eval", "FAILED", "evaluator failed")
    assert read_run_state(run)["status"] == "FAILED"
    with pytest.raises(ValueError):
        transition_step(run, "eval", "RUNNING", "retry in same run")
