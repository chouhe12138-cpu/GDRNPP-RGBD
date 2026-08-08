import pytest

from research.experiment_system.summaries import compare_screening_metrics


def experiment():
    return {
        "schema_version": 1,
        "experiment_id": "EXP-20260808-008-example",
        "title": "示例",
        "stage": "future",
        "status": "PLANNED",
        "legacy_import": False,
        "protocol": {
            "metrics": ["bop19_ar_macro", "add_s_0.1d_macro_object"],
            "gate": {
                "minimum_bop19_ar_macro_delta": 0.005,
                "minimum_add_s_0.1d_macro_object_delta": 0.01,
                "minimum_nonnegative_objects": 5,
            },
        },
        "evidence": {},
    }


def metrics(bop, add, objects, checkpoint):
    return {
        "schema_version": 1,
        "dataset_id": "lmo_bop_test",
        "bbox_type": "gt",
        "checkpoint_id": checkpoint,
        "metrics": {
            "bop19_ar_macro": {"value": bop},
            "add_s_0.1d_macro_object": {"value": add, "object_recalls": objects},
        },
    }


def test_screening_uses_fraction_deltas_and_reports_percentage_points():
    baseline_objects = {str(index): 0.4 for index in range(1, 9)}
    result_objects = {str(index): 0.41 for index in range(1, 9)}
    summary = compare_screening_metrics(
        experiment(),
        metrics(0.69, 0.50, baseline_objects, "official"),
        metrics(0.70, 0.52, result_objects, "epoch_040"),
    )
    assert summary["status"] == "SCREEN_PASS"
    assert summary["metrics"]["bop19_ar_macro"]["delta_percentage_points"] == pytest.approx(1.0)
    assert summary["nonnegative_objects"] == 8
