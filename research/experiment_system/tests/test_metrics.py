import json

import pytest

from research.experiment_system.metrics import (
    index_bop_evaluation,
    verify_indexed_evaluation,
)


def write_scores(root, separator):
    result = root / "method_lmo-test"
    result.mkdir(parents=True)
    (result / "scores_bop19.json").write_text(
        json.dumps({"bop19_average_recall": 0.69})
    )
    add = result / f"error{separator}ad_ntop{separator}1"
    add.mkdir()
    (add / f"scores_th{separator}0.100_min-visib{separator}-1.000.json").write_text(
        json.dumps(
            {
                "recall": 0.5,
                "mean_obj_recall": 0.625,
                "obj_recalls": {"1": 0.625},
                "targets_count": 10,
                "gt_count": 11,
                "tp_count": 5,
            }
        )
    )


@pytest.mark.parametrize("separator", ["=", ":"])
def test_evaluator_index_supports_historical_and_actual_names(tmp_path, separator):
    write_scores(tmp_path, separator)
    index, normalized = index_bop_evaluation(tmp_path, "lmo_bop_test", "gt", "epoch_040")
    assert normalized["metrics"]["bop19_ar_macro"]["value"] == 0.69
    assert normalized["metrics"]["add_s_0.1d_macro_object"]["value"] == 0.625
    assert normalized["metrics"]["add_s_0.1d_micro_target"]["value"] == 0.5
    assert normalized["metrics"]["add_s_0.1d_micro_target"]["tp_count"] == 5
    assert index["raw_files"]["add_s_0.1d"]["path"].startswith("method_lmo-test/error")
    verify_indexed_evaluation(tmp_path)


def test_evaluator_index_rejects_ambiguous_add_scores(tmp_path):
    write_scores(tmp_path, "=")
    result = tmp_path / "method_lmo-test"
    duplicate = result / "error:ad_ntop:1"
    duplicate.mkdir()
    (duplicate / "scores_th:0.100_min-visib:-1.000.json").write_text(
        json.dumps({"recall": 0.5})
    )
    with pytest.raises(RuntimeError, match="exactly one ADD"):
        index_bop_evaluation(tmp_path, "lmo_bop_test", "gt", "epoch_040")


def test_evaluator_index_refuses_to_mislabel_micro_recall_as_macro(tmp_path):
    write_scores(tmp_path, ":")
    add_path = next(tmp_path.rglob("scores_th*0.100_min-visib*-1.000.json"))
    add_path.write_text(json.dumps({"recall": 0.5, "obj_recalls": {"1": 0.5}}))
    with pytest.raises(ValueError, match="missing mean_obj_recall"):
        index_bop_evaluation(tmp_path, "lmo_bop_test", "gt", "epoch_040")
