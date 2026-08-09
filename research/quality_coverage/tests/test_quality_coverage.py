import json
from pathlib import Path

import pytest
import torch
from mmcv import Config

from core.gdrn_modeling.engine.engine import (
    is_better_checkpoint,
    load_bop_selection_metrics,
    should_evaluate_epoch,
)
from core.gdrn_modeling.models.heads.quality_coverage_attention import (
    QualityCoverageAttention,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = PROJECT_ROOT / "configs/gdrn/lmo_pbr"


def test_attention_is_exact_identity_at_initialization():
    torch.manual_seed(7)
    module = QualityCoverageAttention(5, 64, hidden_dim=16, max_residual=0.25)
    coor = torch.randn(2, 5, 8, 8)
    region = torch.softmax(torch.randn(2, 64, 8, 8), dim=1)
    mask = torch.rand(2, 1, 8, 8)
    output = module(coor, region, mask)
    assert torch.equal(output, region)


def test_attention_receives_gradients_without_updating_inputs():
    torch.manual_seed(11)
    module = QualityCoverageAttention(5, 8, hidden_dim=8, max_residual=0.25)
    coor = torch.randn(2, 5, 8, 8)
    region = torch.softmax(torch.randn(2, 8, 8, 8), dim=1)
    output = module(coor, region, torch.rand(2, 1, 8, 8))
    weighted_loss = (output * torch.randn_like(output)).sum()
    weighted_loss.backward()
    assert module.quality_net[-1].weight.grad.abs().sum() > 0
    assert module.coverage_net[-1].weight.grad.abs().sum() > 0
    assert coor.grad is None
    assert region.grad is None


def test_attention_rejects_invalid_shapes():
    module = QualityCoverageAttention(5, 8)
    with pytest.raises(ValueError):
        module(torch.randn(2, 4, 8, 8), torch.randn(2, 8, 8, 8), None)
    with pytest.raises(ValueError):
        module(torch.randn(2, 5, 8, 8), torch.randn(2, 7, 8, 8), None)


def test_formal_and_local_protocol_configs():
    formal = Config.fromfile(str(CONFIG_ROOT / "convnext_stage3c1_quality_coverage_lmo.py"))
    local = Config.fromfile(str(CONFIG_ROOT / "convnext_stage3c1_quality_coverage_local_lmo.py"))

    assert tuple(formal.DATASETS.TRAIN) == ("lmo_pbr_train",)
    assert tuple(formal.DATASETS.TEST) == ("lmo_bop_test",)
    assert formal.MODEL.POSE_NET.BACKBONE.FREEZE
    assert formal.MODEL.POSE_NET.GEO_HEAD.FREEZE
    assert formal.MODEL.POSE_NET.PNP_NET.FREEZE
    assert formal.MODEL.POSE_NET.QUALITY_COVERAGE.ENABLED
    assert not formal.MODEL.POSE_NET.QUALITY_COVERAGE.FREEZE
    assert formal.SOLVER.TOTAL_EPOCHS == 40
    assert formal.SOLVER.CHECKPOINT_PERIOD == 5
    assert formal.SOLVER.MAX_TO_KEEP == 3
    assert formal.SOLVER.BEST_CHECKPOINT.ENABLED
    assert formal.TEST.EVAL_PERIOD == 5
    assert formal.TEST.TEST_BBOX_TYPE == "gt"
    assert not formal.TEST.USE_PNP

    assert tuple(local.DATASETS.TRAIN) == ("lmo_pbr_stage3_local_train",)
    assert tuple(local.DATASETS.TEST) == ()
    assert local.SOLVER.IMS_PER_BATCH == 4
    assert local.SOLVER.REFERENCE_BS == 48
    assert local.SOLVER.TOTAL_EPOCHS == 1


def test_conditional_control_configs_share_formal_budget():
    baseline = Config.fromfile(str(CONFIG_ROOT / "convnext_stage3c0_pnp_only_lmo.py"))
    method = Config.fromfile(str(CONFIG_ROOT / "convnext_stage3c2_pnp_quality_coverage_lmo.py"))
    for cfg in (baseline, method):
        assert tuple(cfg.DATASETS.TRAIN) == ("lmo_pbr_train",)
        assert cfg.SOLVER.TOTAL_EPOCHS == 40
        assert cfg.SOLVER.CHECKPOINT_PERIOD == 5
        assert cfg.TEST.EVAL_PERIOD == 5
    assert not baseline.MODEL.POSE_NET.PNP_NET.FREEZE
    assert not baseline.MODEL.POSE_NET.QUALITY_COVERAGE.ENABLED
    assert not method.MODEL.POSE_NET.PNP_NET.FREEZE
    assert method.MODEL.POSE_NET.QUALITY_COVERAGE.ENABLED
    assert method.MODEL.POSE_NET.PNP_NET.LR_MULT == pytest.approx(0.1)
    for cfg in (baseline, method):
        assert cfg.RUN_ARTIFACTS.STRUCTURED_LAYOUT
        assert cfg.RUN_ARTIFACTS.COMPACT_LOG
        assert cfg.RUN_ARTIFACTS.TENSORBOARD is False
        assert cfg.RUN_ARTIFACTS.SKIP_DUPLICATE_FINAL_EVAL


def test_epoch_evaluation_happens_only_after_completed_epochs():
    iters = 100
    assert not should_evaluate_epoch(399, iters, 5)
    assert should_evaluate_epoch(499, iters, 5)
    assert not should_evaluate_epoch(500, iters, 5)
    assert should_evaluate_epoch(999, iters, 5)
    assert not should_evaluate_epoch(499, iters, 0)


def test_checkpoint_metric_priority_and_tie_break():
    incumbent = {"bop_ar": 0.69, "add_s_0.1d": 0.50}
    assert is_better_checkpoint({"bop_ar": 0.692, "add_s_0.1d": 0.40}, incumbent)
    assert is_better_checkpoint({"bop_ar": 0.6905, "add_s_0.1d": 0.51}, incumbent)
    assert not is_better_checkpoint({"bop_ar": 0.688, "add_s_0.1d": 0.80}, incumbent)


@pytest.mark.parametrize(
    "add_dir_name,score_name",
    [
        ("error=ad_ntop=-1", "scores_th=0.100_min-visib=-1.000.json"),
        ("error:ad_ntop:1", "scores_th:0.100_min-visib:-1.000.json"),
    ],
)
def test_load_bop_selection_metrics(tmp_path, add_dir_name, score_name):
    result = tmp_path / "method_lmo-test"
    result.mkdir()
    (result / "scores_bop19.json").write_text(
        json.dumps({"bop19_average_recall": 0.691}),
        encoding="utf-8",
    )
    add_dir = result / add_dir_name
    add_dir.mkdir()
    (add_dir / score_name).write_text(
        json.dumps({"recall": 0.512}),
        encoding="utf-8",
    )
    assert load_bop_selection_metrics(tmp_path) == {
        "bop_ar": pytest.approx(0.691),
        "add_s_0.1d": pytest.approx(0.512),
        "add_s_obj_recalls": {},
    }


def test_load_bop_selection_metrics_rejects_duplicate_add_scores(tmp_path):
    result = tmp_path / "method_lmo-test"
    result.mkdir()
    (result / "scores_bop19.json").write_text(
        json.dumps({"bop19_average_recall": 0.691}),
        encoding="utf-8",
    )
    for directory, score_name in (
        ("error=ad_ntop=-1", "scores_th=0.100_min-visib=-1.000.json"),
        ("error:ad_ntop:1", "scores_th:0.100_min-visib:-1.000.json"),
    ):
        add_dir = result / directory
        add_dir.mkdir()
        (add_dir / score_name).write_text(
            json.dumps({"recall": 0.512}),
            encoding="utf-8",
        )
    assert load_bop_selection_metrics(tmp_path) == {
        "bop_ar": pytest.approx(0.691),
    }
