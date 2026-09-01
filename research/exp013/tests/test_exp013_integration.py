from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from mmcv import Config

from core.gdrn_modeling.engine.engine_utils import geometry_supervision_enabled

from research.exp013.interventions import (
    EXP013_THREE_PATH_CONDITIONS,
    apply_cpm_xyz_region_intervention,
    cpm_xyz_region_condition,
)
from research.exp013.summarize import attention_effective, versus_exp012


ROOT = Path(__file__).resolve().parents[3]
CONFIGS = {
    "A": ROOT / "configs/gdrn/lmo_pbr/research/exp013/a_xyz_residual/train.py",
    "B": ROOT / "configs/gdrn/lmo_pbr/research/exp013/b_geometry_attention/train.py",
    "C": ROOT / "configs/gdrn/lmo_pbr/research/exp013/c_rt_decoupled/train.py",
    "F": ROOT / "configs/gdrn/lmo_pbr/research/exp013/f_glm_pose_l/train.py",
}


def test_formal_config_contracts_and_head_types():
    expected = {
        "A": "XYZResidualBypassPnPNet",
        "B": "GeometryAttentionResidualPnPNet",
        "C": "RTDecoupledGeometryPnPNet",
        "F": "GLMPoseLNet",
    }
    for variant, path in CONFIGS.items():
        cfg = Config.fromfile(str(path))
        pose = cfg.MODEL.POSE_NET
        assert cfg.SEED == 42
        assert cfg.SOLVER.TOTAL_EPOCHS == 40
        assert cfg.SOLVER.IMS_PER_BATCH == 48
        assert cfg.SOLVER.CHECKPOINT_PERIOD == 5
        assert cfg.TEST.EVAL_PERIOD == 5
        assert pose.BACKBONE.FREEZE and pose.GEO_HEAD.FREEZE
        assert not pose.PNP_NET.FREEZE
        assert pose.PNP_NET.INIT_CFG.type == expected[variant]
        assert pose.PNP_NET.COORD_2D_TYPE == "abs"
        assert pose.PNP_NET.REGION_ATTENTION
        assert pose.PNP_NET.MASK_ATTENTION == "mul"
        assert pose.XYZ_ONLINE is True
        assert geometry_supervision_enabled(cfg) is (variant not in ("C", "F"))
        if variant == "C":
            assert "attention_scale_init" not in pose.PNP_NET.INIT_CFG
            assert pose.PNP_NET.INIT_CFG.geometry_scale_r_init == 0.1
            assert pose.PNP_NET.INIT_CFG.geometry_scale_t_init == 0.1
        if variant == "F":
            assert pose.PNP_NET.INIT_CFG.use_depth_stats is True
            assert cfg.INPUT.HEAD_DEPTH is True


def test_geometry_supervision_disabled_rejects_unfrozen_head():
    cfg = Config.fromfile(str(CONFIGS["C"]))
    cfg.MODEL.POSE_NET.GEO_HEAD.FREEZE = False
    with pytest.raises(ValueError, match="requires GEO_HEAD.FREEZE=True"):
        geometry_supervision_enabled(cfg)


def test_smoke_configs_are_isolated_one_epoch_real_data_runs():
    paths = list(CONFIGS.values()) + [
        ROOT / "configs/gdrn/lmo_pbr/research/exp013/e_official_head_random/train.py"
    ]
    for path in paths:
        cfg = Config.fromfile(str(path.with_name("smoke.py")))
        assert tuple(cfg.DATASETS.TRAIN) == ("lmo_pbr_stage3_local_train",)
        assert tuple(cfg.DATASETS.TEST) == ()
        assert cfg.SOLVER.TOTAL_EPOCHS == 1
        assert cfg.SOLVER.IMS_PER_BATCH == 4
        assert cfg.DATALOADER.NUM_WORKERS == 2
        assert cfg.TEST.EVAL_PERIOD == 0


def test_exp013_three_path_has_five_alphas_and_three_region_sources():
    factors = {cpm_xyz_region_condition(name) for name in EXP013_THREE_PATH_CONDITIONS}
    assert factors == {
        (alpha, source)
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0)
        for source in ("pred", "gt", "zero")
    }


def test_region_zero_intervention_zeros_region_without_changing_roi():
    xyz = np.zeros((2, 2, 3), dtype=np.float32)
    gt_xyz = np.ones_like(xyz)
    roi = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    pred_region = np.ones((2, 2, 64), dtype=np.float32) / 64
    gt_region = np.zeros_like(pred_region)
    support = np.ones((2, 2), dtype=bool)
    xyz_out, roi_out, region_out = apply_cpm_xyz_region_intervention(
        xyz,
        gt_xyz,
        roi,
        pred_region,
        gt_region,
        support,
        "xyz_alpha_050_zero_region",
    )
    np.testing.assert_allclose(xyz_out, 0.5)
    np.testing.assert_array_equal(roi_out, roi)
    np.testing.assert_array_equal(region_out, np.zeros_like(region_out))


def test_preregistered_metric_gates_are_applied_literally():
    a = {
        "bop": 0.6839,
        "add_s": 0.5042,
        "per_object_delta": {str(index): 0.0 for index in range(8)},
    }
    b = {**a, "bop": 0.6849, "add_s": 0.5040}
    assert versus_exp012(a)["passed"]
    assert attention_effective(a, b)["effective"]
    tied = {**a, "bop": a["bop"] + 0.0005, "add_s": a["add_s"] + 0.0001}
    assert attention_effective(a, tied)["effective"]
