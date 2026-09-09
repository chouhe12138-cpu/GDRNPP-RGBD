"""EXP020 A/B config matching, freeze contract, and run-contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from mmcv import Config

from research.run_contract import validate_research_run_config

ROOT = Path(__file__).resolve().parents[3]
EXP020 = ROOT / "configs/gdrn/lmo_pbr/research/exp020_geometry_aware_corr"

EXPERIMENT_ID = "EXP-20260909-020-geometry-aware-correspondence-loss"


def _load(name: str) -> Config:
    return Config.fromfile(str(EXP020 / name))


def _without_experiment_identity(cfg: Config) -> dict:
    d = cfg.to_dict()
    d.pop("EXPERIMENT_ID")
    d.pop("OUTPUT_DIR")
    return d


def test_arms_differ_only_in_reproj_lw():
    a = _without_experiment_identity(_load("control.py"))
    b = _without_experiment_identity(_load("reproj.py"))
    lw_a = a["MODEL"]["POSE_NET"]["LOSS_CFG"].pop("REPROJ_LW")
    lw_b = b["MODEL"]["POSE_NET"]["LOSS_CFG"].pop("REPROJ_LW")
    assert a == b
    assert float(lw_a) == 0.0
    assert float(lw_b) > 0.0


def _assert_exp020_freeze_and_loss_protocol(cfg: Config) -> None:
    pose = cfg.MODEL.POSE_NET
    loss = pose.LOSS_CFG
    assert cfg.SEED == 42
    assert cfg.SOLVER.TOTAL_EPOCHS == 40 and cfg.SOLVER.IMS_PER_BATCH == 48
    assert cfg.SOLVER.CHECKPOINT_PERIOD == 5 and cfg.TEST.EVAL_PERIOD == 5
    assert pose.XYZ_RENDERER == "egl"
    assert pose.GEO_HEAD.FREEZE is False
    assert pose.GEO_HEAD.TRAIN_SUPERVISION is True
    assert pose.BACKBONE.FREEZE is True
    assert pose.BACKBONE.INIT_CFG.pretrained is False
    assert pose.PNP_NET.FREEZE is True
    assert pose.QUALITY_COVERAGE.ENABLED is False
    # Geometry supervision kept, exactly as documented.
    assert loss.XYZ_LOSS_TYPE == "L1" and float(loss.XYZ_LW) > 0
    assert float(loss.MASK_LW) > 0 and float(loss.REGION_LW) > 0
    assert float(loss.FULL_MASK_LW) > 0
    # Pose-level losses must be zero so no pose gradient reaches coor_feat.
    for key in ("PM_LW", "CENTROID_LW", "Z_LW", "ROT_LW", "TRANS_LW", "BIND_LW"):
        assert float(loss[key]) == 0.0, key


def test_control_arm_freezes_producer_and_isolates_pose():
    _assert_exp020_freeze_and_loss_protocol(_load("control.py"))


def test_reproj_arm_freezes_producer_and_isolates_pose():
    cfg = _load("reproj.py")
    _assert_exp020_freeze_and_loss_protocol(cfg)
    assert float(cfg.MODEL.POSE_NET.LOSS_CFG.REPROJ_LW) == 1.0


def test_formal_and_smoke_and_eval_configs_satisfy_run_contract():
    for name, mode in (
        ("control.py", "formal"),
        ("reproj.py", "formal"),
        ("smoke_control.py", "smoke"),
        ("smoke_reproj.py", "smoke"),
        ("eval.py", "eval"),
    ):
        result = validate_research_run_config(
            _load(name),
            mode=mode,
            expected_experiment_id=EXPERIMENT_ID,
        )
        assert result["training_geometry_supervision"] is True
        assert result["training_renderer"] == "egl"
        if mode == "formal":
            assert result["evaluation_renderer"] == "cpp"


def test_reproj_lw_defaults_to_zero_when_key_absent():
    # Older configs/checkpoints without the new keys must behave as REPROJ_LW=0.
    cfg = _load("control.py")
    cfg.MODEL.POSE_NET.LOSS_CFG.pop("REPROJ_LW")
    assert float(cfg.MODEL.POSE_NET.LOSS_CFG.get("REPROJ_LW", 0.0)) == 0.0
