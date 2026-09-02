from __future__ import annotations

import copy
from pathlib import Path

import pytest
from mmcv import Config

from research.run_contract import validate_research_run_config


ROOT = Path(__file__).resolve().parents[2]
EXP017 = ROOT / "configs/gdrn/lmo_pbr/research/exp017/support_aware_rotation_residual"
FULLTRAIN = ROOT / "configs/gdrn/lmo_pbr/research/exp013/d_fulltrain/train.py"


def _load(name: str) -> Config:
    return Config.fromfile(str(EXP017 / name))


def test_exp017_modes_have_distinct_safe_renderer_contracts():
    formal = validate_research_run_config(
        _load("train.py"),
        mode="formal",
        expected_experiment_id="EXP-20260902-017-support-aware-rotation-residual",
    )
    smoke = validate_research_run_config(
        _load("smoke.py"),
        mode="smoke",
        expected_experiment_id="EXP-20260902-017-support-aware-rotation-residual",
    )
    evaluation = validate_research_run_config(
        _load("eval.py"),
        mode="eval",
        expected_experiment_id="EXP-20260902-017-support-aware-rotation-residual",
    )

    assert formal["training_geometry_supervision"] is False
    assert formal["training_renderer"] is None
    assert formal["evaluation_renderer"] == "cpp"
    assert smoke["total_epochs"] == 1 and smoke["evaluation_period"] == 0
    assert evaluation["evaluation_renderer"] == "cpp"


def test_formal_contract_rejects_smoke_or_drifted_protocol():
    with pytest.raises(ValueError, match="formal protocol mismatch"):
        validate_research_run_config(_load("smoke.py"), mode="formal")

    cfg = _load("train.py")
    cfg.SOLVER.TOTAL_EPOCHS = 39
    with pytest.raises(ValueError, match="formal protocol mismatch"):
        validate_research_run_config(cfg, mode="formal")


def test_smoke_contract_rejects_periodic_evaluation():
    cfg = _load("smoke.py")
    cfg.TEST.EVAL_PERIOD = 1
    with pytest.raises(ValueError, match="disable periodic/formal evaluation"):
        validate_research_run_config(cfg, mode="smoke")


def test_frozen_geometry_rejects_supervision_or_training_renderer():
    cfg = _load("train.py")
    cfg.MODEL.POSE_NET.GEO_HEAD.TRAIN_SUPERVISION = True
    cfg.MODEL.POSE_NET.XYZ_RENDERER = "cpp"
    with pytest.raises(ValueError, match="Frozen GEO_HEAD"):
        validate_research_run_config(cfg, mode="formal")

    cfg = _load("train.py")
    cfg.MODEL.POSE_NET.XYZ_RENDERER = "egl"
    with pytest.raises(ValueError, match="must disable"):
        validate_research_run_config(cfg, mode="formal")


def test_unfrozen_geometry_requires_explicit_training_renderer():
    cfg = Config.fromfile(str(FULLTRAIN))
    result = validate_research_run_config(cfg, mode="formal")
    assert result["training_geometry_supervision"] is True
    assert result["training_renderer"] == "egl"

    disabled = copy.deepcopy(cfg)
    disabled.MODEL.POSE_NET.XYZ_RENDERER = None
    with pytest.raises(ValueError, match="requires XYZ_RENDERER"):
        validate_research_run_config(disabled, mode="formal")
