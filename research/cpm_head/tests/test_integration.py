from __future__ import annotations

import json
from pathlib import Path

import pytest
from mmcv import Config

from core.gdrn_modeling.models.heads.cpm_pnp_net import (
    CorrespondenceAwareMomentPnPNet,
)
from core.gdrn_modeling.models.model_utils import get_pnp_net
from core.gdrn_modeling.models.net_factory import HEADS
from research.experiment_system.config_contract import validate_config_contract


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = (
    PROJECT_ROOT / "configs/gdrn/lmo_pbr/convnext_cpm_head_local_lmo.py"
)
EVAL_CONFIG_PATH = (
    PROJECT_ROOT / "configs/gdrn/lmo_pbr/convnext_cpm_head_local_eval_lmo.py"
)
FORMAL_CONFIG_PATH = (
    PROJECT_ROOT / "configs/gdrn/lmo_pbr/research/exp009_cpm_head/train.py"
)
B_CONTROL_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/gdrn/lmo_pbr/research/exp005_pnp_control/train.py"
)
EXPERIMENT_PATH = (
    PROJECT_ROOT
    / "research/experiments/EXP-20260809-009-cpm-head/EXPERIMENT.json"
)


def test_cpm_local_config_contract() -> None:
    cfg = Config.fromfile(str(CONFIG_PATH))
    pose = cfg.MODEL.POSE_NET
    pnp = pose.PNP_NET
    assert pose.BACKBONE.FREEZE
    assert pose.GEO_HEAD.FREEZE
    assert not pnp.FREEZE
    assert pnp.INIT_CFG.type == "CorrespondenceAwareMomentPnPNet"
    assert pnp.WITH_2D_COORD and pnp.COORD_2D_TYPE == "abs"
    assert pnp.REGION_ATTENTION
    assert pnp.MASK_ATTENTION == "mul"
    assert pnp.ROT_TYPE == "allo_rot6d"
    assert pnp.TRANS_TYPE == "centroid_z"
    assert not pose.QUALITY_COVERAGE.ENABLED
    assert tuple(cfg.DATASETS.TRAIN) == ("lmo_pbr_stage3_local_train",)
    assert tuple(cfg.DATASETS.TEST) == ()
    assert int(cfg.SOLVER.TOTAL_EPOCHS) == 1


def test_cpm_is_registered_and_builds_through_model_utils() -> None:
    cfg = Config.fromfile(str(CONFIG_PATH))
    cfg.SOLVER.BASE_LR = float(cfg.SOLVER.OPTIMIZER_CFG.lr)
    assert HEADS["CorrespondenceAwareMomentPnPNet"] is CorrespondenceAwareMomentPnPNet
    module, parameter_groups = get_pnp_net(cfg)
    assert isinstance(module, CorrespondenceAwareMomentPnPNet)
    assert len(parameter_groups) == 1
    assert sum(parameter.numel() for parameter in module.parameters()) == 822_281
    assert all(parameter.requires_grad for parameter in module.parameters())


def test_cpm_local_eval_config_uses_gt_boxes_and_structured_outputs() -> None:
    cfg = Config.fromfile(str(EVAL_CONFIG_PATH))
    assert tuple(cfg.DATASETS.TEST) == ("lmo_bop_test",)
    assert tuple(cfg.DATASETS.DET_FILES_TEST) == ()
    assert cfg.MODEL.LOAD_DETS_TEST is False
    assert cfg.TEST.TEST_BBOX_TYPE == "gt"
    assert cfg.TEST.USE_PNP is False
    assert cfg.TEST.AMP_TEST is False
    assert cfg.RUN_ARTIFACTS.STRUCTURED_LAYOUT is True
    assert cfg.RUN_ARTIFACTS.COMPACT_LOG is True
    assert cfg.MODEL.WEIGHTS.endswith("model_0002047.pth")


def test_cpm_formal_config_matches_mandatory_b_training_protocol() -> None:
    cpm = Config.fromfile(str(FORMAL_CONFIG_PATH))
    control = Config.fromfile(str(B_CONTROL_CONFIG_PATH))
    fields = (
        "IMS_PER_BATCH",
        "REFERENCE_BS",
        "TOTAL_EPOCHS",
        "LR_SCHEDULER_NAME",
        "ANNEAL_METHOD",
        "ANNEAL_POINT",
        "WARMUP_FACTOR",
        "WARMUP_ITERS",
        "CHECKPOINT_PERIOD",
        "CHECKPOINT_BY_EPOCH",
        "MAX_TO_KEEP",
    )
    for field in fields:
        assert cpm.SOLVER[field] == control.SOLVER[field]
    assert dict(cpm.SOLVER.OPTIMIZER_CFG) == dict(control.SOLVER.OPTIMIZER_CFG)
    assert tuple(cpm.DATASETS.TRAIN) == tuple(control.DATASETS.TRAIN)
    assert tuple(cpm.DATASETS.TEST) == tuple(control.DATASETS.TEST)
    assert cpm.TEST.TEST_BBOX_TYPE == control.TEST.TEST_BBOX_TYPE == "gt"
    assert cpm.SEED == control.SEED == 42
    assert cpm.MODEL.POSE_NET.BACKBONE.FREEZE
    assert cpm.MODEL.POSE_NET.GEO_HEAD.FREEZE
    assert not cpm.MODEL.POSE_NET.PNP_NET.FREEZE
    assert not cpm.SOLVER.BEST_CHECKPOINT.ENABLED


def test_cpm_formal_config_satisfies_registered_experiment_contract() -> None:
    experiment = json.loads(EXPERIMENT_PATH.read_text(encoding="utf-8"))
    cfg = Config.fromfile(str(FORMAL_CONFIG_PATH))
    validate_config_contract(
        experiment,
        cfg,
        "formal",
        42,
        "RUN-TEST-CONTRACT",
        Path("/tmp/cpm-contract"),
    )
    assert cfg.EXPERIMENT_ID == experiment["experiment_id"]
    assert cfg.SEED == 42
    assert cfg.RUN_ID == "RUN-TEST-CONTRACT"
    assert cfg.OUTPUT_DIR == "/tmp/cpm-contract"


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("WITH_2D_COORD", False, "absolute ROI 2D"),
        ("REGION_ATTENTION", False, "Region posterior"),
        ("MASK_ATTENTION", "none", "visible-mask support"),
    ],
)
def test_cpm_builder_rejects_missing_required_inputs(
    field: str, value: object, message: str
) -> None:
    cfg = Config.fromfile(str(CONFIG_PATH))
    cfg.SOLVER.BASE_LR = float(cfg.SOLVER.OPTIMIZER_CFG.lr)
    cfg.MODEL.POSE_NET.PNP_NET[field] = value
    with pytest.raises(ValueError, match=message):
        get_pnp_net(cfg)
