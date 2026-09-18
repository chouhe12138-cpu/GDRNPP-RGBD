from pathlib import Path

import pytest
from mmcv import Config

from research.exp022.preflight import check_lmo_full_imagenet_protocol
from research.run_contract import validate_research_run_config


ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "configs/gdrn/lmo_pbr/research/exp022_progressive_pcc"
EXPERIMENT_ID = "EXP-20260919-024-lmo-progressive-pcc-fulltrain"


def test_full_training_protocol_and_formal_lock():
    cfg = Config.fromfile(str(CONFIG_DIR / "train_lmo_full_imagenet.py"))
    protocol = check_lmo_full_imagenet_protocol(cfg)
    assert protocol["accumulation_steps"] == 12
    assert cfg.EXPERIMENT_ID == EXPERIMENT_ID
    assert cfg.MODEL.POSE_NET.PCC_HEAD.INIT_CFG.beam_k == 2
    assert cfg.RESEARCH_PROTOCOL.FORMAL_READY is True
    summary = validate_research_run_config(cfg, mode="formal", expected_experiment_id=EXPERIMENT_ID)
    assert summary["amp_enabled"] is True
    assert summary["training_renderer"] == "egl"
    cfg.RESEARCH_PROTOCOL.FORMAL_READY = False
    with pytest.raises(ValueError, match="not ready"):
        validate_research_run_config(cfg, mode="formal", expected_experiment_id=EXPERIMENT_ID)


def test_full_training_rejects_protocol_drift():
    cfg = Config.fromfile(str(CONFIG_DIR / "train_lmo_full_imagenet.py"))
    cfg.MODEL.POSE_NET.BACKBONE.FREEZE = True
    with pytest.raises(ValueError, match="BACKBONE.FREEZE"):
        check_lmo_full_imagenet_protocol(cfg)
    cfg.MODEL.POSE_NET.BACKBONE.FREEZE = False
    cfg.DATASETS.TRAIN = ("lm_pbr_13_online_train",)
    with pytest.raises(ValueError, match="DATASETS.TRAIN"):
        check_lmo_full_imagenet_protocol(cfg)


def test_full_training_smoke_has_no_periodic_evaluation():
    cfg = Config.fromfile(str(CONFIG_DIR / "smoke_lmo_full_imagenet.py"))
    summary = validate_research_run_config(cfg, mode="smoke", expected_experiment_id=EXPERIMENT_ID)
    assert summary["total_epochs"] == 1
    assert summary["batch_size"] == 4
    assert summary["evaluation_period"] == 0
