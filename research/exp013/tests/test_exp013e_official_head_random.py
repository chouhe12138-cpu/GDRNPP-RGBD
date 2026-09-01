from __future__ import annotations

from pathlib import Path

import pytest
import torch
from mmcv import Config

from core.gdrn_modeling.models.model_utils import get_pnp_net
from core.gdrn_modeling.models.heads.conv_pnp_net import ConvPnPNet
from core.gdrn_modeling.models.heads.official_head_random_init import (
    OfficialConvPnPNetRandomInit,
)
from research.exp013.preflight import load_official_shared_state

ROOT = Path(__file__).resolve().parents[3]
E_CONFIG = ROOT / "configs/gdrn/lmo_pbr/research/exp013/e_official_head_random/train.py"


def _e_cfg() -> Config:
    cfg = Config.fromfile(str(E_CONFIG))
    cfg.SOLVER.BASE_LR = float(cfg.SOLVER.OPTIMIZER_CFG.lr)
    return cfg


def test_e_formal_config_contract_mirrors_official_head():
    cfg = _e_cfg()
    pose = cfg.MODEL.POSE_NET
    pnp = pose.PNP_NET
    assert cfg.SEED == 42
    assert cfg.SOLVER.TOTAL_EPOCHS == 40
    assert cfg.SOLVER.IMS_PER_BATCH == 48
    assert cfg.SOLVER.CHECKPOINT_PERIOD == 5
    assert cfg.TEST.EVAL_PERIOD == 5
    assert cfg.SOLVER.OPTIMIZER_CFG.lr == 8e-4
    assert pose.BACKBONE.FREEZE and pose.GEO_HEAD.FREEZE
    assert not pnp.FREEZE
    assert pnp.INIT_CFG.type == "OfficialConvPnPNetRandomInit"
    assert pnp.INIT_CFG.act == "gelu" and pnp.INIT_CFG.norm == "GN"
    assert pnp.INIT_CFG.flat_op == "flatten"
    assert pnp.INIT_CFG.denormalize_by_extent
    assert "use_region_aux" not in pnp.INIT_CFG
    assert pnp.WITH_2D_COORD and pnp.COORD_2D_TYPE == "abs"
    assert pnp.REGION_ATTENTION
    assert pnp.MASK_ATTENTION == "none"
    assert pnp.ROT_TYPE == "allo_rot6d" and pnp.TRANS_TYPE == "centroid_z"
    # The original official checkpoint (present in every container) provides
    # backbone/geometry weights; key-name mismatch protects the random head.
    assert cfg.MODEL.WEIGHTS.endswith("model_final_wo_optim.pth")
    # Renderer guarantee: supervision off means the engine never builds one.
    assert pose.GEO_HEAD.TRAIN_SUPERVISION is False
    assert pose.XYZ_ONLINE is True


def test_e_pnp_net_builds_official_topology_under_wrapper():
    cfg = _e_cfg()
    torch.manual_seed(7)
    pnp_net, _params = get_pnp_net(cfg)
    assert isinstance(pnp_net, OfficialConvPnPNetRandomInit)
    assert isinstance(pnp_net.head, ConvPnPNet)
    total = sum(p.numel() for p in pnp_net.parameters())
    assert 9_000_000 < total < 10_500_000
    # Random (std=0.001) initialization must be in place at build time.
    assert pnp_net.head.fc1.weight.detach().abs().mean() < 0.01
    # Wrapper key names never collide with the official pnp keys.
    assert all(key.startswith("head.") for key in pnp_net.state_dict())


def test_e_official_pnp_keys_cannot_overwrite_random_init():
    """The EXP013E core guarantee, exercised through the real migration path."""

    class _Holder(torch.nn.Module):
        def __init__(self, head):
            super().__init__()
            self.pnp_net = head

    cfg = _e_cfg()
    torch.manual_seed(7)
    head, _params = get_pnp_net(cfg)
    initial = {k: v.detach().clone() for k, v in head.state_dict().items()}
    official_like = {}
    for key, value in head.head.state_dict().items():
        official_like[f"pnp_net.{key}"] = value + 1.0  # "official" pose weights
    load_official_shared_state(_Holder(head), official_like)
    after = {k: v.detach().clone() for k, v in head.state_dict().items()}
    for key in initial:
        assert torch.equal(initial[key], after[key])


def test_e_smoke_and_audit_are_isolated_one_epoch_runs():
    for name, workers, batch in (("smoke.py", 2, 4), ("audit48.py", 16, 48)):
        cfg = Config.fromfile(str(E_CONFIG.with_name(name)))
        assert tuple(cfg.DATASETS.TRAIN) == ("lmo_pbr_stage3_local_train",)
        assert tuple(cfg.DATASETS.TEST) == ()
        assert cfg.SOLVER.TOTAL_EPOCHS == 1
        assert cfg.SOLVER.IMS_PER_BATCH == batch
        assert cfg.DATALOADER.NUM_WORKERS == workers
        assert cfg.TEST.EVAL_PERIOD == 0
