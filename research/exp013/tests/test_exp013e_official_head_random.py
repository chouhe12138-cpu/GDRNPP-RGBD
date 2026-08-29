from __future__ import annotations

from pathlib import Path

import pytest
import torch
from mmcv import Config

from core.gdrn_modeling.models.model_utils import get_pnp_net
from research.exp013.e_prep import strip_model_state
from research.exp013.preflight import verify_stripped_checkpoint

ROOT = Path(__file__).resolve().parents[3]
E_CONFIG = ROOT / "configs/gdrn/lmo_pbr/research/exp013/e_official_head_random/train.py"


def _e_cfg() -> Config:
    return Config.fromfile(str(E_CONFIG))


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
    assert pnp.INIT_CFG.type == "ConvPnPNet"
    assert pnp.INIT_CFG.act == "gelu" and pnp.INIT_CFG.norm == "GN"
    assert pnp.INIT_CFG.flat_op == "flatten"
    assert pnp.INIT_CFG.denormalize_by_extent
    assert "use_region_aux" not in pnp.INIT_CFG
    assert pnp.WITH_2D_COORD and pnp.COORD_2D_TYPE == "abs"
    assert pnp.REGION_ATTENTION
    assert pnp.MASK_ATTENTION == "none"
    assert pnp.ROT_TYPE == "allo_rot6d" and pnp.TRANS_TYPE == "centroid_z"
    assert cfg.MODEL.WEIGHTS.endswith("model_final_wo_optim_wo_pnp.pth")
    # Renderer guarantee: supervision off means the engine never builds one.
    assert pose.GEO_HEAD.TRAIN_SUPERVISION is False
    assert pose.XYZ_ONLINE is True


def test_e_pnp_net_builds_official_architecture_with_fresh_init():
    cfg = _e_cfg()
    cfg.SOLVER.BASE_LR = float(cfg.SOLVER.OPTIMIZER_CFG.lr)
    torch.manual_seed(7)
    pnp_net, _params = get_pnp_net(cfg)
    assert type(pnp_net).__name__ == "ConvPnPNet"
    total = sum(p.numel() for p in pnp_net.parameters())
    assert 9_000_000 < total < 10_500_000
    # The rebuilt head must keep its random (std=0.001) initialization.
    fc1_weight = pnp_net.fc1.weight.detach()
    assert fc1_weight.abs().mean() < 0.01


def test_e_smoke_and_audit_are_isolated_one_epoch_runs():
    for name, workers, batch in (("smoke.py", 2, 4), ("audit48.py", 16, 48)):
        cfg = Config.fromfile(str(E_CONFIG.with_name(name)))
        assert tuple(cfg.DATASETS.TRAIN) == ("lmo_pbr_stage3_local_train",)
        assert tuple(cfg.DATASETS.TEST) == ()
        assert cfg.SOLVER.TOTAL_EPOCHS == 1
        assert cfg.SOLVER.IMS_PER_BATCH == batch
        assert cfg.DATALOADER.NUM_WORKERS == workers
        assert cfg.TEST.EVAL_PERIOD == 0


def test_verify_stripped_checkpoint_accepts_clean_stripped_state(tmp_path: Path):
    official_state = {
        "_module.backbone.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "_module.pnp_net.fc1.weight": torch.ones(2, 2),
    }
    stripped_state = {"_module.backbone.weight": official_state["_module.backbone.weight"]}
    official = tmp_path / "official.pth"
    stripped = tmp_path / "stripped.pth"
    torch.save({"model": official_state}, official)
    torch.save({"model": stripped_state}, stripped)
    report = verify_stripped_checkpoint(stripped, official)
    assert report == {"official_shared_tensors": 1, "stripped_pnp_tensors": 1}


def test_verify_stripped_checkpoint_rejects_pnp_leak(tmp_path: Path):
    official_state = {
        "_module.backbone.weight": torch.ones(3, 4),
        "_module.pnp_net.fc1.weight": torch.ones(2, 2),
    }
    official = tmp_path / "official.pth"
    torch.save({"model": official_state}, official)
    leaked = tmp_path / "leaked.pth"
    torch.save({"model": official_state}, leaked)
    with pytest.raises(RuntimeError, match="still contains pnp tensors"):
        verify_stripped_checkpoint(leaked, official)


def test_strip_model_state_removes_pnp_and_keeps_prefixes():
    state = {
        "_module.backbone.weight": torch.zeros(1),
        "_module.pnp_net.fc1.weight": torch.zeros(1),
        "iteration": 5,
    }
    kept, removed = strip_model_state(state)
    assert set(kept) == {"_module.backbone.weight", "iteration"}
    assert removed == ["_module.pnp_net.fc1.weight"]
