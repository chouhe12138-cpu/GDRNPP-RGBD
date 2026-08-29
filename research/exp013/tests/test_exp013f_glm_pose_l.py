from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from mmcv import Config

from core.gdrn_modeling.engine.engine_utils import geometry_supervision_enabled
from core.gdrn_modeling.models.model_utils import get_pnp_net
from core.gdrn_modeling.datasets.roi_depth_stats import compute_roi_depth_stats

ROOT = Path(__file__).resolve().parents[3]
F_CONFIG = ROOT / "configs/gdrn/lmo_pbr/research/exp013/f_glm_pose_l/train.py"


def _f_cfg() -> Config:
    cfg = Config.fromfile(str(F_CONFIG))
    cfg.SOLVER.BASE_LR = float(cfg.SOLVER.OPTIMIZER_CFG.lr)
    return cfg


def _inputs(batch: int = 2, size: int = 64):
    torch.manual_seed(101)
    return (
        torch.rand(batch, 5, size, size),
        torch.softmax(torch.randn(batch, 64, size, size), dim=1),
        torch.rand(batch, 3) + 0.05,
        (torch.rand(batch, 1, size, size) > 0.2).float(),
    )


def _small_kwargs() -> dict:
    return dict(
        base_channels=16,
        mid_channels=24,
        high_channels=32,
        use_region_aux=True,
        region_aux_dim=8,
        coarse_grid_size=4,
        dropout=0.0,
        geometry_channels=(8, 12, 8),
        geometry_grid_size=8,
        geometry_scale_init=0.1,
        embed_channels=64,
        attn_heads=4,
        ffn_channels=128,
        shared_channels=64,
    )


def test_f_formal_config_contract():
    cfg = _f_cfg()
    pose = cfg.MODEL.POSE_NET
    pnp = pose.PNP_NET
    assert cfg.SEED == 42
    assert cfg.SOLVER.TOTAL_EPOCHS == 40
    assert cfg.SOLVER.IMS_PER_BATCH == 48
    assert cfg.SOLVER.OPTIMIZER_CFG.lr == 8e-4
    assert pose.BACKBONE.FREEZE and pose.GEO_HEAD.FREEZE
    assert not pnp.FREEZE
    assert pnp.INIT_CFG.type == "GLMPoseLNet"
    assert pnp.INIT_CFG.use_depth_stats is True
    assert pnp.WITH_2D_COORD and pnp.COORD_2D_TYPE == "abs"
    assert pnp.REGION_ATTENTION and pnp.INIT_CFG.use_region_aux
    assert pnp.MASK_ATTENTION == "mul"
    assert pnp.ROT_TYPE == "allo_rot6d" and pnp.TRANS_TYPE == "centroid_z"
    assert cfg.MODEL.WEIGHTS.endswith("model_final_wo_optim.pth")
    assert cfg.INPUT.HEAD_DEPTH is True
    # Renderer guarantee: supervision off means the engine never builds one.
    assert pose.GEO_HEAD.TRAIN_SUPERVISION is False
    assert geometry_supervision_enabled(cfg) is False
    assert pose.XYZ_ONLINE is True


def test_f_smoke_and_audit_are_isolated_one_epoch_runs():
    for name, workers, batch in (("smoke.py", 2, 4), ("audit48.py", 16, 48)):
        cfg = Config.fromfile(str(F_CONFIG.with_name(name)))
        assert tuple(cfg.DATASETS.TRAIN) == ("lmo_pbr_stage3_local_train",)
        assert tuple(cfg.DATASETS.TEST) == ()
        assert cfg.SOLVER.TOTAL_EPOCHS == 1
        assert cfg.SOLVER.IMS_PER_BATCH == batch
        assert cfg.DATALOADER.NUM_WORKERS == workers
        assert cfg.TEST.EVAL_PERIOD == 0
        assert cfg.INPUT.HEAD_DEPTH is True


def test_f_registered_and_budget():
    from core.gdrn_modeling.models.net_factory import HEADS

    from core.gdrn_modeling.models.heads.glm_pose_net import GLMPoseLNet

    assert HEADS["GLMPoseLNet"] is GLMPoseLNet
    cfg = _f_cfg()
    torch.manual_seed(7)
    pnp_net, _params = get_pnp_net(cfg)
    total = sum(p.numel() for p in pnp_net.parameters())
    assert 800_000 <= total <= 1_100_000


def test_f_forward_shapes_gradients_and_no_geometry_path():
    torch.manual_seed(11)
    net, _params = get_pnp_net(_f_cfg())
    head = type(net)(**_small_kwargs())
    coor, region, extents, mask = _inputs()
    rot, t = head(coor, region=region, extents=extents, mask_attention=mask)
    assert rot.shape == (2, 6) and t.shape == (2, 3)
    assert torch.isfinite(rot).all() and torch.isfinite(t).all()
    (rot.square().mean() + t.square().mean()).backward()
    missing = [n for n, p in head.named_parameters() if p.grad is None]
    assert missing == []
    for legacy in (
        "geometry_projection",
        "geometry_scale",
        "pose_fc1",
        "pose_fc2",
    ):
        assert not hasattr(head, legacy)


def test_f_depth_stats_zero_padding_and_rotation_independence():
    torch.manual_seed(11)
    net, _params = get_pnp_net(_f_cfg())
    head = type(net)(**_small_kwargs())
    head.eval()
    coor, region, extents, mask = _inputs(batch=1)
    with torch.no_grad():
        rot_none, t_none = head(coor, region=region, extents=extents, mask_attention=mask)
        rot_zero, t_zero = head(
            coor,
            region=region,
            extents=extents,
            mask_attention=mask,
            depth_stats=torch.zeros(1, 4),
        )
        rot_probe, t_probe = head(
            coor,
            region=region,
            extents=extents,
            mask_attention=mask,
            depth_stats=torch.full((1, 4), 0.5),
        )
    assert torch.equal(t_none, t_zero) and torch.equal(rot_none, rot_zero)
    assert not torch.equal(t_zero, t_probe)
    assert torch.equal(rot_zero, rot_probe)


def test_f_depth_stats_receive_gradient():
    torch.manual_seed(11)
    net, _params = get_pnp_net(_f_cfg())
    head = type(net)(**_small_kwargs())
    coor, region, extents, mask = _inputs(batch=1)
    depth_stats = torch.rand(1, 4)
    rot, t = head(
        coor, region=region, extents=extents, mask_attention=mask, depth_stats=depth_stats
    )
    t.square().mean().backward()
    assert head.pose_translation.weight.grad is not None
    assert torch.isfinite(head.pose_translation.weight.grad).all()


def test_compute_roi_depth_stats_object_band_and_empty_fallback():
    depth = np.full((64, 64), 2.0, dtype=np.float32)
    depth[20:44, 20:44] = 1.0  # object blob at 1 m inside 2 m background
    extent_z = 0.1
    stats = compute_roi_depth_stats(depth, extent_z)
    assert stats.shape == (4,)
    assert stats[0] == pytest.approx(10.0)  # band median 1.0 m / extent_z 0.1
    assert stats[1] == pytest.approx(10.0)  # center depth 1.0 m / extent_z 0.1
    assert 0.0 <= stats[2] <= 1.0
    assert 0.0 < stats[3] <= 1.0
    empty = np.zeros((32, 32), dtype=np.float32)
    assert np.array_equal(compute_roi_depth_stats(empty, extent_z), np.zeros(4, np.float32))
