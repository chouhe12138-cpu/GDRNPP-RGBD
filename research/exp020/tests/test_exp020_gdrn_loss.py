"""EXP020 wiring tests at the ``gdrn_loss`` boundary.

These test the new reprojection term without a full backbone forward: they
feed synthetic continuous-XYZ network outputs (leaves) straight into
``GDRN_DoubleMask.gdrn_loss`` and verify loss keys, gradients, fail-fast
guards, and REPROJ_LW==0 backward compatibility.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from mmcv import Config

from core.gdrn_modeling.models.GDRN_double_mask import GDRN_DoubleMask

ROOT = Path(__file__).resolve().parents[3]
REPROJ_CONFIG = (
    ROOT
    / "configs/gdrn/lmo_pbr/research/exp020_geometry_aware_corr/reproj.py"
)
CONTROL_CONFIG = (
    ROOT
    / "configs/gdrn/lmo_pbr/research/exp020_geometry_aware_corr/control.py"
)


def _make_tensors(height: int = 16, width: int = 16, batch: int = 2, seed: int = 202):
    torch.manual_seed(seed)
    device = torch.device("cpu")
    ys = torch.arange(height)
    xs = torch.arange(width)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    fg = (xx >= width // 4) & (xx < 3 * width // 4) & (yy >= height // 4) & (
        yy < 3 * height // 4
    )
    fg = fg.to(torch.float32).repeat(batch, 1, 1)
    mask_visib = fg.clone()
    mask_trunc = fg.clone()
    mask_full = fg.clone()
    mask_obj = fg.clone()

    out = {}
    out["out_mask_vis"] = torch.rand(batch, 1, height, width) * 0.5
    out["out_mask_full"] = torch.rand(batch, 1, height, width) * 0.5
    out["gt_mask_trunc"] = mask_trunc
    out["gt_mask_visib"] = mask_visib
    out["gt_mask_obj"] = mask_obj
    out["gt_mask_full"] = mask_full
    # Continuous normalized object XYZ prediction leaves.
    out["out_x"] = torch.rand(batch, 1, height, width, requires_grad=True)
    out["out_y"] = torch.rand(batch, 1, height, width, requires_grad=True)
    out["out_z"] = torch.rand(batch, 1, height, width, requires_grad=True)
    gt_xyz = torch.rand(batch, 3, height, width)
    # Keep GT far enough from the output border so a positive t_z keeps every
    # sampled foreground pixel in front of the camera regardless of xyz.
    out["gt_xyz"] = gt_xyz * mask_visib[:, None]
    out["out_region"] = torch.rand(batch, 65, height, width)
    out["gt_region"] = torch.zeros(batch, height, width, dtype=torch.long)
    out["gt_rot"] = torch.eye(3).repeat(batch, 1, 1)
    out["gt_trans"] = torch.tensor([[0.0, 0.0, 2.0]]).repeat(batch, 1)
    out["extents"] = torch.tensor([[0.1, 0.1, 0.15]]).repeat(batch, 1)
    out["gt_xyz_bin"] = None
    out["gt_trans_ratio"] = torch.zeros(batch, 3)
    out["gt_points"] = None
    out["sym_infos"] = None
    return out


def _crop_k(batch: int, height: int, width: int):
    # Cropped+resized camera matrix in output-pixel units; offset principal
    # point so we do not only exercise the identity-centre special case.
    K = torch.tensor(
        [[[200.0, 0.0, width * 0.45], [0.0, 200.0, height * 0.55], [0.0, 0.0, 1.0]]]
    )
    return K.repeat(batch, 1, 1)


def _load(path: Path) -> Config:
    cfg = Config.fromfile(str(path))
    # gdrn_loss is device-agnostic; make sure nothing forces CUDA.
    cfg.MODEL.DEVICE = "cpu"
    return cfg


def _loss(cfg, tensors, roi_zoom_cams):
    return GDRN_DoubleMask.gdrn_loss(
        None,
        cfg,
        out_mask_vis=tensors["out_mask_vis"],
        out_mask_full=tensors["out_mask_full"],
        gt_mask_trunc=tensors["gt_mask_trunc"],
        gt_mask_visib=tensors["gt_mask_visib"],
        gt_mask_obj=tensors["gt_mask_obj"],
        gt_mask_full=tensors["gt_mask_full"],
        out_x=tensors["out_x"],
        out_y=tensors["out_y"],
        out_z=tensors["out_z"],
        gt_xyz=tensors["gt_xyz"],
        gt_xyz_bin=tensors["gt_xyz_bin"],
        out_region=tensors["out_region"],
        gt_region=tensors["gt_region"],
        out_rot=None,
        gt_rot=tensors["gt_rot"],
        out_trans=None,
        gt_trans=tensors["gt_trans"],
        out_centroid=None,
        out_trans_z=None,
        gt_trans_ratio=tensors["gt_trans_ratio"],
        gt_points=tensors["gt_points"],
        sym_infos=tensors["sym_infos"],
        extents=tensors["extents"],
        roi_zoom_cams=roi_zoom_cams,
        vis_extra=None,
    )


def test_reproj_lw_zero_keeps_legacy_geometry_loss_keys():
    cfg = _load(CONTROL_CONFIG)
    t = _make_tensors()
    loss_dict = _loss(cfg, t, roi_zoom_cams=None)
    assert set(loss_dict) == {
        "loss_coor_x",
        "loss_coor_y",
        "loss_coor_z",
        "loss_mask",
        "loss_mask_full",
        "loss_region",
    }
    assert torch.isfinite(sum(loss_dict.values()))


def test_reproj_lw_positive_adds_loss_and_gradients():
    cfg = _load(REPROJ_CONFIG)
    t = _make_tensors()
    K = _crop_k(t["out_x"].shape[0], t["out_x"].shape[2], t["out_x"].shape[3])
    loss_dict = _loss(cfg, t, roi_zoom_cams=K)
    assert "loss_xyz_reproj" in loss_dict
    loss = sum(loss_dict.values())
    assert torch.isfinite(loss)
    assert float(loss_dict["loss_xyz_reproj"]) >= 0.0

    loss.backward()
    for leaf in (t["out_x"], t["out_y"], t["out_z"]):
        assert leaf.grad is not None
        assert torch.isfinite(leaf.grad).all()
        # Geometry supervision plus reprojection must both reach the XYZ head.
        assert float(leaf.grad.abs().sum()) > 0.0


def test_reproj_missing_crop_camera_matrix_fails_fast():
    cfg = _load(REPROJ_CONFIG)
    t = _make_tensors()
    with pytest.raises(ValueError, match="roi_zoom_cams"):
        _loss(cfg, t, roi_zoom_cams=None)


def test_reproj_with_ce_xyz_fails_fast():
    cfg = _load(REPROJ_CONFIG)
    cfg.MODEL.POSE_NET.LOSS_CFG.XYZ_LOSS_TYPE = "CE_coor"
    t = _make_tensors()
    with pytest.raises(ValueError, match="XYZ_LOSS_TYPE"):
        _loss(cfg, t, roi_zoom_cams=_crop_k(2, 16, 16))


def test_reproj_requires_gt_pose_and_extents():
    cfg = _load(REPROJ_CONFIG)
    t = _make_tensors()
    t["gt_rot"] = None
    with pytest.raises(ValueError, match="GT pose"):
        _loss(cfg, t, roi_zoom_cams=_crop_k(2, 16, 16))


def test_reproj_empty_valid_mask_stays_finite():
    cfg = _load(REPROJ_CONFIG)
    t = _make_tensors()
    t["gt_mask_visib"].zero_()
    K = _crop_k(t["out_x"].shape[0], t["out_x"].shape[2], t["out_x"].shape[3])
    loss_dict = _loss(cfg, t, roi_zoom_cams=K)
    assert "loss_xyz_reproj" in loss_dict
    assert float(loss_dict["loss_xyz_reproj"]) == 0.0
    assert torch.isfinite(sum(loss_dict.values()))


def test_reproj_gt_consistent_prediction_is_near_zero_through_gdrn_loss():
    # Feed the XYZ loss a GT-consistent normalized map (points that project
    # back to their own output pixel) so only reprojection remains non-zero;
    # near-zero here proves the gdrn_loss wiring uses the output pixel grid.
    cfg = _load(REPROJ_CONFIG)
    t = _make_tensors(height=8, width=8, batch=1)
    h, w = 8, 8
    ys = torch.arange(h, dtype=torch.float32)
    xs = torch.arange(w, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    f = 100.0
    xyz_obj = torch.stack(
        (2.0 * xx / f, 2.0 * yy / f, torch.zeros_like(xx)), dim=0
    ).unsqueeze(0)
    gt_norm = xyz_obj / t["extents"][:, :, None, None] + 0.5
    t["gt_xyz"] = gt_norm * t["gt_mask_visib"][:, None]
    t["out_x"].data.copy_(gt_norm[:, 0:1])
    t["out_y"].data.copy_(gt_norm[:, 1:2])
    t["out_z"].data.copy_(gt_norm[:, 2:3])
    K = torch.tensor(
        [[[f, 0.0, 0.0], [0.0, f, 0.0], [0.0, 0.0, 1.0]]]
    )
    loss_dict = _loss(cfg, t, roi_zoom_cams=K)
    # All geometry targets coincide, so only the reprojection term matters and
    # it must be at the machine-precision floor (background has no gradient).
    assert float(loss_dict["loss_xyz_reproj"]) < 1e-6
