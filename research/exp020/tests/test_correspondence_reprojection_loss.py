"""Pure-PyTorch unit tests for the EXP020 reprojection loss."""

from __future__ import annotations

import pytest
import torch

from core.gdrn_modeling.losses.correspondence_reprojection_loss import (
    correspondence_reprojection_loss,
    decode_normalized_object_xyz,
    project_object_xyz_with_gt_pose,
)


def _exact_case(height: int = 4, width: int = 5, batch: int = 1):
    """Synthetic GT-consistent geometry (identity pose, t_z=2, z_obj=0).

    GT xyz at each output pixel is chosen so that projecting it under the GT
    pose and crop K returns exactly that output pixel (integer grid), which is
    exactly the consistency the online back-projected XYZ target satisfies.
    """
    gt_rot = torch.eye(3).unsqueeze(0).repeat(batch, 1, 1)
    gt_trans = torch.tensor([[0.0, 0.0, 2.0]]).repeat(batch, 1)
    crop_K = torch.tensor(
        [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]]]
    ).repeat(batch, 1, 1)
    extents = torch.tensor([[1.0, 1.0, 1.0]]).repeat(batch, 1)

    ys = torch.arange(height, dtype=torch.float32)
    xs = torch.arange(width, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    # With z_obj=0 and t_z=2: u = fx * x_obj / 2, so x_obj = 2 * u / fx.
    x_obj = 2.0 * xx / 100.0
    y_obj = 2.0 * yy / 100.0
    z_obj = torch.zeros_like(x_obj)
    xyz_obj = torch.stack((x_obj, y_obj, z_obj), dim=0).unsqueeze(0).repeat(batch, 1, 1, 1)
    pred_xyz_norm = xyz_obj / extents[:, :, None, None] + 0.5
    valid_mask = torch.ones((batch, height, width))
    return pred_xyz_norm, extents, gt_rot, gt_trans, crop_K, valid_mask


def test_exact_geometry_has_near_zero_loss():
    pred, extents, rot, trans, K, mask = _exact_case()
    loss, stats = correspondence_reprojection_loss(
        pred, extents, rot, trans, K, mask, normalize_by_res=False
    )
    assert float(loss) < 1e-7
    assert float(stats["mean_reproj_px"]) < 1e-6


def test_exact_geometry_batch_matches_online_backprojection_convention():
    # The online GT XYZ target is built with the integer output grid and the
    # same crop K (engine_utils / calc_xyz_bp_batch); batch must not change it.
    pred, extents, rot, trans, K, mask = _exact_case(batch=3)
    loss, stats = correspondence_reprojection_loss(
        pred, extents, rot, trans, K, mask, normalize_by_res=False
    )
    assert float(loss) < 1e-6
    assert float(stats["valid_count"]) == 3 * 4 * 5


def test_lateral_xyz_perturbation_increases_loss_and_has_gradient():
    pred, extents, rot, trans, K, mask = _exact_case()
    pred = pred.clone()
    pred[:, 0] += 0.01  # lateral metric shift along object x
    pred.requires_grad_(True)

    loss, stats = correspondence_reprojection_loss(
        pred, extents, rot, trans, K, mask, normalize_by_res=False
    )
    assert float(loss.detach()) > 0.0
    assert float(stats["mean_reproj_px"]) > 0.0

    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert float(pred.grad.abs().sum()) > 0.0


def test_empty_mask_returns_connected_finite_zero():
    pred, extents, rot, trans, K, mask = _exact_case()
    pred = pred.clone().requires_grad_(True)
    mask.zero_()

    loss, stats = correspondence_reprojection_loss(pred, extents, rot, trans, K, mask)
    assert torch.isfinite(loss)
    assert float(loss.detach()) == 0.0

    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert float(pred.grad.abs().sum()) == 0.0
    assert int(stats["valid_count"]) == 0


def test_invalid_negative_camera_depth_is_ignored():
    pred, extents, rot, trans, K, mask = _exact_case()
    # Move the object behind the camera; every pixel has z <= 0.
    trans = torch.tensor([[0.0, 0.0, -2.0]])
    pred = pred.clone().requires_grad_(True)

    loss, stats = correspondence_reprojection_loss(pred, extents, rot, trans, K, mask)
    assert torch.isfinite(loss)
    assert float(loss.detach()) == 0.0
    assert int(stats["valid_count"]) == 0
    loss.backward()
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("poison", [float("nan"), float("inf"), -float("inf"), 1e30])
def test_nonfinite_or_extreme_predictions_do_not_break_loss(poison):
    pred, extents, rot, trans, K, mask = _exact_case()
    pred = pred.clone()
    pred[:, :, 1, 1] = poison
    pred[:, :, 2, 3] = poison
    pred.requires_grad_(True)

    loss, stats = correspondence_reprojection_loss(pred, extents, rot, trans, K, mask)
    assert torch.isfinite(loss)
    # NaN/Inf pixels are excluded by the finite mask; huge-but-finite values
    # stay in the loss but must keep it finite.
    assert int(stats["valid_count"]) >= 4 * 5 - 2
    loss.backward()
    assert torch.isfinite(pred.grad).all()


def test_resolution_normalization_scales_loss():
    pred, extents, rot, trans, K, mask = _exact_case()
    pred = pred.clone()
    pred[:, 0] += 0.01

    raw, _ = correspondence_reprojection_loss(
        pred, extents, rot, trans, K, mask, normalize_by_res=False
    )
    normalized, _ = correspondence_reprojection_loss(
        pred, extents, rot, trans, K, mask, normalize_by_res=True
    )
    assert torch.allclose(normalized, raw / 5.0)


def test_extent_and_pose_round_trip_of_helpers():
    # Reconstructing the metric point must invert the online normalization,
    # and projecting it must give back the integer output pixel.
    pred, extents, rot, trans, K, mask = _exact_case(height=8, width=8)
    xyz_obj = decode_normalized_object_xyz(pred, extents)
    uv, positive_depth = project_object_xyz_with_gt_pose(
        xyz_obj, rot, trans, K
    )
    assert bool(positive_depth.all())
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(8), torch.arange(8), indexing="ij"
        ),
        dim=-1,
    ).flip(-1)  # (x, y) order
    assert torch.allclose(uv[0], grid.float(), atol=1e-5)


def test_shape_validation_rejects_misaligned_inputs():
    pred, extents, rot, trans, K, mask = _exact_case()
    with pytest.raises(ValueError, match="valid_mask"):
        correspondence_reprojection_loss(pred, extents, rot, trans, K, mask[:, 0])


def test_unsupported_loss_type_raises():
    pred, extents, rot, trans, K, mask = _exact_case()
    with pytest.raises(ValueError, match="loss_type"):
        correspondence_reprojection_loss(
            pred, extents, rot, trans, K, mask, loss_type="huber"
        )
