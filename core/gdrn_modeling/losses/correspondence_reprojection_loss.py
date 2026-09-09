"""Geometry-aware per-pixel correspondence reprojection loss (EXP020).

Given the continuous normalized-object XYZ predicted by the geometry head, the
GT object-to-camera pose, the object extents and the crop-resized camera
matrix, this module penalises the distance between

- the projection of each predicted 3D point under the GT pose, and
- the output-map pixel location that produced that prediction.

This is a geometric consistency constraint on the correspondence producer,
not a PnP loss: the GT pose and crop K are used directly, so the gradient chain
stays short and local.  It intentionally knows nothing about cfg, datasets,
PnP, or checkpoints.

Conventions (must match the online training path):
- ``pred_xyz_norm``: ``xyz_obj / extent + 0.5``, shape ``[B,3,H,W]``;
- ``extents``: object extents in the same metric unit as ``gt_trans``;
- ``gt_rot`` / ``gt_trans``: the same GT object-to-camera pose that produced
  the online rendered XYZ target;
- ``crop_K``: ``roi_zoom_K`` from ``engine_utils.batch_data_train_online``,
  i.e. the camera matrix after ROI crop+resize, in output-map pixel units;
- target pixels are the output-map grid ``x in [0,W-1], y in [0,H-1]``, which
  is the coordinate system shared by ``roi_zoom_K`` and the online
  back-projected XYZ target (``lib.pysixd.misc.calc_xyz_bp_batch``).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "decode_normalized_object_xyz",
    "project_object_xyz_with_gt_pose",
    "correspondence_reprojection_loss",
]

_GEOMETRY_DTYPES = (torch.float16, torch.bfloat16)


def decode_normalized_object_xyz(
    pred_xyz_norm: torch.Tensor, extents: torch.Tensor
) -> torch.Tensor:
    """Decode GDRN normalized object XYZ back to metric object coordinates.

    ``xyz_norm = xyz_obj / extent + 0.5``, so
    ``xyz_obj = (xyz_norm - 0.5) * extent``.
    """
    return (pred_xyz_norm - 0.5) * extents[:, :, None, None]


def project_object_xyz_with_gt_pose(
    xyz_obj: torch.Tensor,
    gt_rot: torch.Tensor,
    gt_trans: torch.Tensor,
    crop_K: torch.Tensor,
    *,
    z_eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project dense object XYZ under the GT pose and crop-resized camera K.

    Args:
        xyz_obj: ``[B,3,H,W]`` metric object coordinates.
        gt_rot: ``[B,3,3]`` GT object-to-camera rotation.
        gt_trans: ``[B,3]`` GT object-to-camera translation.
        crop_K: ``[B,3,3]`` crop-resized camera matrix in output pixels.

    Returns:
        uv_pred: ``[B,H,W,2]`` predicted pixel coordinates.
        positive_depth: ``[B,H,W]`` mask of camera-space ``Z > z_eps``.
    """
    if xyz_obj.ndim != 4 or xyz_obj.shape[1] != 3:
        raise ValueError(
            f"xyz_obj must have shape [B,3,H,W], got {tuple(xyz_obj.shape)}"
        )

    b, _, h, w = xyz_obj.shape
    xyz_flat = xyz_obj.flatten(2)  # [B,3,N]
    xyz_cam = torch.bmm(gt_rot, xyz_flat) + gt_trans[:, :, None]  # [B,3,N]

    z = xyz_cam[:, 2, :]
    positive_depth = z > float(z_eps)

    # Safe denominator for numerical stability; invalid depths are masked later.
    safe_z = torch.where(positive_depth, z, torch.ones_like(z))
    xy1 = torch.bmm(crop_K, xyz_cam)  # [B,3,N]
    u = xy1[:, 0, :] / safe_z
    v = xy1[:, 1, :] / safe_z
    uv = torch.stack((u, v), dim=-1).view(b, h, w, 2)

    return uv, positive_depth.view(b, h, w)


def _validate_shapes(
    pred_xyz_norm: torch.Tensor,
    extents: torch.Tensor,
    gt_rot: torch.Tensor,
    gt_trans: torch.Tensor,
    crop_K: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[int, int, int]:
    if pred_xyz_norm.ndim != 4 or pred_xyz_norm.shape[1] != 3:
        raise ValueError(
            f"pred_xyz_norm must have shape [B,3,H,W], got {tuple(pred_xyz_norm.shape)}"
        )
    b, _, h, w = pred_xyz_norm.shape
    expected = {
        "extents": (b, 3),
        "gt_rot": (b, 3, 3),
        "gt_trans": (b, 3),
        "crop_K": (b, 3, 3),
        "valid_mask": (b, h, w),
    }
    actual = {
        "extents": tuple(extents.shape),
        "gt_rot": tuple(gt_rot.shape),
        "gt_trans": tuple(gt_trans.shape),
        "crop_K": tuple(crop_K.shape),
        "valid_mask": tuple(valid_mask.shape),
    }
    for name, shape in expected.items():
        if actual[name] != shape:
            raise ValueError(f"{name} must have shape {shape}, got {actual[name]}")
    return b, h, w


def _pixel_grid(
    batch: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Output-map pixel coordinates with shape ``[B,H,W,2]``."""
    ys = torch.arange(height, device=device, dtype=dtype)
    xs = torch.arange(width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    uv = torch.stack((xx, yy), dim=-1)
    return uv.unsqueeze(0).expand(batch, -1, -1, -1)


def correspondence_reprojection_loss(
    pred_xyz_norm: torch.Tensor,
    extents: torch.Tensor,
    gt_rot: torch.Tensor,
    gt_trans: torch.Tensor,
    crop_K: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    loss_type: str = "smooth_l1",
    smooth_l1_beta_px: float = 1.0,
    normalize_by_res: bool = True,
    z_eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Dense per-pixel correspondence reprojection loss (EXP020).

    The loss compares the projection of each predicted object-space point under
    the GT pose against the output-map pixel that produced that prediction.
    Pixels are kept only where the (GT XYZ) mask is foreground and the camera
    depth of the predicted point is positive and finite, mirroring the XYZ L1
    supervision so background predictions receive no gradient here either.

    Returns:
        loss: scalar ``[0, inf)``; an empty valid set yields a connected zero.
        stats: ``mean_reproj_px`` (output pixels), ``valid_ratio``,
            ``valid_count`` (all detached).
    """
    b, h, w = _validate_shapes(
        pred_xyz_norm, extents, gt_rot, gt_trans, crop_K, valid_mask
    )

    # Run the geometry math in a stable float type while preserving gradients
    # back to the (possibly low-precision) network outputs.
    dtype = pred_xyz_norm.dtype
    calc_dtype = torch.float32 if dtype in _GEOMETRY_DTYPES else dtype

    xyz_obj = decode_normalized_object_xyz(
        pred_xyz_norm.to(calc_dtype), extents.to(calc_dtype)
    )
    uv_pred, positive_depth = project_object_xyz_with_gt_pose(
        xyz_obj,
        gt_rot.to(calc_dtype),
        gt_trans.to(calc_dtype),
        crop_K.to(calc_dtype),
        z_eps=z_eps,
    )
    uv_target = _pixel_grid(
        b,
        h,
        w,
        device=pred_xyz_norm.device,
        dtype=calc_dtype,
    )

    finite = torch.isfinite(uv_pred).all(dim=-1)
    valid = (valid_mask > 0.5) & positive_depth & finite

    # Keep a connected zero so backward() stays valid for an empty mask.
    if not torch.any(valid):
        zero = pred_xyz_norm.sum() * 0.0
        stats = {
            "mean_reproj_px": zero.detach(),
            "valid_ratio": valid.float().mean().detach(),
            "valid_count": valid.sum().detach(),
        }
        return zero, stats

    pred_valid = uv_pred[valid]
    target_valid = uv_target[valid]

    loss_type = str(loss_type).lower()
    if loss_type == "smooth_l1":
        loss = F.smooth_l1_loss(
            pred_valid,
            target_valid,
            reduction="mean",
            beta=float(smooth_l1_beta_px),
        )
    elif loss_type == "l1":
        loss = F.l1_loss(pred_valid, target_valid, reduction="mean")
    else:
        raise ValueError(f"Unsupported reprojection loss_type: {loss_type!r}")

    if normalize_by_res:
        loss = loss / float(max(h, w))

    with torch.no_grad():
        px_error = torch.linalg.vector_norm(pred_valid - target_valid, dim=-1)
        stats = {
            "mean_reproj_px": px_error.mean(),
            "valid_ratio": valid.float().mean(),
            "valid_count": valid.sum(),
        }

    return loss, stats
