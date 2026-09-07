from dataclasses import dataclass

import cv2
import numpy as np

from .types import DiagnosticSample


@dataclass(frozen=True)
class Correspondences:
    x3d_m: np.ndarray
    x2d_px: np.ndarray
    flat_indices: np.ndarray
    K: np.ndarray

    @property
    def n(self) -> int:
        return int(self.x3d_m.shape[0])


def interpolate_xyz(pred_xyz_norm, gt_xyz_norm, alpha, support_mask=None):
    """Historical EXP004 intervention: change XYZ only on frozen support."""

    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"alpha must be in [0,1], got {alpha}")
    pred = np.asarray(pred_xyz_norm, dtype=np.float64)
    gt = np.asarray(gt_xyz_norm, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {pred.shape} vs {gt.shape}")
    if support_mask is None:
        return (1.0 - float(alpha)) * pred + float(alpha) * gt
    support = np.asarray(support_mask, dtype=bool)
    if support.shape != pred.shape[1:]:
        raise ValueError("support_mask shape must match XYZ spatial shape")
    result = pred.copy()
    result[:, support] = (1.0 - float(alpha)) * pred[:, support] + float(alpha) * gt[:, support]
    return result


def xyz_norm_to_metric(xyz_norm, extent_m):
    xyz = np.asarray(xyz_norm, dtype=np.float64)
    extent = np.asarray(extent_m, dtype=np.float64).reshape(3, 1, 1)
    if xyz.ndim != 3 or xyz.shape[0] != 3:
        raise ValueError(f"xyz_norm must be [3,H,W], got {xyz.shape}")
    return (xyz - 0.5) * extent


def roi2d_norm_to_pixels(roi2d_norm, image_hw):
    roi = np.asarray(roi2d_norm, dtype=np.float64)
    h, w = [float(v) for v in np.asarray(image_hw).reshape(2)]
    if roi.ndim != 3 or roi.shape[0] != 2:
        raise ValueError(f"roi2d_norm must be [2,H,W], got {roi.shape}")
    out = roi.copy()
    out[0] *= w
    out[1] *= h
    return out


def build_correspondences(sample, xyz_alpha_norm, min_correspondences=6, max_correspondences=0):
    """Build the single fixed 3D-2D set consumed by RANSAC and EPro."""

    sample.validate()
    xyz_m = xyz_norm_to_metric(xyz_alpha_norm, sample.extent_m)
    roi_px = roi2d_norm_to_pixels(sample.roi2d_norm, sample.image_hw)
    support = np.asarray(sample.support_mask, dtype=bool)
    valid = support & np.isfinite(xyz_m).all(axis=0) & np.isfinite(roi_px).all(axis=0)
    flat_idx = np.flatnonzero(valid.reshape(-1))
    if flat_idx.size < int(min_correspondences):
        raise ValueError(
            f"{sample.target_id}: only {flat_idx.size} valid fixed-support points; "
            f"need >= {min_correspondences}"
        )
    if max_correspondences and flat_idx.size > max_correspondences:
        choose = np.linspace(0, flat_idx.size - 1, max_correspondences, dtype=np.int64)
        flat_idx = flat_idx[choose]
    x3d = xyz_m.reshape(3, -1).T[flat_idx]
    x2d = roi_px.reshape(2, -1).T[flat_idx]
    return Correspondences(
        x3d_m=np.ascontiguousarray(x3d, dtype=np.float64),
        x2d_px=np.ascontiguousarray(x2d, dtype=np.float64),
        flat_indices=np.ascontiguousarray(flat_idx),
        K=np.ascontiguousarray(sample.K, dtype=np.float64),
    )


def project_points(x3d_m, R, t, K):
    x3d = np.asarray(x3d_m, dtype=np.float64)
    camera = x3d @ np.asarray(R, dtype=np.float64).reshape(3, 3).T
    camera += np.asarray(t, dtype=np.float64).reshape(1, 3)
    uvw = camera @ np.asarray(K, dtype=np.float64).reshape(3, 3).T
    return uvw[:, :2] / uvw[:, 2:3]


def historical_gt_reprojection_errors(x3d_m, image_points, support, R, t, K):
    """Match EXP004's GT-XYZ sanity check, including Rodrigues normalization.

    Some LM-O ``cam_R_m2c`` matrices are measurably non-orthogonal.  EXP004
    passed them through OpenCV's matrix-to-Rodrigues conversion before
    projection, while retaining the raw matrix for depth-to-object conversion.
    Reproducing both details is required for its recorded <0.5 px check.
    """

    valid = np.asarray(support, dtype=bool)
    points = np.asarray(x3d_m, dtype=np.float64)[valid]
    expected = np.asarray(image_points, dtype=np.float64)[valid]
    if not len(points):
        return np.empty(0, dtype=np.float64)
    rotation_vector = cv2.Rodrigues(np.asarray(R, dtype=np.float64).reshape(3, 3))[0]
    projected = cv2.projectPoints(
        points,
        rotation_vector,
        np.asarray(t, dtype=np.float64).reshape(3, 1),
        np.asarray(K, dtype=np.float64).reshape(3, 3),
        np.zeros((8, 1), dtype=np.float64),
    )[0].reshape(-1, 2)
    return np.linalg.norm(projected - expected, axis=1)
