"""Small, reusable EXP013 XYZ/Region diagnostic interventions."""

from __future__ import annotations

import numpy as np


XYZ_ALPHA_VALUES = (0.0, 0.25, 0.50, 0.75, 1.0)
EXP013_THREE_PATH_CONDITIONS = tuple(
    f"xyz_alpha_{int(round(alpha * 100)):03d}_{region_source}_region"
    for region_source in ("pred", "gt", "zero")
    for alpha in XYZ_ALPHA_VALUES
)


def cpm_xyz_region_condition(condition: str) -> tuple[float, str]:
    """Resolve an XYZ interpolation factor and Region source."""

    for alpha in XYZ_ALPHA_VALUES:
        token = int(round(alpha * 100))
        for region_source in ("pred", "gt", "zero"):
            if condition == f"xyz_alpha_{token:03d}_{region_source}_region":
                return alpha, region_source
    raise ValueError(f"Unknown EXP013 XYZ/Region condition: {condition}")


def apply_cpm_xyz_region_intervention(
    xyz: np.ndarray,
    gt_xyz: np.ndarray,
    roi_2d: np.ndarray,
    pred_region: np.ndarray,
    gt_region: np.ndarray,
    support: np.ndarray,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate XYZ and select Region values on the supplied support."""

    alpha, region_source = cpm_xyz_region_condition(condition)
    mask = np.asarray(support, dtype=bool)
    xyz_out = np.asarray(xyz).copy()
    xyz_out[mask] = (1.0 - alpha) * xyz_out[mask] + alpha * np.asarray(gt_xyz)[mask]
    region_out = np.asarray(pred_region).copy()
    if region_source == "gt":
        region_out[mask] = np.asarray(gt_region)[mask]
    elif region_source == "zero":
        region_out[...] = 0.0
    return xyz_out, np.asarray(roi_2d).copy(), region_out
