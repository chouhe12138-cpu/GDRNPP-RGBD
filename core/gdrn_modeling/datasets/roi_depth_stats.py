# -*- coding: utf-8 -*-
"""EXP013F: mask-free ROI depth statistics for the pose-head translation
branch. Kept in its own module so both ``data_loader`` (test loader) and
``data_loader_online`` (train loader, used whenever ``XYZ_ONLINE=True``) can
import it without a circular dependency."""

import numpy as np


def compute_roi_depth_stats(
    depth: np.ndarray, extent_z: float, eps: float = 1.0e-6
) -> np.ndarray:
    """Mask-free ROI depth statistics for the EXP013F translation branch.

    ``depth`` is the ROI-cropped metric depth in meters (HxW, values <= 0 are
    invalid). A center depth is measured in a small central window and used as
    an anchor; the statistics are taken over the "anchor band"
    ``|depth - center| <= extent_z``, which selects object-like pixels without
    needing a segmentation mask (training renders carry object-only depth;
    at test time the band rejects most background). Returns four normalized
    values: ``[median/ext_z, center/ext_z, variance/median^2, band_fraction]``.
    """
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim == 3:
        # cv2 resize/warpAffine may or may not keep a singleton channel
        # depending on the input layout; normalize to HxW before statistics.
        d = d.reshape(d.shape[0], d.shape[1])
    valid = d > 0
    if not bool(valid.any()):
        return np.zeros(4, dtype=np.float32)
    height, width = d.shape
    cy, cx = height // 2, width // 2
    radius = max(2, min(height, width) // 32)
    window = d[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1]
    window_valid = window[valid[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1]]
    center = float(np.median(window_valid)) if window_valid.size else float(np.median(d[valid]))
    band_width = max(float(extent_z), eps)
    band = valid & (np.abs(d - center) <= band_width)
    values = d[band]
    if values.size == 0:
        band = valid
        values = d[valid]
    median = float(np.median(values))
    variance = float(np.var(values))
    fraction = float(values.size) / max(float(valid.sum()), 1.0)
    safe_ez = max(float(extent_z), eps)
    return np.array(
        [
            median / safe_ez,
            center / safe_ez,
            variance / max(median * median, eps * eps),
            fraction,
        ],
        dtype=np.float32,
    )
