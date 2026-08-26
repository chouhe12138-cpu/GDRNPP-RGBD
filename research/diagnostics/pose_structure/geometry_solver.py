from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


def _image_wh(sample: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    for wk, hk in (("im_W", "im_H"), ("width", "height"), ("image_width", "image_height")):
        if wk in sample and hk in sample:
            w, h = sample[wk], sample[hk]
            if torch.is_tensor(w):
                w = float(w.flatten()[0])
            if torch.is_tensor(h):
                h = float(h.flatten()[0])
            return float(w), float(h)
    return None


def solve_pnp_from_correspondence(
    xyz_normalized: torch.Tensor,
    roi_coord_2d: torch.Tensor,
    extents: torch.Tensor,
    cams: torch.Tensor,
    support: torch.Tensor,
    raw_data: List[Dict[str, Any]],
    max_points: int = 1000,
    reprojection_error_px: float = 3.0,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any]]:
    """Optional explicit geometry reference.

    roi_coord_2d in this repository is absolute full-image [0,1] coordinates.
    The function is skipped when image width/height are unavailable in raw samples.
    """
    b = xyz_normalized.shape[0]
    out_r, out_t = [], []
    failures = []
    for i in range(b):
        wh = _image_wh(raw_data[i]) if i < len(raw_data) else None
        if wh is None:
            failures.append("missing_image_wh")
            continue
        w, h = wh
        xyz = (xyz_normalized[i] - 0.5) * extents[i, :, None, None]
        xy = roi_coord_2d[i]
        mask = support[i, 0] > 0.5
        mask &= torch.isfinite(xyz).all(dim=0)
        ids = mask.flatten().nonzero(as_tuple=False).flatten()
        if ids.numel() < 6:
            failures.append("too_few_points")
            continue
        if ids.numel() > max_points:
            step = max(1, ids.numel() // max_points)
            ids = ids[::step][:max_points]
        obj = xyz.permute(1, 2, 0).reshape(-1, 3)[ids].detach().cpu().numpy().astype(np.float32)
        uv01 = xy.permute(1, 2, 0).reshape(-1, 2)[ids].detach().cpu().numpy().astype(np.float32)
        img = uv01 * np.asarray([w, h], dtype=np.float32)[None]
        K = cams[i].detach().cpu().numpy().astype(np.float64)
        ok, rvec, tvec, _inliers = cv2.solvePnPRansac(
            obj,
            img,
            K,
            None,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=float(reprojection_error_px),
            iterationsCount=100,
            confidence=0.99,
        )
        if not ok:
            failures.append("solve_failed")
            continue
        R, _ = cv2.Rodrigues(rvec)
        out_r.append(torch.as_tensor(R, device=xyz_normalized.device, dtype=xyz_normalized.dtype))
        out_t.append(torch.as_tensor(tvec.reshape(3), device=xyz_normalized.device, dtype=xyz_normalized.dtype))
        failures.append("")

    if any(failures):
        return None, None, {"status": "skipped_or_partial", "failures": failures}
    return torch.stack(out_r), torch.stack(out_t), {"status": "ok", "n": b}
