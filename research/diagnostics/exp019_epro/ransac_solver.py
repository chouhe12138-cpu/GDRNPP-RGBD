import cv2
import numpy as np

from .types import PoseResult, failed_pose


def solve_ransac_pnp(
    corr,
    seed=20260730,
    reprojection_error_px=3.0,
    confidence=0.99,
    iterations_count=100,
):
    """Exact historical EXP004 OpenCV RANSAC settings."""

    if corr.n < 4:
        return failed_pose("ransac_epnp", "fewer_than_four_points", corr.n)
    cv2.setRNGSeed(int(seed))
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            corr.x3d_m,
            corr.x2d_px,
            corr.K,
            np.zeros((4, 1), dtype=np.float64),
            flags=cv2.SOLVEPNP_EPNP,
            iterationsCount=int(iterations_count),
            reprojectionError=float(reprojection_error_px),
            confidence=float(confidence),
        )
    except cv2.error as exc:
        return failed_pose("ransac_epnp", f"opencv:{exc.code}", corr.n)
    if not ok:
        return failed_pose("ransac_epnp", "cv2.solvePnPRansac failed", corr.n)
    R, _ = cv2.Rodrigues(rvec)
    return PoseResult(
        R=R.astype(np.float64),
        t=tvec.reshape(3).astype(np.float64),
        success=True,
        solver="ransac_epnp",
        message=f"inliers={0 if inliers is None else len(inliers)}",
        num_points=corr.n,
    )
