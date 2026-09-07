from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class DiagnosticSample:
    """One LM-O target produced by one frozen official-GDRNPP forward."""

    target_id: str
    scene_id: int
    im_id: int
    instance_id: int
    obj_id: int
    pred_xyz_norm: np.ndarray
    gt_xyz_norm: np.ndarray
    roi2d_norm: np.ndarray
    extent_m: np.ndarray
    K: np.ndarray
    image_hw: np.ndarray
    support_mask: np.ndarray
    mask_prob: Optional[np.ndarray] = None
    region: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.pred_xyz_norm.ndim != 3 or self.pred_xyz_norm.shape[0] != 3:
            raise ValueError(f"pred_xyz_norm must be [3,H,W], got {self.pred_xyz_norm.shape}")
        if self.gt_xyz_norm.shape != self.pred_xyz_norm.shape:
            raise ValueError("gt_xyz_norm shape must match pred_xyz_norm")
        if self.roi2d_norm.shape != (2,) + self.pred_xyz_norm.shape[1:]:
            raise ValueError("roi2d_norm must be [2,H,W] aligned with XYZ")
        if self.support_mask.shape != self.pred_xyz_norm.shape[1:]:
            raise ValueError("support_mask shape must match XYZ")
        if np.asarray(self.extent_m).shape != (3,):
            raise ValueError("extent_m must be [3]")
        if np.asarray(self.K).shape != (3, 3):
            raise ValueError("K must be [3,3]")
        if np.asarray(self.image_hw).shape != (2,):
            raise ValueError("image_hw must be [2] in (H,W) order")


@dataclass
class PoseResult:
    R: np.ndarray
    t: np.ndarray
    success: bool
    solver: str
    message: str = ""
    num_points: int = 0

    def validate(self) -> None:
        if np.asarray(self.R).shape != (3, 3):
            raise ValueError(f"R must be [3,3], got {np.asarray(self.R).shape}")
        if np.asarray(self.t).shape not in ((3,), (3, 1)):
            raise ValueError(f"t must be [3] or [3,1], got {np.asarray(self.t).shape}")


def failed_pose(solver: str, message: str, num_points: int = 0) -> PoseResult:
    return PoseResult(
        R=np.eye(3, dtype=np.float64),
        t=np.zeros(3, dtype=np.float64),
        success=False,
        solver=solver,
        message=message,
        num_points=int(num_points),
    )

