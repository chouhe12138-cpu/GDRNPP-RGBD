from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class CapturedPoseCall:
    coor_feat: torch.Tensor
    region: Optional[torch.Tensor]
    extents: Optional[torch.Tensor]
    mask_attention: Optional[torch.Tensor]
    raw_rot: torch.Tensor
    raw_t: torch.Tensor

    def detached(self) -> "CapturedPoseCall":
        def _d(x):
            return None if x is None else x.detach()

        return CapturedPoseCall(
            coor_feat=self.coor_feat.detach(),
            region=_d(self.region),
            extents=_d(self.extents),
            mask_attention=_d(self.mask_attention),
            raw_rot=self.raw_rot.detach(),
            raw_t=self.raw_t.detach(),
        )


@dataclass
class DiagnosticBatch:
    raw_data: List[Dict[str, Any]]
    batch: Dict[str, Any]
    pose_call: CapturedPoseCall
    pred_rot: torch.Tensor
    pred_trans: torch.Tensor


@dataclass
class PosePrediction:
    rot: torch.Tensor
    trans: torch.Tensor
    raw_rot: Optional[torch.Tensor] = None
    raw_t: Optional[torch.Tensor] = None


@dataclass
class VariantResult:
    name: str
    prediction: PosePrediction
    extras: Dict[str, Any]


def tensor_rms(x: torch.Tensor) -> torch.Tensor:
    return x.float().pow(2).mean().sqrt()


def cosine_scalar(a: torch.Tensor, b: torch.Tensor, eps: float = 1.0e-12) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    denom = a.norm() * b.norm()
    if float(denom) <= eps:
        return float("nan")
    return float(torch.dot(a, b) / denom)


def safe_float(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return float(x.detach().cpu())
        return x.detach().cpu().tolist()
    if isinstance(x, (float, int, str, bool)) or x is None:
        return x
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass
    return str(x)
