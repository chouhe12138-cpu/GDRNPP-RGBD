from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch


def rotation_error_deg(pred_rot: torch.Tensor, gt_rot: torch.Tensor) -> torch.Tensor:
    gt_rot = gt_rot.to(device=pred_rot.device, dtype=pred_rot.dtype)
    rel = torch.bmm(pred_rot, gt_rot.transpose(1, 2))
    trace = rel.diagonal(dim1=1, dim2=2).sum(dim=1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def translation_error_cm(pred_trans: torch.Tensor, gt_trans: torch.Tensor) -> torch.Tensor:
    gt_trans = gt_trans.to(device=pred_trans.device, dtype=pred_trans.dtype)
    return torch.linalg.vector_norm(pred_trans - gt_trans, dim=1) * 100.0


def _quantiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p90": float(np.quantile(a, 0.90)),
    }

class PoseMetricAccumulator:
    """Diagnostic pose metrics. These are not official BOP metrics."""

    def __init__(self) -> None:
        self.rot_deg: List[float] = []
        self.trans_cm: List[float] = []
        self.classes: List[int] = []

    def add(
        self,
        pred_rot: torch.Tensor,
        pred_trans: torch.Tensor,
        gt_rot: torch.Tensor,
        gt_trans: torch.Tensor,
        classes: Optional[torch.Tensor] = None,
    ) -> None:
        re = rotation_error_deg(pred_rot, gt_rot).detach().cpu().tolist()
        te = translation_error_cm(pred_trans, gt_trans).detach().cpu().tolist()
        self.rot_deg.extend(float(v) for v in re)
        self.trans_cm.extend(float(v) for v in te)
        if classes is None:
            self.classes.extend([-1] * len(re))
        else:
            self.classes.extend(int(v) for v in classes.detach().cpu().tolist())

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "n": len(self.rot_deg),
            "rotation_deg": _quantiles(self.rot_deg),
            "translation_cm": _quantiles(self.trans_cm),
        }
        if self.rot_deg:
            r = np.asarray(self.rot_deg)
            t = np.asarray(self.trans_cm)
            out["rotation_recall_deg"] = {
                "5": float((r < 5).mean()),
                "10": float((r < 10).mean()),
                "20": float((r < 20).mean()),
            }
            out["translation_recall_cm"] = {
                "2": float((t < 2).mean()),
                "5": float((t < 5).mean()),
                "10": float((t < 10).mean()),
            }
        per_class: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: {"r": [], "t": []})
        for cls, r, t in zip(self.classes, self.rot_deg, self.trans_cm):
            per_class[int(cls)]["r"].append(float(r))
            per_class[int(cls)]["t"].append(float(t))
        out["per_class"] = {
            str(cls): {
                "n": len(v["r"]),
                "rotation_deg": _quantiles(v["r"]),
                "translation_cm": _quantiles(v["t"]),
            }
            for cls, v in sorted(per_class.items())
        }
        return out


class ScalarAccumulator:
    def __init__(self) -> None:
        self.values: Dict[str, List[float]] = defaultdict(list)

    def add(self, **kwargs: float) -> None:
        for k, v in kwargs.items():
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fv):
                self.values[k].append(fv)

    def summary(self) -> Dict[str, Dict[str, float]]:
        return {k: _quantiles(v) for k, v in sorted(self.values.items())}


def monotonicity(values: Iterable[float], increasing: bool = True, atol: float = 1.0e-9) -> Dict[str, Any]:
    vals = [float(v) for v in values]
    if len(vals) < 2:
        return {"pairs": 0, "satisfied_fraction": float("nan"), "strict": True}
    diffs = np.diff(np.asarray(vals))
    ok = diffs >= -atol if increasing else diffs <= atol
    return {
        "pairs": int(len(diffs)),
        "satisfied_fraction": float(ok.mean()),
        "strict": bool(ok.all()),
        "diffs": [float(x) for x in diffs],
    }
