"""Pure helpers for the frozen XYZ-to-pose utilization diagnostic."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np


DEFAULT_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def alpha_token(alpha: float) -> str:
    value = float(alpha)
    if not 0.0 <= value <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return f"a{int(round(value * 100)):03d}"


def metric_xyz_to_normalized(xyz_m: np.ndarray, extent_m: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz_m, dtype=np.float64)
    extent = np.asarray(extent_m, dtype=np.float64).reshape(3)
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz_m must have shape HxWx3, got {xyz.shape}")
    if not np.isfinite(extent).all() or np.any(extent <= 0):
        raise ValueError("extent_m must contain three positive finite values")
    return xyz / extent.reshape(1, 1, 3) + 0.5


def interpolate_xyz(
    predicted_normalized: np.ndarray,
    gt_metric_m: np.ndarray,
    extent_m: np.ndarray,
    correction_support: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Move predicted XYZ toward GT only on a frozen support.

    Pixels outside ``correction_support`` are copied exactly from the network
    prediction. This prevents an XYZ intervention from also changing mask or
    correspondence support.
    """

    predicted = np.asarray(predicted_normalized, dtype=np.float64)
    gt_metric = np.asarray(gt_metric_m, dtype=np.float64)
    support = np.asarray(correction_support, dtype=bool)
    value = float(alpha)
    if predicted.ndim != 3 or predicted.shape[-1] != 3:
        raise ValueError(f"predicted_normalized must have shape HxWx3, got {predicted.shape}")
    if gt_metric.shape != predicted.shape:
        raise ValueError("predicted and GT XYZ shapes differ")
    if support.shape != predicted.shape[:2]:
        raise ValueError("correction_support has the wrong spatial shape")
    if not 0.0 <= value <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    gt_normalized = metric_xyz_to_normalized(gt_metric, extent_m)
    result = predicted.copy()
    result[support] = (1.0 - value) * predicted[support] + value * gt_normalized[support]
    return result


def strictly_monotonic_non_decreasing(values: Sequence[float], tolerance: float = 1e-12) -> bool:
    array = np.asarray(values, dtype=np.float64)
    return bool(np.isfinite(array).all() and np.all(np.diff(array) >= -float(tolerance)))


def object_nonnegative_count(
    per_object: Sequence[dict],
    method: str,
    baseline: str,
    metric: str = "add_s_0.1d_recall",
) -> int:
    method_rows = {int(row["obj_id"]): row for row in per_object if row["method"] == method}
    baseline_rows = {int(row["obj_id"]): row for row in per_object if row["method"] == baseline}
    if method_rows.keys() != baseline_rows.keys():
        raise ValueError("method and baseline object sets differ")
    return sum(
        float(method_rows[obj_id][metric]) >= float(baseline_rows[obj_id][metric])
        for obj_id in method_rows
    )


def utilization_decision(
    per_method: Sequence[dict],
    per_object: Sequence[dict],
    alphas: Iterable[float] = DEFAULT_ALPHAS,
) -> Dict[str, object]:
    """Classify whether frozen Patch-PnP converts improved XYZ into pose gains."""

    alpha_values = tuple(float(value) for value in alphas)
    summary = {row["method"]: row for row in per_method}
    patch_methods = [f"patch_{alpha_token(value)}" for value in alpha_values]
    ransac_methods = [f"ransac_{alpha_token(value)}" for value in alpha_values]
    missing = [name for name in patch_methods + ransac_methods if name not in summary]
    if missing:
        raise ValueError(f"missing methods: {missing}")

    patch_add_curve = [float(summary[name]["add_s_0.1d_recall"]) for name in patch_methods]
    ransac_add_curve = [float(summary[name]["add_s_0.1d_recall"]) for name in ransac_methods]
    patch_bop_curve = [float(summary[name].get("bop_ar", np.nan)) for name in patch_methods]
    ransac_bop_curve = [float(summary[name].get("bop_ar", np.nan)) for name in ransac_methods]

    patch_add_gain = patch_add_curve[-1] - patch_add_curve[0]
    ransac_add_gain = ransac_add_curve[-1] - ransac_add_curve[0]
    patch_bop_gain = patch_bop_curve[-1] - patch_bop_curve[0]
    ransac_bop_gain = ransac_bop_curve[-1] - ransac_bop_curve[0]
    conversion_ratio = (
        patch_add_gain / ransac_add_gain if ransac_add_gain > 1e-12 else float("nan")
    )
    patch_objects_nonnegative = object_nonnegative_count(
        per_object, patch_methods[-1], patch_methods[0]
    )
    ransac_objects_nonnegative = object_nonnegative_count(
        per_object, ransac_methods[-1], ransac_methods[0]
    )
    patch_monotonic = strictly_monotonic_non_decreasing(patch_add_curve)
    ransac_monotonic = strictly_monotonic_non_decreasing(ransac_add_curve)
    bop_available = bool(
        np.isfinite(patch_bop_curve).all() and np.isfinite(ransac_bop_curve).all()
    )

    reference_uses_xyz = bool(
        ransac_add_gain >= 0.05
        and ransac_objects_nonnegative >= 6
        and ransac_monotonic
        and (not bop_available or ransac_bop_gain >= 0.01)
    )
    patch_uses_xyz = bool(
        patch_add_gain >= 0.05
        and patch_objects_nonnegative >= 6
        and patch_monotonic
        and conversion_ratio >= 0.50
        and (not bop_available or patch_bop_gain >= 0.01)
    )
    patch_underutilizes = bool(
        reference_uses_xyz
        and patch_add_gain < max(0.01, 0.30 * ransac_add_gain)
        and (ransac_add_gain - patch_add_gain) >= 0.05
    )

    if patch_uses_xyz:
        status = "PATCH_PNP_USES_IMPROVED_XYZ"
        next_action = "IMPROVE_XYZ_GEOMETRY"
    elif patch_underutilizes:
        status = "PATCH_PNP_UNDERUTILIZATION"
        next_action = "TRAIN_DIRECT_QUALITY_COVERAGE_ATTENTION"
    else:
        status = "MIXED_OR_INCONCLUSIVE"
        next_action = "ANALYZE_ROTATION_TRANSLATION_CURVES_BEFORE_TRAINING"

    return {
        "status": status,
        "next_action": next_action,
        "patch_add_curve": patch_add_curve,
        "ransac_add_curve": ransac_add_curve,
        "patch_bop_curve": patch_bop_curve,
        "ransac_bop_curve": ransac_bop_curve,
        "patch_add_gain": float(patch_add_gain),
        "ransac_add_gain": float(ransac_add_gain),
        "patch_bop_gain": float(patch_bop_gain),
        "ransac_bop_gain": float(ransac_bop_gain),
        "patch_to_ransac_add_conversion_ratio": float(conversion_ratio),
        "patch_objects_nonnegative": int(patch_objects_nonnegative),
        "ransac_objects_nonnegative": int(ransac_objects_nonnegative),
        "patch_add_monotonic": bool(patch_monotonic),
        "ransac_add_monotonic": bool(ransac_monotonic),
        "bop_available": bop_available,
        "rule": {
            "reference_add_gain_min": 0.05,
            "patch_add_gain_min": 0.05,
            "bop_gain_min_when_available": 0.01,
            "objects_nonnegative_min": 6,
            "patch_conversion_ratio_adequate_min": 0.50,
            "patch_conversion_ratio_underutilized_max": 0.30,
            "underutilization_add_gap_min": 0.05,
        },
    }
