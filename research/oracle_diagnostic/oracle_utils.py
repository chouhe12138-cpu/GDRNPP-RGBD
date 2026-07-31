"""Pure NumPy helpers for the GDRNPP causal-oracle diagnostic.

Depth, ground-truth masks, and poses are used only to construct diagnostic
oracles. None of the helpers in this module form a deployable RGB method.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.stats import spearmanr

from research.pose_aggregation.solvers import Correspondences


def normalized_image_points(
    coord_2d_normalized: np.ndarray, image_height: int, image_width: int
) -> np.ndarray:
    coords = np.asarray(coord_2d_normalized, dtype=np.float64).copy()
    if coords.ndim != 3 or coords.shape[-1] != 2:
        raise ValueError(f"coord_2d_normalized must be HxWx2, got {coords.shape}")
    coords[..., 0] *= float(image_width)
    coords[..., 1] *= float(image_height)
    return coords


def normalized_xyz_to_metric(xyz_normalized: np.ndarray, extent_m: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz_normalized, dtype=np.float64)
    extent = np.asarray(extent_m, dtype=np.float64).reshape(3)
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz_normalized must be HxWx3, got {xyz.shape}")
    return (xyz - 0.5) * extent.reshape(1, 1, 3)


def prediction_valid_mask(
    xyz_normalized: np.ndarray,
    mask_probability: np.ndarray,
    extent_m: np.ndarray,
    mask_threshold: float,
) -> np.ndarray:
    xyz_normalized = np.asarray(xyz_normalized, dtype=np.float64)
    mask = np.asarray(mask_probability, dtype=np.float64).squeeze()
    extent = np.asarray(extent_m, dtype=np.float64).reshape(3)
    if mask.shape != xyz_normalized.shape[:2]:
        raise ValueError("mask and xyz spatial shapes differ")
    xyz_m = normalized_xyz_to_metric(xyz_normalized, extent)
    epsilon = 1e-4 * extent.reshape(1, 1, 3)
    return (
        np.isfinite(xyz_m).all(axis=2)
        & np.isfinite(mask)
        & (np.abs(xyz_m) > epsilon).all(axis=2)
        & (mask > float(mask_threshold))
    )


def region_confidence(region_logits: np.ndarray) -> np.ndarray:
    region = np.asarray(region_logits, dtype=np.float64)
    if region.ndim != 3:
        raise ValueError(f"region_logits must be CxHxW, got {region.shape}")
    foreground = region[1:] if region.shape[0] > 1 else region
    shifted = foreground - foreground.max(axis=0, keepdims=True)
    exp_values = np.exp(shifted)
    probabilities = exp_values / np.maximum(exp_values.sum(axis=0, keepdims=True), 1e-12)
    return probabilities.max(axis=0)


def sample_nearest(image: np.ndarray, image_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a full-resolution image at aligned dense image coordinates."""

    image = np.asarray(image)
    points = np.asarray(image_points, dtype=np.float64)
    height, width = image.shape[:2]
    finite = np.isfinite(points).all(axis=2)
    safe_points = np.where(np.isfinite(points), points, 0.0)
    x = np.rint(safe_points[..., 0]).astype(np.int64)
    y = np.rint(safe_points[..., 1]).astype(np.int64)
    valid = finite & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    sampled_shape = points.shape[:2] + image.shape[2:]
    sampled = np.zeros(sampled_shape, dtype=image.dtype)
    sampled[valid] = image[y[valid], x[valid]]
    return sampled, valid


def depth_to_object_coordinates(
    depth_m: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    rotation_gt: np.ndarray,
    translation_gt_m: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project sampled depth and transform camera points to object space."""

    sampled_depth, in_image = sample_nearest(depth_m, image_points)
    sampled_depth = np.asarray(sampled_depth, dtype=np.float64)
    points = np.asarray(image_points, dtype=np.float64)
    camera = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    rotation = np.asarray(rotation_gt, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation_gt_m, dtype=np.float64).reshape(3)
    z = sampled_depth
    x = (points[..., 0] - camera[0, 2]) * z / camera[0, 0]
    y = (points[..., 1] - camera[1, 2]) * z / camera[1, 1]
    camera_points = np.stack((x, y, z), axis=-1)
    object_points = (camera_points - translation.reshape(1, 1, 3)) @ rotation
    valid = in_image & np.isfinite(object_points).all(axis=2) & np.isfinite(z) & (z > 0)
    object_points[~valid] = 0.0
    return object_points, valid


def build_correspondences_from_dense(
    image_points: np.ndarray,
    model_points_m: np.ndarray,
    support: np.ndarray,
    reliability: Optional[np.ndarray] = None,
) -> Correspondences:
    image_points = np.asarray(image_points, dtype=np.float64)
    model_points = np.asarray(model_points_m, dtype=np.float64)
    support = np.asarray(support, dtype=bool)
    if image_points.shape != support.shape + (2,):
        raise ValueError("image_points and support shapes differ")
    if model_points.shape != support.shape + (3,):
        raise ValueError("model_points and support shapes differ")
    valid = support & np.isfinite(image_points).all(axis=2) & np.isfinite(model_points).all(axis=2)
    if reliability is None:
        reliability_array = np.ones(support.shape, dtype=np.float64)
    else:
        reliability_array = np.asarray(reliability, dtype=np.float64)
        if reliability_array.shape != support.shape:
            raise ValueError("reliability and support shapes differ")
        valid &= np.isfinite(reliability_array)
    return Correspondences(
        image_points=np.ascontiguousarray(image_points[valid], dtype=np.float64),
        model_points=np.ascontiguousarray(model_points[valid], dtype=np.float64),
        reliability=np.ascontiguousarray(reliability_array[valid], dtype=np.float64),
    )


def subset_top_fraction(
    correspondences: Correspondences,
    score: np.ndarray,
    keep_fraction: float = 0.5,
    largest: bool = True,
) -> Correspondences:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    if score.size != correspondences.size:
        raise ValueError("score length must equal correspondence count")
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be in (0, 1]")
    if correspondences.size < 4:
        return correspondences
    keep = max(4, int(np.ceil(correspondences.size * keep_fraction)))
    order = np.argsort(-score if largest else score, kind="stable")[:keep]
    return Correspondences(
        image_points=np.ascontiguousarray(correspondences.image_points[order]),
        model_points=np.ascontiguousarray(correspondences.model_points[order]),
        reliability=np.ascontiguousarray(correspondences.reliability[order]),
    )


def mask_metrics(predicted: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    predicted = np.asarray(predicted, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    intersection = int(np.count_nonzero(predicted & truth))
    union = int(np.count_nonzero(predicted | truth))
    pred_count = int(np.count_nonzero(predicted))
    truth_count = int(np.count_nonzero(truth))
    return {
        "iou": intersection / union if union else 1.0,
        "precision": intersection / pred_count if pred_count else float(truth_count == 0),
        "recall": intersection / truth_count if truth_count else 1.0,
    }


def dense_geometry_statistics(
    image_points: np.ndarray,
    model_points: np.ndarray,
    support: np.ndarray,
) -> Dict[str, float]:
    image = np.asarray(image_points, dtype=np.float64)[support]
    model = np.asarray(model_points, dtype=np.float64)[support]
    result = {
        "support_points": int(len(image)),
        "image_spread_min_eigen": float("nan"),
        "image_spread_eigen_ratio": float("nan"),
        "model_spread_min_eigen": float("nan"),
        "model_spread_eigen_ratio": float("nan"),
    }
    if len(image) >= 3:
        image_eigen = np.linalg.eigvalsh(np.cov(image, rowvar=False))
        result["image_spread_min_eigen"] = float(image_eigen[0])
        result["image_spread_eigen_ratio"] = float(image_eigen[0] / max(image_eigen[-1], 1e-12))
    if len(model) >= 4:
        model_eigen = np.linalg.eigvalsh(np.cov(model, rowvar=False))
        result["model_spread_min_eigen"] = float(model_eigen[0])
        result["model_spread_eigen_ratio"] = float(model_eigen[0] / max(model_eigen[-1], 1e-12))
    return result


def safe_spearman(values_a: np.ndarray, values_b: np.ndarray) -> float:
    a = np.asarray(values_a, dtype=np.float64).reshape(-1)
    b = np.asarray(values_b, dtype=np.float64).reshape(-1)
    valid = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(valid) < 3 or np.ptp(a[valid]) == 0 or np.ptp(b[valid]) == 0:
        return float("nan")
    return float(spearmanr(a[valid], b[valid]).statistic)


def paired_cluster_bootstrap(
    rows_a: Sequence[dict],
    rows_b: Sequence[dict],
    value_field: str,
    cluster_fields: Iterable[str] = ("scene_id", "im_id"),
    iterations: int = 10000,
    seed: int = 20260730,
) -> Dict[str, float]:
    """Paired cluster bootstrap for method-A minus method-B means."""

    key_fields = ("scene_id", "im_id", "instance_id", "obj_id")
    cluster_fields = tuple(cluster_fields)
    lookup_b = {tuple(row[field] for field in key_fields): row for row in rows_b}
    cluster_sums: Dict[tuple, float] = {}
    cluster_counts: Dict[tuple, int] = {}
    for row_a in rows_a:
        key = tuple(row_a[field] for field in key_fields)
        if key not in lookup_b:
            raise ValueError(f"missing paired row for {key}")
        row_b = lookup_b[key]
        difference = float(row_a[value_field]) - float(row_b[value_field])
        cluster = tuple(row_a[field] for field in cluster_fields)
        cluster_sums[cluster] = cluster_sums.get(cluster, 0.0) + difference
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
    clusters = sorted(cluster_sums)
    sums = np.asarray([cluster_sums[key] for key in clusters], dtype=np.float64)
    counts = np.asarray([cluster_counts[key] for key in clusters], dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sample = rng.integers(0, len(clusters), size=len(clusters))
        estimates[index] = sums[sample].sum() / counts[sample].sum()
    return {
        "delta": float(sums.sum() / counts.sum()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "clusters": len(clusters),
        "iterations": int(iterations),
    }


def reprojection_error_for_gt(
    object_points_m: np.ndarray,
    image_points: np.ndarray,
    support: np.ndarray,
    camera_matrix: np.ndarray,
    rotation_gt: np.ndarray,
    translation_gt_m: np.ndarray,
) -> np.ndarray:
    points = np.asarray(object_points_m, dtype=np.float64)[support]
    expected = np.asarray(image_points, dtype=np.float64)[support]
    if not len(points):
        return np.empty(0, dtype=np.float64)
    projected, _ = cv2.projectPoints(
        points,
        cv2.Rodrigues(np.asarray(rotation_gt, dtype=np.float64))[0],
        np.asarray(translation_gt_m, dtype=np.float64).reshape(3, 1),
        np.asarray(camera_matrix, dtype=np.float64),
        np.zeros((8, 1), dtype=np.float64),
    )
    return np.linalg.norm(projected.reshape(-1, 2) - expected, axis=1)
