"""Evaluation helpers for the pose-aggregation diagnostic."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.spatial import cKDTree


SYMMETRIC_LMO_OBJECT_IDS = frozenset({10, 11})


def rotation_error_deg(
    rotation_est: np.ndarray,
    rotation_gt: np.ndarray,
    symmetry_rotations: Optional[np.ndarray] = None,
) -> float:
    rotation_est = np.asarray(rotation_est, dtype=np.float64)
    rotation_gt = np.asarray(rotation_gt, dtype=np.float64)
    if symmetry_rotations is None:
        symmetry_rotations = np.eye(3, dtype=np.float64)[None]
    equivalent_gt = rotation_gt[None] @ np.asarray(symmetry_rotations, dtype=np.float64)
    relatives = rotation_est[None] @ np.transpose(equivalent_gt, (0, 2, 1))
    cosines = np.clip((np.trace(relatives, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.min(np.degrees(np.arccos(cosines))))


def translation_error_mm(translation_est: np.ndarray, translation_gt: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(translation_est) - np.asarray(translation_gt)) * 1000.0)


def add_distance_m(
    rotation_est: np.ndarray,
    translation_est: np.ndarray,
    rotation_gt: np.ndarray,
    translation_gt: np.ndarray,
    model_points: np.ndarray,
    symmetric: bool,
) -> float:
    points = np.asarray(model_points, dtype=np.float64)
    estimated = (np.asarray(rotation_est) @ points.T).T + np.asarray(translation_est).reshape(1, 3)
    ground_truth = (np.asarray(rotation_gt) @ points.T).T + np.asarray(translation_gt).reshape(1, 3)
    if symmetric:
        distances, _ = cKDTree(ground_truth).query(estimated, k=1, workers=1)
        return float(np.mean(distances))
    return float(np.mean(np.linalg.norm(estimated - ground_truth, axis=1)))


def pose_metrics(
    rotation_est: np.ndarray,
    translation_est: np.ndarray,
    rotation_gt: np.ndarray,
    translation_gt: np.ndarray,
    model_points: np.ndarray,
    diameter_m: float,
    obj_id: int,
    symmetry_rotations: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    add_m = add_distance_m(
        rotation_est,
        translation_est,
        rotation_gt,
        translation_gt,
        model_points,
        symmetric=obj_id in SYMMETRIC_LMO_OBJECT_IDS,
    )
    return {
        "rotation_error_deg": rotation_error_deg(rotation_est, rotation_gt, symmetry_rotations),
        "translation_error_mm": translation_error_mm(translation_est, translation_gt),
        "add_s_m": add_m,
        "add_s_0.1d": float(add_m < 0.1 * float(diameter_m)),
    }
