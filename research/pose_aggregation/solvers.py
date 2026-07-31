"""Deterministic pose solvers used by the GDRNPP aggregation diagnostic.

This module is intentionally independent from the upstream evaluator.  Every
method consumes the same dense prediction, which makes the comparison isolate
pose aggregation rather than network inference.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


METHODS = (
    "patch_pnp",
    "epnp_all",
    "ransac_epnp",
    "reliable_ransac",
    "geom_R_net_t",
    "net_R_geom_t",
)


@dataclass(frozen=True)
class Correspondences:
    image_points: np.ndarray
    model_points: np.ndarray
    reliability: np.ndarray

    @property
    def size(self) -> int:
        return int(self.image_points.shape[0])


@dataclass(frozen=True)
class PoseSolution:
    success: bool
    rotation: np.ndarray
    translation: np.ndarray
    num_points: int
    num_inliers: int
    median_reprojection_error: float
    failure_reason: str = ""
    solver_time_ms: float = 0.0

    @property
    def pose(self) -> np.ndarray:
        return np.hstack((self.rotation, self.translation.reshape(3, 1)))


def _failed(num_points: int, reason: str) -> PoseSolution:
    return PoseSolution(
        success=False,
        rotation=np.full((3, 3), np.nan, dtype=np.float64),
        translation=np.full(3, np.nan, dtype=np.float64),
        num_points=int(num_points),
        num_inliers=0,
        median_reprojection_error=float("nan"),
        failure_reason=reason,
    )


def _softmax(values: np.ndarray, axis: int = 0) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.maximum(exp_values.sum(axis=axis, keepdims=True), 1e-12)


def build_correspondences(
    mask_probability: np.ndarray,
    xyz_normalized: np.ndarray,
    coord_2d_normalized: np.ndarray,
    image_height: int,
    image_width: int,
    extent: np.ndarray,
    region_logits: Optional[np.ndarray] = None,
    mask_threshold: float = 0.5,
) -> Correspondences:
    """Convert dense crop predictions to image/object correspondences.

    The background test matches the official GDRNPP evaluator, but all arrays
    are copied before conversion to prevent cross-method mutation.
    """

    mask = np.asarray(mask_probability, dtype=np.float64).squeeze().copy()
    xyz = np.asarray(xyz_normalized, dtype=np.float64).copy()
    coords = np.asarray(coord_2d_normalized, dtype=np.float64).copy()
    extent = np.asarray(extent, dtype=np.float64).reshape(3)

    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz_normalized must have shape HxWx3, got {xyz.shape}")
    if coords.shape != xyz.shape[:2] + (2,):
        raise ValueError(f"coord_2d_normalized must have shape HxWx2, got {coords.shape}")
    if mask.shape != xyz.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} does not match xyz shape {xyz.shape[:2]}")

    xyz = (xyz - 0.5) * extent.reshape(1, 1, 3)
    coords[..., 0] *= float(image_width)
    coords[..., 1] *= float(image_height)

    epsilon = 1e-4 * extent.reshape(1, 1, 3)
    finite = np.isfinite(xyz).all(axis=2) & np.isfinite(coords).all(axis=2) & np.isfinite(mask)
    non_background = (np.abs(xyz) > epsilon).all(axis=2)
    valid = finite & non_background & (mask > float(mask_threshold))

    if region_logits is None:
        region_confidence = np.ones_like(mask, dtype=np.float64)
    else:
        region = np.asarray(region_logits, dtype=np.float64)
        if region.ndim != 3 or region.shape[1:] != mask.shape:
            raise ValueError(f"region_logits must have shape CxHxW, got {region.shape}")
        foreground = region[1:] if region.shape[0] > 1 else region
        region_confidence = np.max(_softmax(foreground, axis=0), axis=0)

    reliability = np.clip(mask, 0.0, 1.0) * np.clip(region_confidence, 0.0, 1.0)
    return Correspondences(
        image_points=np.ascontiguousarray(coords[valid].reshape(-1, 2), dtype=np.float64),
        model_points=np.ascontiguousarray(xyz[valid].reshape(-1, 3), dtype=np.float64),
        reliability=np.ascontiguousarray(reliability[valid].reshape(-1), dtype=np.float64),
    )


def reliable_subset(
    correspondences: Correspondences,
    keep_fraction: float = 0.5,
    minimum_before_filter: int = 32,
) -> Correspondences:
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be in (0, 1]")
    if correspondences.size < minimum_before_filter:
        return correspondences

    keep = max(4, int(np.ceil(correspondences.size * keep_fraction)))
    order = np.argsort(-correspondences.reliability, kind="stable")[:keep]
    return Correspondences(
        image_points=np.ascontiguousarray(correspondences.image_points[order]),
        model_points=np.ascontiguousarray(correspondences.model_points[order]),
        reliability=np.ascontiguousarray(correspondences.reliability[order]),
    )


def reprojection_errors(
    rotation: np.ndarray,
    translation: np.ndarray,
    correspondences: Correspondences,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    if correspondences.size == 0:
        return np.empty(0, dtype=np.float64)
    projected, _ = cv2.projectPoints(
        correspondences.model_points,
        cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))[0],
        np.asarray(translation, dtype=np.float64).reshape(3, 1),
        np.asarray(camera_matrix, dtype=np.float64),
        np.zeros((8, 1), dtype=np.float64),
    )
    return np.linalg.norm(projected.reshape(-1, 2) - correspondences.image_points, axis=1)


def solution_from_pose(
    rotation: np.ndarray,
    translation: np.ndarray,
    correspondences: Correspondences,
    camera_matrix: np.ndarray,
    num_inliers: Optional[int] = None,
) -> PoseSolution:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        return _failed(correspondences.size, "non_finite_pose")
    if translation[2] <= 0:
        return _failed(correspondences.size, "non_positive_depth")
    errors = reprojection_errors(rotation, translation, correspondences, camera_matrix)
    median_error = float(np.median(errors)) if errors.size else float("nan")
    return PoseSolution(
        success=True,
        rotation=rotation,
        translation=translation,
        num_points=correspondences.size,
        num_inliers=correspondences.size if num_inliers is None else int(num_inliers),
        median_reprojection_error=median_error,
    )


def solve_epnp(correspondences: Correspondences, camera_matrix: np.ndarray) -> PoseSolution:
    if correspondences.size < 4:
        return _failed(correspondences.size, "fewer_than_four_points")
    try:
        success, rvec, tvec = cv2.solvePnP(
            correspondences.model_points,
            correspondences.image_points,
            np.asarray(camera_matrix, dtype=np.float64),
            np.zeros((8, 1), dtype=np.float64),
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error as error:
        return _failed(correspondences.size, f"opencv:{error.code}")
    if not success:
        return _failed(correspondences.size, "opencv_returned_false")
    rotation = cv2.Rodrigues(rvec)[0]
    return solution_from_pose(rotation, tvec, correspondences, camera_matrix)


def solve_ransac_epnp(
    correspondences: Correspondences,
    camera_matrix: np.ndarray,
    seed: int = 0,
    reprojection_threshold: float = 3.0,
    iterations: int = 100,
) -> PoseSolution:
    if correspondences.size < 4:
        return _failed(correspondences.size, "fewer_than_four_points")
    cv2.setRNGSeed(int(seed))
    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            correspondences.model_points,
            correspondences.image_points,
            np.asarray(camera_matrix, dtype=np.float64),
            np.zeros((8, 1), dtype=np.float64),
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=float(reprojection_threshold),
            iterationsCount=int(iterations),
            confidence=0.99,
        )
    except cv2.error as error:
        return _failed(correspondences.size, f"opencv:{error.code}")
    if not success:
        return _failed(correspondences.size, "opencv_returned_false")
    rotation = cv2.Rodrigues(rvec)[0]
    num_inliers = 0 if inliers is None else len(inliers)
    return solution_from_pose(rotation, tvec, correspondences, camera_matrix, num_inliers=num_inliers)


def solve_translation_with_fixed_rotation(
    correspondences: Correspondences,
    camera_matrix: np.ndarray,
    rotation: np.ndarray,
) -> PoseSolution:
    """Solve translation from 2D-3D pairs while holding rotation fixed."""

    if correspondences.size < 2:
        return _failed(correspondences.size, "fewer_than_two_points")

    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    homogeneous = np.column_stack(
        (correspondences.image_points, np.ones(correspondences.size, dtype=np.float64))
    )
    normalized = (np.linalg.inv(camera_matrix) @ homogeneous.T).T
    x = normalized[:, 0] / normalized[:, 2]
    y = normalized[:, 1] / normalized[:, 2]
    rotated = (rotation @ correspondences.model_points.T).T

    system = np.zeros((2 * correspondences.size, 3), dtype=np.float64)
    target = np.zeros(2 * correspondences.size, dtype=np.float64)
    system[0::2, 0] = 1.0
    system[0::2, 2] = -x
    system[1::2, 1] = 1.0
    system[1::2, 2] = -y
    target[0::2] = x * rotated[:, 2] - rotated[:, 0]
    target[1::2] = y * rotated[:, 2] - rotated[:, 1]

    weights = np.sqrt(np.clip(correspondences.reliability, 1e-6, 1.0))
    row_weights = np.repeat(weights, 2)
    weighted_system = system * row_weights[:, None]
    weighted_target = target * row_weights
    translation, _, rank, _ = np.linalg.lstsq(weighted_system, weighted_target, rcond=None)
    if rank < 3:
        return _failed(correspondences.size, "rank_deficient_system")
    return solution_from_pose(rotation, translation, correspondences, camera_matrix)


def solve_methods(
    correspondences: Correspondences,
    camera_matrix: np.ndarray,
    network_rotation: np.ndarray,
    network_translation: np.ndarray,
    seed: int = 0,
) -> Dict[str, PoseSolution]:
    """Run the pre-registered six-method diagnostic."""

    def timed(callable_):
        start = time.perf_counter()
        solution = callable_()
        return replace(solution, solver_time_ms=(time.perf_counter() - start) * 1000.0)

    reliable = reliable_subset(correspondences)
    patch = timed(
        lambda: solution_from_pose(
            network_rotation,
            network_translation,
            correspondences,
            camera_matrix,
            num_inliers=correspondences.size,
        )
    )
    epnp = timed(lambda: solve_epnp(correspondences, camera_matrix))
    ransac = timed(lambda: solve_ransac_epnp(correspondences, camera_matrix, seed=seed))
    reliable_ransac = timed(lambda: solve_ransac_epnp(reliable, camera_matrix, seed=seed))

    if reliable_ransac.success:
        geom_r_net_t = timed(
            lambda: solution_from_pose(
                reliable_ransac.rotation,
                network_translation,
                reliable,
                camera_matrix,
                num_inliers=reliable_ransac.num_inliers,
            )
        )
    else:
        geom_r_net_t = _failed(reliable.size, f"rotation_source_failed:{reliable_ransac.failure_reason}")

    net_r_geom_t = timed(
        lambda: solve_translation_with_fixed_rotation(
            reliable,
            camera_matrix,
            network_rotation,
        )
    )

    return {
        "patch_pnp": patch,
        "epnp_all": epnp,
        "ransac_epnp": ransac,
        "reliable_ransac": reliable_ransac,
        "geom_R_net_t": geom_r_net_t,
        "net_R_geom_t": net_r_geom_t,
    }
