from __future__ import annotations

import cv2
import numpy as np

from research.pose_aggregation.solvers import (
    Correspondences,
    METHODS,
    build_correspondences,
    reliable_subset,
    solve_methods,
    solve_ransac_epnp,
    solve_translation_with_fixed_rotation,
)


CAMERA = np.array(
    [
        [572.4, 0.0, 325.3],
        [0.0, 573.6, 242.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def make_correspondences(
    seed: int = 7,
    count: int = 200,
    noise_px: float = 0.0,
    outlier_fraction: float = 0.0,
) -> tuple[Correspondences, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    model_points = rng.uniform(-0.06, 0.06, size=(count, 3))
    rotation = cv2.Rodrigues(np.array([0.25, -0.12, 0.08], dtype=np.float64))[0]
    translation = np.array([0.015, -0.02, 0.72], dtype=np.float64)
    projected, _ = cv2.projectPoints(
        model_points,
        cv2.Rodrigues(rotation)[0],
        translation,
        CAMERA,
        np.zeros((8, 1), dtype=np.float64),
    )
    image_points = projected.reshape(-1, 2)
    if noise_px:
        image_points += rng.normal(0.0, noise_px, size=image_points.shape)
    reliability = np.full(count, 0.95, dtype=np.float64)
    outlier_count = int(round(count * outlier_fraction))
    if outlier_count:
        indices = rng.choice(count, size=outlier_count, replace=False)
        image_points[indices] = rng.uniform([0.0, 0.0], [640.0, 480.0], size=(outlier_count, 2))
        reliability[indices] = 0.02
    return Correspondences(image_points, model_points, reliability), rotation, translation


def rotation_error_deg(estimate: np.ndarray, truth: np.ndarray) -> float:
    relative = estimate @ truth.T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def test_clean_ransac_recovers_pose() -> None:
    correspondences, rotation, translation = make_correspondences(noise_px=0.05)
    solution = solve_ransac_epnp(correspondences, CAMERA, seed=11)
    assert solution.success
    assert rotation_error_deg(solution.rotation, rotation) < 0.1
    assert np.linalg.norm(solution.translation - translation) < 1e-3


def test_reliability_filter_rejects_low_confidence_outliers() -> None:
    correspondences, _, _ = make_correspondences(outlier_fraction=0.3)
    filtered = reliable_subset(correspondences)
    assert filtered.size == correspondences.size // 2
    assert filtered.reliability.min() > 0.5


def test_fixed_rotation_translation_recovers_translation() -> None:
    correspondences, rotation, translation = make_correspondences(noise_px=0.1)
    solution = solve_translation_with_fixed_rotation(correspondences, CAMERA, rotation)
    assert solution.success
    assert np.linalg.norm(solution.translation - translation) < 1e-3


def test_all_methods_return_named_results() -> None:
    correspondences, rotation, translation = make_correspondences(noise_px=0.1, outlier_fraction=0.3)
    solutions = solve_methods(correspondences, CAMERA, rotation, translation, seed=13)
    assert tuple(solutions) == METHODS
    assert solutions["patch_pnp"].success
    assert solutions["reliable_ransac"].success


def test_degenerate_inputs_fail_explicitly() -> None:
    too_few = Correspondences(
        image_points=np.zeros((3, 2), dtype=np.float64),
        model_points=np.zeros((3, 3), dtype=np.float64),
        reliability=np.ones(3, dtype=np.float64),
    )
    assert not solve_ransac_epnp(too_few, CAMERA).success

    collinear = Correspondences(
        image_points=np.column_stack((np.arange(5), np.arange(5))).astype(np.float64),
        model_points=np.column_stack((np.arange(5), np.zeros((5, 2)))).astype(np.float64),
        reliability=np.ones(5, dtype=np.float64),
    )
    result = solve_translation_with_fixed_rotation(collinear, CAMERA, np.eye(3))
    assert not result.success or np.isfinite(result.translation).all()


def test_dense_conversion_does_not_mutate_predictions() -> None:
    mask = np.ones((2, 2), dtype=np.float64)
    xyz = np.array(
        [
            [[0.2, 0.3, 0.4], [0.6, 0.7, 0.8]],
            [[0.3, 0.4, 0.5], [0.7, 0.8, 0.9]],
        ],
        dtype=np.float64,
    )
    coords = np.array(
        [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
        ],
        dtype=np.float64,
    )
    region_logits = np.zeros((65, 2, 2), dtype=np.float64)
    xyz_before = xyz.copy()
    coords_before = coords.copy()

    correspondences = build_correspondences(
        mask,
        xyz,
        coords,
        image_height=480,
        image_width=640,
        extent=np.array([0.2, 0.3, 0.4]),
        region_logits=region_logits,
    )

    assert correspondences.size == 3
    np.testing.assert_array_equal(xyz, xyz_before)
    np.testing.assert_array_equal(coords, coords_before)

