from __future__ import annotations

import numpy as np

from research.oracle_diagnostic.oracle_utils import (
    build_correspondences_from_dense,
    depth_to_object_coordinates,
    mask_metrics,
    paired_cluster_bootstrap,
    reprojection_error_for_gt,
    subset_top_fraction,
)


CAMERA = np.array(
    [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64
)


def test_depth_object_coordinates_reproject_exactly() -> None:
    depth = np.full((480, 640), 0.8, dtype=np.float64)
    image_points = np.array([[[300.25, 220.5], [340.5, 260.25]]], dtype=np.float64)
    rotation = np.eye(3)
    translation = np.array([0.01, -0.02, 0.7])
    xyz, valid = depth_to_object_coordinates(depth, image_points, CAMERA, rotation, translation)
    error = reprojection_error_for_gt(xyz, image_points, valid, CAMERA, rotation, translation)
    assert valid.all()
    assert np.max(error) < 1e-9


def test_mask_metrics_separate_precision_and_recall() -> None:
    truth = np.array([[1, 1], [0, 0]], dtype=bool)
    predicted = np.array([[1, 0], [1, 0]], dtype=bool)
    result = mask_metrics(predicted, truth)
    assert result == {"iou": 1 / 3, "precision": 0.5, "recall": 0.5}


def test_top_fraction_uses_requested_order() -> None:
    image = np.column_stack((np.arange(10), np.arange(10))).reshape(2, 5, 2)
    model = np.column_stack((np.arange(10), np.arange(10), np.arange(10))).reshape(2, 5, 3)
    corr = build_correspondences_from_dense(image, model, np.ones((2, 5), dtype=bool))
    best = subset_top_fraction(corr, np.arange(10), keep_fraction=0.5, largest=False)
    assert best.size == 5
    np.testing.assert_array_equal(best.model_points[:, 0], np.arange(5))


def test_cluster_bootstrap_preserves_pairing() -> None:
    rows_a = []
    rows_b = []
    for image_id in range(5):
        for instance_id in range(2):
            base = {"scene_id": 2, "im_id": image_id, "instance_id": instance_id, "obj_id": 1}
            rows_a.append({**base, "score": 1.0})
            rows_b.append({**base, "score": 0.0})
    result = paired_cluster_bootstrap(rows_a, rows_b, "score", iterations=100, seed=3)
    assert result["delta"] == 1.0
    assert result["ci95_low"] == 1.0
    assert result["ci95_high"] == 1.0
    assert result["clusters"] == 5
