from __future__ import annotations

import numpy as np

from research.pose_aggregation.metrics import pose_metrics, rotation_error_deg, translation_error_mm


def test_identity_pose_has_zero_error() -> None:
    rotation = np.eye(3)
    translation = np.array([0.0, 0.0, 0.7])
    points = np.array(
        [
            [-0.05, -0.05, 0.0],
            [0.05, -0.05, 0.0],
            [0.05, 0.05, 0.0],
            [-0.05, 0.05, 0.0],
        ]
    )
    result = pose_metrics(rotation, translation, rotation, translation, points, 0.2, obj_id=1)
    assert result["rotation_error_deg"] == 0.0
    assert result["translation_error_mm"] == 0.0
    assert result["add_s_m"] == 0.0
    assert result["add_s_0.1d"] == 1.0


def test_error_units_are_degrees_and_millimetres() -> None:
    ninety_degrees = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert np.isclose(rotation_error_deg(ninety_degrees, np.eye(3)), 90.0)
    assert np.isclose(translation_error_mm(np.array([0.0, 0.0, 0.701]), np.array([0.0, 0.0, 0.7])), 1.0)


def test_rotation_error_respects_discrete_symmetry() -> None:
    half_turn = np.diag([-1.0, -1.0, 1.0])
    symmetries = np.stack((np.eye(3), half_turn))
    assert np.isclose(rotation_error_deg(half_turn, np.eye(3), symmetries), 0.0)
