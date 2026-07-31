from __future__ import annotations

import json

import numpy as np
import pytest

from research.pose_head_utilization.utilization_utils import (
    DEFAULT_ALPHAS,
    alpha_token,
    interpolate_xyz,
    utilization_decision,
)


def test_interpolation_changes_only_frozen_support_and_scales_error() -> None:
    predicted = np.full((2, 2, 3), 0.5, dtype=np.float64)
    gt_metric = np.full((2, 2, 3), 0.2, dtype=np.float64)
    extent = np.ones(3, dtype=np.float64)
    support = np.array([[True, False], [False, True]])
    halfway = interpolate_xyz(predicted, gt_metric, extent, support, alpha=0.5)
    expected_gt = 0.7
    np.testing.assert_allclose(halfway[support], (0.5 + expected_gt) / 2.0)
    np.testing.assert_array_equal(halfway[~support], predicted[~support])


def test_alpha_token_is_stable() -> None:
    assert [alpha_token(value) for value in DEFAULT_ALPHAS] == [
        "a000",
        "a025",
        "a050",
        "a075",
        "a100",
    ]
    with pytest.raises(ValueError):
        alpha_token(1.1)


def _decision_rows(patch_curve, ransac_curve):
    methods = []
    objects = []
    for alpha, patch, ransac in zip(DEFAULT_ALPHAS, patch_curve, ransac_curve):
        for prefix, value in (("patch", patch), ("ransac", ransac)):
            method = f"{prefix}_{alpha_token(alpha)}"
            methods.append({"method": method, "add_s_0.1d_recall": value, "bop_ar": value})
            for obj_id in range(1, 9):
                objects.append(
                    {
                        "method": method,
                        "obj_id": obj_id,
                        "add_s_0.1d_recall": value,
                    }
                )
    return methods, objects


def test_decision_detects_pose_head_underutilization() -> None:
    methods, objects = _decision_rows(
        [0.50, 0.50, 0.505, 0.505, 0.51],
        [0.53, 0.58, 0.63, 0.68, 0.73],
    )
    result = utilization_decision(methods, objects)
    assert result["status"] == "PATCH_PNP_UNDERUTILIZATION"
    json.dumps(result)


def test_decision_detects_pose_head_use_of_improved_xyz() -> None:
    methods, objects = _decision_rows(
        [0.50, 0.56, 0.62, 0.68, 0.75],
        [0.53, 0.59, 0.65, 0.71, 0.78],
    )
    result = utilization_decision(methods, objects)
    assert result["status"] == "PATCH_PNP_USES_IMPROVED_XYZ"
    assert result["next_action"] == "IMPROVE_XYZ_GEOMETRY"
