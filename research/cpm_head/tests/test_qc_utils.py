from __future__ import annotations

import numpy as np
import pytest

from research.cpm_head.qc_utils import (
    bin_indices,
    derive_moment_scales,
    moment_group_summaries,
    scalar_summary,
)


def test_scalar_summary_contract() -> None:
    result = scalar_summary(np.array([-2.0, 0.0, 1.0, np.nan]))
    assert result["count"] == 4
    assert result["finite_count"] == 3
    assert result["finite_ratio"] == 0.75
    assert result["zero_ratio"] == 1 / 3
    assert result["absolute_max"] == 2.0


def test_moment_group_summaries_use_only_selected_regions() -> None:
    descriptor = np.ones((2, 3, 21), dtype=np.float32)
    descriptor[0, 0, 1:4] = 9.0
    valid = np.zeros((2, 3), dtype=bool)
    valid[0, 0] = True
    result = moment_group_summaries(descriptor, valid)
    assert result["mu_x"]["median"] == 9.0
    assert result["mu_u"]["median"] == 1.0


def test_scale_rule_is_identity_below_one_order_of_magnitude() -> None:
    descriptor = np.ones((2, 2, 21), dtype=np.float32)
    valid = np.ones((2, 2), dtype=bool)
    result = derive_moment_scales(descriptor, valid)
    assert result["status"] == "PASS"
    assert result["rule"] == "identity"
    assert set(result["scales"].values()) == {1.0}


def test_scale_rule_uses_group_p95_for_large_mismatch() -> None:
    descriptor = np.ones((2, 2, 21), dtype=np.float32)
    descriptor[..., 15:21] = 0.001
    valid = np.ones((2, 2), dtype=bool)
    result = derive_moment_scales(descriptor, valid)
    assert result["status"] == "PASS"
    assert result["rule"] == "p95_abs_group_rescaling"
    assert result["scales"]["c_xu"] == pytest.approx(0.001)


def test_bin_indices_include_extreme_endpoints() -> None:
    result = bin_indices(np.array([0.0, 0.1, 0.2, 1.0]), (0.0, 0.1, 0.2, 1.0))
    assert result.tolist() == [0, 0, 1, 2]
