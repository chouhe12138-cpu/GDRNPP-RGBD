"""Pure aggregation helpers for CPM moment and soft-support audits."""

from __future__ import annotations

from typing import Iterable

import numpy as np


MOMENT_GROUP_SLICES = {
    "mu_x": slice(1, 4),
    "mu_u": slice(4, 6),
    "c_xx": slice(6, 12),
    "c_uu": slice(12, 15),
    "c_xu": slice(15, 21),
}
COVERAGE_EDGES = (0.0, 1e-4, 1e-3, 1e-2, 1.0)
EFFECTIVE_SAMPLE_EDGES = (0.0, 2.0, 8.0, 32.0, 4096.0)


def scalar_summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(array)
    finite_values = array[finite]
    result: dict[str, float | int] = {
        "count": int(array.size),
        "finite_count": int(finite.sum()),
        "finite_ratio": float(finite.mean()) if array.size else float("nan"),
    }
    if not finite_values.size:
        result.update(
            {
                "zero_ratio": float("nan"),
                "median": float("nan"),
                "p5": float("nan"),
                "p95": float("nan"),
                "p95_abs": float("nan"),
                "absolute_max": float("nan"),
            }
        )
        return result
    result.update(
        {
            "zero_ratio": float(np.mean(finite_values == 0.0)),
            "median": float(np.median(finite_values)),
            "p5": float(np.percentile(finite_values, 5)),
            "p95": float(np.percentile(finite_values, 95)),
            "p95_abs": float(np.percentile(np.abs(finite_values), 95)),
            "absolute_max": float(np.max(np.abs(finite_values))),
        }
    )
    return result


def moment_group_summaries(
    descriptors: np.ndarray, region_selector: np.ndarray
) -> dict[str, dict[str, float | int]]:
    values = np.asarray(descriptors)
    selector = np.asarray(region_selector, dtype=bool)
    if values.ndim != 3 or values.shape[-1] != 21:
        raise ValueError(f"descriptors must have shape SxKx21, got {values.shape}")
    if selector.shape != values.shape[:2]:
        raise ValueError("region selector shape differs from descriptors")
    return {
        group: scalar_summary(values[..., group_slice][selector])
        for group, group_slice in MOMENT_GROUP_SLICES.items()
    }


def derive_moment_scales(
    descriptors: np.ndarray, valid: np.ndarray
) -> dict[str, object]:
    summaries = moment_group_summaries(descriptors, valid)
    robust = {group: float(summary["p95_abs"]) for group, summary in summaries.items()}
    invalid = [
        group
        for group, value in robust.items()
        if not np.isfinite(value) or value <= 1e-8
    ]
    if invalid:
        return {
            "status": "BLOCKED",
            "reason": "invalid_group_scale",
            "invalid_groups": invalid,
            "raw_p95_abs": robust,
        }
    ratio = max(robust.values()) / min(robust.values())
    use_rescaling = ratio >= 10.0
    scales = {
        group: max(value, 1e-6) if use_rescaling else 1.0
        for group, value in robust.items()
    }
    return {
        "status": "PASS",
        "rule": "p95_abs_group_rescaling" if use_rescaling else "identity",
        "trigger_ratio": float(ratio),
        "trigger_threshold": 10.0,
        "raw_p95_abs": robust,
        "scales": scales,
    }


def interval_labels(edges: Iterable[float], prefix: str) -> tuple[str, ...]:
    values = tuple(float(value) for value in edges)
    return tuple(
        f"{prefix}_{values[index]:g}_to_{values[index + 1]:g}"
        for index in range(len(values) - 1)
    )


def bin_indices(values: np.ndarray, edges: Iterable[float]) -> np.ndarray:
    boundaries = np.asarray(tuple(edges), dtype=np.float64)
    if boundaries.ndim != 1 or boundaries.size < 2:
        raise ValueError("bin edges must contain at least two values")
    # Include the rightmost finite endpoint in the final bin.
    return np.clip(np.digitize(values, boundaries[1:-1], right=True), 0, boundaries.size - 2)
