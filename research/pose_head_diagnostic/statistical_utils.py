"""Aggregation helpers that never persist instance-level diagnostic features."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np


SUMMARY_FIELDS = (
    "count",
    "finite_count",
    "nonfinite_count",
    "mean",
    "std",
    "median",
    "p05",
    "p25",
    "p75",
    "p95",
    "ci95_low",
    "ci95_high",
    "positive_fraction",
    "negative_fraction",
)


def summarize_values(
    values: Iterable[float],
    seed: int,
    bootstrap_samples: int = 1000,
) -> dict:
    """Return compact descriptive statistics and a mean bootstrap interval."""

    raw = np.asarray(list(values), dtype=np.float64)
    finite = raw[np.isfinite(raw)]
    result = {
        "count": int(len(raw)),
        "finite_count": int(len(finite)),
        "nonfinite_count": int(len(raw) - len(finite)),
    }
    if not len(finite):
        return {
            **result,
            **{field: float("nan") for field in SUMMARY_FIELDS if field not in result},
        }
    rng = np.random.default_rng(int(seed))
    if len(finite) == 1:
        ci_low = ci_high = float(finite[0])
    else:
        means = np.empty(int(bootstrap_samples), dtype=np.float64)
        for index in range(int(bootstrap_samples)):
            means[index] = np.mean(rng.choice(finite, len(finite), replace=True))
        ci_low, ci_high = np.quantile(means, (0.025, 0.975))
    return {
        **result,
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "median": float(np.median(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "p25": float(np.quantile(finite, 0.25)),
        "p75": float(np.quantile(finite, 0.75)),
        "p95": float(np.quantile(finite, 0.95)),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "positive_fraction": float(np.mean(finite > 0)),
        "negative_fraction": float(np.mean(finite < 0)),
    }


def aggregate_scalar_records(
    records: Sequence[dict],
    group_fields: tuple[str, ...],
    metric_fields: Sequence[str],
    seed: int,
    bootstrap_samples: int = 1000,
) -> list[dict]:
    """Aggregate in-memory scalar records into one row per group and metric."""

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        groups[tuple(record[field] for field in group_fields)].append(record)
    rows = []
    for group_index, (key, group) in enumerate(sorted(groups.items())):
        prefix = dict(zip(group_fields, key))
        for metric_index, metric in enumerate(metric_fields):
            rows.append(
                {
                    **prefix,
                    "metric": metric,
                    **summarize_values(
                        (record[metric] for record in group),
                        seed=seed + group_index * 1009 + metric_index,
                        bootstrap_samples=bootstrap_samples,
                    ),
                }
            )
    return rows


def assign_quartile_labels(records: Sequence[dict], field: str) -> list[str]:
    values = np.asarray([float(record[field]) for record in records], dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return ["not_finite"] * len(records)
    thresholds = np.quantile(finite, (0.25, 0.50, 0.75))
    labels = []
    for value in values:
        if not np.isfinite(value):
            labels.append("not_finite")
        elif value <= thresholds[0]:
            labels.append("q1")
        elif value <= thresholds[1]:
            labels.append("q2")
        elif value <= thresholds[2]:
            labels.append("q3")
        else:
            labels.append("q4")
    return labels
