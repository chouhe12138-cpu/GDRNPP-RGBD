"""Model-free CAD diagnostics. Coverage statistics do not impose scientific gates."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from .geometry import assign_paths


def surface_representation(levels: dict, object_ids: tuple[int, ...],
                           surface_points: dict[int, np.ndarray], *, selected_depths=None,
                           sample_seed=None) -> dict:
    """Sampled surface coverage at caller-selected depths.

    The depth-derived GT sits several millimetres off the mesh, which is larger than
    most leaf radii, so sampling the mesh itself separates that data deficit from the
    hierarchy's own capability:

    * ``global``    -- inside the globally nearest anchor's ball, not the union
    * ``traversal`` -- the ball of the leaf reached by the 8-ary nearest-child walk
                       (what the head supervises and infers) contains it
    """
    selected_depths = tuple(sorted(levels)) if selected_depths is None else tuple(selected_depths)
    level_depth = {f"t{d}": d for d in selected_depths}
    if not selected_depths or any(d not in levels for d in selected_depths):
        raise ValueError("selected_depths must name existing levels")
    if not surface_points or any(len(p) == 0 for p in surface_points.values()):
        raise ValueError("Surface coverage requires nonempty samples for every object")
    report = {"samples_per_object": int(len(next(iter(surface_points.values())))),
              "sample_seed": sample_seed, "per_object": {},
              "coverage_definition": "global means nearest-anchor ball; traversal means assigned ball",
              "sample_source": "caller-provided surface points; may include mesh vertices",
              "empty_nodes_check": "NOT_CHECKED; unhit sampled nodes do not prove empty cells"}
    pooled = {level: [] for level in level_depth}
    agreement = {depth: [] for depth in range(2, len(levels) + 1)}
    for index, obj_id in enumerate(object_ids):
        points = surface_points[obj_id]
        ids = assign_paths(points, index, levels)
        entry = {"num_points": int(len(points))}
        for depth in range(2, len(levels) + 1):
            nearest = cKDTree(levels[depth]["anchors"][index]).query(points, k=1)[1]
            agreement[depth].append(float((nearest == ids[:, depth - 1]).mean()))
        for level, depth in level_depth.items():
            anchors = levels[depth]["anchors"][index][ids[:, depth - 1]]
            radii = levels[depth]["radii"][index][ids[:, depth - 1]]
            traversal = np.linalg.norm(points - anchors, axis=1) / radii
            nearest = cKDTree(levels[depth]["anchors"][index]).query(points, k=1)
            entry[f"global_coverage_{level}"] = float(
                (nearest[0] / levels[depth]["radii"][index][nearest[1]] <= 1.0).mean())
            entry[f"traversal_coverage_{level}"] = float((traversal <= 1.0).mean())
            entry[f"traversal_p95_{level}"] = float(np.percentile(traversal, 95))
            entry[f"unhit_sampled_nodes_{level}"] = int(np.count_nonzero(
                np.bincount(ids[:, depth - 1], minlength=levels[depth]['radii'].shape[1]) == 0))
            pooled[level].append(traversal)
        report["per_object"][str(obj_id)] = entry
    for depth in range(2, len(levels) + 1):
        report[f"traversal_matches_global_level{depth}"] = float(np.mean(agreement[depth]))
    for level in level_depth:
        norms = np.concatenate(pooled[level])
        report[f"pooled_traversal_coverage_{level}"] = float((norms <= 1.0).mean())
        report[f"pooled_traversal_p95_{level}"] = float(np.percentile(norms, 95))
        report[f"pooled_global_coverage_{level}"] = float(
            np.mean([item[f"global_coverage_{level}"] for item in report["per_object"].values()]))
    return report


def hierarchy_shape_report(levels: dict, object_ids: tuple[int, ...]) -> dict:
    """Report malformed input before attempting any parent/child reshape."""
    num_objects = len(object_ids)
    report = {"object_ids": [int(v) for v in object_ids], "levels": {}, "parent_coverage": {}}
    report['failed_relations'] = []
    if (not num_objects or sorted(levels) != list(range(1, len(levels) + 1))
            or not levels or any(not {'anchors', 'normals', 'radii'} <= set(v) for v in levels.values())
            or levels[1]['anchors'].ndim != 3 or levels[1]['anchors'].shape[1] < 2):
        report.update(result='FAIL', error='Invalid tree levels, fields or root shape')
        return report
    branch = levels[1]["anchors"].shape[1]
    counts = tuple(branch ** d for d in range(1, len(levels) + 1))
    for depth, count in enumerate(counts, start=1):
        anchors, radii = levels[depth]["anchors"], levels[depth]["radii"]
        normals = levels[depth]['normals']
        report["levels"][f"level{depth}"] = {
            "anchors_shape": list(anchors.shape),
            "radii_shape": list(radii.shape),
            "expected_anchors_shape": [num_objects, count, 3],
            "expected_radii_shape": [num_objects, count],
            "shape_ok": list(anchors.shape) == [num_objects, count, 3]
            and list(radii.shape) == [num_objects, count]
            and list(normals.shape) == [num_objects, count, 3],
            "all_finite": bool(np.isfinite(anchors).all() and np.isfinite(radii).all()
                               and np.isfinite(normals).all()),
            "radius_min_m": float(radii.min()) if radii.size else None,
            "radius_max_m": float(radii.max()) if radii.size else None,
            "all_radii_positive": bool((radii > 0).all()),
        }
    shapes_ok = all(item["shape_ok"] and item["all_finite"] and item["all_radii_positive"]
                    for item in report["levels"].values())
    report['result'] = 'PASS' if shapes_ok else 'FAIL'
    return report


def parent_child_coverage(levels, object_ids):
    """Coverage ratios for every adjacent relation; requires valid shapes."""
    shape_report = hierarchy_shape_report(levels, object_ids)
    if shape_report['result'] != 'PASS':
        raise ValueError('Parent coverage requires valid hierarchy shapes and values')
    num_objects = len(object_ids)
    branch = levels[1]['anchors'].shape[1]
    counts = tuple(branch ** d for d in range(1, len(levels) + 1))
    report = {'parent_coverage': {}}
    for child_depth in range(2, len(levels) + 1):
        parent_count = counts[child_depth - 2]
        child_anchors = levels[child_depth]["anchors"].reshape(num_objects, parent_count, branch, 3)
        child_radii = levels[child_depth]["radii"].reshape(num_objects, parent_count, branch)
        required = (
            np.linalg.norm(child_anchors - levels[child_depth - 1]["anchors"][:, :, None, :],
                           axis=-1) + child_radii
        ).max(-1)
        ratio = (levels[child_depth - 1]["radii"] / np.maximum(required, 1e-12)).reshape(-1)
        report["parent_coverage"][f"level{child_depth - 1}_over_level{child_depth}"] = {
            "count": int(ratio.size),
            "min": float(ratio.min()),
            "p1": float(np.percentile(ratio, 1)),
            "p50": float(np.percentile(ratio, 50)),
            "mean": float(ratio.mean()),
            "below_one": int(np.count_nonzero(ratio < 1.0)),
            "below_1.049": int(np.count_nonzero(ratio < 1.049)),
        }
    return report['parent_coverage']


def hierarchy_sanity(levels, object_ids):
    """Structural validity and all parent balls; no sampled-surface gate."""
    report = hierarchy_shape_report(levels, object_ids)
    if report['result'] != 'PASS':
        return report
    report['parent_coverage'] = parent_child_coverage(levels, object_ids)
    report["failed_relations"] = [key for key, value in report["parent_coverage"].items()
                                  if value["below_one"] > 0]
    report["result"] = "PASS" if not report["failed_relations"] else "FAIL"
    return report

def _finite(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def residual_stats(norms) -> dict | None:
    array = _finite(norms)
    if not array.size:
        return None
    stats = {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }
    for threshold in (0.25, 0.50, 0.75, 1.00):
        stats[f"fraction_le_{threshold:.2f}"] = float((array <= threshold).mean())
    stats["fraction_gt_1.00"] = float((array > 1.0).mean())
    return stats

