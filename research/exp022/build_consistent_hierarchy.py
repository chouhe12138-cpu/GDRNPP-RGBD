#!/usr/bin/env python3
"""Build the traversal-consistent 8^4 CAD surface hierarchy (EXP022 v3).

The EXP022 read-out rule is a nested nearest-anchor walk
(``ProgressivePCCHead._nearest_child`` / ``_target_paths``): a surface point
descends to the child whose anchor is closest.  A hierarchy is *consistent*
only when each level is a nearest-anchor partition of its parent's cell *and*
each node's residual ball covers the cell the walk hands it.  The two existing
LM-O artifacts fail that in different ways:

* ``reused_v1`` re-groups EXP021's flat 64x64 cells into an 8-ary tree and takes
  T3 anchors as child means, so the walk and the cells disagree wherever a point
  is nearest to an anchor outside its intended parent (T2 agreement 85.3%,
  T3 81.2%, T4 70.6% on ideal surface samples);
* ``independent_v2`` is built level by level with nearest-anchor labels, but its
  balance pass rewrites labels after they were assigned, so the fitted radii no
  longer cover the points the walk actually selects.

This builder derives all four levels from one recursive nearest-anchor partition
and fits normals and radii to the points that walk assigns.  Two conventions
keep the fit honest:

* the partition runs on the area-weighted surface samples only -- the appended
  mesh vertices would make cell size follow tessellation density instead of area;
* the radius is fitted on those samples *plus* every mesh vertex, so a cell whose
  extreme point is a polyhedral vertex still has a ball that covers it.

Every radius keeps one ``*1.05`` margin, and each parent radius is widened to the
maximum of its own covering radius and ``max_j(||c_j - p|| + r_j)`` over its eight
children, so the levels nest and the artifact satisfies the ``_parent_stats``
contract at every level rather than only at level 3/4.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from mmcv import Config

import ref
from research.exp021.build_cad_hierarchy import (
    _fps,
    _nearest,
    _normalize,
    _sample_surface,
)
from research.exp022.build_hierarchy import (
    LMO_CONFIG,
    _symmetry_arrays,
)
from research.exp022.dataset_context import resolve_dataset_context
from lib.pysixd import inout


ROOT = Path(__file__).resolve().parents[2]
GENERATOR_VERSION = 3
MODE = "consistent"
SEED = 20260919
# A cell with n < 8 surface samples can fill at most n of its eight children, so any
# microscopic CAD feature starves the level-4 split and leaves unreachable leaves.
# 200k samples (the EXP021/022 density) starves one or two cells per object; 800k
# clears it and 1M leaves headroom.  The denser set also tightens the covering-radius
# estimate, which is what the held-out surface coverage depends on.
DEFAULT_BUILD_SAMPLES = 1_000_000
RADIUS_MARGIN = 1.05
RADIUS_FLOOR_M = 1e-5
LEVEL_COUNTS = (8, 64, 512, 4096)


def _fps_indices(points: np.ndarray, count: int, first_point: np.ndarray) -> np.ndarray:
    """Farthest-point indices, tolerating cells with fewer samples than children."""
    if len(points) >= count:
        return _fps(points, count, first_point)
    return np.arange(count) % len(points)


def _partition(points: np.ndarray, level: int, node: int, anchors: np.ndarray,
               degenerate: list[int]) -> None:
    """Fill ``anchors[level]`` with the 8 children of ``node`` and recurse."""
    indices = _fps_indices(points, 8, points.mean(axis=0))
    if len(np.unique(indices)) < 8:
        degenerate.append(node)
    children = points[indices]
    anchors[level][node * 8:(node + 1) * 8] = children
    if level == len(LEVEL_COUNTS):
        return
    labels = _nearest(points, children)
    for child in range(8):
        _partition(points[labels == child], level + 1, node * 8 + child, anchors, degenerate)


def _walk(points: np.ndarray, anchors: dict[int, np.ndarray]) -> np.ndarray:
    """Nested nearest-child walk; returns the [P, 4] id path, identical to the head."""
    parent = np.zeros(len(points), dtype=np.int64)
    ids = np.empty((len(points), 4), dtype=np.int64)
    for depth in range(1, 5):
        grouped = anchors[depth].reshape(-1, 8, 3)
        delta = points[:, None, :] - grouped[parent]
        parent = parent * 8 + np.einsum("pkj,pkj->pk", delta, delta).argmin(-1)
        ids[:, depth - 1] = parent
    return ids


def _object(obj_id: int, seed: int, sample_count: int, model_dir: Path,
            vertex_scale: float) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict]:
    model = inout.load_ply(str(model_dir / f"obj_{obj_id:06d}.ply"), vertex_scale=vertex_scale)
    sampled, sampled_normals = _sample_surface(
        model, sample_count, np.random.default_rng(seed + obj_id)
    )
    sampled = sampled.astype(np.float64)
    sampled_normals = sampled_normals.astype(np.float64)
    # ``_sample_surface`` returns the area-weighted draws first, then every mesh vertex.
    assert len(sampled) == sample_count + len(np.asarray(model["pts"])), "sample layout changed"
    points = sampled
    normals = sampled_normals

    anchors = {depth: np.empty((count, 3), dtype=np.float64)
               for depth, count in enumerate(LEVEL_COUNTS, start=1)}
    degenerate: list[int] = []
    _partition(points[:sample_count], 1, 0, anchors, degenerate)

    ids = _walk(points, anchors)
    radii = {}
    node_normals = {}
    for depth in range(1, 5):
        count = LEVEL_COUNTS[depth - 1]
        chosen = ids[:, depth - 1]
        distance = np.linalg.norm(points - anchors[depth][chosen], axis=1)
        covering = np.zeros(count, dtype=np.float64)
        np.maximum.at(covering, chosen, distance)
        counts = np.bincount(chosen, minlength=count)
        normal_sum = np.zeros((count, 3), dtype=np.float64)
        np.add.at(normal_sum, chosen, normals)
        mean_normal = normal_sum / np.maximum(counts, 1)[:, None]
        radii[depth] = np.maximum(covering * RADIUS_MARGIN, RADIUS_FLOOR_M)
        node_normals[depth] = _normalize(
            mean_normal, fallback=np.tile(np.array([0.0, 0.0, 1.0]), (count, 1)))

    # Widen each parent to also cover its children's balls (bottom-up, so the
    # child radii used here are already final).
    nest_ratio_before = {}
    for depth in (3, 2, 1):
        child_anchors = anchors[depth + 1].reshape(-1, 8, 3)
        required = (np.linalg.norm(child_anchors - anchors[depth][:, None, :], axis=-1)
                    + radii[depth + 1].reshape(-1, 8)).max(-1) * RADIUS_MARGIN
        nest_ratio_before[f"level{depth + 1}_over_level{depth}"] = float(
            (radii[depth] / np.maximum(required, 1e-12)).min())
        radii[depth] = np.maximum(radii[depth], required)

    empty = [int(np.count_nonzero(np.bincount(ids[:, depth - 1],
                                              minlength=LEVEL_COUNTS[depth - 1]) == 0))
             for depth in range(1, 5)]
    if any(empty) or degenerate:
        raise ValueError(
            f"obj {obj_id}: {sample_count} surface samples leave empty cells {empty} "
            f"and {len(degenerate)} cells with fewer than eight distinct anchors; "
            "increase --sample-count")

    stats = {
        "object_id": int(obj_id),
        "degenerate_cells": degenerate,
        "empty_cells_per_level": empty,
        "nest_ratio_min_before_widening": nest_ratio_before,
        "radius_m": {f"level{depth}": {
            "min": float(radii[depth].min()), "p50": float(np.median(radii[depth])),
            "max": float(radii[depth].max())} for depth in range(1, 5)},
        "anchor_to_nearest_sibling_m": {
            f"level{depth}": float(_sibling_gap(anchors[depth])) for depth in range(1, 5)},
    }
    return anchors, {"radii": radii, "normals": node_normals}, stats


def _sibling_gap(anchors: np.ndarray) -> float:
    grouped = anchors.reshape(-1, 8, 3)
    difference = np.linalg.norm(grouped[:, :, None, :] - grouped[:, None, :, :], axis=-1)
    difference[:, np.arange(8), np.arange(8)] = np.inf
    return float(difference.min())


def build(context, seed: int = SEED, sample_count: int = DEFAULT_BUILD_SAMPLES) -> tuple[dict, dict]:
    object_ids = context.object_ids
    models_info = ref.__dict__[context.cad_ref_key].get_models_info()
    counts, transforms = _symmetry_arrays(models_info, object_ids, context.cad_vertex_scale)
    output = {
        "object_ids": np.asarray(object_ids, dtype=np.int64),
        "extents": [], "diameters": np.asarray(
            [models_info[str(obj_id)]["diameter"] * context.cad_vertex_scale
             for obj_id in object_ids], dtype=np.float32),
        "symmetry_counts": counts, "symmetry_transforms": transforms,
        "source_leaf_indices": np.full((len(object_ids), LEVEL_COUNTS[-1]), -1, dtype=np.int64),
    }
    per_level = {depth: {"anchors": [], "normals": [], "radii": []} for depth in range(1, 5)}
    per_object = {}
    for index, obj_id in enumerate(object_ids):
        started = time.perf_counter()
        model_path = context.cad_model_dir / f"obj_{obj_id:06d}.ply"
        anchors, fitted, stats = _object(obj_id, seed, sample_count,
                                         context.cad_model_dir, context.cad_vertex_scale)
        model = inout.load_ply(str(model_path), vertex_scale=context.cad_vertex_scale)
        output["extents"].append(np.ptp(np.asarray(model["pts"], dtype=np.float64), axis=0))
        for depth in range(1, 5):
            per_level[depth]["anchors"].append(anchors[depth])
            per_level[depth]["normals"].append(fitted["normals"][depth])
            per_level[depth]["radii"].append(fitted["radii"][depth])
        stats["seconds"] = time.perf_counter() - started
        per_object[str(obj_id)] = stats
        print(f"  obj {obj_id}: {stats['seconds']:.1f}s "
              f"radii l4 {stats['radius_m']['level4']['min']*1000:.3f}-"
              f"{stats['radius_m']['level4']['max']*1000:.3f} mm "
              f"degenerate {len(stats['degenerate_cells'])}", flush=True)
    for depth in range(1, 5):
        for field in ("anchors", "normals", "radii"):
            output[f"level{depth}_{field}"] = np.stack(per_level[depth][field]).astype(np.float32)
    output["extents"] = np.stack(output["extents"]).astype(np.float32)
    output["mode"] = np.asarray(MODE)
    output["generator_version"] = np.asarray(GENERATOR_VERSION, dtype=np.int64)
    output["dataset_key"] = np.asarray(context.key)
    output["cad_ref_key"] = np.asarray(context.cad_ref_key)
    output["seed"] = np.asarray(seed, dtype=np.int64)
    output["sample_count"] = np.asarray(sample_count, dtype=np.int64)
    output["radius_floor_m"] = np.asarray(RADIUS_FLOOR_M, dtype=np.float32)
    return output, per_object


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=LMO_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_BUILD_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    context = resolve_dataset_context(Config.fromfile(str(args.config)), require_hierarchy=False)
    output = args.output or (ROOT / ".local/dataset_cache/exp022/consistent_v3.npz")
    if output.exists():
        raise FileExistsError(output)
    hierarchy, per_object = build(context, args.seed, args.sample_count)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **hierarchy)
    report = {"mode": MODE, "generator_version": GENERATOR_VERSION, "seed": args.seed,
              "sample_count": args.sample_count, "radius_margin": RADIUS_MARGIN,
              "radius_floor_m": RADIUS_FLOOR_M, "object_ids": [int(v) for v in context.object_ids],
              "per_object": per_object}
    output.with_suffix(".stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"EXP022 {MODE} hierarchy: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
