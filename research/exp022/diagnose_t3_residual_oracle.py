#!/usr/bin/env python3
"""EXP022 T3=512 bounded-residual oracle diagnostic (geometry capability only).

One question, asked before any redesign: under *oracle* GT surface identity, can a
T3=512 region anchor plus the current bounded residual (``||r|| <= 1``) reconstruct
the true visible surface well enough to match T4=4096 under an identical
fixed-support RANSAC-PnP?

No checkpoint is loaded and no network forward is run; every producer is built
analytically from GT geometry:

    Arm A  GT XYZ        -> pipeline/solver ceiling
    Arm B  T4 oracle     -> 4096 + bounded residual ceiling
    Arm C  T3 oracle     -> 512 + bounded residual ceiling
    Arm D  T3 anchor     -> how much the residual is actually doing

Two GT targets are reported, because they differ by several millimetres:

* ``depth``    -- the depth-reprojected GT the existing matched evaluators use.
* ``rendered`` -- exact Möller-Trumbore ray/mesh intersection at the same pixel,
                  i.e. the true visible surface point (the 2026-09-13 round used
                  the same construction).  Also on-surface and exactly consistent
                  with the pixel, so it isolates hierarchy capability from sensor
                  and calibration error in the depth maps.

All arms of one target share a single frozen 2D support (pixel coordinates, valid
mask, selected indices, K, RANSAC seed/threshold/iterations); only the 3D producer
changes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from detectron2.data import MetadataCatalog
from mmcv import Config
from scipy.spatial import cKDTree

from core.gdrn_modeling.datasets.data_loader import build_gdrn_test_loader
from core.gdrn_modeling.datasets.dataset_factory import register_datasets_in_cfg
from core.gdrn_modeling.models.heads.progressive_pcc_head import (
    LEVEL_SIZES,
    load_pcc_hierarchy,
)
from lib.pysixd import inout
from lib.utils.mask_utils import cocosegm2mask
from research.diagnostics.exp019_epro.correspondence import (
    historical_gt_reprojection_errors,
    project_points,
    roi2d_norm_to_pixels,
)
from research.diagnostics.exp019_epro.repo_adapter import (
    _dataset_lookups,
    _depth_to_object,
    _sample_nearest,
)
from research.exp021.build_cad_hierarchy import DEFAULT_SEED, _sample_surface
from research.exp020.matched_pnp_eval import (
    MIN_FIXED_SUPPORT,
    RANSAC_CONFIDENCE,
    RANSAC_ITERATIONS,
    RANSAC_REPROJ_ERR_PX,
    SEED_BASE,
    _decode_selected,
    _full_target_total,
    build_fixed_plan,
    solve_arm_fixed,
)
from research.exp022.dataset_context import resolve_dataset_context
from research.exp022.preflight import LMO_CONFIG_ROOT


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "EXP-20260916-022-progressive-pcc"
ARMS = ("a_gt", "b_t4_oracle", "c_t3_oracle", "d_t3_anchor")
ARM_LABELS = {
    "a_gt": "GT XYZ",
    "b_t4_oracle": "T4 Oracle",
    "c_t3_oracle": "T3 Oracle",
    "d_t3_anchor": "T3 Anchor Only",
}
# The two geometry producers each carry their own oracle map.
FAMILY_LEVEL = {"b_t4_oracle": "t4", "c_t3_oracle": "t3", "d_t3_anchor": None}
LEVELS = ("t4", "t3")
LEVEL_DEPTH = {"t4": 4, "t3": 3}
CORR_KEYS = ("mean_mm", "median_mm", "p90_mm", "p95_mm", "p99_mm")
REPROJ_KEYS = ("mean_px", "median_px", "p90_px", "p95_px")
# The BOP toolkit parses the method name out of "<method>_<dataset>-<split>.csv",
# so it must not contain an underscore.
BOP_METHOD = {"a_gt": "gtxyz", "b_t4_oracle": "t4oracle",
              "c_t3_oracle": "t3oracle", "d_t3_anchor": "t3anchor"}
NORM_BINS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, math.inf)
SURFACE_SAMPLES = 200_000
_MT_EPS = 1e-12
_T_EPS = 1e-9
# Pre-registered decision thresholds (diagnostic document, section 17).
COVERAGE_PASS = 0.99
COVERAGE_CONDITIONAL = 0.97
P95_PASS = 0.8
POSE_DELTA_PASS_PT = 1.0
POSE_DELTA_CONDITIONAL_PT = 2.0


# --------------------------------------------------------------------------- #
# exact ray/mesh GT correspondence
# --------------------------------------------------------------------------- #
@dataclass
class TriMesh:
    v0: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    cent: np.ndarray
    rad: np.ndarray


def load_tri_mesh(path: Path) -> TriMesh:
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    vertices = np.asarray(mesh.vertices, dtype=np.float64) / 1000.0
    faces = np.asarray(mesh.faces, dtype=np.int64)
    v0 = vertices[faces[:, 0]]
    e1 = vertices[faces[:, 1]] - v0
    e2 = vertices[faces[:, 2]] - v0
    triangles = vertices[faces]
    cent = triangles.mean(axis=1)
    return TriMesh(v0=v0, e1=e1, e2=e2, cent=cent,
                   rad=np.linalg.norm(triangles - cent[:, None, :], axis=2).max(axis=1))


def raycast(mesh: TriMesh, origin: np.ndarray, directions: np.ndarray):
    """Nearest double-sided Möller-Trumbore hit; exact cone pre-selection.

    A perspective ray that hits triangle T makes an angle of at most the angular
    radius of T's bounding sphere with T's centroid direction, so querying the
    centroid-direction k-d tree with the largest such radius is a conservative
    superset and cannot drop a hit.
    """
    n_rays = directions.shape[0]
    best_t = np.full(n_rays, np.inf)
    best_face = np.full(n_rays, -1, dtype=np.int64)
    if not n_rays:
        return best_t, best_face
    relative = mesh.cent - origin[None]
    distance = np.linalg.norm(relative, axis=1)
    ratio = np.clip(mesh.rad / np.maximum(distance, mesh.rad + 1e-12), 0.0, 1.0)
    radius = float(min((1.5 * np.arcsin(ratio) + 1e-3).max(), np.pi / 2))
    tree = cKDTree(relative / distance[:, None])
    unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    candidates = tree.query_ball_point(unit, radius)
    counts = np.array([len(item) for item in candidates], dtype=np.int64)
    if not counts.sum():
        return best_t, best_face
    ray_index = np.repeat(np.arange(n_rays, dtype=np.int64), counts)
    face_index = np.fromiter((face for group in candidates for face in group),
                             dtype=np.int64, count=int(counts.sum()))
    p = np.cross(directions[ray_index], mesh.e2[face_index])
    det = (mesh.e1[face_index] * p).sum(-1)
    ok = np.abs(det) > _MT_EPS
    inverse = 1.0 / np.where(ok, det, 1.0)
    tvec = origin[None] - mesh.v0[face_index]
    u = (tvec * p).sum(-1) * inverse
    ok &= (u >= 0.0) & (u <= 1.0)
    q = np.cross(tvec, mesh.e1[face_index])
    v = (directions[ray_index] * q).sum(-1) * inverse
    ok &= (v >= 0.0) & (u + v <= 1.0)
    t = (mesh.e2[face_index] * q).sum(-1) * inverse
    ok &= t > _T_EPS
    if not ok.any():
        return best_t, best_face
    ray_index, face_index, t = ray_index[ok], face_index[ok], t[ok]
    order = np.lexsort((t, ray_index))
    ray_index, face_index, t = ray_index[order], face_index[order], t[order]
    first = np.ones(ray_index.shape[0], dtype=bool)
    first[1:] = ray_index[1:] != ray_index[:-1]
    best_t[ray_index[first]] = t[first]
    best_face[ray_index[first]] = face_index[first]
    return best_t, best_face


def render_object_points(mesh: TriMesh, R: np.ndarray, t: np.ndarray, K: np.ndarray,
                         xy_px: np.ndarray):
    """Object-frame metric points where the posed mesh is hit by each pixel ray."""
    origin = -(np.asarray(R, dtype=np.float64).T @ np.asarray(t, dtype=np.float64).reshape(3))
    camera = np.stack(((xy_px[:, 0] - K[0, 2]) / K[0, 0],
                       (xy_px[:, 1] - K[1, 2]) / K[1, 1],
                       np.ones(xy_px.shape[0])), axis=1)
    directions = camera @ np.asarray(R, dtype=np.float64)
    hits, _ = raycast(mesh, origin, directions)
    hit = np.isfinite(hits)
    points = origin[None] + hits[:, None] * directions
    points[~hit] = np.nan
    return points, hit


# --------------------------------------------------------------------------- #
# hierarchy
# --------------------------------------------------------------------------- #
def load_hierarchy_artifact(path: Path, expected_object_ids, dataset_key):
    """Load an EXP022-shaped artifact without the head's ``(mode, version)`` whitelist.

    ``load_pcc_hierarchy`` additionally pins the artifact generation so the training
    path cannot silently pick up a tree no config points at.  This diagnostic is
    model-free and is used to compare two generations of the same 8^4 contract, so it
    re-checks every shape, finiteness and positivity rule by hand and returns the
    provenance instead of rejecting it.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"EXP022 hierarchy missing: {path}")
    with np.load(path, allow_pickle=False) as data:
        expected = {"object_ids", "extents", "diameters", "symmetry_counts",
                    "symmetry_transforms", "generator_version", "mode", "source_leaf_indices"}
        for depth in range(1, 5):
            expected.update({f"level{depth}_{field}" for field in ("anchors", "normals", "radii")})
        missing = expected.difference(data.files)
        if missing:
            raise ValueError(f"EXP022 hierarchy missing arrays: {sorted(missing)}")
        provenance = {"mode": str(data["mode"]),
                      "generator_version": int(data["generator_version"])}
        arrays = {name: torch.from_numpy(np.asarray(data[name]).copy()) for name in expected
                  if name not in {"generator_version", "mode"}}
        stored_dataset = str(data["dataset_key"]) if "dataset_key" in data else None
    object_ids = tuple(int(value) for value in arrays["object_ids"].tolist())
    if object_ids != tuple(expected_object_ids):
        raise ValueError(f"EXP022 object order mismatch: {object_ids}")
    if dataset_key is not None and stored_dataset != dataset_key:
        if not (dataset_key == "lmo" and stored_dataset is None):
            raise ValueError("EXP022 hierarchy dataset mismatch")
    num_objects = len(object_ids)
    if (tuple(arrays["extents"].shape) != (num_objects, 3)
            or tuple(arrays["diameters"].shape) != (num_objects,)):
        raise ValueError("EXP022 extent/diameter shape mismatch")
    if tuple(arrays["source_leaf_indices"].shape) != (num_objects, 4096):
        raise ValueError("EXP022 source_leaf_indices shape mismatch")
    counts = arrays["symmetry_counts"]
    transforms = arrays["symmetry_transforms"]
    if tuple(counts.shape) != (num_objects,) or counts.dtype not in (torch.int32, torch.int64):
        raise ValueError("EXP022 symmetry_counts must contain N integer counts")
    if transforms.ndim != 4 or transforms.shape[0] != num_objects or tuple(transforms.shape[2:]) != (4, 4):
        raise ValueError("EXP022 symmetry_transforms must have shape [N,S,4,4]")
    max_sym = int(counts.max().item())
    if max_sym > 2:
        raise RuntimeError(f"EXP022 V1 symmetry branch supports at most 2 equivalents, got {max_sym}")
    if torch.any(counts < 1) or max_sym > transforms.shape[1]:
        raise ValueError("EXP022 symmetry_counts are outside stored transform bounds")
    for depth, count in enumerate(LEVEL_SIZES, start=1):
        for field, shape in (("anchors", (num_objects, count, 3)),
                             ("normals", (num_objects, count, 3)),
                             ("radii", (num_objects, count))):
            value = arrays[f"level{depth}_{field}"]
            if tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid EXP022 level{depth}_{field}: {tuple(value.shape)}")
            if field == "radii" and torch.any(value <= 0):
                raise ValueError(f"Non-positive EXP022 level{depth} radius")
    return arrays, provenance


def _levels_from_hierarchy(hierarchy: dict) -> dict[int, dict[str, np.ndarray]]:
    """Expose the model's own loader contract as float64 numpy level arrays."""
    return {
        depth: {
            "anchors": hierarchy[f"level{depth}_anchors"].numpy().astype(np.float64),
            "radii": hierarchy[f"level{depth}_radii"].numpy().astype(np.float64),
            "normals": hierarchy[f"level{depth}_normals"].numpy().astype(np.float64),
        }
        for depth in (1, 2, 3, 4)
    }


def assign_paths(points: np.ndarray, object_index: int, levels: dict) -> np.ndarray:
    """8-ary parent-child traversal identical to the head's ``_target_paths``.

    Returns ``[P,4]`` int64 ids for T1..T4.  ``points`` are object-space metric XYZ
    that have *not* been transformed, matching the head's canonical branch.
    """
    parent = np.zeros(len(points), dtype=np.int64)
    ids = np.empty((len(points), 4), dtype=np.int64)
    for depth in range(1, 5):
        grouped = levels[depth]["anchors"][object_index].reshape(-1, 8, 3)
        delta = points[:, None, :] - grouped[parent]
        child = np.einsum("pkj,pkj->pk", delta, delta).argmin(-1)
        parent = parent * 8 + child
        ids[:, depth - 1] = parent
    return ids


def residual_norms(points: np.ndarray, ids: np.ndarray, object_index: int, levels: dict,
                   depth: int) -> np.ndarray:
    """``||(x - anchor) / radius||`` under the stored level anchors/radii."""
    anchors = levels[depth]["anchors"][object_index][ids[:, depth - 1]]
    radii = levels[depth]["radii"][object_index][ids[:, depth - 1]]
    return np.linalg.norm((points - anchors) / radii[:, None], axis=-1)


def oracle_xyz(points: np.ndarray, ids: np.ndarray, object_index: int, levels: dict,
               depth: int) -> np.ndarray:
    """Bounded-residual optimum: project onto the closed unit ball, then decode."""
    anchors = levels[depth]["anchors"][object_index][ids[:, depth - 1]]
    radii = levels[depth]["radii"][object_index][ids[:, depth - 1]]
    raw = (points - anchors) / radii[:, None]
    norm = np.linalg.norm(raw, axis=-1, keepdims=True)
    return anchors + radii[:, None] * (raw / np.maximum(norm, 1.0))


def anchor_xyz(ids: np.ndarray, object_index: int, levels: dict, depth: int) -> np.ndarray:
    return levels[depth]["anchors"][object_index][ids[:, depth - 1]]


def producer_map(producer: str, points: np.ndarray, ids: np.ndarray, object_index: int,
                 levels: dict, template: np.ndarray, valid: np.ndarray,
                 extent: np.ndarray) -> np.ndarray:
    """Metric producer -> normalized ``[3,H,W]`` map consumed by the shared solver."""
    if producer == "a_gt":
        metric = points
    elif FAMILY_LEVEL[producer] is None:
        metric = anchor_xyz(ids, object_index, levels, 3)
    else:
        metric = oracle_xyz(points, ids, object_index, levels, LEVEL_DEPTH[FAMILY_LEVEL[producer]])
    out = np.zeros_like(template)
    out[valid] = metric
    return np.ascontiguousarray(((out / extent.reshape(1, 1, 3)) + 0.5).transpose(2, 0, 1))


def surface_representation(levels: dict, object_ids: tuple[int, ...],
                           surface_points: dict[int, np.ndarray]) -> dict:
    """Static coverage of the *ideal* CAD surface by the T3/T4 residual balls.

    The depth-derived GT sits several millimetres off the mesh, which is larger than
    most leaf radii, so sampling the mesh itself separates that data deficit from the
    hierarchy's own capability:

    * ``global``    -- an ideal surface point is inside *some* ball of this level
    * ``traversal`` -- the ball of the leaf reached by the 8-ary nearest-child walk
                       (what the head supervises and infers) contains it
    """
    report = {"samples_per_object": int(len(next(iter(surface_points.values())))),
              "sample_seed": int(DEFAULT_SEED), "per_object": {}}
    pooled = {level: [] for level in LEVELS}
    agreement = {depth: [] for depth in (2, 3, 4)}
    for index, obj_id in enumerate(object_ids):
        points = surface_points[obj_id]
        ids = assign_paths(points, index, levels)
        entry = {"num_points": int(len(points))}
        for depth in (2, 3, 4):
            nearest = cKDTree(levels[depth]["anchors"][index]).query(points, k=1)[1]
            agreement[depth].append(float((nearest == ids[:, depth - 1]).mean()))
        for level, depth in LEVEL_DEPTH.items():
            anchors = levels[depth]["anchors"][index][ids[:, depth - 1]]
            radii = levels[depth]["radii"][index][ids[:, depth - 1]]
            traversal = np.linalg.norm(points - anchors, axis=1) / radii
            nearest = cKDTree(levels[depth]["anchors"][index]).query(points, k=1)
            entry[f"global_coverage_{level}"] = float(
                (nearest[0] / levels[depth]["radii"][index][nearest[1]] <= 1.0).mean())
            entry[f"traversal_coverage_{level}"] = float((traversal <= 1.0).mean())
            entry[f"traversal_p95_{level}"] = float(np.percentile(traversal, 95))
            pooled[level].append(traversal)
        report["per_object"][str(obj_id)] = entry
    for depth in (2, 3, 4):
        report[f"traversal_matches_global_level{depth}"] = float(np.mean(agreement[depth]))
    for level in LEVELS:
        norms = np.concatenate(pooled[level])
        report[f"pooled_traversal_coverage_{level}"] = float((norms <= 1.0).mean())
        report[f"pooled_traversal_p95_{level}"] = float(np.percentile(norms, 95))
        report[f"pooled_global_coverage_{level}"] = float(
            np.mean([item[f"global_coverage_{level}"] for item in report["per_object"].values()]))
    return report


def hierarchy_sanity(levels: dict, object_ids: tuple[int, ...]) -> dict:
    """Static artifact contract plus parent coverage of the child residual balls."""
    num_objects = len(object_ids)
    report = {"object_ids": [int(v) for v in object_ids], "levels": {}, "parent_coverage": {}}
    for depth, count in enumerate(LEVEL_SIZES, start=1):
        anchors, radii = levels[depth]["anchors"], levels[depth]["radii"]
        report["levels"][f"level{depth}"] = {
            "anchors_shape": list(anchors.shape),
            "radii_shape": list(radii.shape),
            "expected_anchors_shape": [num_objects, count, 3],
            "expected_radii_shape": [num_objects, count],
            "shape_ok": list(anchors.shape) == [num_objects, count, 3]
            and list(radii.shape) == [num_objects, count],
            "all_finite": bool(np.isfinite(anchors).all() and np.isfinite(radii).all()),
            "radius_min_m": float(radii.min()),
            "radius_max_m": float(radii.max()),
            "all_radii_positive": bool((radii > 0).all()),
        }
    for child_depth in (2, 3, 4):
        parent_count = LEVEL_SIZES[child_depth - 2]
        child_anchors = levels[child_depth]["anchors"].reshape(num_objects, parent_count, 8, 3)
        child_radii = levels[child_depth]["radii"].reshape(num_objects, parent_count, 8)
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
    coverage = report["parent_coverage"]["level3_over_level4"]
    shapes_ok = all(item["shape_ok"] and item["all_finite"] and item["all_radii_positive"]
                    for item in report["levels"].values())
    report["result"] = "PASS" if shapes_ok and coverage["below_one"] == 0 else "FAIL"
    return report


# --------------------------------------------------------------------------- #
# statistics helpers
# --------------------------------------------------------------------------- #
def _finite(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def _mean(values) -> float | None:
    array = _finite([value for value in values if value is not None])
    return float(array.mean()) if array.size else None


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


def error_stats(values_mm) -> dict | None:
    array = _finite(values_mm)
    if not array.size:
        return None
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def _geodesic_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    cosine = (np.trace(rotation_a @ rotation_b.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _pose_symmetry_errors(pose: dict, R_gt: np.ndarray, t_gt: np.ndarray,
                          transforms: np.ndarray) -> dict:
    if not pose["success"]:
        return {"rot_err_deg": None, "trans_err_mm": None}
    R_pred = np.asarray(pose["R"], dtype=np.float64)
    t_pred = np.asarray(pose["t"], dtype=np.float64)
    best_rotation, best_translation = None, None
    for transform in transforms:
        rotation = np.asarray(transform[:3, :3], dtype=np.float64)
        translation = np.asarray(transform[:3, 3], dtype=np.float64)
        rotation_error = _geodesic_deg(R_pred, R_gt @ rotation)
        translation_error = float(np.linalg.norm(t_pred - (R_gt @ translation + t_gt)) * 1000.0)
        if best_rotation is None or rotation_error < best_rotation:
            best_rotation, best_translation = rotation_error, translation_error
    return {"rot_err_deg": best_rotation, "trans_err_mm": best_translation}


def _add_metrics(pose: dict, R_gt: np.ndarray, t_gt: np.ndarray, vertices: np.ndarray,
                 diameter_m: float) -> dict:
    """Empirical ADD / ADD-S in mm on the CAD vertices, with the 0.1-diameter recall."""
    if not pose["success"]:
        return {"add_mm": None, "adds_mm": None, "add_0.1d": None, "adds_0.1d": None}
    gt_points = vertices @ R_gt.T + t_gt
    pred_points = (vertices @ np.asarray(pose["R"], dtype=np.float64).T
                   + np.asarray(pose["t"], dtype=np.float64))
    add = np.linalg.norm(gt_points - pred_points, axis=1) * 1000.0
    adds = cKDTree(pred_points).query(gt_points, k=1)[0] * 1000.0
    threshold = 0.1 * diameter_m * 1000.0
    return {
        "add_mm": float(add.mean()),
        "adds_mm": float(adds.mean()),
        "add_0.1d": bool(add.mean() <= threshold),
        "adds_0.1d": bool(adds.mean() <= threshold),
    }


def target_counts_for(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    counts: dict[tuple[str, int], int] = {}
    for item in json.loads(path.read_text(encoding="utf-8")):
        key = (f"{int(item['scene_id'])}/{int(item['im_id'])}", int(item["obj_id"]))
        counts[key] = counts.get(key, 0) + int(item.get("inst_count", 1))
    return counts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state() -> tuple[str, str, bool]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    return commit, branch, dirty


def _producer_stats(plan, xyz_norm: np.ndarray) -> dict:
    """Metric and GT-pose reprojection behaviour of one producer on the support."""
    x3d = _decode_selected(plan, xyz_norm)
    if not x3d.size or not np.isfinite(x3d).all():
        return {"corr_err_mm": None, "reproj_err_px": None, "corr": {}, "reproj": {}}
    error = np.linalg.norm(x3d - plan.xyz_gt_m, axis=1) * 1000.0
    reprojection = np.linalg.norm(
        project_points(x3d, plan.R_gt, plan.t_gt, plan.K) - plan.xy2d_px, axis=1)
    return {
        "corr_err_mm": float(error.mean()),
        "reproj_err_px": float(reprojection.mean()),
        "corr": {"mean_mm": float(error.mean()), "median_mm": float(np.median(error)),
                 "p90_mm": float(np.percentile(error, 90)),
                 "p95_mm": float(np.percentile(error, 95)),
                 "p99_mm": float(np.percentile(error, 99))},
        "reproj": {"mean_px": float(reprojection.mean()),
                   "median_px": float(np.median(reprojection)),
                   "p90_px": float(np.percentile(reprojection, 90)),
                   "p95_px": float(np.percentile(reprojection, 95))},
    }


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def _arm_summary(rows: list[dict], family: str, arm: str) -> dict:
    entries = [row[family][arm] for row in rows]
    solved = [item for item in entries if item["pose"]["success"]]
    result = {
        "solve_successes": len(solved),
        "solve_failures": len(entries) - len(solved),
        "solve_success_rate": len(solved) / len(entries) if entries else None,
    }
    for name in ("corr_err_mm", "reproj_err_px", "rot_err_deg", "trans_err_mm",
                 "add_mm", "adds_mm"):
        values = _finite([item.get(name) for item in entries if item.get(name) is not None])
        result[f"mean_{name}"] = float(values.mean()) if values.size else None
        if name in ("rot_err_deg", "trans_err_mm", "adds_mm"):
            result[f"median_{name}"] = float(np.median(values)) if values.size else None
            result[f"p90_{name}"] = float(np.percentile(values, 90)) if values.size else None
    for name in ("add_0.1d", "adds_0.1d"):
        values = [bool(item[name]) for item in entries if item.get(name) is not None]
        result[f"recall_{name}"] = float(np.mean(values)) if values else None
    for name, keys in (("corr", CORR_KEYS), ("reproj", REPROJ_KEYS)):
        for key in keys:
            result[f"{name}_{key}"] = _mean(
                [item[name][key] for item in entries if item.get(name) and key in item[name]])
    return result


def summarize(rows: list[dict]) -> dict:
    summary = {"num_targets": len(rows), "families": {}, "residual": {}, "residual_render": {},
               "xyz_error": {}, "xyz_error_render": {}, "surface_distance_mm": None,
               "support": {
                   "mean_num_valid_gt_points": _mean([row["num_valid_gt_points"] for row in rows]),
                   "mean_num_ray_hits": _mean([row["num_ray_hits"] for row in rows]),
                   "mean_num_selected_depth": _mean([row["num_selected"] for row in rows]),
                   "mean_num_selected_render": _mean([row["num_selected_surface"] for row in rows]),
               }}
    for family in ("depth", "render"):
        summary["families"][family] = {
            arm: {"label": ARM_LABELS[arm], **_arm_summary(rows, family, arm)} for arm in ARMS}
    for group, key in (("residual", "residual"), ("residual_render", "residual_render"),
                       ("xyz_error", "xyz_error"), ("xyz_error_render", "xyz_error_render")):
        for level in LEVELS:
            values = (np.concatenate([row[key][level] for row in rows]) if rows else [])
            summary[group][level] = (residual_stats(values) if "residual" in group
                                     else error_stats(values))
    if rows:
        summary["xyz_error"]["t3_minus_t4"] = error_stats(
            np.concatenate([row["xyz_error"]["t3"] - row["xyz_error"]["t4"] for row in rows]))
        summary["xyz_error_render"]["t3_minus_t4"] = error_stats(
            np.concatenate([row["xyz_error_render"]["t3"] - row["xyz_error_render"]["t4"]
                            for row in rows]))
        summary["surface_distance_mm"] = error_stats(
            np.concatenate([row["surface_distance_mm"] for row in rows]))
    return summary


def _decision_from(residual: dict | None, family: dict, bop_family: dict | None,
                   label: str) -> dict:
    coverage = 1.0 - residual["fraction_gt_1.00"] if residual else None
    p95 = residual["p95"] if residual else None
    pose_delta = {}
    for key in ("adds_0.1d", "add_0.1d"):
        left = family["c_t3_oracle"].get(f"recall_{key}")
        right = family["b_t4_oracle"].get(f"recall_{key}")
        if left is not None and right is not None:
            pose_delta[f"sampled_{key}"] = left - right
    if bop_family:
        for key in ("bop", "add"):
            left = bop_family.get("c_t3_oracle", {}).get(key)
            right = bop_family.get("b_t4_oracle", {}).get(key)
            if left is not None and right is not None:
                pose_delta[key] = left - right
    worst = max((abs(value) for value in pose_delta.values()), default=None)
    if coverage is None or p95 is None:
        verdict = "UNDETERMINED"
    elif (coverage >= COVERAGE_PASS and p95 <= P95_PASS
          and worst is not None and worst <= POSE_DELTA_PASS_PT / 100.0):
        verdict = "PASS"
    elif (coverage >= COVERAGE_CONDITIONAL and worst is not None
          and worst <= POSE_DELTA_CONDITIONAL_PT / 100.0):
        verdict = "CONDITIONAL PASS"
    else:
        verdict = "FAIL"
    return {
        "metric_source": label,
        "label": verdict,
        "t3_coverage_le_one": coverage,
        "t3_p95_raw_norm": p95,
        "t3_minus_t4_pt": {key: value * 100.0 for key, value in pose_delta.items()},
        "max_abs_pose_delta_pt": worst * 100.0 if worst is not None else None,
        "thresholds": {"coverage_pass": COVERAGE_PASS,
                       "coverage_conditional": COVERAGE_CONDITIONAL,
                       "p95_pass": P95_PASS,
                       "pose_delta_pass_pt": POSE_DELTA_PASS_PT,
                       "pose_delta_conditional_pt": POSE_DELTA_CONDITIONAL_PT},
    }


def decision(summary: dict, bop: dict | None) -> dict:
    """Apply the diagnostic document's pre-registered thresholds to both GT targets."""
    return {
        "depth_gt_protocol": _decision_from(
            summary["residual"].get("t3"), summary["families"]["depth"],
            bop.get("depth") if bop else None,
            "depth-derived GT (documented protocol)"),
        "rendered_surface_gt": _decision_from(
            summary["residual_render"].get("t3"), summary["families"]["render"],
            bop.get("render") if bop else None,
            "rendered surface GT (ray/mesh intersection)"),
        "bop_evaluated": bool(bop),
    }


def per_object_summary(rows: list[dict]) -> dict:
    result = {}
    for obj_id in sorted({row["obj_id"] for row in rows}):
        subset = [row for row in rows if row["obj_id"] == obj_id]
        result[str(obj_id)] = {
            "num_targets": len(subset),
            "surface_distance_mm": error_stats(np.concatenate(
                [row["surface_distance_mm"] for row in subset])),
            "residual": {level: residual_stats(np.concatenate(
                [row["residual"][level] for row in subset])) for level in LEVELS},
            "residual_render": {level: residual_stats(np.concatenate(
                [row["residual_render"][level] for row in subset])) for level in LEVELS},
            "xyz_error": {level: error_stats(np.concatenate(
                [row["xyz_error"][level] for row in subset])) for level in LEVELS},
            "xyz_error_render": {level: error_stats(np.concatenate(
                [row["xyz_error_render"][level] for row in subset])) for level in LEVELS},
            "families": {family: {arm: _arm_summary(subset, family, arm) for arm in ARMS}
                         for family in ("depth", "render")},
        }
    return result


def histogram_csv(rows: list[dict], path: Path) -> None:
    edges = list(NORM_BINS)
    keys = ("depth_t4", "depth_t3", "render_t4", "render_t3")
    counts = {key: np.zeros(len(edges) - 1, dtype=np.int64) for key in keys}
    for row in rows:
        for level in LEVELS:
            for family, key in (("residual", "depth"), ("residual_render", "render")):
                values = _finite(row[family][level])
                if values.size:
                    counts[f"{key}_{level}"] += np.histogram(values, bins=edges)[0]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("bin_low", "bin_high", *[f"{key}_count" for key in keys]))
        for index in range(len(edges) - 1):
            writer.writerow((edges[index], edges[index + 1],
                             *[int(counts[key][index]) for key in keys]))


# --------------------------------------------------------------------------- #
# main evaluation
# --------------------------------------------------------------------------- #
def run(config: Path, output: Path, limit: int | None, max_correspondences: int,
        device: str, bop_eval: bool, hierarchy_path: Path | None = None) -> dict:
    cfg = Config.fromfile(str(config))
    cfg.MODEL.DEVICE = device
    cfg.DATASETS.DET_FILES_TEST = ()
    cfg.MODEL.LOAD_DETS_TEST = False
    cfg.TEST.TEST_BBOX_TYPE = "gt"
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.DATALOADER.PERSISTENT_WORKERS = False
    context = resolve_dataset_context(cfg, require_hierarchy=hierarchy_path is None)
    if context.test_dataset is None:
        raise ValueError("EXP022 T3 residual oracle needs a test split")
    if context.key != "lmo" or context.cad_ref_key != "lm_full":
        raise ValueError("This diagnostic is defined for LM-O CAD hierarchies")

    path = Path(hierarchy_path) if hierarchy_path is not None else context.hierarchy_path
    if hierarchy_path is None:
        # The configured tree must also satisfy the head's own loader contract.
        load_pcc_hierarchy(path, expected_object_ids=context.object_ids, dataset_key=context.key)
    hierarchy, provenance = load_hierarchy_artifact(path, context.object_ids, context.key)
    levels = _levels_from_hierarchy(hierarchy)
    sanity = hierarchy_sanity(levels, context.object_ids)
    output.mkdir(parents=True, exist_ok=False)
    (output / "hierarchy_sanity.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")
    print(json.dumps({"hierarchy_sanity": sanity["result"],
                      "t3_parent_coverage": sanity["parent_coverage"]["level3_over_level4"]},
                     indent=2), flush=True)

    commit, branch, dirty = _git_state()
    metadata = {
        "status": "RUNNING",
        "experiment_id": EXPERIMENT_ID,
        "diagnostic": "exp022_t3_residual_oracle",
        "network_forward": False,
        "training": False,
        "checkpoint_loaded": None,
        "config": str(config),
        "hierarchy": str(path),
        "hierarchy_sha256": _sha256(path),
        "hierarchy_mode": provenance["mode"],
        "hierarchy_generator_version": provenance["generator_version"],
        "dataset": context.key,
        "test_dataset": context.test_dataset,
        "object_ids": [int(v) for v in context.object_ids],
        "bop_targets": str(context.bop_targets),
        "cad_model_dir": str(context.cad_model_dir),
        "source_commit": commit,
        "source_branch": branch,
        "source_tree_dirty": dirty,
        "command": " ".join([sys.executable, "-m",
                             "research.exp022.diagnose_t3_residual_oracle", *sys.argv[1:]]),
        "device": device,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": SEED_BASE,
        "ransac": {"seed_base": SEED_BASE, "reprojection_error_px": RANSAC_REPROJ_ERR_PX,
                   "confidence": RANSAC_CONFIDENCE, "iterations": RANSAC_ITERATIONS,
                   "min_fixed_support": MIN_FIXED_SUPPORT,
                   "max_correspondences": int(max_correspondences)},
        "gt_targets": {
            "depth": "depth-reprojected GT (repo_adapter._depth_to_object)",
            "render": "Moeller-Trumbore ray/mesh intersection at the same pixel",
        },
        "fixed_support_source": "gt_visible INTERSECT valid_depth (model-free diagnostic)",
        "surface_samples_per_object": SURFACE_SAMPLES,
        "limit": limit,
        "num_targets": 0,
    }
    metadata_path = output / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    register_datasets_in_cfg(cfg)
    metadata_catalog = MetadataCatalog.get(context.test_dataset)
    object_ids = [int(v) for v in context.object_ids]
    images, annotations = _dataset_lookups(context.test_dataset)
    target_counts = target_counts_for(context.bop_targets)
    if set(object_ids) != {obj_id for _, obj_id in target_counts}:
        raise ValueError("EXP022 target object IDs do not match dataset context")
    vertices, diameters, meshes, surface_points = {}, {}, {}, {}
    for index, obj_id in enumerate(object_ids):
        model_path = context.cad_model_dir / f"obj_{obj_id:06d}.ply"
        model = inout.load_ply(str(model_path), vertex_scale=context.cad_vertex_scale)
        vertices[obj_id] = np.asarray(model["pts"], dtype=np.float64)
        diameters[obj_id] = float(hierarchy["diameters"][index].item())
        meshes[obj_id] = load_tri_mesh(model_path)
        sampled, _ = _sample_surface(model, SURFACE_SAMPLES,
                                     np.random.default_rng(DEFAULT_SEED + obj_id))
        surface_points[obj_id] = sampled.astype(np.float64)
    sanity["surface_representation"] = surface_representation(levels, context.object_ids,
                                                              surface_points)
    (output / "hierarchy_sanity.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in sanity["surface_representation"].items()
                      if key != "per_object"}, indent=2), flush=True)
    loader = build_gdrn_test_loader(cfg, context.test_dataset,
                                    train_objs=metadata_catalog.objs, batch_size=1)

    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    np.random.seed(SEED_BASE)
    depth_cache: dict[str, np.ndarray] = {}
    occurrences: dict[tuple, int] = {}
    selected_per_object = {obj: 0 for obj in object_ids}
    quota = math.ceil(limit / len(object_ids)) if limit is not None and limit <= 32 else None
    rows, max_gt_reproj, max_render_reproj = [], 0.0, 0.0
    id3_equals_id4_div8 = id_checked = 0
    started = time.perf_counter()
    try:
        for inputs in loader:
            if not isinstance(inputs, list):
                inputs = [inputs]
            flat_index = -1
            for item in inputs:
                for local_index in range(len(item["roi_img"])):
                    flat_index += 1
                    class_index = int(item["roi_cls"][local_index])
                    obj_id = object_ids[class_index]
                    scene_im_id = item["scene_im_id"][local_index]
                    target_key = (scene_im_id, obj_id)
                    occurrence = occurrences.get(target_key, 0)
                    if occurrence >= target_counts.get(target_key, 0):
                        continue
                    occurrences[target_key] = occurrence + 1
                    if quota is not None and selected_per_object[obj_id] >= quota:
                        continue
                    instance_id = int(item["inst_id"][local_index])
                    gt = annotations[(scene_im_id, instance_id)]
                    image = images[scene_im_id]
                    scene_id, im_id = map(int, scene_im_id.split("/"))
                    K = item["cam"][local_index].cpu().numpy().astype(np.float64)
                    height = int(item["im_H"][local_index])
                    width = int(item["im_W"][local_index])
                    extent = item["roi_extent"][local_index].cpu().numpy().astype(np.float64)
                    roi2d_px = np.moveaxis(roi2d_norm_to_pixels(
                        item["roi_coord_2d"][local_index].cpu().numpy(),
                        np.array([height, width])), 0, -1)
                    if scene_im_id not in depth_cache:
                        raw = cv2.imread(str((ROOT / image["depth_file"]).resolve()),
                                         cv2.IMREAD_UNCHANGED)
                        if raw is None:
                            raise FileNotFoundError(image["depth_file"])
                        depth_cache[scene_im_id] = raw.astype(np.float64) / float(
                            image["depth_factor"])
                    visible_full = cocosegm2mask(gt["segmentation"], height, width).astype(bool)
                    visible_sampled, in_image = _sample_nearest(visible_full, roi2d_px)
                    R_gt = np.asarray(gt["pose"][:, :3], dtype=np.float64)
                    t_gt = np.asarray(gt["pose"][:, 3], dtype=np.float64)
                    gt_xyz, depth_valid = _depth_to_object(
                        depth_cache[scene_im_id], roi2d_px, K, R_gt, t_gt)
                    gt_visible = visible_sampled & in_image & depth_valid

                    sanity_px = historical_gt_reprojection_errors(
                        gt_xyz, roi2d_px, gt_visible, R_gt, t_gt, K)
                    if len(sanity_px):
                        max_gt_reproj = max(max_gt_reproj, float(sanity_px.max()))
                        if sanity_px.max() >= 0.5:
                            raise RuntimeError(
                                f"GT XYZ reprojection {sanity_px.max():.4f}px >= 0.5px")

                    # Exact on-surface GT at the same pixels (ray/mesh intersection).
                    rendered, ray_hit = render_object_points(
                        meshes[obj_id], R_gt, t_gt, K, roi2d_px[gt_visible])
                    rendered_px = np.linalg.norm(
                        project_points(rendered[ray_hit], R_gt, t_gt, K)
                        - roi2d_px[gt_visible][ray_hit], axis=1)
                    if rendered_px.size:
                        max_render_reproj = max(max_render_reproj, float(rendered_px.max()))
                    surface_visible = np.zeros_like(gt_visible)
                    surface_visible[gt_visible] = ray_hit
                    surface_xyz = np.zeros_like(gt_xyz)
                    surface_xyz[surface_visible] = rendered[ray_hit]

                    points = gt_xyz[gt_visible]
                    ids = assign_paths(points, class_index, levels)
                    id_checked += points.shape[0]
                    id3_equals_id4_div8 += int(np.count_nonzero(ids[:, 2] == ids[:, 3] // 8))
                    surface_points_here = surface_xyz[surface_visible]
                    surface_distance = np.linalg.norm(
                        surface_points_here - points[ray_hit], axis=1) * 1000.0

                    residuals, xyz_error = {}, {}
                    for level, depth in LEVEL_DEPTH.items():
                        residuals[level] = residual_norms(points, ids, class_index, levels, depth)
                        xyz_error[level] = np.linalg.norm(
                            oracle_xyz(points, ids, class_index, levels, depth) - points,
                            axis=1) * 1000.0

                    # Noise-controlled twin: identical pipeline on the exact surface GT.
                    surface_ids = assign_paths(surface_points_here, class_index, levels)
                    render_residual, render_error = {}, {}
                    for level, depth in LEVEL_DEPTH.items():
                        render_residual[level] = residual_norms(
                            surface_points_here, surface_ids, class_index, levels, depth)
                        render_error[level] = np.linalg.norm(
                            oracle_xyz(surface_points_here, surface_ids, class_index, levels,
                                       depth) - surface_points_here, axis=1) * 1000.0

                    plan_kwargs = dict(scene_id=scene_id, im_id=im_id, obj_id=obj_id,
                                       instance_id=instance_id, K=K, extent_m=extent,
                                       R_gt=R_gt, t_gt=t_gt,
                                       max_correspondences=max_correspondences)
                    plan = build_fixed_plan(
                        target_id=f"{scene_id:06d}/{im_id:06d}/{obj_id:06d}/{instance_id:03d}",
                        support_mask=gt_visible, roi2d_px_map=roi2d_px, xyz_gt_m=gt_xyz,
                        **plan_kwargs)
                    surface_plan = build_fixed_plan(
                        target_id=f"{scene_id:06d}/{im_id:06d}/{obj_id:06d}/{instance_id:03d}",
                        support_mask=surface_visible, roi2d_px_map=roi2d_px,
                        xyz_gt_m=surface_xyz, **plan_kwargs)

                    count = int(hierarchy["symmetry_counts"][class_index].item())
                    transforms = hierarchy["symmetry_transforms"][class_index].numpy()[:count]
                    seed = SEED_BASE + len(rows)
                    families = {}
                    for family, target_plan, source, source_ids in (
                            ("depth", plan, points, ids),
                            ("render", surface_plan, surface_points_here, surface_ids)):
                        arms = {}
                        for arm in ARMS:
                            xyz_norm = producer_map(arm, source, source_ids, class_index, levels,
                                                    gt_xyz, (gt_visible if family == "depth"
                                                             else surface_visible), extent)
                            solved = solve_arm_fixed(target_plan, xyz_norm, seed)
                            stats = _producer_stats(target_plan, xyz_norm)
                            stats["pose"] = solved["pose"]
                            stats.update(_pose_symmetry_errors(solved["pose"], R_gt, t_gt,
                                                              transforms))
                            stats.update(_add_metrics(solved["pose"], R_gt, t_gt,
                                                      vertices[obj_id], diameters[obj_id]))
                            arms[arm] = stats
                        families[family] = arms

                    rows.append({
                        "target_id": plan.target_id, "scene_id": scene_id, "im_id": im_id,
                        "obj_id": obj_id, "instance_id": instance_id,
                        "num_valid_gt_points": int(np.count_nonzero(gt_visible)),
                        "num_ray_hits": int(np.count_nonzero(surface_visible)),
                        "num_fixed_support": int(plan.num_fixed_support),
                        "num_selected": int(plan.num_selected),
                        "num_selected_surface": int(surface_plan.num_selected),
                        "surface_distance_mm": surface_distance,
                        "residual": residuals, "residual_render": render_residual,
                        "xyz_error": xyz_error, "xyz_error_render": render_error,
                        **families,
                    })
                    selected_per_object[obj_id] += 1
                    if len(rows) % 100 == 0:
                        print(f"EXP022 T3 oracle targets: {len(rows)}", flush=True)
                    if limit is not None and len(rows) >= limit:
                        break
                if limit is not None and len(rows) >= limit:
                    break
            if limit is not None and len(rows) >= limit:
                break
    except Exception as exc:
        metadata.update(status="FAILED", error=f"{type(exc).__name__}: {exc}",
                        num_targets=len(rows))
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        raise

    expected = limit if limit is not None else _full_target_total(target_counts)
    if len(rows) != expected:
        raise RuntimeError(f"Processed {len(rows)} targets; expected {expected}")
    if id_checked and id3_equals_id4_div8 != id_checked:
        raise RuntimeError(f"id3 == id4 // 8 held for {id3_equals_id4_div8}/{id_checked} points")

    summary = summarize(rows)
    summary["id3_equals_id4_div8_fraction"] = id3_equals_id4_div8 / id_checked if id_checked else None
    bop = None
    if bop_eval:
        bop = {"depth": bop_evaluate(rows, output / "bop_depth", "depth", context.bop_dataset,
                                     context.bop_targets.name),
               "render": bop_evaluate(rows, output / "bop_render", "render",
                                      context.bop_dataset, context.bop_targets.name)}
        summary["bop"] = bop
    summary["decision"] = decision(summary, bop)

    with (output / "per_target.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            serializable = dict(row)
            for key in ("residual", "residual_render", "xyz_error", "xyz_error_render"):
                serializable[key] = {name: value.tolist() for name, value in row[key].items()}
            serializable["surface_distance_mm"] = row["surface_distance_mm"].tolist()
            stream.write(json.dumps(serializable) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "per_object.json").write_text(
        json.dumps(per_object_summary(rows), indent=2), encoding="utf-8")
    histogram_csv(rows, output / "residual_norm_histogram.csv")
    (output / "command.txt").write_text(
        " ".join([sys.executable, "-m", "research.exp022.diagnose_t3_residual_oracle",
                  *sys.argv[1:]]) + "\n", encoding="utf-8")
    metadata.update(status="COMPLETE", num_targets=len(rows),
                    max_gt_xyz_reprojection_px=max_gt_reproj,
                    max_rendered_gt_reprojection_px=max_render_reproj,
                    elapsed_seconds=time.perf_counter() - started,
                    hierarchy_sanity=sanity["result"])
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return summary


def bop_evaluate(rows: list[dict], result_dir: Path, family: str, dataset: str,
                 targets_filename: str) -> dict:
    """Optional full BOP scoring on the same frozen-support poses."""
    result_dir.mkdir(parents=True, exist_ok=False)
    names = []
    for variant in ARMS:
        name = f"{BOP_METHOD[variant]}_{dataset}-test.csv"
        names.append(name)
        with (result_dir / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=("scene_id", "im_id", "obj_id", "score", "R", "t", "time"))
            writer.writeheader()
            for row in rows:
                pose = row[family][variant]["pose"]
                if not pose["success"]:
                    continue
                writer.writerow({
                    "scene_id": row["scene_id"], "im_id": row["im_id"], "obj_id": row["obj_id"],
                    "score": 1.0,
                    "R": " ".join(f"{v:.9g}" for v in np.asarray(pose["R"]).reshape(-1)),
                    "t": " ".join(f"{v:.9g}" for v in np.asarray(pose["t"]) * 1000.0),
                    "time": -1.0})
    eval_dir = result_dir.parent / f"bop_eval_{family}"
    toolkit, renderer = ROOT / ".local/bop_toolkit", ROOT / ".local/bop_renderer/build"
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": os.pathsep.join(
            [str(ROOT), str(toolkit), str(renderer), environment.get("PYTHONPATH", "")]),
        "BOP_PATH": str((ROOT / "datasets/BOP_DATASETS").resolve()),
        "BOP_RESULTS_PATH": str(result_dir), "BOP_EVAL_PATH": str(eval_dir),
        "BOP_RENDERER_PATH": str(renderer), "BOP_NUM_WORKERS": "1"})
    subprocess.run([
        sys.executable, str(ROOT / "lib/pysixd/scripts/eval_pose_results_more.py"),
        f"--results_path={result_dir}", f"--eval_path={eval_dir}",
        f"--result_filenames={','.join(names)}", "--renderer_type=cpp",
        "--error_types=mspd,mssd,vsd,ad,reS,teS", f"--targets_filename={targets_filename}",
        "--n_top=1", f"--dataset={dataset}"], check=True, cwd=ROOT, env=environment)
    metrics = {}
    for variant in ARMS:
        root = eval_dir / f"{BOP_METHOD[variant]}_{dataset}-test"
        bop = json.loads((root / "scores_bop19.json").read_text(encoding="utf-8"))
        add_files = list(root.glob("error=ad_ntop=*/scores_th=0.100_min-visib=-1.000.json"))
        add_files += list(root.glob("error:ad_ntop:*/scores_th:0.100_min-visib:-1.000.json"))
        if len(add_files) != 1:
            raise RuntimeError(f"Expected one ADD score for {variant}, got {add_files}")
        add = json.loads(add_files[0].read_text(encoding="utf-8"))
        metrics[variant] = {
            "bop": float(bop["bop19_average_recall"]),
            "add": float(add["recall"]),
            "reS": float(bop["bop19_average_recall_reS"]),
            "teS": float(bop["bop19_average_recall_teS"]),
        }
    (result_dir.parent / f"bop_metrics_{family}.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=LMO_CONFIG_ROOT / "train_reused.py")
    parser.add_argument("--hierarchy", type=Path,
                        help="override the config's PCC_HEAD.HIERARCHY_PATH")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-correspondences", type=int, default=0)
    parser.add_argument("--hierarchy-only", action="store_true")
    parser.add_argument("--bop-eval", action="store_true")
    args = parser.parse_args()
    if args.bop_eval and args.limit is not None:
        raise ValueError("--bop-eval requires a complete run")
    if args.hierarchy_only:
        cfg = Config.fromfile(str(args.config))
        context = resolve_dataset_context(cfg, require_hierarchy=args.hierarchy is None)
        path = args.hierarchy or context.hierarchy_path
        hierarchy, provenance = load_hierarchy_artifact(path, context.object_ids, context.key)
        sanity = hierarchy_sanity(_levels_from_hierarchy(hierarchy), context.object_ids)
        sanity["hierarchy"] = str(path)
        sanity["hierarchy_sha256"] = _sha256(path)
        sanity["hierarchy_mode"] = provenance["mode"]
        sanity["hierarchy_generator_version"] = provenance["generator_version"]
        print(json.dumps(sanity, indent=2))
        return 0 if sanity["result"] == "PASS" else 1
    if args.output is None:
        parser.error("--output is required unless --hierarchy-only is given")
    summary = run(args.config, args.output.resolve(), args.limit,
                  args.max_correspondences, args.device, args.bop_eval, args.hierarchy)
    print(json.dumps({"status": "COMPLETE", "decision": summary["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
