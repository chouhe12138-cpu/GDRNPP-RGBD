"""Numerical CAD utilities. Sampling/FPS preserve the EXP021 reference arithmetic."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def normalize_vectors(values: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    normalized = values / np.maximum(norm, 1e-12)
    if fallback is not None:
        invalid = norm[..., 0] <= 1e-12
        normalized[invalid] = fallback[invalid]
    return normalized.astype(np.float32)


def sample_surface(model: dict, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(model["pts"], dtype=np.float64)
    faces = np.asarray(model["faces"], dtype=np.int64)
    triangles = points[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > 1e-15
    if not np.any(valid):
        raise ValueError("CAD mesh has no non-degenerate triangle")
    faces = faces[valid]
    triangles = triangles[valid]
    cross = cross[valid]
    probability = double_area[valid] / double_area[valid].sum()
    selected = rng.choice(len(triangles), size=count, replace=True, p=probability)
    tri = triangles[selected]
    u = np.sqrt(rng.random(count))
    v = rng.random(count)
    sampled = (
        (1.0 - u)[:, None] * tri[:, 0]
        + (u * (1.0 - v))[:, None] * tri[:, 1]
        + (u * v)[:, None] * tri[:, 2]
    )
    face_normals = normalize_vectors(cross)[selected]
    vertex_normals = np.asarray(model.get("normals", np.zeros_like(points)), dtype=np.float64)
    vertex_normals = normalize_vectors(vertex_normals, fallback=np.tile(np.array([0, 0, 1]), (len(points), 1)))
    return (
        np.concatenate([sampled.astype(np.float32), points.astype(np.float32)], axis=0),
        np.concatenate([face_normals, vertex_normals], axis=0),
    )


def nearest_anchor(points: np.ndarray, anchors: np.ndarray, chunk: int = 20_000) -> np.ndarray:
    labels = np.empty(len(points), dtype=np.int64)
    for start in range(0, len(points), chunk):
        stop = min(start + chunk, len(points))
        labels[start:stop] = cdist(points[start:stop], anchors).argmin(axis=1)
    return labels


def farthest_point_sampling(points: np.ndarray, count: int, first_point: np.ndarray) -> np.ndarray:
    if len(points) < count:
        raise ValueError(f"Need at least {count} surface candidates, got {len(points)}")
    selected = np.empty(count, dtype=np.int64)
    selected[0] = np.linalg.norm(points - first_point[None], axis=1).argmin()
    min_distance = np.linalg.norm(points - points[selected[0]][None], axis=1)
    for index in range(1, count):
        selected[index] = int(min_distance.argmax())
        min_distance = np.minimum(
            min_distance, np.linalg.norm(points - points[selected[index]][None], axis=1)
        )
    return selected



def children_of(parent, branch_factor=8):
    return np.asarray(parent)[..., None] * branch_factor + np.arange(branch_factor)


def parent_of(child, branch_factor=8):
    return np.asarray(child) // branch_factor


def traverse_hierarchy(points, level_anchors, branch_factor=8):
    """Nested nearest-child walk on one object's tree, returning [P, depth]."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be finite [P,3]")
    if branch_factor < 2 or not level_anchors:
        raise ValueError("A nonempty tree with branch_factor >= 2 is required")
    parent = np.zeros(len(points), dtype=np.int64)
    ids = np.empty((len(points), len(level_anchors)), dtype=np.int64)
    for depth, anchors in enumerate(level_anchors, 1):
        anchors = np.asarray(anchors)
        if anchors.shape != (branch_factor ** depth, 3) or not np.isfinite(anchors).all():
            raise ValueError(f"Invalid anchors at depth {depth}")
        grouped = anchors.reshape(-1, branch_factor, 3)
        delta = points[:, None, :] - grouped[parent]
        parent = parent * branch_factor + np.einsum("pkj,pkj->pk", delta, delta).argmin(-1)
        ids[:, depth - 1] = parent
    return ids


def encode_residual(points, anchors, radii):
    radii = np.asarray(radii)
    if not np.isfinite(radii).all() or np.any(radii <= 0):
        raise ValueError("radii must be finite and positive")
    return (np.asarray(points) - anchors) / radii[..., None]


def clip_residual_unit_ball(raw):
    raw = np.asarray(raw)
    return raw / np.maximum(np.linalg.norm(raw, axis=-1, keepdims=True), 1.0)


def decode_residual(residual, anchors, radii):
    radii = np.asarray(radii)
    if not np.isfinite(radii).all() or np.any(radii <= 0):
        raise ValueError("radii must be finite and positive")
    return anchors + radii[..., None] * residual


def oracle_decode(points, anchors, radii):
    return decode_residual(clip_residual_unit_ball(encode_residual(points, anchors, radii)),
                           anchors, radii)


def assign_paths(points, object_index, levels):
    return traverse_hierarchy(points, [levels[d]["anchors"][object_index] for d in sorted(levels)],
                              levels[1]["anchors"].shape[1])


def residual_norms(points, ids, object_index, levels, depth):
    anchors = levels[depth]["anchors"][object_index][ids[:, depth - 1]]
    radii = levels[depth]["radii"][object_index][ids[:, depth - 1]]
    return np.linalg.norm(encode_residual(points, anchors, radii), axis=-1)


def oracle_xyz(points, ids, object_index, levels, depth):
    anchors = levels[depth]["anchors"][object_index][ids[:, depth - 1]]
    radii = levels[depth]["radii"][object_index][ids[:, depth - 1]]
    return oracle_decode(points, anchors, radii)


def anchor_xyz(ids, object_index, levels, depth):
    return levels[depth]["anchors"][object_index][ids[:, depth - 1]]
