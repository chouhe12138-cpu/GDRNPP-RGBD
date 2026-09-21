"""Offline geometry-adaptive hierarchy primitives; no EXP025 runtime dependency."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix

from .geometry import (farthest_point_sampling, nearest_anchor, normalize_vectors,
                       traverse_hierarchy)


BRANCH = 8
LEVEL_COUNTS = (8, 64, 512)
RADIUS_MARGIN = 1.05
RADIUS_FLOOR_M = 1e-5


def face_complexity(model: dict, quantile: float = .95):
    """Return per-face normalized dispersion, raw dispersion, validity and edge graph."""
    vertices = np.asarray(model['pts'], dtype=np.float64)
    faces = np.asarray(model['faces'], dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or not 0 < quantile < 1:
        raise ValueError('Expected triangular faces and a quantile in (0, 1)')
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > 1e-15
    if not valid.any():
        raise ValueError('CAD mesh has no non-degenerate triangle')
    area = .5 * double_area
    normals = np.zeros_like(cross)
    normals[valid] = cross[valid] / double_area[valid, None]

    owners = defaultdict(list)
    for index, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            owners[(min(int(u), int(v)), max(int(u), int(v)))].append(index)
    rows, cols = [], []
    for group in owners.values():
        for face in group:
            rows.extend([face] * len(group))
            cols.extend(group)
    graph = csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)),
                       shape=(len(faces), len(faces)))
    graph.data[:] = 1
    weighted = graph @ (area[:, None] * normals)
    weight_sum = np.asarray(graph @ area).reshape(-1)
    mean = weighted / np.maximum(weight_sum[:, None], 1e-12)
    raw = np.maximum(0., 1. - np.einsum('ij,ij->i', mean, mean))
    raw[~valid] = 0.
    scale = float(np.quantile(raw[valid], quantile))
    normalized = (np.zeros_like(raw) if scale <= 1e-12 else
                  np.clip(raw / scale, 0., 1.))
    return normalized.astype(np.float32), raw.astype(np.float32), valid, graph


def sample_surface_with_complexity(model: dict, count: int, rng: np.random.Generator,
                                   complexity: np.ndarray):
    """Match sample_surface arithmetic/RNG while retaining sampled face complexity."""
    if count < 1:
        raise ValueError('count must be positive')
    points = np.asarray(model['pts'], dtype=np.float64)
    faces = np.asarray(model['faces'], dtype=np.int64)
    if np.asarray(complexity).shape != (len(faces),):
        raise ValueError('complexity must contain one value per mesh face')
    triangles = points[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > 1e-15
    if not valid.any():
        raise ValueError('CAD mesh has no non-degenerate triangle')
    triangles = triangles[valid]
    cross = cross[valid]
    probability = double_area[valid] / double_area[valid].sum()
    selected = rng.choice(len(triangles), size=count, replace=True, p=probability)
    tri = triangles[selected]
    u = np.sqrt(rng.random(count))
    v = rng.random(count)
    sampled = ((1. - u)[:, None] * tri[:, 0]
               + (u * (1. - v))[:, None] * tri[:, 1]
               + (u * v)[:, None] * tri[:, 2])
    face_normals = normalize_vectors(cross)[selected]
    vertex_normals = np.asarray(model.get('normals', np.zeros_like(points)), dtype=np.float64)
    vertex_normals = normalize_vectors(
        vertex_normals, fallback=np.tile(np.array([0, 0, 1]), (len(points), 1)))
    all_points = np.concatenate((sampled.astype(np.float32), points.astype(np.float32)), axis=0)
    all_normals = np.concatenate((face_normals, vertex_normals), axis=0)
    selected_faces = np.flatnonzero(valid)[selected]
    return all_points, all_normals, np.asarray(complexity)[selected_faces], selected_faces


def geometry_adaptive_fps(points, complexity, count, first_point, lambda_geo):
    points = np.asarray(points, dtype=np.float64)
    complexity = np.asarray(complexity, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 3 or complexity.shape != (len(points),)
            or count < 1 or len(points) < count or not np.isfinite(points).all()
            or not np.isfinite(complexity).all() or np.any((complexity < 0) | (complexity > 1))
            or not np.isfinite(lambda_geo) or lambda_geo < 0):
        raise ValueError('Invalid GA-FPS points, complexity, count or lambda')
    if lambda_geo == 0:
        return farthest_point_sampling(points, count, first_point)
    selected = np.empty(count, dtype=np.int64)
    selected[0] = np.linalg.norm(points - np.asarray(first_point)[None], axis=1).argmin()
    chosen = np.zeros(len(points), dtype=bool)
    chosen[selected[0]] = True
    minimum = np.linalg.norm(points - points[selected[0]][None], axis=1)
    for index in range(1, count):
        score = minimum * (1. + lambda_geo * complexity)
        score[chosen] = -np.inf
        selected[index] = int(score.argmax())
        chosen[selected[index]] = True
        minimum = np.minimum(minimum, np.linalg.norm(points - points[selected[index]][None], axis=1))
    return selected


def build_anchors(sampled_points, sampled_complexity, lambda_geo):
    """Recursive 8-way FPS and nearest-child partition, stopping at T3."""
    anchors = {depth: np.empty((count, 3), dtype=np.float64)
               for depth, count in enumerate(LEVEL_COUNTS, 1)}

    def partition(points, complexity, depth, node):
        if len(points) < BRANCH:
            raise ValueError(f'Empty/undersampled node at level {depth}, node {node}: {len(points)}')
        indices = geometry_adaptive_fps(points, complexity, BRANCH, points.mean(0), lambda_geo)
        children = points[indices]
        if len(np.unique(children, axis=0)) != BRANCH:
            raise ValueError(f'Degenerate anchors at level {depth}, node {node}')
        anchors[depth][node * BRANCH:(node + 1) * BRANCH] = children
        if depth == len(LEVEL_COUNTS):
            return
        labels = nearest_anchor(points, children)
        for child in range(BRANCH):
            mask = labels == child
            partition(points[mask], complexity[mask], depth + 1, node * BRANCH + child)

    partition(np.asarray(sampled_points, dtype=np.float64),
              np.asarray(sampled_complexity, dtype=np.float64), 1, 0)
    return anchors


def fit_node_geometry(all_points, all_normals, anchors):
    """Fit traversal support; only T3->T2->T1 ball propagation is permitted."""
    points = np.asarray(all_points, dtype=np.float64)
    normals = np.asarray(all_normals, dtype=np.float64)
    paths = traverse_hierarchy(points, [anchors[d] for d in (1, 2, 3)], BRANCH)
    radii, node_normals, empty = {}, {}, []
    for depth, count in enumerate(LEVEL_COUNTS, 1):
        selected = paths[:, depth - 1]
        distance = np.linalg.norm(points - anchors[depth][selected], axis=1)
        covering = np.zeros(count, dtype=np.float64)
        np.maximum.at(covering, selected, distance)
        counts = np.bincount(selected, minlength=count)
        empty.append(int(np.count_nonzero(counts == 0)))
        normal_sum = np.zeros((count, 3), dtype=np.float64)
        np.add.at(normal_sum, selected, normals)
        mean_normal = normal_sum / np.maximum(counts, 1)[:, None]
        fallback = np.tile(np.array([0., 0., 1.]), (count, 1))
        node_normals[depth] = normalize_vectors(mean_normal, fallback=fallback)
        radii[depth] = np.maximum(covering * RADIUS_MARGIN, RADIUS_FLOOR_M)
    if any(empty):
        raise ValueError(f'Empty traversal cells: {empty}')
    for depth in (2, 1):
        children = anchors[depth + 1].reshape(-1, BRANCH, 3)
        required = (np.linalg.norm(children - anchors[depth][:, None], axis=-1)
                    + radii[depth + 1].reshape(-1, BRANCH)).max(-1) * RADIUS_MARGIN
        radii[depth] = np.maximum(radii[depth], required)
    return paths, node_normals, radii, empty
