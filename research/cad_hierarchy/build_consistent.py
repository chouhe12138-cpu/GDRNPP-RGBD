#!/usr/bin/env python3
"""Build a deterministic traversal-consistent 8^4 CAD hierarchy.

The recursive partition and radius fit are the retained, dataset-neutral form of the
EXP022 consistent-v3 builder. Artifacts are machine-local and must be registered by
SHA256 in ``research.exp025.configuration`` before EXP025 may consume them.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from mmcv import Config

import ref
from lib.pysixd import inout, misc
from research.cad_hierarchy.geometry import (
    farthest_point_sampling,
    nearest_anchor,
    normalize_vectors,
    sample_surface,
    traverse_hierarchy,
)
from research.exp025.dataset_context import resolve_dataset_context


GENERATOR_VERSION = 3
MODE = 'consistent'
DEFAULT_SEED = 20260919
DEFAULT_BUILD_SAMPLES = 1_000_000
RADIUS_MARGIN = 1.05
RADIUS_FLOOR_M = 1e-5
LEVEL_COUNTS = (8, 64, 512, 4096)


def _symmetry_arrays(models_info, object_ids, vertex_scale):
    groups = []
    for object_id in object_ids:
        transforms = misc.get_symmetry_transformations(
            models_info[str(object_id)], max_sym_disc_step=.01)
        matrices = []
        for item in transforms:
            matrix = np.eye(4, dtype=np.float32)
            matrix[:3, :3] = item['R']
            matrix[:3, 3] = np.asarray(item['t']).reshape(3) * vertex_scale
            matrices.append(matrix)
        groups.append(np.stack(matrices))
    counts = np.asarray([len(group) for group in groups], dtype=np.int64)
    padded = np.tile(np.eye(4, dtype=np.float32), (len(groups), int(counts.max()), 1, 1))
    for index, group in enumerate(groups):
        padded[index, :len(group)] = group
    return counts, padded


def _fps_indices(points, count, first_point):
    if not len(points):
        raise ValueError('Traversal-consistent hierarchy encountered an empty cell')
    if len(points) >= count:
        return farthest_point_sampling(points, count, first_point)
    return np.arange(count) % len(points)


def _partition(points, level, node, anchors, degenerate):
    indices = _fps_indices(points, 8, points.mean(axis=0))
    if len(np.unique(indices)) < 8:
        degenerate.append(node)
    children = points[indices]
    anchors[level][node * 8:(node + 1) * 8] = children
    if level == len(LEVEL_COUNTS):
        return
    labels = nearest_anchor(points, children)
    for child in range(8):
        _partition(points[labels == child], level + 1, node * 8 + child,
                   anchors, degenerate)


def _object(object_id, seed, sample_count, model_dir, vertex_scale):
    model = inout.load_ply(str(model_dir / f'obj_{object_id:06d}.ply'),
                           vertex_scale=vertex_scale)
    points, normals = sample_surface(model, sample_count,
                                      np.random.default_rng(seed + object_id))
    points = points.astype(np.float64)
    normals = normals.astype(np.float64)
    if len(points) != sample_count + len(np.asarray(model['pts'])):
        raise RuntimeError('Surface sampling layout changed')
    anchors = {depth: np.empty((count, 3), dtype=np.float64)
               for depth, count in enumerate(LEVEL_COUNTS, 1)}
    degenerate = []
    _partition(points[:sample_count], 1, 0, anchors, degenerate)
    paths = traverse_hierarchy(points, [anchors[d] for d in range(1, 5)], 8)
    radii, node_normals = {}, {}
    for depth, count in enumerate(LEVEL_COUNTS, 1):
        selected = paths[:, depth - 1]
        distance = np.linalg.norm(points - anchors[depth][selected], axis=1)
        covering = np.zeros(count, dtype=np.float64)
        np.maximum.at(covering, selected, distance)
        counts = np.bincount(selected, minlength=count)
        normal_sum = np.zeros((count, 3), dtype=np.float64)
        np.add.at(normal_sum, selected, normals)
        mean_normal = normal_sum / np.maximum(counts, 1)[:, None]
        radii[depth] = np.maximum(covering * RADIUS_MARGIN, RADIUS_FLOOR_M)
        fallback = np.tile(np.array([0., 0., 1.]), (count, 1))
        node_normals[depth] = normalize_vectors(mean_normal, fallback=fallback)
    for depth in (3, 2, 1):
        children = anchors[depth + 1].reshape(-1, 8, 3)
        required = (np.linalg.norm(children - anchors[depth][:, None], axis=-1)
                    + radii[depth + 1].reshape(-1, 8)).max(-1) * RADIUS_MARGIN
        radii[depth] = np.maximum(radii[depth], required)
    empty = [int(np.count_nonzero(np.bincount(paths[:, depth - 1],
                                              minlength=count) == 0))
             for depth, count in enumerate(LEVEL_COUNTS, 1)]
    if any(empty) or degenerate:
        raise ValueError(f'object {object_id}: empty={empty}, degenerate={len(degenerate)}; '
                         'increase --sample-count')
    return model, anchors, node_normals, radii, {
        'empty_cells_per_level': empty,
        'radius_m': {f'level{depth}': {
            'min': float(radii[depth].min()),
            'median': float(np.median(radii[depth])),
            'max': float(radii[depth].max())} for depth in range(1, 5)},
    }


def build(context, seed=DEFAULT_SEED, sample_count=DEFAULT_BUILD_SAMPLES):
    models_info = ref.__dict__[context.cad_ref_key].get_models_info()
    symmetry_counts, symmetry_transforms = _symmetry_arrays(
        models_info, context.object_ids, context.cad_vertex_scale)
    output = {
        'object_ids': np.asarray(context.object_ids, dtype=np.int64),
        'extents': [],
        'diameters': np.asarray([
            models_info[str(object_id)]['diameter'] * context.cad_vertex_scale
            for object_id in context.object_ids], dtype=np.float32),
        'symmetry_counts': symmetry_counts,
        'symmetry_transforms': symmetry_transforms,
        'source_leaf_indices': np.full((len(context.object_ids), LEVEL_COUNTS[-1]),
                                       -1, dtype=np.int64),
    }
    levels = {depth: {field: [] for field in ('anchors', 'normals', 'radii')}
              for depth in range(1, 5)}
    report = {}
    for object_id in context.object_ids:
        started = time.perf_counter()
        model, anchors, normals, radii, stats = _object(
            object_id, seed, sample_count, context.cad_model_dir,
            context.cad_vertex_scale)
        output['extents'].append(np.ptp(np.asarray(model['pts'], dtype=np.float64), axis=0))
        for depth in range(1, 5):
            levels[depth]['anchors'].append(anchors[depth])
            levels[depth]['normals'].append(normals[depth])
            levels[depth]['radii'].append(radii[depth])
        stats['seconds'] = time.perf_counter() - started
        report[str(object_id)] = stats
        print(f'object={object_id} seconds={stats["seconds"]:.1f}', flush=True)
    output['extents'] = np.stack(output['extents']).astype(np.float32)
    for depth in range(1, 5):
        for field in ('anchors', 'normals', 'radii'):
            output[f'level{depth}_{field}'] = np.stack(levels[depth][field]).astype(np.float32)
    output.update(
        mode=np.asarray(MODE), generator_version=np.asarray(GENERATOR_VERSION, dtype=np.int64),
        dataset_key=np.asarray(context.key), cad_ref_key=np.asarray(context.cad_ref_key),
        seed=np.asarray(seed, dtype=np.int64), sample_count=np.asarray(sample_count, dtype=np.int64),
        radius_floor_m=np.asarray(RADIUS_FLOOR_M, dtype=np.float32))
    return output, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-count', type=int, default=DEFAULT_BUILD_SAMPLES)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = Config.fromfile(str(args.config))
    context = resolve_dataset_context(cfg, require_hierarchy=False)
    hierarchy, objects = build(context, args.seed, args.sample_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **hierarchy)
    summary = dict(mode=MODE, generator_version=GENERATOR_VERSION, dataset=context.key,
                   object_ids=list(context.object_ids), seed=args.seed,
                   sample_count=args.sample_count, radius_margin=RADIUS_MARGIN,
                   radius_floor_m=RADIUS_FLOOR_M, per_object=objects)
    args.output.with_suffix('.stats.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')
    print(f'CONSISTENT_HIERARCHY {args.output}')


if __name__ == '__main__':
    main()
