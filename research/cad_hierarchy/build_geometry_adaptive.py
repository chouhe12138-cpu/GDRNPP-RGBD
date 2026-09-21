#!/usr/bin/env python3
"""Build matched, independent 3-level LM-O uniform and GA-HFPS CAD artifacts."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from mmcv import Config

import ref
from lib.pysixd import inout, misc
from research.cad_hierarchy.geometry_adaptive import (
    BRANCH, LEVEL_COUNTS, RADIUS_FLOOR_M, RADIUS_MARGIN, build_anchors,
    face_complexity, fit_node_geometry, sample_surface_with_complexity)
from research.exp025.dataset_context import resolve_dataset_context


DEFAULT_SEED = 20260919
DEFAULT_BUILD_SAMPLES = 1_000_000
VARIANTS = {'uniform_512': 0., 'adaptive_512_l1': 1., 'adaptive_512_l2': 2.}


def symmetry_arrays(models_info, object_ids, vertex_scale):
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


def build(context, seed=DEFAULT_SEED, sample_count=DEFAULT_BUILD_SAMPLES):
    if sample_count < LEVEL_COUNTS[-1]:
        raise ValueError(f'At least {LEVEL_COUNTS[-1]} surface samples are required')
    models_info = ref.__dict__[context.cad_ref_key].get_models_info()
    symmetry_counts, symmetry_transforms = symmetry_arrays(
        models_info, context.object_ids, context.cad_vertex_scale)
    common = {
        'object_ids': np.asarray(context.object_ids, dtype=np.int64),
        'extents': [],
        'diameters': np.asarray([
            models_info[str(oid)]['diameter'] * context.cad_vertex_scale
            for oid in context.object_ids], dtype=np.float32),
        'symmetry_counts': symmetry_counts,
        'symmetry_transforms': symmetry_transforms,
    }
    levels = {name: {depth: {field: [] for field in ('anchors', 'normals', 'radii')}
                     for depth in (1, 2, 3)} for name in VARIANTS}
    report = {}
    for oid in context.object_ids:
        started = time.perf_counter()
        model = inout.load_ply(str(context.cad_model_dir / f'obj_{oid:06d}.ply'),
                               vertex_scale=context.cad_vertex_scale)
        complexity, raw, valid, _ = face_complexity(model)
        points, normals, sampled_complexity, _ = sample_surface_with_complexity(
            model, sample_count, np.random.default_rng(seed + oid), complexity)
        common['extents'].append(np.ptp(np.asarray(model['pts'], dtype=np.float64), axis=0))
        entry = {'faces': int(len(valid)), 'degenerate_faces': int((~valid).sum()),
                 'complexity_raw_q95': float(np.quantile(raw[valid], .95)), 'variants': {}}
        for name, lambda_geo in VARIANTS.items():
            anchors = build_anchors(points[:sample_count], sampled_complexity, lambda_geo)
            _, node_normals, radii, empty = fit_node_geometry(points, normals, anchors)
            for depth in (1, 2, 3):
                levels[name][depth]['anchors'].append(anchors[depth])
                levels[name][depth]['normals'].append(node_normals[depth])
                levels[name][depth]['radii'].append(radii[depth])
            entry['variants'][name] = {
                'empty_cells_per_level': empty,
                't3_radius_median_m': float(np.median(radii[3])),
                't3_radius_max_m': float(radii[3].max())}
        entry['seconds'] = time.perf_counter() - started
        report[str(oid)] = entry
        print(f'object={oid} seconds={entry["seconds"]:.1f}', flush=True)
    common['extents'] = np.stack(common['extents']).astype(np.float32)
    artifacts = {}
    for name, lambda_geo in VARIANTS.items():
        output = dict(common)
        for depth in (1, 2, 3):
            for field in ('anchors', 'normals', 'radii'):
                output[f'level{depth}_{field}'] = np.stack(levels[name][depth][field]).astype(np.float32)
        output.update(
            mode=np.asarray('geometry_adaptive'),
            generator_version=np.asarray(1, dtype=np.int64),
            branch_factor=np.asarray(BRANCH, dtype=np.int64),
            depth=np.asarray(3, dtype=np.int64),
            level_counts=np.asarray(LEVEL_COUNTS, dtype=np.int64),
            variant=np.asarray(name),
            lambda_geo=np.asarray(lambda_geo, dtype=np.float32),
            complexity_metric=np.asarray('normal_dispersion_1ring_area_weighted'),
            complexity_q=np.asarray(.95, dtype=np.float32),
            dataset_key=np.asarray(context.key), cad_ref_key=np.asarray(context.cad_ref_key),
            seed=np.asarray(seed, dtype=np.int64),
            sample_count=np.asarray(sample_count, dtype=np.int64),
            radius_margin=np.asarray(RADIUS_MARGIN, dtype=np.float32),
            radius_floor_m=np.asarray(RADIUS_FLOOR_M, dtype=np.float32))
        artifacts[name] = output
    return artifacts, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sample-count', type=int, default=DEFAULT_BUILD_SAMPLES)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if 'exp025' in args.output_dir.resolve().parts:
        raise ValueError('EXP026 artifacts must not be placed inside the locked EXP025 cache')
    cfg = Config.fromfile(str(args.config))
    context = resolve_dataset_context(cfg, require_hierarchy=False)
    artifacts, report = build(context, args.seed, args.sample_count)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, artifact in artifacts.items():
        path = args.output_dir / f'{name}.npz'
        np.savez_compressed(path, **artifact)
        print(f'ARTIFACT {path}', flush=True)
    (args.output_dir / 'build_stats.json').write_text(json.dumps({
        'dataset': context.key, 'object_ids': list(context.object_ids),
        'seed': args.seed, 'sample_count': args.sample_count,
        'radius_margin': RADIUS_MARGIN, 'radius_floor_m': RADIUS_FLOOR_M,
        'per_object': report}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
