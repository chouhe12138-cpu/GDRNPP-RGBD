#!/usr/bin/env python3
"""Held-out LM-O comparison of matched 3-level uniform and GA-HFPS artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from mmcv import Config
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

from core.gdrn_modeling.cad.hierarchy import load_cad_hierarchy
from lib.pysixd import inout
from research.cad_hierarchy.build_geometry_adaptive import VARIANTS
from research.cad_hierarchy.diagnostics import hierarchy_sanity
from research.cad_hierarchy.geometry import traverse_hierarchy
from research.cad_hierarchy.geometry_adaptive import face_complexity, sample_surface_with_complexity
from research.exp025.configuration import require_hierarchy
from research.exp025.dataset_context import resolve_dataset_context


DEFAULT_HELDOUT_SEED = 20261019
DEFAULT_HELDOUT_COUNT = 200_000


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def summary(distance, radius):
    if not len(distance):
        return None
    normalized = distance / radius
    return {'count': int(len(distance)),
            'anchor_distance_mean_m': float(distance.mean()),
            'anchor_distance_median_m': float(np.median(distance)),
            'anchor_distance_p90_m': float(np.percentile(distance, 90)),
            'anchor_distance_p95_m': float(np.percentile(distance, 95)),
            'radius_mean_m': float(radius.mean()),
            'radius_p95_m': float(np.percentile(radius, 95)),
            'normalized_residual_mean': float(normalized.mean()),
            'normalized_residual_p95': float(np.percentile(normalized, 95)),
            'traversal_coverage': float((normalized <= 1.).mean())}


def connectivity(graph, valid, labels):
    """Components of each T3 label on valid mesh faces; diagnostic only."""
    subgraph = graph[valid][:, valid].tocoo()
    labels = labels[valid]
    same = labels[subgraph.row] == labels[subgraph.col]
    same_graph = csr_matrix((np.ones(int(same.sum()), dtype=np.uint8),
                             (subgraph.row[same], subgraph.col[same])), shape=subgraph.shape)
    n_components, component = connected_components(same_graph, directed=False)
    _, first = np.unique(component, return_index=True)
    per_node = np.bincount(labels[first], minlength=512)
    occupied = int(np.count_nonzero(per_node))
    return {'components': int(n_components), 'occupied_nodes': occupied,
            'extra_components_per_occupied_node': float((n_components - occupied) / occupied),
            'fragmented_node_fraction': float((per_node > 1).sum() / occupied)}


def node_relation(ids, complexity, radius):
    counts = np.bincount(ids, minlength=512)
    means = np.bincount(ids, weights=complexity, minlength=512) / np.maximum(counts, 1)
    occupied = counts > 0
    if occupied.sum() < 3 or np.ptp(means[occupied]) <= 1e-12:
        return {'occupied_nodes': int(occupied.sum()), 'complexity_vs_cell_size_spearman': None,
                'complexity_vs_radius_spearman': None}
    return {'occupied_nodes': int(occupied.sum()),
            'complexity_vs_cell_size_spearman': float(spearmanr(means[occupied], counts[occupied]).statistic),
            'complexity_vs_radius_spearman': float(spearmanr(means[occupied], radius[occupied]).statistic)}


def compare(context, artifact_dir, old_artifact, heldout_count=DEFAULT_HELDOUT_COUNT,
            heldout_seed=DEFAULT_HELDOUT_SEED):
    if heldout_count < 1:
        raise ValueError('heldout_count must be positive')
    expected = tuple(context.object_ids)
    loaded, sanity, digests = {}, {}, {}
    for name, lambda_geo in VARIANTS.items():
        path = artifact_dir / f'{name}.npz'
        h = load_cad_hierarchy(path, expected_object_ids=expected, dataset_key=context.key)
        if (h.depth != 3 or h.level_counts != (8, 64, 512)
                or h.metadata.get('mode') != 'geometry_adaptive'
                or h.metadata.get('generator_version') != 1
                or h.metadata.get('variant') != name
                or not np.isclose(h.metadata.get('lambda_geo', -1), lambda_geo)):
            raise ValueError(f'Unexpected 3-level GA-HFPS contract: {path}')
        loaded[name] = h
        sanity[name] = hierarchy_sanity(h.numpy_levels(), expected)
        digests[name] = sha256(path)
    metadata = loaded['uniform_512'].metadata
    for h in loaded.values():
        if (h.metadata.get('seed') != metadata.get('seed')
                or h.metadata.get('sample_count') != metadata.get('sample_count')
                or h.metadata.get('radius_margin') != metadata.get('radius_margin')):
            raise ValueError('Artifacts are not matched on build protocol')
    old_digest = require_hierarchy(old_artifact, context.key)
    old = load_cad_hierarchy(old_artifact, expected_object_ids=expected, dataset_key=context.key)
    anchor_match = all(np.array_equal(loaded['uniform_512'].level(d).anchors.numpy(),
                                      old.level(d).anchors.numpy()) for d in (1, 2, 3))
    if not anchor_match:
        raise ValueError('Uniform T1-T3 anchors differ from locked old artifact; T4 effect is confounded')

    pooled = {name: {group: {'distance': [], 'radius': []}
                     for group in ('global', 'complex_q75', 'smooth_q25')}
              for name in VARIANTS}
    per_object = {}
    for oi, oid in enumerate(expected):
        model = inout.load_ply(str(context.cad_model_dir / f'obj_{oid:06d}.ply'),
                               vertex_scale=context.cad_vertex_scale)
        complexity_face, _, valid, graph = face_complexity(model)
        sampled, _, complexity, _ = sample_surface_with_complexity(
            model, heldout_count, np.random.default_rng(heldout_seed + oid), complexity_face)
        points = sampled[:heldout_count].astype(np.float64)
        complexity = complexity.astype(np.float64)
        if np.ptp(complexity) <= 1e-12:
            complex_mask = smooth_mask = np.zeros(len(points), dtype=bool)
        else:
            q25, q75 = np.quantile(complexity, [.25, .75])
            complex_mask = complexity >= q75
            smooth_mask = complexity <= q25
        masks = {'global': np.ones(len(points), dtype=bool),
                 'complex_q75': complex_mask, 'smooth_q25': smooth_mask}
        vertices = np.asarray(model['pts'], dtype=np.float64)
        faces = np.asarray(model['faces'], dtype=np.int64)
        centroids = vertices[faces].mean(axis=1)
        entry = {'face_count': int(len(faces)), 'valid_faces': int(valid.sum()),
                 'complexity_q25': float(np.quantile(complexity, .25)),
                 'complexity_q75': float(np.quantile(complexity, .75)), 'variants': {}}
        for name, h in loaded.items():
            levels = [h.level(d).anchors[oi].numpy().astype(np.float64) for d in (1, 2, 3)]
            leaf = traverse_hierarchy(points, levels, 8)[:, 2]
            anchor = levels[2][leaf]
            radius_nodes = h.level(3).radii[oi].numpy().astype(np.float64)
            radius = radius_nodes[leaf]
            distance = np.linalg.norm(points - anchor, axis=1)
            metrics = {}
            for group, mask in masks.items():
                metrics[group] = summary(distance[mask], radius[mask])
                if mask.any():
                    pooled[name][group]['distance'].append(distance[mask])
                    pooled[name][group]['radius'].append(radius[mask])
            face_leaf = traverse_hierarchy(centroids, levels, 8)[:, 2]
            metrics['connectivity'] = connectivity(graph, valid, face_leaf)
            metrics['cell_relation'] = node_relation(leaf, complexity, radius_nodes)
            entry['variants'][name] = metrics
        per_object[str(oid)] = entry
        print(f'heldout_object={oid} count={heldout_count}', flush=True)

    pooled_report = {name: {group: (summary(np.concatenate(values['distance']),
                                            np.concatenate(values['radius']))
                                    if values['distance'] else None)
                            for group, values in groups.items()}
                     for name, groups in pooled.items()}
    for name in VARIANTS:
        components = [per_object[str(oid)]['variants'][name]['connectivity'] for oid in expected]
        occupied = sum(item['occupied_nodes'] for item in components)
        extra = sum(item['components'] - item['occupied_nodes'] for item in components)
        pooled_report[name]['connectivity'] = {
            'components': int(sum(item['components'] for item in components)),
            'occupied_nodes': int(occupied),
            'extra_components_per_occupied_node': float(extra / occupied)}
    base = pooled_report['uniform_512']
    gates = {}
    for name in ('adaptive_512_l1', 'adaptive_512_l2'):
        candidate = pooled_report[name]
        complex_change = (candidate['complex_q75']['anchor_distance_mean_m'] /
                          base['complex_q75']['anchor_distance_mean_m'] - 1.)
        global_change = (candidate['global']['anchor_distance_mean_m'] /
                         base['global']['anchor_distance_mean_m'] - 1.)
        base_extra = base['connectivity']['extra_components_per_occupied_node']
        new_extra = candidate['connectivity']['extra_components_per_occupied_node']
        connectivity_warning = bool(new_extra - base_extra >= .1 and
                                    (base_extra == 0 or new_extra / base_extra > 1.2))
        integrity = all(item['result'] == 'PASS' for item in sanity.values())
        gates[name] = {'complex_q75_mean_distance_change': float(complex_change),
                       'global_mean_distance_change': float(global_change),
                       'connectivity_warning': connectivity_warning,
                       'engineering_pass': bool(integrity and complex_change <= -.10
                                                and global_change <= .05 and not connectivity_warning)}
    recommendation = ('adaptive_512_l1' if gates['adaptive_512_l1']['engineering_pass'] else
                      'adaptive_512_l2' if gates['adaptive_512_l2']['engineering_pass'] else None)
    return {'dataset': context.key, 'object_ids': list(expected),
            'build_seed': metadata['seed'], 'build_sample_count': metadata['sample_count'],
            'heldout_seed': heldout_seed, 'heldout_sample_count_per_object': heldout_count,
            'old_exp025_sha256': old_digest, 'uniform_anchors_match_old_t1_t3': anchor_match,
            'artifact_sha256': digests, 'sanity': sanity, 'per_object': per_object,
            'pooled': pooled_report, 'gates': gates,
            'recommended_candidate': recommendation,
            'decision_scope': 'offline engineering only; no formal pose accuracy or server authorization'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    parser.add_argument('--old-artifact', type=Path, required=True)
    parser.add_argument('--heldout-count', type=int, default=DEFAULT_HELDOUT_COUNT)
    parser.add_argument('--heldout-seed', type=int, default=DEFAULT_HELDOUT_SEED)
    args = parser.parse_args()
    report_path = args.artifact_dir / 'offline_compare.json'
    if report_path.exists():
        raise FileExistsError(report_path)
    cfg = Config.fromfile(str(args.config))
    context = resolve_dataset_context(cfg, require_hierarchy=False)
    report = compare(context, args.artifact_dir, args.old_artifact,
                     args.heldout_count, args.heldout_seed)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'OFFLINE_COMPARE {report_path} recommendation={report["recommended_candidate"]}')


if __name__ == '__main__':
    main()
