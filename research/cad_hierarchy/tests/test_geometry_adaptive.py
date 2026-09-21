"""GA-HFPS primitives and strict three-level geometry contracts."""
import numpy as np
import pytest

from core.gdrn_modeling.cad.hierarchy import load_cad_hierarchy
from research.cad_hierarchy.diagnostics import hierarchy_sanity
from research.cad_hierarchy.geometry import farthest_point_sampling, sample_surface
from research.cad_hierarchy.geometry_adaptive import (
    LEVEL_COUNTS, build_anchors, face_complexity, fit_node_geometry,
    geometry_adaptive_fps, sample_surface_with_complexity)


def tetrahedron():
    return {'pts': np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
            'faces': np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])}


def test_face_complexity_flat_fold_and_degenerate():
    flat = {'pts': np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]]),
            'faces': np.array([[0, 1, 2], [0, 2, 3]])}
    normalized, raw, valid, graph = face_complexity(flat)
    assert valid.all() and graph[0, 1] and np.max(np.abs(raw)) < 1e-8
    assert np.max(np.abs(normalized)) < 1e-8
    folded = tetrahedron()
    normalized, raw, valid, _ = face_complexity(folded)
    assert valid.all() and raw.max() > 0 and normalized.max() > 0
    folded['faces'] = np.concatenate((folded['faces'], [[0, 0, 1]]))
    normalized, raw, valid, _ = face_complexity(folded)
    assert not valid[-1] and normalized[-1] == 0 and raw[-1] == 0


def test_sampling_matches_existing_arithmetic_and_lambda_zero():
    model = tetrahedron()
    complexity, _, _, _ = face_complexity(model)
    actual = sample_surface_with_complexity(model, 1000, np.random.default_rng(42), complexity)
    expected = sample_surface(model, 1000, np.random.default_rng(42))
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2], complexity[actual[3]])
    points = actual[0][:100]
    first = points.mean(0)
    np.testing.assert_array_equal(
        geometry_adaptive_fps(points, actual[2][:100], 8, first, 0.),
        farthest_point_sampling(points.astype(np.float64), 8, first))
    with pytest.raises(ValueError):
        geometry_adaptive_fps(points, np.ones(100), 8, first, -1.)


def test_three_level_deterministic_loader_and_coverage(tmp_path):
    model = tetrahedron()
    complexity, _, _, _ = face_complexity(model)
    points, normals, sampled_complexity, _ = sample_surface_with_complexity(
        model, 20000, np.random.default_rng(7), complexity)
    anchors = build_anchors(points[:20000], sampled_complexity, 1.)
    anchors_again = build_anchors(points[:20000], sampled_complexity, 1.)
    assert tuple(anchors) == (1, 2, 3) and LEVEL_COUNTS == (8, 64, 512)
    for depth, count in enumerate(LEVEL_COUNTS, 1):
        assert anchors[depth].shape == (count, 3)
        np.testing.assert_array_equal(anchors[depth], anchors_again[depth])
    paths, node_normals, radii, empty = fit_node_geometry(points, normals, anchors)
    assert paths.shape == (len(points), 3) and empty == [0, 0, 0]
    data = {'object_ids': np.array([1]), 'extents': np.array([[1., 1., 1.]]),
            'diameters': np.array([1.]), 'symmetry_counts': np.array([1]),
            'symmetry_transforms': np.eye(4)[None, None],
            'mode': np.array('geometry_adaptive'), 'generator_version': np.array(1),
            'branch_factor': np.array(8), 'depth': np.array(3),
            'level_counts': np.array(LEVEL_COUNTS)}
    for depth in (1, 2, 3):
        data[f'level{depth}_anchors'] = anchors[depth][None].astype(np.float32)
        data[f'level{depth}_normals'] = node_normals[depth][None]
        data[f'level{depth}_radii'] = radii[depth][None].astype(np.float32)
    path = tmp_path / 'three_levels.npz'
    np.savez(path, **data)
    hierarchy = load_cad_hierarchy(path, expected_object_ids=(1,))
    assert hierarchy.depth == 3 and hierarchy.level_counts == LEVEL_COUNTS
    assert hierarchy_sanity(hierarchy.numpy_levels(), (1,))['result'] == 'PASS'
    assert not any(key.startswith('level4') for key in data)
    direct = np.zeros(512)
    np.maximum.at(direct, paths[:, 2], np.linalg.norm(points - anchors[3][paths[:, 2]], axis=1))
    np.testing.assert_allclose(radii[3], np.maximum(direct * 1.05, 1e-5), rtol=0, atol=0)
