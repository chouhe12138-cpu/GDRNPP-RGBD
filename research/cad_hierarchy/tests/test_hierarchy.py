"""Synthetic contracts and optional local-artifact regressions."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from core.gdrn_modeling.cad.hierarchy import load_cad_hierarchy
from research.cad_hierarchy import geometry as g
from research.cad_hierarchy.diagnostics import hierarchy_sanity, surface_representation


def tree(depth=3, branch=2):
    data = dict(object_ids=np.array([1]), extents=np.ones((1, 3)), diameters=np.ones(1),
                symmetry_counts=np.array([3]),
                symmetry_transforms=np.tile(np.eye(4), (1, 3, 1, 1)))
    for d in range(1, depth + 1):
        data[f'level{d}_anchors'] = np.zeros((1, branch ** d, 3))
        data[f'level{d}_normals'] = np.zeros((1, branch ** d, 3))
        data[f'level{d}_radii'] = np.full((1, branch ** d), float(depth - d + 1))
    return data


def save(tmp_path, data):
    path = tmp_path / 'tree.npz'
    np.savez(path, **data)
    return path


@pytest.mark.parametrize('depth,branch', [(1, 2), (2, 2), (3, 8)])
def test_generic_tree(tmp_path, depth, branch):
    h = load_cad_hierarchy(save(tmp_path, tree(depth, branch)))
    assert h.depth == depth and h.branch_factor == branch
    assert h.source_leaf_indices is None
    assert h.symmetry_counts.tolist() == [3]
    assert hierarchy_sanity(h.numpy_levels(), (1,))['result'] == 'PASS'
    with pytest.raises(ValueError):
        h.level(0)


@pytest.mark.parametrize('field,value', [
    ('object_ids', np.array([1.5])), ('object_ids', np.array([1, 1])),
    ('level2_radii', np.zeros((1, 4))), ('level2_radii', -np.ones((1, 4))),
    ('level2_radii', np.full((1, 4), np.nan)),
    ('level2_normals', np.full((1, 4, 3), np.inf)),
    ('level2_anchors', np.zeros((1, 5, 3))),
    ('extents', np.zeros((1, 3))), ('diameters', np.array([np.inf])),
    ('symmetry_counts', np.array([4])), ('symmetry_counts', np.array([0])),
    ('symmetry_counts', np.array([1.])), ('depth', np.array(4)),
    ('branch_factor', np.array(8)), ('level_counts', np.array([2, 4, 9])),
    ('schema_version', np.array('bad')),
])
def test_invalid_artifact(tmp_path, field, value):
    data = tree()
    data[field] = value
    with pytest.raises(ValueError):
        load_cad_hierarchy(save(tmp_path, data))


def test_dataset_order_and_missing_arrays(tmp_path):
    data = tree()
    path = save(tmp_path, data)
    with pytest.raises(ValueError, match='order'):
        load_cad_hierarchy(path, expected_object_ids=(2,))
    with pytest.raises(ValueError, match='dataset'):
        load_cad_hierarchy(path, dataset_key='lmo')
    load_cad_hierarchy(path, dataset_key='lmo', allow_missing_dataset=True)
    data['dataset_key'] = np.array('other')
    with pytest.raises(ValueError, match='dataset'):
        load_cad_hierarchy(save(tmp_path, data), dataset_key='lmo', allow_missing_dataset=True)
    del data['level2_normals']
    with pytest.raises(ValueError, match='missing'):
        load_cad_hierarchy(save(tmp_path, data))


def test_middle_relation_failure(tmp_path):
    data = tree(depth=4)
    data['level2_radii'][:] = 1.5
    h = load_cad_hierarchy(save(tmp_path, data))
    report = hierarchy_sanity(h.numpy_levels(), (1,))
    assert report['result'] == 'FAIL'
    assert report['failed_relations'] == ['level2_over_level3']


def test_bad_shapes_report_without_reshape(tmp_path):
    h = load_cad_hierarchy(save(tmp_path, tree()))
    levels = h.numpy_levels()
    levels[2]['anchors'] = np.empty((1, 0, 3))
    assert hierarchy_sanity(levels, (1,))['result'] == 'FAIL'
    assert hierarchy_sanity({}, (1,))['result'] == 'FAIL'


def test_surface_report_dynamic_depth_and_evidence(tmp_path):
    h = load_cad_hierarchy(save(tmp_path, tree(depth=2)))
    report = surface_representation(h.numpy_levels(), (1,), {1: np.zeros((3, 3))},
                                    selected_depths=(1, 2), sample_seed=42)
    assert report['pooled_traversal_coverage_t1'] == 1
    assert report['per_object']['1']['unhit_sampled_nodes_t2'] == 3
    assert report['empty_nodes_check'].startswith('NOT_CHECKED')


def test_paths_and_residuals():
    anchors = [np.array([[-2., 0, 0], [2., 0, 0]]),
               np.array([[-3., 0, 0], [-1., 0, 0], [1., 0, 0], [3., 0, 0]])]
    points = np.array([[-2.8, 0, 0], [1.2, 0, 0], [0., 0, 0]])
    np.testing.assert_array_equal(g.traverse_hierarchy(points, anchors, 2), [[0, 0], [1, 2], [0, 1]])
    assert g.traverse_hierarchy(np.empty((0, 3)), anchors, 2).shape == (0, 2)
    np.testing.assert_array_equal(g.parent_of(g.children_of(np.arange(4), 2), 2),
                                  np.repeat(np.arange(4)[:, None], 2, axis=1))
    points = np.array([[.5, 0, 0], [2., 0, 0], [1., 0, 0]])
    decoded = g.oracle_decode(points, np.zeros_like(points), np.ones(3))
    np.testing.assert_array_equal(decoded, [[.5, 0, 0], [1, 0, 0], [1, 0, 0]])
    with pytest.raises(ValueError):
        g.encode_residual(points, points, np.zeros(3))
    with pytest.raises(ValueError):
        g.traverse_hierarchy(points, anchors, 8)


def test_sampling_fps_nearest_are_deterministic():
    mesh = dict(pts=np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1]]),
                faces=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))
    a = g.sample_surface(mesh, 1000, np.random.default_rng(42))
    b = g.sample_surface(mesh, 1000, np.random.default_rng(42))
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    first = a[0].mean(0)
    ids = g.farthest_point_sampling(a[0], 8, first)
    assert len(np.unique(ids)) == len(ids)
    nearest = g.nearest_anchor(a[0], a[0][ids])
    np.testing.assert_array_equal(nearest[ids], np.arange(len(ids)))


def test_real_exp025_artifact():
    path = Path('.local/dataset_cache/exp025/consistent_v3.npz')
    if not path.is_file():
        pytest.skip('local artifact unavailable')
    h = load_cad_hierarchy(path, dataset_key='lmo')
    report = hierarchy_sanity(h.numpy_levels(), h.object_ids.tolist())
    assert report['result'] == 'PASS'
    assert [value['below_one'] for value in report['parent_coverage'].values()] == [0, 0, 0]


def test_import_isolation():
    code = '''import sys
import core.gdrn_modeling.cad.hierarchy
import research.cad_hierarchy.geometry
import research.cad_hierarchy.diagnostics
assert not any(k.startswith(('research.exp021', 'research.exp022', 'detectron2')) for k in sys.modules)
'''
    subprocess.run([sys.executable, '-c', code], check=True)
