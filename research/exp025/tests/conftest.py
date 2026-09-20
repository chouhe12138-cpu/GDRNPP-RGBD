"""Shared synthetic hierarchy fixture for the EXP025 test modules."""
import numpy as np
import pytest
import torch

from core.gdrn_modeling.models.heads.hierarchical_cad_attention_head import HierarchicalCADAttentionHead


@pytest.fixture(scope='module')
def tree(tmp_path_factory):
    rng = np.random.default_rng(42)
    arrays = dict(object_ids=np.array([1, 2]), extents=np.ones((2, 3), np.float32),
                  diameters=np.ones(2, np.float32), symmetry_counts=np.array([1, 2]),
                  symmetry_transforms=np.tile(np.eye(4, dtype=np.float32), (2, 2, 1, 1)),
                  mode=np.array('consistent'), generator_version=np.array(3), dataset_key=np.array('synthetic'))
    arrays['symmetry_transforms'][1, 1, :2, :2] *= -1
    for depth in (1, 2, 3):
        arrays[f'level{depth}_anchors'] = rng.normal(0, .1, (2, 8**depth, 3)).astype(np.float32)
        arrays[f'level{depth}_normals'] = np.ones((2, 8**depth, 3), np.float32)
        arrays[f'level{depth}_radii'] = np.ones((2, 8**depth), np.float32)
    path = tmp_path_factory.mktemp('exp025') / 'tree.npz'
    np.savez(path, **arrays)
    return path


@pytest.fixture(scope='module')
def head(tree):
    torch.set_num_threads(4)
    return HierarchicalCADAttentionHead(tree, token_dim=16, num_heads=4, dataset_key='synthetic')
