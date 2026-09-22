"""EXP026 alignment, artifact identity, and matched-arm contracts."""
from pathlib import Path

import numpy as np
import pytest
import torch
from mmcv import Config

from core.gdrn_modeling.models.heads.hierarchical_cad_attention_head import HierarchicalCADAttentionHead
from core.gdrn_modeling.models.GDRN_CAD import dataset_context
from research.cad_hierarchy.contracts import require_exact_hierarchy
from research.exp026.preflight import ARMS
from research.run_contract import validate_research_run_config


@pytest.fixture
def heads(tmp_path_factory):
    arrays = dict(object_ids=np.array([1]), extents=np.ones((1, 3), np.float32),
                  diameters=np.ones(1, np.float32), symmetry_counts=np.array([1]),
                  symmetry_transforms=np.eye(4, dtype=np.float32).reshape(1, 1, 4, 4),
                  mode=np.array('geometry_adaptive'), generator_version=np.array(1),
                  dataset_key=np.array('synthetic'))
    for depth in (1, 2, 3):
        count = 8 ** depth
        arrays[f'level{depth}_anchors'] = np.zeros((1, count, 3), np.float32)
        arrays[f'level{depth}_normals'] = np.ones((1, count, 3), np.float32)
        arrays[f'level{depth}_radii'] = np.ones((1, count), np.float32)
    arrays['level3_anchors'][0, 1, 0] = .1
    path = tmp_path_factory.mktemp('exp026') / 'tree.npz'
    np.savez(path, **arrays)
    return tuple(HierarchicalCADAttentionHead(path, token_dim=16, num_heads=4,
                 dataset_key='synthetic', residual_target_mode=mode)
                 for mode in ('gt_route', 'predicted_route'))


def sample(pred_id, xyz=.5):
    logits = torch.full((1, 512, 1, 1), -20., requires_grad=True)
    logits = logits + torch.nn.functional.one_hot(torch.tensor(pred_id), 512).reshape(1, 512, 1, 1) * 40
    residual = torch.zeros(1, 3, 1, 1, requires_grad=True)
    prediction = dict(t3_logits=logits, residual=residual,
                      mask_logit=torch.zeros(1, 1, 1, 1))
    return prediction, torch.tensor([0]), torch.tensor([xyz, .5, .5]).reshape(1, 3, 1, 1), torch.ones(1, 1, 1, 1)


def test_same_route_target_is_numerically_identical(heads):
    args = sample(0, .52)
    old, _ = heads[0].loss(*args)
    new, stats = heads[1].loss(*args)
    torch.testing.assert_close(old['loss_cad_residual'], new['loss_cad_residual'])
    assert stats['cad_gt_route_equals_pred_route'] == 1


def test_wrong_representable_route_uses_predicted_cell_and_roundtrips(heads):
    prediction, classes, xyz, mask = sample(1, .5)
    old, _ = heads[0].loss(prediction, classes, xyz, mask)
    new, stats = heads[1].loss(prediction, classes, xyz, mask)
    assert old['loss_cad_residual'] == 0
    assert new['loss_cad_residual'] > 0
    assert stats['cad_pred_route_representable'] == 1
    point = (xyz.flatten(2).transpose(1, 2) - .5) * heads[1].extents[classes, None]
    anchor = heads[1].level3_anchors[classes, 1:2]
    radius = heads[1].level3_radii[classes, 1:2]
    target = (point - anchor) / radius[..., None]
    torch.testing.assert_close(anchor + radius[..., None] * target, point)


def test_unreachable_masks_residual_but_keeps_route(heads):
    with torch.no_grad():
        heads[1].level3_anchors[0, 1, 0] = 5.
    prediction, classes, xyz, mask = sample(1)
    losses, stats = heads[1].loss(prediction, classes, xyz, mask)
    assert losses['loss_cad_residual'] == 0
    assert losses['loss_cad_t3'] > 0
    assert stats['cad_pred_route_residual_valid_points'] == 0
    with torch.no_grad():
        heads[1].level3_anchors[0, 1, 0] = .1


def test_residual_target_does_not_create_classifier_gradient(heads):
    prediction, classes, xyz, mask = sample(1)
    losses, _ = heads[1].loss(prediction, classes, xyz, mask)
    gradient = torch.autograd.grad(losses['loss_cad_residual'], prediction['t3_logits'],
                                   allow_unused=True)[0]
    assert gradient is None or torch.count_nonzero(gradient) == 0
    route_gradient = torch.autograd.grad(losses['loss_cad_t3'], prediction['t3_logits'])[0]
    assert route_gradient is not None and route_gradient.abs().sum() > 0


def test_symmetry_branch_selection_keeps_historical_score(heads, tmp_path):
    original = heads[0]
    arrays = {name: value.detach().numpy().copy() for name, value in original.named_buffers()
              if name in ('object_ids', 'extents', 'diameters', 'symmetry_counts', 'symmetry_transforms')
              or name.startswith('level')}
    arrays['symmetry_counts'][:] = 2
    arrays['symmetry_transforms'] = np.tile(np.eye(4, dtype=np.float32), (1, 2, 1, 1))
    arrays['symmetry_transforms'][0, 1, :2, :2] *= -1
    arrays.update(mode=np.array('geometry_adaptive'), generator_version=np.array(1),
                  dataset_key=np.array('synthetic'))
    path = tmp_path / 'symmetric.npz'
    np.savez(path, **arrays)
    old = HierarchicalCADAttentionHead(path, token_dim=16, num_heads=4, dataset_key='synthetic')
    new = HierarchicalCADAttentionHead(path, token_dim=16, num_heads=4, dataset_key='synthetic',
                                       residual_target_mode='predicted_route')
    logits = torch.full((1, 512, 1, 1), -10.)
    logits[:, 0] = 10.
    prediction = dict(t3_logits=logits, residual=torch.tensor([-.1, -.2, 0.]).reshape(1, 3, 1, 1),
                      mask_logit=torch.zeros(1, 1, 1, 1))
    classes = torch.tensor([0])
    xyz = torch.tensor([.6, .7, .5]).reshape(1, 3, 1, 1)
    mask = torch.ones(1, 1, 1, 1)
    old_losses, old_stats = old.loss(prediction, classes, xyz, mask)
    new_losses, new_stats = new.loss(prediction, classes, xyz, mask)
    assert old_stats['cad_symmetry_branch'] == new_stats['cad_symmetry_branch'] == 1
    for key in ('loss_cad_t1', 'loss_cad_t2', 'loss_cad_t3'):
        torch.testing.assert_close(old_losses[key], new_losses[key])


def flatten(data, prefix=''):
    result = {}
    for key, value in data.items():
        path = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, path))
        else:
            result[path] = value
    return result


def test_resolved_configs_have_only_predeclared_arm_differences():
    uniform, adaptive = (Config.fromfile(str(path)) for path in ARMS.values())
    left, right = flatten(uniform._cfg_dict), flatten(adaptive._cfg_dict)
    different = {key for key in left.keys() | right.keys() if left.get(key) != right.get(key)}
    assert different == {'EXP026_ARM', 'OUTPUT_DIR', 'MODEL.POSE_NET.CAD_ATTENTION_HEAD.HIERARCHY_PATH',
                         'CAD_HIERARCHY_CONTRACT.SHA256', 'CAD_HIERARCHY_CONTRACT.VARIANT',
                         'CAD_HIERARCHY_CONTRACT.LAMBDA_GEO'}
    assert uniform.MODEL.WEIGHTS == adaptive.MODEL.WEIGHTS == ''
    assert uniform.SOLVER.IMS_PER_BATCH == adaptive.SOLVER.IMS_PER_BATCH == 48
    assert uniform.RESEARCH_PROTOCOL.SERVER_RELEASE_ALLOWED is False
    for cfg in (uniform, adaptive):
        validate_research_run_config(cfg, mode='prepare')
        for mode in ('smoke', 'formal', 'eval'):
            with pytest.raises(ValueError, match='SERVER_BLOCKED'):
                validate_research_run_config(cfg, mode=mode)


@pytest.mark.parametrize('arm', tuple(ARMS))
def test_exact_artifact_contract(arm):
    cfg = Config.fromfile(str(ARMS[arm]))
    path = Path(cfg.MODEL.POSE_NET.CAD_ATTENTION_HEAD.HIERARCHY_PATH)
    digest = require_exact_hierarchy(path, cfg.CAD_HIERARCHY_CONTRACT,
                                      dataset_key='lmo', object_ids=(1, 5, 6, 8, 9, 10, 11, 12))
    assert digest == cfg.CAD_HIERARCHY_CONTRACT.SHA256
    bad = dict(cfg.CAD_HIERARCHY_CONTRACT)
    bad['SHA256'] = '0' * 64
    with pytest.raises(ValueError, match='SHA mismatch'):
        require_exact_hierarchy(path, bad, dataset_key='lmo',
                                object_ids=(1, 5, 6, 8, 9, 10, 11, 12))
    bad = dict(cfg.CAD_HIERARCHY_CONTRACT)
    bad['VARIANT'] = 'wrong_variant'
    with pytest.raises(ValueError, match='variant mismatch'):
        require_exact_hierarchy(path, bad, dataset_key='lmo',
                                object_ids=(1, 5, 6, 8, 9, 10, 11, 12))


def test_exp025_sha_gate_cannot_be_bypassed_by_new_contract():
    cfg = Config.fromfile('configs/gdrn/lmo_pbr/research/exp025_hierarchical_cad_attention/train_imagenet_full.py')
    new = Config.fromfile(str(ARMS['uniform_full']))
    cfg.MODEL.POSE_NET.CAD_ATTENTION_HEAD.HIERARCHY_PATH = new.MODEL.POSE_NET.CAD_ATTENTION_HEAD.HIERARCHY_PATH
    cfg.CAD_HIERARCHY_CONTRACT = new.CAD_HIERARCHY_CONTRACT
    with pytest.raises(ValueError, match='EXP025 lmo hierarchy requires'):
        dataset_context(cfg)


def test_exp026_arm_name_cannot_be_swapped_without_artifact():
    cfg = Config.fromfile(str(ARMS['uniform_full']))
    cfg.EXP026_ARM = 'adaptive_l1_full'
    with pytest.raises(ValueError, match='artifact identity mismatch'):
        dataset_context(cfg)
