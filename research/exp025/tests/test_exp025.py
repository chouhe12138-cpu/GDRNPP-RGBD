"""EXP025 distribution, geometry, feature-flow and runtime contracts."""
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from core.gdrn_modeling.models.heads.hierarchical_cad_attention_head import (
    hierarchy_log_probabilities, nested_targets, bounded_residual)
from core.gdrn_modeling.models.heads.cad_attention_blocks import CADStageTransition, WindowAttentionBlock
from research.cad_hierarchy.geometry import traverse_hierarchy
from research.exp025.preflight import read_config
from research.exp025.configuration import set_mode

# `tree` and `head` fixtures live in conftest.py, shared with the diagnostic tests.


@pytest.mark.parametrize('scale', [0., 1., 1000.])
def test_marginals_and_gradients(scale):
    logits = (torch.randn(2, 512, 3, 5)*scale).requires_grad_()
    t1, t2, t3 = hierarchy_log_probabilities(logits)
    for lp in (t1, t2, t3):
        torch.testing.assert_close(lp.exp().sum(1), torch.ones_like(lp[:, 0]), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(t2.exp(), t3.exp().reshape(2, 64, 8, 3, 5).sum(2))
    torch.testing.assert_close(t1.exp(), t2.exp().reshape(2, 8, 8, 3, 5).sum(2))
    target = torch.randint(512, (2, 3, 5))
    torch.testing.assert_close(F.nll_loss(t3, target), F.cross_entropy(logits, target))
    for lp, divisor in ((t1, 64), (t2, 8), (t3, 1)):
        grad, = torch.autograd.grad(F.nll_loss(lp, target//divisor), logits, retain_graph=True)
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    if scale == 0:
        for lp, n in ((t1, 8), (t2, 64), (t3, 512)):
            torch.testing.assert_close(lp.exp(), torch.full_like(lp, 1/n))


def test_marginal_argmax_is_not_leaf_ancestor(head):
    logits = torch.full((1, 512, 1, 1), -100.)
    logits[:, 0] = 2.
    logits[:, 64:128] = 1.
    t1, _, _ = hierarchy_log_probabilities(logits)
    assert logits.argmax(1).item() == 0 and t1.argmax(1).item() == 1
    pred = dict(t3_logits=logits, residual=torch.zeros(1, 3, 1, 1))
    xyz = head.decode(pred, torch.tensor([0]))
    torch.testing.assert_close(xyz.flatten(), head.level3_anchors[0, 0]+.5)


def test_targets_nested_and_all_ids(head):
    points = torch.randn(2, 100, 3)*.1
    anchors = [getattr(head, f'level{d}_anchors') for d in (1, 2, 3)]
    paths = nested_targets(points, torch.tensor([0, 1]), anchors)
    torch.testing.assert_close(paths[0], paths[2]//64)
    torch.testing.assert_close(paths[1], paths[2]//8)
    for obj in range(2):
        ref = traverse_hierarchy(points[obj].numpy(), [a[obj].numpy() for a in anchors])
        np.testing.assert_array_equal(ref, torch.stack(paths, -1)[obj].numpy())
    ids = torch.arange(512)
    torch.testing.assert_close((ids//64)*64 + ((ids//8)%8)*8 + ids%8, ids)


def test_targets_ignore_autocast(head):
    xyz = torch.rand(2, 3, 3, 5)
    mask = torch.ones(2, 1, 3, 5)
    classes = torch.tensor([0, 1])
    expected = head.targets(xyz, mask, classes, 1)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = head.targets(xyz, mask, classes, 1)
    assert actual[1].dtype == torch.float32
    for a, b in zip(actual[0], expected[0]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_forward_loss_no_gt_flow_or_t4(head):
    head.zero_grad(set_to_none=True)
    feature = torch.randn(2, 1024, 8, 8, requires_grad=True)
    classes = torch.tensor([0, 1])
    output = head(feature, classes)
    assert output['t3_logits'].shape == (2, 512, 64, 64)
    assert output['residual'].norm(dim=1).max() <= 1.00001
    assert not any('level4' in k for k, _ in head.named_buffers())
    assert not any('c1' in k or 'c2' in k or 'c3' in k for k, _ in head.named_parameters())
    xyz = torch.rand(2, 3, 64, 64)
    mask = torch.ones(2, 1, 64, 64)
    losses, stats = head.loss(output, classes, xyz, mask)
    assert len(losses) == 5 and 'cad_route_sum' in stats
    sum(losses.values()).backward()
    assert feature.grad is not None and torch.isfinite(feature.grad).all()
    assert head.t3_classifier.weight.grad.abs().sum() > 0
    head.loss(output, classes, xyz+1, mask)
    with torch.no_grad():
        again = head(feature, classes)
    torch.testing.assert_close(output['t3_logits'], again['t3_logits'], rtol=0, atol=0)


def test_empty_support_nan_gt(head):
    pred = dict(t3_logits=torch.randn(1, 512, 2, 2, requires_grad=True),
                residual=bounded_residual(torch.randn(1, 3, 2, 2, requires_grad=True)),
                mask_logit=torch.zeros(1, 1, 2, 2, requires_grad=True))
    losses, stats = head.loss(pred, torch.tensor([0]), torch.full((1, 3, 2, 2), float('nan')),
                             torch.zeros(1, 1, 2, 2))
    assert all(v == 0 for k, v in losses.items() if k != 'loss_cad_mask')
    assert stats['cad_valid_points'] == 0
    sum(losses.values()).backward()


def test_symmetry_instance_selection(head):
    classes = torch.tensor([1])
    xyz = torch.tensor([.6, .7, .5]).reshape(1, 3, 1, 1)
    mask = torch.ones(1, 1, 1, 1)
    path, res, _ = head.targets(xyz, mask, classes, 1)
    logits = torch.full((1, 512, 1, 1), -10.)
    logits[0, path[2].item()] = 10.
    pred = dict(t3_logits=logits, residual=res.transpose(1, 2).reshape(1, 3, 1, 1), mask_logit=torch.zeros_like(mask))
    losses, stats = head.loss(pred, classes, xyz, mask)
    assert stats['cad_symmetry_branch'] == 1
    assert losses['loss_cad_residual'] == 0


def test_transition_and_window():
    block = CADStageTransition(16, 8)
    x = torch.randn(1, 16, 8, 8, requires_grad=True)
    y = block(x)
    assert y.shape == (1, 8, 16, 16)
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    attention = WindowAttentionBlock(16, 4, resolution=16, window=8, shift=4)
    assert attention.allowed.dtype == torch.bool and (~attention.allowed).any()
    tokens = torch.randn(1, 256, 16, requires_grad=True)
    attention(tokens).sum().backward()
    assert torch.isfinite(tokens.grad).all()


@pytest.mark.parametrize('train,init', [(False, 'official_lmo'), (True, 'official_lmo'), (True, 'imagenet'), (False, 'imagenet')])
def test_modes_and_batch_contract(train, init):
    cfg = set_mode(read_config(), train, init)
    assert cfg.MODEL.POSE_NET.BACKBONE.FREEZE == (not train)
    assert cfg.MODEL.WEIGHTS == ('' if init == 'imagenet' else 'pretrained_models/lmo_pbr/model_final_wo_optim.pth')
    from research.run_contract import validate_research_run_config
    from core.gdrn_modeling.models.GDRN_CAD import dataset_context
    assert dataset_context(cfg).hierarchy_path.name == 'consistent_v3.npz'
    validate_research_run_config(cfg, mode='prepare')
    with pytest.raises(ValueError, match='not ready'):
        validate_research_run_config(cfg, mode='formal')


def test_wrapper_grad_range(head):
    from core.gdrn_modeling.models.GDRN_CAD import GDRN_CAD
    backbone = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((8, 8)), torch.nn.Conv2d(3, 1024, 1))
    model = GDRN_CAD(backbone, head)
    for train in (False, True):
        model.zero_grad(set_to_none=True)
        backbone.requires_grad_(train)
        model.train()
        assert backbone.training == train
        pred = model.predict(torch.randn(1, 3, 256, 256), torch.tensor([0]))
        pred['t3_logits'].square().mean().backward()
        assert any(p.grad is not None for p in backbone.parameters()) == train
