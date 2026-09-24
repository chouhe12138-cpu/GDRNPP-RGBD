"""EXP027 config isolation and causal paths through both architectures."""
from pathlib import Path

import numpy as np
import pytest
import torch
from mmcv import Config

from core.gdrn_modeling.models.heads.multiscale_cad_head import (
    MultiscaleImageQueryHead, HierarchicalCADRegionQueryHead)
from core.gdrn_modeling.models.heads.hierarchical_cad_attention_head import HierarchicalCADAttentionHead
from research.exp027.preflight import ARMS, inspect_config
from research.run_contract import validate_research_run_config

SPEC = dict(backbone_channels=(128, 256, 512, 1024),
            feature_resolutions=(64, 32, 16, 8),
            pyramid_channels=(64, 128, 256, 512))
BASELINE = 'configs/gdrn/lmo_pbr/research/exp026_residual_aligned_sampling_ablation/train_adaptive_full.py'


def _flat(value, prefix=''):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(_flat(child, f'{prefix}.{key}' if prefix else key))
        return result
    return {prefix: value}


def test_matched_configs_and_legacy_isolation():
    a, b = (Config.fromfile(str(path)) for path in ARMS.values())
    assert inspect_config(a)[1] == inspect_config(b)[1]
    fa, fb = _flat(a._cfg_dict), _flat(b._cfg_dict)
    assert {key for key in fa | fb if fa.get(key) != fb.get(key)} == {
        'EXP027_ARM', 'OUTPUT_DIR', 'MODEL.POSE_NET.CAD_ATTENTION_HEAD.ARCHITECTURE'}
    old = Config.fromfile(BASELINE)
    assert tuple(old.MODEL.POSE_NET.BACKBONE.INIT_CFG.out_indices) == (3,)
    assert 'ARCHITECTURE' not in old.MODEL.POSE_NET.CAD_ATTENTION_HEAD
    assert 'backbone_channels' not in old.MODEL.POSE_NET.CAD_ATTENTION_HEAD.INIT_CFG
    assert 'feature_resolutions' not in old.MODEL.POSE_NET.CAD_ATTENTION_HEAD.INIT_CFG
    assert 'pyramid_channels' not in old.MODEL.POSE_NET.CAD_ATTENTION_HEAD.INIT_CFG
    assert not HierarchicalCADAttentionHead.requires_multiscale_features
    assert MultiscaleImageQueryHead.requires_multiscale_features
    assert HierarchicalCADRegionQueryHead.requires_multiscale_features
    for cfg in (a, b):
        validate_research_run_config(cfg, mode='prepare')
        with pytest.raises(ValueError, match='release not authorized'):
            validate_research_run_config(cfg, mode='formal')


def test_exp027_a_vs_exp026_adaptive_resolved_config():
    baseline = Config.fromfile(BASELINE)
    a = Config.fromfile(str(ARMS['A_multiscale_fpn']))
    fb, fa = _flat(baseline._cfg_dict), _flat(a._cfg_dict)
    diff = {key for key in fb | fa if fb.get(key) != fa.get(key)}
    allowed = {
        'EXP026_ARM', 'EXP027_ARM', 'EXPERIMENT_ID', 'OUTPUT_DIR', 'TRAIN_PROTOCOL.NAME',
        'MODEL.POSE_NET.BACKBONE.INIT_CFG.out_indices',
        'MODEL.POSE_NET.CAD_ATTENTION_HEAD.ARCHITECTURE',
        'MODEL.POSE_NET.CAD_ATTENTION_HEAD.INIT_CFG.backbone_channels',
        'MODEL.POSE_NET.CAD_ATTENTION_HEAD.INIT_CFG.feature_resolutions',
        'MODEL.POSE_NET.CAD_ATTENTION_HEAD.INIT_CFG.pyramid_channels',
        'SOLVER.AMP.INIT_SCALE', 'RESEARCH_PROTOCOL.FORMAL_READY',
        'RESEARCH_PROTOCOL.LOCAL_FORMAL_READY', 'RESEARCH_PROTOCOL.SERVER_RELEASE_ALLOWED'}
    assert not diff - allowed, {key: (fb.get(key), fa.get(key)) for key in sorted(diff - allowed)}
    for key in ('SEED', 'DATASETS.TRAIN', 'DATASETS.TEST', 'CAD_HIERARCHY_CONTRACT.SHA256',
                'CAD_HIERARCHY_CONTRACT.VARIANT', 'CAD_HIERARCHY_CONTRACT.LAMBDA_GEO',
                'SOLVER.IMS_PER_BATCH', 'SOLVER.TOTAL_EPOCHS', 'SOLVER.OPTIMIZER_CFG.type',
                'SOLVER.OPTIMIZER_CFG.lr', 'SOLVER.OPTIMIZER_CFG.weight_decay',
                'SOLVER.OPTIMIZER_CFG.betas', 'INPUT.COLOR_AUG_CODE', 'TEST.PNP_TYPE',
                'TEST.EVAL_PERIOD', 'VAL.ERROR_TYPES',
                'MODEL.POSE_NET.CAD_ATTENTION_HEAD.INIT_CFG.residual_target_mode'):
        assert fb[key] == fa[key], key


def test_multiscale_spec_and_backbone_contract(hierarchy):
    from core.gdrn_modeling.models.GDRN_CAD import GDRN_CAD, build_model_optimizer
    head = MultiscaleImageQueryHead(hierarchy, dataset_key='synthetic', token_dim=16,
                                    num_heads=4, **SPEC)
    assert head.feature_shapes == ((128, 64), (256, 32), (512, 16), (1024, 8))
    assert [(layer.in_channels, layer.out_channels) for layer in head.laterals] == [
        (512, 256), (256, 128), (128, 64)]
    for key, invalid in (('backbone_channels', (128, 256, 0, 1024)),
                         ('feature_resolutions', (64, 32, 8, 16)),
                         ('pyramid_channels', (64, 128, 256))):
        spec = dict(SPEC, **{key: invalid})
        with pytest.raises(ValueError):
            MultiscaleImageQueryHead(hierarchy, dataset_key='synthetic', token_dim=16,
                                     num_heads=4, **spec)
    alternative = dict(backbone_channels=(64, 128, 256, 512),
                       feature_resolutions=(32, 16, 8, 4),
                       pyramid_channels=(32, 64, 128, 256))
    changed = MultiscaleImageQueryHead(hierarchy, dataset_key='synthetic', token_dim=16,
                                       num_heads=4, **alternative)
    prediction = changed([torch.randn(1, c, s, s) for c, s in changed.feature_shapes],
                         torch.tensor([0]))
    assert prediction['t3_logits'].shape == (1, 512, 32, 32)
    current_features = [torch.randn(1, c, s, s) for c, s in head.feature_shapes]
    for index, replacement in ((0, torch.randn(1, 127, 64, 64)),
                               (1, torch.randn(1, 256, 16, 16))):
        wrong = list(current_features)
        wrong[index] = replacement
        with pytest.raises(ValueError, match='feature shape mismatch'):
            head(wrong, torch.tensor([0]))
    class FeatureBackbone(torch.nn.Module):
        def __init__(self, features):
            super().__init__()
            self.features = features

        def forward(self, _image):
            return self.features

    image = torch.empty(1, 3, 256, 256)
    assert len(GDRN_CAD(FeatureBackbone(current_features), head).backbone_feature(image)) == 4
    with pytest.raises(ValueError, match='four feature maps'):
        GDRN_CAD(FeatureBackbone(current_features[-1:]), head).backbone_feature(image)
    legacy = HierarchicalCADAttentionHead(hierarchy, dataset_key='synthetic', token_dim=16,
                                          num_heads=4)
    assert GDRN_CAD(FeatureBackbone(current_features[-1:]), legacy).backbone_feature(image).shape == (
        1, 1024, 8, 8)
    with pytest.raises(ValueError, match='one feature map'):
        GDRN_CAD(FeatureBackbone(current_features), legacy).backbone_feature(image)
    cfg = Config.fromfile(str(ARMS['A_multiscale_fpn']))
    cfg.MODEL.POSE_NET.BACKBONE.INIT_CFG.out_indices = (3,)
    with pytest.raises(ValueError, match='out_indices'):
        build_model_optimizer(cfg)


def test_query_checkpoint_requires_query_tensors(tmp_path):
    from core.gdrn_modeling.models.GDRN_CAD import require_full_checkpoint
    path = tmp_path / 'partial.pth'
    torch.save({'model': {'backbone.stem_0.weight': torch.zeros(1),
                          'cad_attention_head.residual_predictor.final.weight': torch.zeros(1),
                          'cad_attention_head.mask_predictor.weight': torch.zeros(1),
                          'cad_attention_head.t3_classifier.weight': torch.zeros(1)}}, path)
    with pytest.raises(ValueError, match='query_parents'):
        require_full_checkpoint(path, 'hierarchical_cad_query')
    complete_except_gate = {name: torch.zeros(1) for name in (
        'backbone.stem_0.weight', 'cad_attention_head.residual_predictor.final.weight',
        'cad_attention_head.mask_predictor.weight',
        'cad_attention_head.laterals.0.weight', 'cad_attention_head.laterals.1.weight',
        'cad_attention_head.laterals.2.weight',
        'cad_attention_head.query_parents.0.weight', 'cad_attention_head.query_parents.1.weight',
        'cad_attention_head.query_parents.2.weight',
        'cad_attention_head.pixel_projection.weight', 'cad_attention_head.query_projection.weight')}
    torch.save({'model': complete_except_gate}, path)
    with pytest.raises(ValueError, match='query_parent_alpha'):
        require_full_checkpoint(path, 'hierarchical_cad_query')
    complete_except_gate['cad_attention_head.query_parent_alpha'] = torch.full((3,), .01)
    torch.save({'model': complete_except_gate}, path)
    assert require_full_checkpoint(path, 'hierarchical_cad_query') == str(path)


def test_server_gate_evidence_requires_unique_exact_match(tmp_path):
    import json
    from research.exp027.gate_evidence import require_unique_gate
    config = str(ARMS['A_multiscale_fpn'])
    sha = '7ac75daa756ed7ae59463614737ccc46cd746173452cd1943a4a024d0817c631'
    with pytest.raises(RuntimeError, match='found 0'):
        require_unique_gate(tmp_path, 'a'*40, config, 4096, sha)
    run = tmp_path / 'RUN-20260924-120000-gate-s42-a01'
    (run / 'gate').mkdir(parents=True)
    (run / 'run_metadata.json').write_text(json.dumps(dict(mode='gate', source_commit='a'*40, config=config)))
    (run / 'gate' / 'report.json').write_text(json.dumps(dict(status='PASS', config=config,
        amp_init_scale=4096, hierarchy_sha256=sha, batch_size=48, steps=8,
        amp_skipped_steps=0, checkpoint_roundtrip='PASS')))
    (run / 'exit_code').write_text('0\n')
    assert require_unique_gate(tmp_path, 'a'*40, config, 4096, sha) == run
    with pytest.raises(RuntimeError, match='found 0'):
        require_unique_gate(tmp_path, 'b'*40, config, 4096, sha)
    second = tmp_path / 'RUN-20260924-120001-gate-s42-a01'
    import shutil
    shutil.copytree(run, second)
    with pytest.raises(RuntimeError, match='found 2'):
        require_unique_gate(tmp_path, 'a'*40, config, 4096, sha)


@pytest.fixture(scope='module')
def hierarchy(tmp_path_factory):
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
    path = tmp_path_factory.mktemp('exp027') / 'tree.npz'
    np.savez(path, **arrays)
    return path


@pytest.mark.parametrize('head_class,coarse_name', [
    (MultiscaleImageQueryHead, 'cross_attention.0.attention.in_proj_weight'),
    (HierarchicalCADRegionQueryHead, 'query_parents.0.weight')])
def test_coarse_path_reaches_t3(hierarchy: Path, head_class, coarse_name):
    torch.manual_seed(5)
    head = head_class(hierarchy, dataset_key='synthetic', token_dim=16, num_heads=4,
                      residual_target_mode='predicted_route', **SPEC)
    features = [torch.randn(1, c, s, s) for c, s in head.feature_shapes]
    prediction = head(features, torch.tensor([0]))
    assert prediction['t3_logits'].shape == (1, 512, 64, 64)
    assert prediction['residual'].shape == (1, 3, 64, 64)
    assert prediction['mask_logit'].shape == (1, 1, 64, 64)
    prediction['t3_logits'].square().mean().backward()
    named = dict(head.named_parameters())
    assert named[coarse_name].grad is not None
    assert named[coarse_name].grad.abs().sum() > 0
    assert head.lateral_alpha.grad is not None
    assert head.lateral_alpha.grad.abs().sum() > 0
    if isinstance(head, HierarchicalCADRegionQueryHead):
        assert head.query_parent_alpha.grad is not None
        assert torch.isfinite(head.query_parent_alpha.grad).all()
        assert (head.query_parent_alpha.grad != 0).all()
        before = head.query_parent_alpha.detach().clone()
        torch.optim.SGD([head.query_parent_alpha], lr=1e-3).step()
        assert not torch.equal(before, head.query_parent_alpha)
        restored = head_class(hierarchy, dataset_key='synthetic', token_dim=16,
                              num_heads=4, residual_target_mode='predicted_route', **SPEC)
        restored.load_state_dict(head.state_dict(), strict=True)
        assert torch.equal(restored.query_parent_alpha, head.query_parent_alpha)
    with pytest.raises(ValueError, match='four ConvNeXt'):
        head(features[-1], torch.tensor([0]))
