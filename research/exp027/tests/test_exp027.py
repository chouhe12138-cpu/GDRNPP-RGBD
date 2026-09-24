"""EXP027 config isolation and causal paths through both architectures."""
from pathlib import Path

import numpy as np
import pytest
import torch
from mmcv import Config

from core.gdrn_modeling.models.heads.multiscale_cad_head import (
    MultiscaleImageQueryHead, HierarchicalCADRegionQueryHead)
from research.exp027.preflight import ARMS, inspect_config
from research.run_contract import validate_research_run_config


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
    old = Config.fromfile('configs/gdrn/lmo_pbr/research/exp026_residual_aligned_sampling_ablation/train_adaptive_full.py')
    assert tuple(old.MODEL.POSE_NET.BACKBONE.INIT_CFG.out_indices) == (3,)
    assert 'ARCHITECTURE' not in old.MODEL.POSE_NET.CAD_ATTENTION_HEAD
    for cfg in (a, b):
        validate_research_run_config(cfg, mode='prepare')
        with pytest.raises(ValueError, match='release not authorized'):
            validate_research_run_config(cfg, mode='formal')


def test_query_checkpoint_requires_query_tensors(tmp_path):
    from core.gdrn_modeling.models.GDRN_CAD import require_full_checkpoint
    path = tmp_path / 'partial.pth'
    torch.save({'model': {'backbone.stem_0.weight': torch.zeros(1),
                          'cad_attention_head.residual_predictor.final.weight': torch.zeros(1),
                          'cad_attention_head.mask_predictor.weight': torch.zeros(1),
                          'cad_attention_head.t3_classifier.weight': torch.zeros(1)}}, path)
    with pytest.raises(ValueError, match='query_parents'):
        require_full_checkpoint(path, 'hierarchical_cad_query')


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
                      residual_target_mode='predicted_route')
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
    with pytest.raises(ValueError, match='four ConvNeXt'):
        head(features[-1], torch.tensor([0]))
