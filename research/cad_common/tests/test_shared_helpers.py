"""Shared CAD helper behavior and EXP025 import compatibility."""
from types import SimpleNamespace

import pytest
import torch
from mmcv import Config

from research.cad_common import configuration, preflight, runtime


def test_exp025_compatibility_exports_are_shared_implementations():
    from research.exp025 import configuration as old_configuration
    from research.exp025 import preflight as old_preflight
    from research.exp025 import runtime as old_runtime

    assert old_configuration.backbone_settings is configuration.backbone_settings
    assert old_preflight.audit_optimizer is preflight.audit_optimizer
    assert old_preflight.verify_imagenet_backbone is preflight.verify_imagenet_backbone
    for name in ('seed_all', 'rng_state', 'restore_rng', 'amp_step',
                 'tensor_stats', 'grad_norm_stats'):
        assert getattr(old_runtime, name) is getattr(runtime, name)


def test_optimizer_audit_keeps_group_and_lr_contract():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD([
        {'params': list(model.parameters()), 'lr': 1e-3, 'name': 'cad_head'}])
    cfg = Config(dict(SOLVER=dict(OPTIMIZER_CFG=dict(lr=1e-3)),
                      BACKBONE_LR_MULT=.1))
    assert preflight.audit_optimizer(model, optimizer, cfg) == [
        {'name': 'cad_head', 'lr': 1e-3, 'parameters': 3}]
    optimizer.param_groups[0]['lr'] = 2e-3
    with pytest.raises(RuntimeError, match='LR mismatch'):
        preflight.audit_optimizer(model, optimizer, cfg)


def test_amp_step_rejects_non_finite_loss():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    with pytest.raises(runtime.NonFiniteTrainingError, match='Non-finite loss'):
        runtime.amp_step(model, optimizer, None, torch.tensor(float('nan')))


def _saved_batch(path, dataset_key=None):
    batch = {name: torch.zeros(2, 1) for name in
             ('roi_img', 'roi_cls', 'roi_xyz', 'roi_mask_visib')}
    if dataset_key is not None:
        batch['dataset_key'] = dataset_key
    torch.save(batch, path)


def test_shared_real_batch_uses_active_dataset_identity(tmp_path, monkeypatch):
    import core.gdrn_modeling.models.GDRN_CAD as model_module

    context = SimpleNamespace(key='lm13', object_ids=(1, 2))
    monkeypatch.setattr(model_module, 'dataset_context', lambda _cfg: context)
    cfg = Config(dict(DATASET_CONTEXT=dict(KEY='lm13')))
    good = tmp_path / 'lm.pt'
    wrong = tmp_path / 'lmo.pt'
    legacy = tmp_path / 'legacy.pt'
    _saved_batch(good, 'lm13')
    _saved_batch(wrong, 'lmo')
    _saved_batch(legacy)
    assert len(runtime.real_batch(cfg, 'cpu', 1, 'cpp', load_batch=good)['roi_img']) == 1
    with pytest.raises(ValueError, match='dataset mismatch'):
        runtime.real_batch(cfg, 'cpu', 1, 'cpp', load_batch=wrong)
    with pytest.raises(ValueError, match='dataset mismatch'):
        runtime.real_batch(cfg, 'cpu', 1, 'cpp', load_batch=legacy)


def test_exp025_wrapper_only_accepts_legacy_unlabelled_lmo(tmp_path, monkeypatch):
    import core.gdrn_modeling.models.GDRN_CAD as model_module
    from research.exp025.runtime import real_batch

    context = SimpleNamespace(key='lmo', object_ids=(1, 5))
    monkeypatch.setattr(model_module, 'dataset_context', lambda _cfg: context)
    legacy = tmp_path / 'legacy_lmo.pt'
    _saved_batch(legacy)
    cfg = Config(dict(DATASET_CONTEXT=dict(KEY='lmo')))
    assert len(real_batch(cfg, 'cpu', 1, 'cpp', load_batch=legacy)['roi_img']) == 1
