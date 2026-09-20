"""Small helpers shared by EXP025 smoke, fixed-batch and numerical diagnostics."""
from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    """Undo `rng_state()` so a run continues from the recorded stream."""
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def metadata(cfg):
    from .configuration import require_hierarchy
    hierarchy = cfg.MODEL.POSE_NET.CAD_ATTENTION_HEAD.HIERARCHY_PATH
    return dict(experiment_id=cfg.EXPERIMENT_ID, seed=42, backbone_init=cfg.BACKBONE_INIT,
                train_backbone=bool(cfg.TRAIN_BACKBONE), config=cfg.filename,
                source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                source_tree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
                hierarchy=hierarchy, hierarchy_sha256=require_hierarchy(hierarchy, cfg.DATASET_CONTEXT.KEY),
                training_kind='diagnostic', formal=False)


def amp_init_scale(cfg, requested=None):
    """The scale a diagnostic's own GradScaler starts from.

    An explicit CLI value wins.  Otherwise the run follows `SOLVER.AMP.INIT_SCALE` when the
    config pins one -- the same value production would hand to `LightningLite` -- and falls
    back to the torch default (65536) that Lite builds when nothing is pinned.
    """
    if requested is not None:
        return float(requested)
    scale = cfg.SOLVER.get('AMP', {}).get('INIT_SCALE', None)
    return 65536. if scale is None else float(scale)


def save_report(output, report):
    with (output / 'report.json').open('w') as stream:
        json.dump(report, stream, indent=2)


def tensor_stats(value):
    """Finite flag plus absolute magnitude of a tensor, as recorded for failures."""
    data = value.detach().float()
    if data.numel() == 0:
        return dict(finite=True, abs_max=0.0, abs_mean=0.0)
    return dict(finite=bool(torch.isfinite(data).all()), abs_mean=float(data.abs().mean()),
                abs_max=float(data.abs().max()))


def token_norm_stats(tokens):
    norm = tokens.detach().float().norm(dim=-1).flatten()
    if norm.numel() == 0:
        return dict(mean=0.0, p95=0.0, max=0.0)
    return dict(mean=float(norm.mean()), p95=float(norm.quantile(.95)), max=float(norm.max()))


def raw_residual_stats(raw):
    magnitude = raw.detach().float().abs()
    return dict(abs_max=float(magnitude.max()), fraction_abs_gt_5=float((magnitude > 5).float().mean()),
                fraction_abs_gt_9=float((magnitude > 9).float().mean()),
                tanh_exact_saturation=float((torch.tanh(raw.detach().float()).abs() == 1).float().mean()))


def head_telemetry(head, diagnostics):
    """Reduce the head's detached diagnostics into JSON-safe numbers."""
    stats = {f'{name}_token_norm': token_norm_stats(diagnostics[name])
             for name in ('image_tokens', 'cad_tokens')}
    stats['t3_logits'] = tensor_stats(diagnostics['t3_logits'])
    stats['mask_logit'] = tensor_stats(diagnostics['mask_logit'])
    stats['raw_residual'] = raw_residual_stats(diagnostics['raw_residual'])
    if 'soft_t3_context' in diagnostics:  # residual V2 conditioning
        stats['soft_t3_context_token_norm'] = token_norm_stats(diagnostics['soft_t3_context'])
    stats['bounded_residual_norm_max'] = float(diagnostics['residual'].detach().float().norm(dim=1).max())
    stats['t3_classifier_weight_abs_max'] = float(head.t3_classifier.weight.detach().abs().max())
    return stats


def grad_norm_stats(model):
    """Per-module gradient norms and element maxima, plus the names that went non-finite.

    The element maximum is what decides whether a scaled fp16 gradient overflows, so it
    is reported next to the norm rather than inferred from it.
    """
    totals, maxima, bad = {}, {}, []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        parts = name.split('.')
        module = '.'.join(parts[:2]) if parts[0] == 'cad_attention_head' else parts[0]
        gradient = parameter.grad.detach().float()
        if not torch.isfinite(gradient).all():
            bad.append(name)
        totals[module] = totals.get(module, 0.0) + float(gradient.square().sum())
        maxima[module] = max(maxima.get(module, 0.0), float(gradient.abs().max()))
    # Non-finite values are reported as null so the report stays valid JSON.
    keep = lambda value: value if value == value and abs(value) != float('inf') else None
    return dict(grad_norm={name: keep(value**.5) for name, value in sorted(totals.items())},
                grad_abs_max={name: keep(value) for name, value in sorted(maxima.items())},
                non_finite_parameters=bad)


def save_last_good(path, model, optimizer, scaler, step, extra=None, probe=None):
    """Model/optimizer/scaler plus RNG so a failure can be replayed from this state.

    A diagnostic probe that owns parameters the model does not carry (the residual
    probe's conditioning branch) must be saved alongside, or the state is incomplete.
    """
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                 gradscaler=None if scaler is None else scaler.state_dict(), step=int(step),
                 rng=rng_state(), extra=extra or {})
    if probe is not None:
        state['probe'] = probe.state_dict()
    torch.save(state, path)


def load_last_good(path):
    """Read a last-good checkpoint written by `save_last_good`."""
    return torch.load(path, map_location='cpu')


class NonFiniteTrainingError(RuntimeError):
    """A non-finite loss or gradient, carrying the telemetry of the failing step."""

    def __init__(self, message, telemetry):
        super().__init__(message)
        self.telemetry = telemetry


def amp_step(model, optimizer, scaler, total, step=None, losses=None, diagnostics=None, head=None):
    """One optimizer update; `scaler=None` means FP32.

    Raises NonFiniteTrainingError carrying the telemetry of the failing step instead of
    a bare message, so a run can report which step, which parameters and which magnitude.
    """
    prefix = 'AMP' if scaler is not None else 'FP32'
    telemetry = dict(step=None if step is None else int(step),
                     amp_scale=None if scaler is None else float(scaler.get_scale()),
                     losses=None if losses is None else {k: float(v.detach()) for k, v in losses.items()})
    if diagnostics:
        telemetry['head'] = head_telemetry(head, diagnostics)
    if not torch.isfinite(total):
        telemetry.update(loss_finite=False)
        raise NonFiniteTrainingError('Non-finite loss', telemetry)
    if scaler is None:
        total.backward()
    else:
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)  # once per step; `scaler.step` below reuses this unscaling
    grad_stats = grad_norm_stats(model)
    telemetry['grad'] = grad_stats
    bad = grad_stats['non_finite_parameters']
    if bad:
        telemetry.update(non_finite_parameters=bad)
        raise NonFiniteTrainingError(f'Non-finite gradient ({prefix}): {bad[:8]}', telemetry)
    if scaler is None:
        optimizer.step()
        return
    before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    if scaler.get_scale() < before:
        raise NonFiniteTrainingError('AMP skipped optimizer step', telemetry)


def real_batch(cfg, device, batch_size, renderer_type, load_batch=None, save_batch=None):
    from core.gdrn_modeling.models.GDRN_CAD import dataset_context
    context = dataset_context(cfg)
    required = ('roi_img', 'roi_cls', 'roi_xyz', 'roi_mask_visib')
    if load_batch:
        batch = torch.load(load_batch, map_location='cpu')
        if not set(required) <= batch.keys() or len(batch['roi_img']) < batch_size:
            raise ValueError('Saved batch lacks required tensors/instances')
        if batch.get('dataset_key', 'lmo') != 'lmo' or tuple(batch.get('object_ids', context.object_ids)) != context.object_ids:
            raise ValueError('Saved batch dataset mismatch')
        result = {key: batch[key][:batch_size].to(device) for key in required}
    else:
        import ref
        from detectron2.data import MetadataCatalog
        from core.gdrn_modeling.datasets.data_loader import build_gdrn_train_loader
        from core.gdrn_modeling.engine.engine_utils import get_renderer, batch_data
        cfg.SOLVER.IMS_PER_BATCH = batch_size
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.DATALOADER.PERSISTENT_WORKERS = False
        cfg.MODEL.POSE_NET.XYZ_RENDERER = renderer_type
        meta = MetadataCatalog.get(context.train_dataset)
        renderer = get_renderer(cfg, ref.__dict__[meta.ref_key], meta.objs, gpu_id=0)
        try:
            raw = next(iter(build_gdrn_train_loader(cfg, cfg.DATASETS.TRAIN)))
            result = batch_data(cfg, raw, renderer=renderer, device=device, phase='train')
            result = {key: result[key] for key in required}
        finally:
            if hasattr(renderer, 'close'):
                renderer.close()
    if save_batch:
        save_batch = Path(save_batch)
        save_batch.parent.mkdir(parents=True, exist_ok=True)
        with save_batch.open('xb') as stream:
            torch.save({**{key: value.detach().cpu() for key, value in result.items()},
                        'dataset_key': context.key, 'object_ids': context.object_ids,
                        'source_batch': str(load_batch) if load_batch else 'online',
                        'subset': f'first {batch_size}', 'renderer': renderer_type}, stream)
    return result
