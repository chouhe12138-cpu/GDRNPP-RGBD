"""Small helpers shared by EXP025 smoke and fixed-batch diagnostics."""
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


def metadata(cfg):
    return dict(experiment_id=cfg.EXPERIMENT_ID, seed=42, backbone_init=cfg.BACKBONE_INIT,
                train_backbone=bool(cfg.TRAIN_BACKBONE), config=cfg.filename,
                source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                source_tree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
                hierarchy=cfg.MODEL.POSE_NET.CAD_ATTENTION_HEAD.HIERARCHY_PATH,
                training_kind='diagnostic', formal=False)


def save_report(output, report):
    with (output / 'report.json').open('w') as stream:
        json.dump(report, stream, indent=2)


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


def amp_step(model, optimizer, scaler, total):
    if not torch.isfinite(total):
        raise RuntimeError('Non-finite loss')
    scaler.scale(total).backward()
    scaler.unscale_(optimizer)
    bad = [name for name, p in model.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    if bad:
        raise RuntimeError(f'Non-finite gradient (AMP scale={scaler.get_scale()}): {bad[:8]}')
    before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    if scaler.get_scale() < before:
        raise RuntimeError('AMP skipped optimizer step')
