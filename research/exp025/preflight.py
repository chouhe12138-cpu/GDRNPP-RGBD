"""EXP025 CPU contract and initialization audit; never authorizes formal training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from mmcv import Config

from core.gdrn_modeling.models.GDRN_CAD import build_model_optimizer, dataset_context
from core.gdrn_modeling.models.heads.hierarchical_cad_attention_head import hierarchy_log_probabilities
from core.gdrn_modeling.cad.hierarchy import load_cad_hierarchy
from research.cad_hierarchy.diagnostics import hierarchy_sanity
from research.run_contract import validate_research_run_config
from .configuration import CONSISTENT_V3_SHA256, require_consistent_v3, set_mode

CONFIG = Path('configs/gdrn/lmo_pbr/research/exp025_hierarchical_cad_attention/train.py')


def read_config(path=CONFIG, train_backbone=None, backbone_init=None):
    cfg = Config.fromfile(str(path))
    if train_backbone is not None or backbone_init is not None:
        set_mode(cfg, cfg.TRAIN_BACKBONE if train_backbone is None else train_backbone,
                 cfg.BACKBONE_INIT if backbone_init is None else backbone_init)
    return cfg


def audit_optimizer(model, optimizer, cfg):
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    registered = [id(p) for group in optimizer.param_groups for p in group['params']]
    if len(registered) != len(set(registered)) or set(registered) != trainable:
        raise RuntimeError('Optimizer parameter coverage/duplicates')
    groups = []
    for group in optimizer.param_groups:
        expected = float(cfg.SOLVER.OPTIMIZER_CFG.lr)
        if group['name'] == 'backbone':
            expected *= float(cfg.BACKBONE_LR_MULT)
        if abs(group['lr'] - expected) > 1e-12:
            raise RuntimeError('Optimizer LR mismatch')
        groups.append(dict(name=group['name'], lr=group['lr'], parameters=sum(p.numel() for p in group['params'])))
    return groups


def run(cfg):
    validate_research_run_config(cfg, mode='prepare')
    if cfg.TRAIN_PROTOCOL.NAME != 'exp025_lmo':
        raise ValueError('EXP025 protocol required')
    context = dataset_context(cfg)
    if context.key != 'lmo' or context.object_ids != (1, 5, 6, 8, 9, 10, 11, 12):
        raise ValueError('EXP025 V1 is LM-O only')
    h = load_cad_hierarchy(context.hierarchy_path, expected_object_ids=context.object_ids, dataset_key='lmo')
    sanity = hierarchy_sanity({d: v for d, v in h.numpy_levels().items() if d <= 3}, context.object_ids)
    if sanity['result'] != 'PASS':
        raise RuntimeError(sanity)
    # Identity, not just shape: the geometry buffers do not travel with a checkpoint.
    hierarchy_sha256 = require_consistent_v3(context.hierarchy_path)
    cfg.MODEL.DEVICE = 'cpu'
    torch.manual_seed(42)
    model, optimizer = build_model_optimizer(cfg)
    if cfg.BACKBONE_INIT == 'imagenet':
        from research.exp022.preflight import verify_imagenet_backbone
        loaded = verify_imagenet_backbone(model, Path(cfg.MODEL.POSE_NET.BACKBONE.INIT_CFG.checkpoint_path))
    else:
        loaded = len(model.backbone.state_dict())
    model.train()
    if model.backbone.training != bool(cfg.TRAIN_BACKBONE):
        raise RuntimeError('Backbone train/eval mismatch')
    groups = audit_optimizer(model, optimizer, cfg)
    image = torch.randn(1, 3, 256, 256)
    classes = torch.zeros(1, dtype=torch.long)
    diagnostics = {}
    prediction = model.predict(image, classes, diagnostics=diagnostics)
    # The four-stage image ladder and the shared prediction heads, checked on the server
    # as well: a stale release must fail here instead of producing numbers.
    expected = {'t3_logits': (1, 512, 64, 64), 'residual': (1, 3, 64, 64), 'mask_logit': (1, 1, 64, 64)}
    for name, shape in expected.items():
        if tuple(prediction[name].shape) != shape:
            raise RuntimeError(f'Unexpected {name} shape {tuple(prediction[name].shape)}, expected {shape}')
    if tuple(diagnostics['image_tokens'].shape) != (1, 4096, model.cad_attention_head.token_dim):
        raise RuntimeError(f'Unexpected final image tokens: {tuple(diagnostics["image_tokens"].shape)}')
    counts = [bank.shape[1] for bank in model.cad_attention_head.token_banks(classes)]
    if counts != [1, 8, 64, 512]:
        raise RuntimeError(f'Unexpected CAD bank token counts: {counts}')
    xyz = torch.full((1, 3, 64, 64), .5)
    mask = torch.ones(1, 1, 64, 64)
    losses, stats = model.cad_attention_head.loss(prediction, classes, xyz, mask)
    total = sum(losses.values())
    total.backward()
    if not torch.isfinite(total) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError('Non-finite loss/gradient')
    has_grad = any(p.grad is not None for p in model.backbone.parameters())
    if has_grad != bool(cfg.TRAIN_BACKBONE):
        raise RuntimeError('Backbone gradient range mismatch')
    with torch.no_grad():
        for lp in hierarchy_log_probabilities(prediction['t3_logits']):
            torch.testing.assert_close(lp.exp().sum(1), torch.ones_like(lp[:, 0]), atol=1e-6, rtol=1e-5)
        decoded = model.cad_attention_head.decode(prediction, classes)
        if not torch.isfinite(decoded).all() or prediction['residual'].norm(dim=1).max() > 1.00001:
            raise RuntimeError('Invalid XYZ/residual')
    return dict(status='PASS', kind='cpu_preflight', backbone_init=cfg.BACKBONE_INIT,
                train_backbone=bool(cfg.TRAIN_BACKBONE), backbone_tensors=loaded, optimizer_groups=groups,
                total_parameters=sum(p.numel() for p in model.parameters()),
                backbone_parameters=sum(p.numel() for p in model.backbone.parameters()),
                head_parameters=sum(p.numel() for p in model.cad_attention_head.parameters()),
                hierarchy=str(context.hierarchy_path), hierarchy_sha256=hierarchy_sha256,
                hierarchy_sha_match=hierarchy_sha256 == CONSISTENT_V3_SHA256, hierarchy_sanity=sanity,
                losses={k: float(v.detach()) for k, v in losses.items()}, formal_ready=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--train-backbone', choices=('yes', 'no'))
    parser.add_argument('--backbone-init', choices=('official_lmo', 'imagenet'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = read_config(args.config, None if args.train_backbone is None else args.train_backbone == 'yes', args.backbone_init)
    result = run(cfg)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
