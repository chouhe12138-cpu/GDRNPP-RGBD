"""Matched local CUDA AMP smoke and fixed-batch learnability for EXP026."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from mmcv import Config

from core.gdrn_modeling.models.GDRN_CAD import build_model_optimizer
from research.cad_common.preflight import audit_optimizer
from research.cad_common.runtime import (amp_step, real_batch, seed_all)
from .preflight import inspect_config


@torch.no_grad()
def measure(model, batch):
    model.eval()
    head = model.cad_attention_head
    with torch.cuda.amp.autocast():
        prediction = model.predict(batch['roi_img'], batch['roi_cls'])
        losses, stats = head.loss(prediction, batch['roi_cls'], batch['roi_xyz'], batch['roi_mask_visib'])
    ids = prediction['t3_logits'].detach().argmax(1).flatten(1)
    classes = batch['roi_cls']
    anchor = head.level3_anchors[classes[:, None], ids].float()
    full = (head.decode(prediction, classes).flatten(2).transpose(1, 2) - .5) * head.extents[classes, None]
    # For symmetric objects use the single branch closest to the anchor for both
    # errors. This is a local engineering diagnostic, not a formal correspondence metric.
    candidate = []
    for branch in range(head.symmetry_transforms.shape[1]):
        _, _, valid, points = head.targets(batch['roi_xyz'], batch['roi_mask_visib'], classes,
                                           branch, return_points=True)
        anchor_error = ((anchor - points).norm(dim=-1) * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        full_error = ((full - points).norm(dim=-1) * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        allowed = branch < head.symmetry_counts[classes]
        candidate.append((anchor_error.masked_fill(~allowed, float('inf')), full_error))
    chosen = torch.stack([item[0] for item in candidate], 1).argmin(1)
    anchor_mm = torch.stack([item[0] for item in candidate], 1).gather(1, chosen[:, None]).mean() * 1000
    full_mm = torch.stack([item[1] for item in candidate], 1).gather(1, chosen[:, None]).mean() * 1000
    raw = prediction['residual'].detach()
    return dict(route_loss=float(sum(losses[f'loss_cad_t{depth}'] for depth in (1, 2, 3))),
                loss_cad_t1=float(losses['loss_cad_t1']),
                loss_cad_t2=float(losses['loss_cad_t2']),
                loss_cad_t3=float(losses['loss_cad_t3']),
                loss_cad_mask=float(losses['loss_cad_mask']),
                residual_loss=float(losses['loss_cad_residual']),
                representable=float(stats['cad_pred_route_representable']),
                residual_valid_points=int(stats['cad_pred_route_residual_valid_points']),
                anchor_error_mm=float(anchor_mm), full_error_mm=float(full_mm),
                residual_gain_mm=float(anchor_mm - full_mm),
                residual_norm_max=float(raw.norm(dim=1).max()),
                residual_saturated_fraction=float((raw.abs() >= .999).float().mean()))


def run(args):
    if not torch.cuda.is_available() or not args.device.startswith('cuda'):
        raise RuntimeError('Local CUDA unavailable; rerun the exact project command with authorized GPU access')
    cfg = Config.fromfile(str(args.config))
    _, digest, _ = inspect_config(cfg)
    if args.amp_scale is None:
        args.amp_scale = float(cfg.SOLVER.AMP.INIT_SCALE)
    cfg.MODEL.DEVICE = args.device
    cfg.SOLVER.IMS_PER_BATCH = cfg.SOLVER.REFERENCE_BS = args.batch_size
    seed_all(42)
    torch.set_num_threads(4)
    model, optimizer = build_model_optimizer(cfg)
    groups = audit_optimizer(model, optimizer, cfg)
    batch = real_batch(cfg, args.device, args.batch_size, args.renderer,
                       load_batch=args.load_batch, save_batch=args.save_batch)
    scaler = torch.cuda.amp.GradScaler(init_scale=args.amp_scale)
    initial = dict(backbone=next(model.backbone.parameters()).detach().clone(),
                   classifier=model.cad_attention_head.t3_classifier.weight.detach().clone(),
                   residual=model.cad_attention_head.residual_predictor.final.weight.detach().clone())
    history = [dict(step=0, **measure(model, batch))]
    seen_route_grad = seen_residual_grad = False
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            prediction = model.predict(batch['roi_img'], batch['roi_cls'])
            losses, stats = model.cad_attention_head.loss(prediction, batch['roi_cls'],
                batch['roi_xyz'], batch['roi_mask_visib'])
            total = sum(losses.values())
        amp_step(model, optimizer, scaler, total, step=step, losses=losses,
                 head=model.cad_attention_head)
        route_grad = model.cad_attention_head.t3_classifier.weight.grad
        residual_grad = model.cad_attention_head.residual_predictor.final.weight.grad
        seen_route_grad |= route_grad is not None and bool(route_grad.abs().sum() > 0)
        seen_residual_grad |= residual_grad is not None and bool(residual_grad.abs().sum() > 0)
        if step % (20 if args.mode == 'fixed' else 1) == 0 or step == args.steps:
            result = dict(step=step, amp_scale=float(scaler.get_scale()), **measure(model, batch))
            history.append(result)
            print(json.dumps(dict(arm=cfg.EXP026_ARM, **result)), flush=True)
            (args.output / 'report.json').write_text(json.dumps(dict(status='RUNNING', history=history), indent=2))
    if not seen_route_grad or not seen_residual_grad:
        raise RuntimeError('Route or residual predictor had no gradient')
    if torch.equal(initial['backbone'], next(model.backbone.parameters())) \
            or torch.equal(initial['classifier'], model.cad_attention_head.t3_classifier.weight) \
            or torch.equal(initial['residual'], model.cad_attention_head.residual_predictor.final.weight):
        raise RuntimeError('Backbone, classifier, or residual predictor did not update')
    if args.mode == 'fixed':
        tail = history[-3:]
        if history[-1]['route_loss'] >= history[0]['route_loss'] \
                or any(row['residual_valid_points'] == 0 for row in tail) \
                or sum(row['residual_gain_mm'] for row in tail) <= 0 \
                or any(row['residual_saturated_fraction'] > .05 for row in tail):
            raise RuntimeError('Fixed-batch route/residual learnability gate failed')
    checkpoint_status = 'covered_by_matched_smoke'
    if args.mode == 'smoke':
        checkpoint = args.output / 'checkpoint_roundtrip.pth'
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        gradscaler=scaler.state_dict()), checkpoint)
        restored = torch.load(checkpoint, map_location=args.device)
        model.load_state_dict(restored['model'], strict=True)
        optimizer.load_state_dict(restored['optimizer'])
        scaler.load_state_dict(restored['gradscaler'])
        checkpoint_status = 'PASS'
    report = dict(status='PASS', mode=args.mode, arm=cfg.EXP026_ARM,
                  hierarchy_sha256=digest, config=str(args.config), source_batch=str(args.load_batch or 'online'),
                  batch_size=args.batch_size, steps=args.steps, amp_init_scale=args.amp_scale,
                  optimizer_groups=groups, route_gradient=seen_route_grad,
                  residual_gradient=seen_residual_grad, checkpoint_roundtrip=checkpoint_status, history=history)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--renderer', choices=('cpp', 'egl'), default='cpp')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--amp-scale', type=float)
    parser.add_argument('--mode', choices=('smoke', 'fixed'), default='smoke')
    parser.add_argument('--load-batch', type=Path)
    parser.add_argument('--save-batch', type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.steps < 2 or (args.amp_scale is not None and args.amp_scale < 1):
        parser.error('Require positive batch, scale, and at least two steps')
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except Exception as exc:
        failure = dict(status='FAIL', error=str(exc),
                       telemetry=getattr(exc, 'telemetry', None))
        (args.output / 'failure.json').write_text(json.dumps(failure, indent=2))
        raise


if __name__ == '__main__':
    main()
