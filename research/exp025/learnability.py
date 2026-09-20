"""Matched fixed-batch residual-only and full marginal-classification diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from core.gdrn_modeling.models.GDRN_CAD import build_model_optimizer
from core.gdrn_modeling.models.heads.hierarchical_cad_attention_head import hierarchy_log_probabilities, masked_mean
from .preflight import CONFIG, read_config
from .runtime import (NonFiniteTrainingError, seed_all, real_batch, amp_init_scale, amp_step,
                      head_telemetry, metadata, save_last_good, save_report)


@torch.no_grad()
def measure(model, batch, route_weight):
    model.eval()
    diagnostics = {}
    with torch.cuda.amp.autocast():
        pred = model.predict(batch['roi_img'], batch['roi_cls'], diagnostics=diagnostics)
    return measure_prediction(model.cad_attention_head, batch, pred, route_weight, diagnostics)


@torch.no_grad()
def measure_prediction(head, batch, pred, route_weight, diagnostics=None):
    """Metrics for one prediction dict, shared by the residual-only and probe arms."""
    losses, stats, ids, valid = head.loss(pred, batch['roi_cls'], batch['roi_xyz'],
                                       batch['roi_mask_visib'], route_weight=route_weight, return_targets=True)
    # Geometry is evaluated against the same instance-consistent branch chosen by this arm.
    branches = []
    for branch in range(min(2, head.symmetry_transforms.shape[1])):
        path, target, _ = head.targets(batch['roi_xyz'], batch['roi_mask_visib'], batch['roi_cls'], branch)
        branches.append((path[2], target))
    # Recompute branch selection exactly through supervised losses, not by ID matching.
    lp = hierarchy_log_probabilities(pred['t3_logits'])
    residual = pred['residual'].flatten(2).transpose(1, 2)
    scores = []
    for id3, target in branches:
        route = sum(masked_mean(torch.nn.functional.nll_loss(logp.flatten(2), label, reduction='none'), valid)
                    for logp, label in zip(lp, (id3//64, id3//8, id3)))
        res = masked_mean(torch.nn.functional.smooth_l1_loss(residual, target, beta=.1,
                          reduction='none').mean(-1), valid)
        scores.append(route_weight*route + res)
    scores = torch.stack(scores, 1)
    allowed = torch.arange(len(branches), device=ids.device)[None] < head.symmetry_counts[batch['roi_cls'], None]
    choice = scores.masked_fill(~allowed, float('inf')).argmin(1)
    target = torch.stack([t for _, t in branches], 1)[torch.arange(len(ids), device=ids.device), choice]
    anchor = head.level3_anchors[batch['roi_cls'][:, None], ids.flatten(1)]
    radius = head.level3_radii[batch['roi_cls'][:, None], ids.flatten(1)]
    target_xyz = anchor + radius[..., None]*target
    clipped = target / target.norm(dim=-1, keepdim=True).clamp_min(1.)
    metrics = {k: float(v.detach()) for k, v in losses.items()}
    metrics.update({k: float(v) for k, v in stats.items()})
    for index, (logp, label) in enumerate(zip(lp, (ids//64, ids//8, ids)), 1):
        metrics[f't{index}_marginal_accuracy'] = float(masked_mean(
            (logp.argmax(1) == label).flatten(1).float(), valid).mean())
        metrics[f't{index}_nll'] = float(masked_mean(torch.nn.functional.nll_loss(
            logp.flatten(2), label.flatten(1), reduction='none'), valid).mean())
    predicted = pred['t3_logits'].argmax(1)
    for depth, divisor in ((1, 64), (2, 8)):
        metrics[f'predicted_t3_ancestor_t{depth}_accuracy'] = float(masked_mean(
            (predicted//divisor == ids//divisor).flatten(1).float(), valid).mean())
    for name, decode_ids in (('gt_path', ids), ('predicted_path', None)):
        xyz = head.decode(pred, batch['roi_cls'], decode_ids).flatten(2).transpose(1, 2)
        xyz = (xyz-.5)*head.extents[batch['roi_cls'], None]
        metrics[f'{name}_xyz_error_mm'] = float(masked_mean((xyz-target_xyz).norm(dim=-1)*1000, valid).mean())
    metrics['oracle_error_mm'] = float(masked_mean((radius[..., None]*(clipped-target)).norm(dim=-1)*1000, valid).mean())
    if diagnostics:
        metrics.update(head_telemetry(head, diagnostics))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--load-batch', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--last-good-period', type=int, default=20,
                        help='steps between last-good checkpoints used for matched replay')
    parser.add_argument('--amp-scale', type=float, default=None,
                        help='initial GradScaler scale; defaults to SOLVER.AMP.INIT_SCALE '
                             'when the config pins one, otherwise 65536.  The current '
                             'structure can overflow at 65536 on a single-step batch4 path -- '
                             'see amp_boundary_probe for the measured boundary')
    parser.add_argument('--arms', default='residual_only,full')
    args = parser.parse_args()
    weights = dict(residual_only=0., full=1.)
    arms = tuple((name, weights[name]) for name in args.arms.split(',') if name in weights)
    if len(arms) != len(set(args.arms.split(','))) or args.steps < 1 or args.batch_size < 1 \
            or args.last_good_period < 1:
        parser.error(f'Require positive steps/batch size/last-good period and arms within {tuple(weights)}')
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = read_config(args.config, False, 'official_lmo')
    cfg.MODEL.DEVICE = args.device
    args.amp_scale = amp_init_scale(cfg, args.amp_scale)
    if args.amp_scale < 1:
        parser.error(f'Require a positive amp scale, got {args.amp_scale}')
    report = dict(**metadata(cfg), run_id=args.output.name, status='RUNNING', steps=args.steps,
                  batch_size=args.batch_size, source_batch=str(args.load_batch), arms={},
                  last_good_period=args.last_good_period, amp_init_scale=args.amp_scale,
                  schedule='constant 3e-4, no formal warmup', interpretation='fixed-batch diagnostic, not generalization')
    save_report(args.output, report)
    try:
        if not torch.cuda.is_available() or not args.device.startswith('cuda'):
            raise RuntimeError('BLOCKED: P2 requires CUDA')
        torch.set_num_threads(4)
        batch = real_batch(cfg, args.device, args.batch_size, 'cpp', load_batch=args.load_batch)
        for name, weight in arms:
            seed_all(42)
            model, optimizer = build_model_optimizer(cfg)
            scaler = torch.cuda.amp.GradScaler(init_scale=args.amp_scale)
            history = [dict(step=0, **measure(model, batch, weight))]
            report['arms'][name] = history
            last_good = args.output / f'{name}_last_good.pth'
            save_last_good(last_good, model, optimizer, scaler, 0, dict(arm=name, route_weight=weight))
            report['arms'][name + '_last_good'] = dict(step=0, path=last_good.name)
            for step in range(1, args.steps+1):
                report.update(current_arm=name, current_step=step)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                diagnostics = {}
                with torch.cuda.amp.autocast():
                    pred = model.predict(batch['roi_img'], batch['roi_cls'], diagnostics=diagnostics)
                    losses, _ = model.cad_attention_head.loss(pred, batch['roi_cls'], batch['roi_xyz'],
                        batch['roi_mask_visib'], route_weight=weight)
                amp_step(model, optimizer, scaler, sum(losses.values()), step=step,
                         losses=losses, diagnostics=diagnostics, head=model.cad_attention_head)
                if step % args.last_good_period == 0 and step < args.steps:
                    save_last_good(last_good, model, optimizer, scaler, step, dict(arm=name, route_weight=weight))
                    report['arms'][name + '_last_good'] = dict(step=step, path=last_good.name)
                if step % 20 == 0 or step == args.steps:
                    history.append(dict(step=step, amp_scale=float(scaler.get_scale()),
                                        **measure(model, batch, weight)))
                    save_report(args.output, report)
                    print(json.dumps(dict(arm=name, **history[-1])), flush=True)
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), iteration=args.steps-1),
                       args.output / f'{name}_diagnostic.pth')
            del model, optimizer, scaler, pred, losses
            torch.cuda.empty_cache()
        report.update(status='COMPLETE', amp_skipped_steps=0)
    except NonFiniteTrainingError as exc:
        # Keep the completed history and the failing step's telemetry; the last-good
        # checkpoint on disk is the state the replay starts from.
        report.update(status='FAIL', error=str(exc), failure=exc.telemetry,
                      failure_arm=report.get('current_arm'), amp_skipped_steps=0)
        raise
    except Exception as exc:
        report.update(status='BLOCKED' if str(exc).startswith('BLOCKED:') else 'FAIL', error=str(exc))
        raise
    finally:
        save_report(args.output, report)


if __name__ == '__main__':
    main()
