"""EXP027 real-batch AMP gate and fixed-batch learnability diagnostic."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from mmcv import Config

from core.gdrn_modeling.models.GDRN_CAD import build_model_optimizer
from research.cad_common.preflight import audit_optimizer
from research.cad_common.runtime import grad_norm_stats, real_batch, seed_all
from research.exp026.local_validation import measure
from .preflight import inspect_config


def run(args):
    cfg = Config.fromfile(str(args.config))
    baseline = str(cfg.EXPERIMENT_ID) == 'EXP-20260922-026-residual-aligned-sampling-ablation'
    if baseline:
        from research.exp026.preflight import inspect_config as inspect_exp026
        _, hierarchy_sha, _ = inspect_exp026(cfg)
    else:
        _, hierarchy_sha = inspect_config(cfg)
    if not torch.cuda.is_available() or not args.device.startswith('cuda'):
        raise RuntimeError('EXP027 real-batch gate requires CUDA')
    scale = float(args.amp_scale if args.amp_scale is not None else cfg.SOLVER.AMP.INIT_SCALE)
    if scale < 1:
        raise ValueError('AMP scale must be positive')
    seed_all(42)
    torch.set_num_threads(4)
    cfg.MODEL.DEVICE = args.device
    cfg.SOLVER.IMS_PER_BATCH = cfg.SOLVER.REFERENCE_BS = args.batch_size
    model, optimizer = build_model_optimizer(cfg)
    groups = audit_optimizer(model, optimizer, cfg)
    batch = real_batch(cfg, args.device, args.batch_size, args.renderer,
                       load_batch=args.load_batch, save_batch=args.save_batch)
    head = model.cad_attention_head
    tracked = (['stages'] if baseline else ['laterals', 'lateral_alpha']) + [
        'cross_attention', 'residual_predictor', 'mask_predictor']
    tracked += ['t3_classifier'] if hasattr(head, 't3_classifier') else [
        'query_parents', 'pixel_projection', 'query_projection']
    before_params = {name: {key: value.detach().clone() for key, value in
                    getattr(head, name).named_parameters()} if name != 'lateral_alpha' else
                    {'value': head.lateral_alpha.detach().clone()} for name in tracked}
    gradients = {name: False for name in tracked}
    timings, history = [], []
    scaler = torch.cuda.amp.GradScaler(init_scale=scale)
    torch.cuda.reset_peak_memory_stats()
    history.append(dict(step=0, **measure(model, batch)))
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        begin = time.perf_counter()
        with torch.cuda.amp.autocast():
            _, losses = model(batch['roi_img'], roi_classes=batch['roi_cls'],
                              gt_xyz=batch['roi_xyz'], gt_mask_visib=batch['roi_mask_visib'], do_loss=True)
            total = sum(losses.values())
        torch.cuda.synchronize()
        forward_end = time.perf_counter()
        if not torch.isfinite(total):
            raise RuntimeError(f'Non-finite loss at step {step}')
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        audit = grad_norm_stats(model)
        if audit['non_finite_parameters']:
            raise RuntimeError(f'Non-finite gradients at step {step}: {audit["non_finite_parameters"][:8]}')
        for name in tracked:
            item = getattr(head, name)
            params = (item,) if isinstance(item, torch.nn.Parameter) else tuple(item.parameters())
            gradients[name] |= any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in params)
        torch.cuda.synchronize()
        backward_end = time.perf_counter()
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < old_scale:
            raise RuntimeError(f'AMP skipped step {step}')
        torch.cuda.synchronize()
        finish = time.perf_counter()
        timings.append(dict(forward_ms=1000*(forward_end-begin),
                            backward_ms=1000*(backward_end-forward_end),
                            optimizer_ms=1000*(finish-backward_end), full_step_ms=1000*(finish-begin)))
        if step % (20 if args.mode == 'fixed' else 1) == 0 or step == args.steps:
            history.append(dict(step=step, **measure(model, batch)))
    updates = {}
    for name, old in before_params.items():
        item = getattr(head, name)
        current = {'value': item} if name == 'lateral_alpha' else dict(item.named_parameters())
        updates[name] = sum(not torch.equal(value, current[key]) for key, value in old.items())
    if not all(gradients.values()) or not all(updates.values()):
        raise RuntimeError(f'EXP027 trainability failed: gradients={gradients}, updates={updates}')
    if args.mode == 'fixed':
        if history[-1]['route_loss'] >= history[0]['route_loss'] or \
                any(row['residual_valid_points'] == 0 for row in history[-3:]):
            raise RuntimeError('EXP027 fixed-batch route/residual learnability failed')
    checkpoint = 'not_required_for_fixed_batch'
    if args.mode == 'smoke':
        checkpoint_path = args.output / 'checkpoint_roundtrip.pth'
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        gradscaler=scaler.state_dict()), checkpoint_path)
        restored = torch.load(checkpoint_path, map_location=args.device)
        model.load_state_dict(restored['model'], strict=True)
        optimizer.load_state_dict(restored['optimizer'])
        scaler.load_state_dict(restored['gradscaler'])
        checkpoint = 'PASS'
    steady = timings[1:] if len(timings) > 1 else timings
    report = dict(status='PASS', arm=cfg.EXP026_ARM if baseline else cfg.EXP027_ARM,
                  hierarchy_sha256=hierarchy_sha,
                  config=str(args.config), mode=args.mode, batch_size=args.batch_size,
                  steps=args.steps, amp_init_scale=scale, amp_skipped_steps=0,
                  optimizer_groups=groups, gradients=gradients, parameter_updates=updates,
                  checkpoint_roundtrip=checkpoint, history=history,
                  timing_median_ms={key: statistics.median(row[key] for row in steady)
                                    for key in steady[0]},
                  peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9,
                  peak_reserved_gb=torch.cuda.max_memory_reserved()/1e9,
                  lateral_alpha=[] if baseline else [float(v) for v in head.lateral_alpha.detach().cpu()])
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
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
    if args.steps < 2 or args.batch_size < 1:
        parser.error('Require at least two steps and positive batch size')
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        print(json.dumps(run(args), indent=2), flush=True)
    except Exception as exc:
        (args.output / 'failure.json').write_text(json.dumps(dict(status='FAIL', error=str(exc)), indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
