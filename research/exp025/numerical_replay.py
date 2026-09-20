"""Matched AMP / low-scale AMP / FP32 replay from a recorded last-good state.

Answers whether the non-finite gradient is an AMP precision boundary or survives in
FP32, and which parameters go bad first.  Replay only: no loss, model or protocol change.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from core.gdrn_modeling.models.GDRN_CAD import build_model_optimizer
from .preflight import CONFIG, read_config
from .runtime import (NonFiniteTrainingError, amp_step, grad_norm_stats,
                      metadata, real_batch, save_report)

ARMS = ('amp_current', 'amp_low_scale', 'fp32')


def restore_rng(rng):
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch'])
    if rng['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng['cuda'])


def arm_report(name, cfg, args, batch, checkpoint):
    seed_state = torch.load(checkpoint, map_location='cpu')
    model, optimizer = build_model_optimizer(cfg)
    model.load_state_dict(seed_state['model'], strict=True)
    optimizer.load_state_dict(seed_state['optimizer'])
    record = dict(arm=name, start_step=int(seed_state['step']), status='RUNNING', steps=[],
                  amp_scale=None, non_finite_parameters=[], first_bad_step=None)
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    restore_rng(seed_state['rng'])
    scaler = None
    if name != 'fp32':
        scale = float(seed_state['gradscaler']['scale']) if name == 'amp_current' else float(args.scale)
        scaler = torch.cuda.amp.GradScaler(init_scale=scale)
        record['amp_scale'] = scale
    try:
        for offset in range(1, args.steps + 1):
            step = record['start_step'] + offset
            model.train()
            optimizer.zero_grad(set_to_none=True)
            diagnostics = {}
            with torch.cuda.amp.autocast(enabled=name != 'fp32'):
                _, losses = model(batch['roi_img'], roi_classes=batch['roi_cls'], gt_xyz=batch['roi_xyz'],
                                  gt_mask_visib=batch['roi_mask_visib'], do_loss=True, diagnostics=diagnostics)
                total = sum(losses.values())
            amp_step(model, optimizer, scaler, total, step=step, losses=losses,
                     diagnostics=diagnostics, head=model.cad_attention_head)
            stats = grad_norm_stats(model)
            record['steps'].append(dict(step=step, total_loss=float(total.detach()),
                                        grad=stats['grad_norm'], grad_abs_max=stats['grad_abs_max']))
    except NonFiniteTrainingError as exc:
        record.update(status='NON_FINITE', first_bad_step=exc.telemetry.get('step'),
                      error=str(exc), failure=exc.telemetry)
    except RuntimeError as exc:
        record.update(status='ERROR', error=f'{type(exc).__name__}: {exc}')
    else:
        record['status'] = 'FINITE'
    if record['steps']:
        record['last_finite_step'] = record['steps'][-1]['step']
        record['last_finite_total_loss'] = record['steps'][-1]['total_loss']
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--from-checkpoint', type=Path, required=True)
    parser.add_argument('--load-batch', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--scale', type=float, default=1024.)
    parser.add_argument('--arms', default=','.join(ARMS))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    arms = tuple(args.arms.split(','))
    if args.steps < 1 or not set(arms) <= set(ARMS):
        parser.error(f'Require positive steps and arms within {ARMS}')
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = read_config(args.config, False, 'official_lmo')
    cfg.MODEL.DEVICE = args.device
    report = dict(**metadata(cfg), run_id=args.output.name, status='RUNNING', steps=args.steps,
                  batch_size=args.batch_size, source_batch=str(args.load_batch),
                  from_checkpoint=str(args.from_checkpoint), arms={},
                  interpretation='matched replay from a recorded last-good state; not generalization')
    save_report(args.output, report)
    try:
        if not torch.cuda.is_available() or not args.device.startswith('cuda'):
            raise RuntimeError('BLOCKED: numerical replay requires CUDA')
        torch.set_num_threads(4)
        batch = real_batch(cfg, args.device, args.batch_size, 'cpp', load_batch=args.load_batch)
        for name in arms:
            report['current_arm'] = name
            save_report(args.output, report)
            report['arms'][name] = arm_report(name, cfg, args, batch, args.from_checkpoint)
            print(json.dumps(report['arms'][name]), flush=True)
        report['status'] = 'COMPLETE'
    except Exception as exc:
        report.update(status='BLOCKED' if str(exc).startswith('BLOCKED:') else 'FAIL', error=str(exc))
        raise
    finally:
        save_report(args.output, report)
    print(json.dumps(dict(status=report['status'],
                          arms={k: v['status'] for k, v in report['arms'].items()}), indent=2))


if __name__ == '__main__':
    main()
