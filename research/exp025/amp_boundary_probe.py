"""One step FP32 versus scaled FP16, from the same initialization and the same batch.

`real_smoke --amp-scale` finds the scale at which a run survives; this probe says why it
does not survive above it.  Both arms start from seed 42 and the same saved batch, so the
only difference is the GradScaler: the FP32 arm reports the true gradient magnitude, the
AMP arm reports what survives the fp16 backward at the requested scale.  A module whose
`scale * abs_max` exceeds the fp16 maximum (65504) is the one that overflows -- a scaling
boundary, not a divergent loss.  Run the same command in a worktree of an older commit to
compare two structures under identical input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from core.gdrn_modeling.models.GDRN_CAD import build_model_optimizer
from .preflight import CONFIG, read_config
from .runtime import metadata, real_batch, save_report, seed_all

FP16_MAX = 65504.


def one_step(cfg, batch, scale):
    """Forward/backward once; `scale=None` is the FP32 control."""
    seed_all(42)
    model, optimizer = build_model_optimizer(cfg)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=scale is not None):
        _, losses = model(batch['roi_img'], roi_classes=batch['roi_cls'], gt_xyz=batch['roi_xyz'],
                          gt_mask_visib=batch['roi_mask_visib'], do_loss=True)
        total = sum(losses.values())
    if scale is None:
        total.backward()
    else:
        scaler = torch.cuda.amp.GradScaler(init_scale=scale)
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
    groups = {}
    for name, module in model.cad_attention_head.named_children():
        gradients = [p.grad.detach() for p in module.parameters() if p.grad is not None]
        if not gradients:
            continue
        magnitude = max(float(g.abs().max()) for g in gradients)
        groups[name] = dict(abs_max=magnitude, finite=all(bool(torch.isfinite(g).all()) for g in gradients))
    parameters = {name: value for name, value in model.cad_attention_head.named_parameters()}
    return dict(losses={name: float(value.detach()) for name, value in losses.items()},
                total_loss=float(total.detach()), groups=groups,
                head_parameters=sum(p.numel() for p in parameters.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--load-batch', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--amp-scale', type=float, default=65536.,
                        help='scale whose fp16 backward the AMP arm must survive')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.batch_size < 1 or args.amp_scale < 1:
        parser.error('Require a positive batch size and scale')
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = read_config(args.config, False, 'official_lmo')
    cfg.MODEL.DEVICE = args.device
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    report = dict(**metadata(cfg), run_id=args.output.name, status='RUNNING', batch_size=args.batch_size,
                  source_batch=str(args.load_batch), amp_scale=args.amp_scale, fp16_max=FP16_MAX,
                  interpretation='one-step numerical boundary probe, not a performance result')
    save_report(args.output, report)
    try:
        if not torch.cuda.is_available() or not args.device.startswith('cuda'):
            raise RuntimeError('BLOCKED: the AMP boundary probe requires CUDA')
        torch.set_num_threads(4)
        batch = real_batch(cfg, args.device, args.batch_size, 'cpp', load_batch=args.load_batch)
        report['fp32'] = one_step(cfg, batch, None)
        torch.cuda.empty_cache()
        report['amp'] = one_step(cfg, batch, args.amp_scale)
        report['overflowed'] = sorted(
            f"{name}:{args.amp_scale * entry['abs_max']:.0f}"
            for name, entry in report['fp32']['groups'].items()
            if args.amp_scale * entry['abs_max'] > FP16_MAX)
        report['non_finite_under_amp'] = sorted(
            name for name, entry in report['amp']['groups'].items() if not entry['finite'])
        report['status'] = 'PASS'
    except Exception as exc:
        report.update(status='BLOCKED' if str(exc).startswith('BLOCKED:') else 'FAIL', error=str(exc))
        raise
    finally:
        save_report(args.output, report)
    print(json.dumps({key: report[key] for key in
                      ('status', 'amp_scale', 'fp32', 'amp', 'overflowed', 'non_finite_under_amp')},
                     indent=2, default=str))


if __name__ == '__main__':
    main()
