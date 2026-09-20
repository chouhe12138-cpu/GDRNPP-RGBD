"""Real batch AMP smoke, optimizer update and checkpoint round-trip."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from core.gdrn_modeling.models.GDRN_CAD import build_model_optimizer
from core.gdrn_modeling.models.heads.hierarchical_cad_attention_head import hierarchy_log_probabilities
from .preflight import CONFIG, read_config, audit_optimizer
from .runtime import seed_all, real_batch, amp_step, metadata, save_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-backbone', choices=('yes', 'no'), default='no')
    parser.add_argument('--backbone-init', choices=('official_lmo', 'imagenet'), default='official_lmo')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--renderer', choices=('cpp', 'egl'), default='cpp')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--load-batch', type=Path)
    parser.add_argument('--save-batch', type=Path)
    args = parser.parse_args()
    if args.steps < 2 or args.batch_size < 1:
        parser.error('Require steps >= 2 and positive batch size')
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = read_config(args.config, args.train_backbone == 'yes', args.backbone_init)
    report = dict(**metadata(cfg), run_id=args.output.name, status='RUNNING',
                  batch_source=str(args.load_batch or 'online'), renderer=args.renderer,
                  batch_size=args.batch_size, steps=args.steps, device=args.device)
    save_report(args.output, report)
    try:
        if not torch.cuda.is_available() or not args.device.startswith('cuda'):
            raise RuntimeError('BLOCKED: real AMP smoke requires CUDA')
        seed_all(42)
        torch.set_num_threads(4)
        cfg.MODEL.DEVICE = args.device
        model, optimizer = build_model_optimizer(cfg)
        model.train()
        report['optimizer_groups'] = audit_optimizer(model, optimizer, cfg)
        batch = real_batch(cfg, args.device, args.batch_size, args.renderer, args.load_batch, args.save_batch)
        frozen = {k: v.detach().cpu().clone() for k, v in model.backbone.state_dict().items()} if not cfg.TRAIN_BACKBONE else None
        initial_backbone = next(model.backbone.parameters()).detach().clone()
        initial_head = model.cad_attention_head.t3_classifier.weight.detach().clone()
        scaler = torch.cuda.amp.GradScaler()
        history, timings = [], []
        report.update(losses=history, timings=timings)
        torch.cuda.reset_peak_memory_stats()
        for step in range(args.steps):
            report['current_step'] = step + 1
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.cuda.amp.autocast():
                _, losses = model(batch['roi_img'], roi_classes=batch['roi_cls'], gt_xyz=batch['roi_xyz'],
                                  gt_mask_visib=batch['roi_mask_visib'], do_loss=True)
                total = sum(losses.values())
            torch.cuda.synchronize()
            forward_end = time.perf_counter()
            amp_step(model, optimizer, scaler, total)
            torch.cuda.synchronize()
            timings.append(dict(forward_ms=(forward_end-start)*1000,
                                backward_and_update_ms=(time.perf_counter()-forward_end)*1000))
            history.append({k: float(v.detach()) for k, v in losses.items()})
        if torch.equal(initial_head, model.cad_attention_head.t3_classifier.weight):
            raise RuntimeError('Classifier did not update')
        if frozen is not None:
            if any(not torch.equal(v.cpu(), frozen[k]) for k, v in model.backbone.state_dict().items()):
                raise RuntimeError('Frozen backbone changed')
        elif torch.equal(initial_backbone, next(model.backbone.parameters())):
            raise RuntimeError('Trainable backbone did not update')
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            out = model(batch['roi_img'], roi_classes=batch['roi_cls'], return_cad_debug=True)
            if not torch.isfinite(out['xyz_norm']).all() or out['residual'].norm(dim=1).max() > 1.00001:
                raise RuntimeError('Invalid decode')
            for lp in hierarchy_log_probabilities(out['t3_logits']):
                torch.testing.assert_close(lp.exp().sum(1), torch.ones_like(lp[:, 0]), atol=1e-6, rtol=1e-5)
        checkpoint = args.output / 'smoke_checkpoint.pth'
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), gradscaler=scaler.state_dict(), iteration=args.steps-1), checkpoint)
        restored = torch.load(checkpoint, map_location=args.device)
        model.load_state_dict(restored['model'], strict=True)
        optimizer.load_state_dict(restored['optimizer'])
        scaler.load_state_dict(restored['gradscaler'])
        with torch.no_grad(), torch.cuda.amp.autocast():
            after = model(batch['roi_img'], roi_classes=batch['roi_cls'], return_cad_debug=True)
        torch.testing.assert_close(after['t3_logits'], out['t3_logits'], rtol=0, atol=0)
        report.update(status='PASS', losses=history, timings=timings, amp_skipped_steps=0,
            step_median_ms=statistics.median(sum(t.values()) for t in timings[2:] or timings),
            peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9,
            peak_reserved_gb=torch.cuda.max_memory_reserved()/1e9,
            total_parameters=sum(p.numel() for p in model.parameters()), checkpoint=checkpoint.name,
            timing_scope='fixed batch; excludes repeated loader/render', checkpoint_roundtrip=True)
    except Exception as exc:
        report.update(status='BLOCKED' if str(exc).startswith('BLOCKED:') else 'FAIL', error=str(exc))
        raise
    finally:
        save_report(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
