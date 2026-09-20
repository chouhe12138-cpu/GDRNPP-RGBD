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
from .runtime import (NonFiniteTrainingError, seed_all, real_batch, amp_init_scale, amp_step,
                      metadata, save_last_good, save_report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-backbone', choices=('yes', 'no'))
    parser.add_argument('--backbone-init', choices=('official_lmo', 'imagenet'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--renderer', choices=('cpp', 'egl'), default='cpp')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--amp-scale', type=float, default=None,
                        help='initial GradScaler scale; defaults to SOLVER.AMP.INIT_SCALE '
                             'when the config pins one, otherwise 65536')
    parser.add_argument('--load-batch', type=Path)
    parser.add_argument('--save-batch', type=Path)
    args = parser.parse_args()
    if args.steps < 2 or args.batch_size < 1:
        parser.error('Require steps >= 2 and positive batch size')
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = read_config(args.config,
                      None if args.train_backbone is None else args.train_backbone == 'yes',
                      args.backbone_init)
    args.amp_scale = amp_init_scale(cfg, args.amp_scale)
    report = dict(**metadata(cfg), run_id=args.output.name, status='RUNNING',
                  batch_source=str(args.load_batch or 'online'), renderer=args.renderer,
                  batch_size=args.batch_size, steps=args.steps, device=args.device,
                  amp_init_scale=args.amp_scale)
    save_report(args.output, report)
    try:
        if not torch.cuda.is_available() or not args.device.startswith('cuda'):
            raise RuntimeError('BLOCKED: real AMP smoke requires CUDA')
        seed_all(42)
        torch.set_num_threads(4)
        cfg.MODEL.DEVICE = args.device
        # This diagnostic runs its own local batch; the formal config stays at batch 48.
        cfg.SOLVER.IMS_PER_BATCH = cfg.SOLVER.REFERENCE_BS = args.batch_size
        model, optimizer = build_model_optimizer(cfg)
        model.train()
        report['optimizer_groups'] = audit_optimizer(model, optimizer, cfg)
        batch = real_batch(cfg, args.device, args.batch_size, args.renderer, args.load_batch, args.save_batch)
        frozen = {k: v.detach().cpu().clone() for k, v in model.backbone.state_dict().items()} if not cfg.TRAIN_BACKBONE else None
        initial_backbone = next(model.backbone.parameters()).detach().clone()
        initial_head = model.cad_attention_head.t3_classifier.weight.detach().clone()
        initial_stages = {name: value.detach().clone()
                          for name, value in model.cad_attention_head.stages.named_parameters()}
        scaler = torch.cuda.amp.GradScaler(init_scale=args.amp_scale)
        history, timings = [], []
        report.update(losses=history, timings=timings)
        torch.cuda.reset_peak_memory_stats()
        last_good = args.output / 'last_good.pth'
        for step in range(args.steps):
            report['current_step'] = step + 1
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            diagnostics = {}
            with torch.cuda.amp.autocast():
                _, losses = model(batch['roi_img'], roi_classes=batch['roi_cls'], gt_xyz=batch['roi_xyz'],
                                  gt_mask_visib=batch['roi_mask_visib'], do_loss=True, diagnostics=diagnostics)
                total = sum(losses.values())
            torch.cuda.synchronize()
            forward_end = time.perf_counter()
            amp_step(model, optimizer, scaler, total, step=step + 1, losses=losses,
                     diagnostics=diagnostics, head=model.cad_attention_head)
            torch.cuda.synchronize()
            timings.append(dict(forward_ms=(forward_end-start)*1000,
                                backward_and_update_ms=(time.perf_counter()-forward_end)*1000))
            # After the timing capture: an 8-step smoke may fail at any step, so this stays
            # per-step, but the checkpoint write must not be reported as step time.
            save_last_good(last_good, model, optimizer, scaler, step + 1, dict(kind='real_smoke'))
            history.append({k: float(v.detach()) for k, v in losses.items()})
        if torch.equal(initial_head, model.cad_attention_head.t3_classifier.weight):
            raise RuntimeError('Classifier did not update')
        if frozen is not None:
            if any(not torch.equal(v.cpu(), frozen[k]) for k, v in model.backbone.state_dict().items()):
                raise RuntimeError('Frozen backbone changed')
        elif torch.equal(initial_backbone, next(model.backbone.parameters())):
            raise RuntimeError('Trainable backbone did not update')
        # Every image-branch stage must participate in training. Requiring every
        # individual tensor to change bitwise is invalid for a short low-LR smoke:
        # small norm gradients can be below one fp32 update quantum.
        stages_now = dict(model.cad_attention_head.stages.named_parameters())
        updated = [name for name, value in initial_stages.items() if not torch.equal(value, stages_now[name])]
        stage_updates, stage_gradients = {}, {}
        for index, stage in enumerate(model.cad_attention_head.stages):
            prefix = f'{index}.'
            stage_updates[str(index)] = sum(name.startswith(prefix) for name in updated)
            gradients = [parameter.grad for parameter in stage.parameters()
                         if parameter.grad is not None]
            stage_gradients[str(index)] = bool(gradients) and all(
                torch.isfinite(gradient).all() for gradient in gradients) and any(
                gradient.abs().max() > 0 for gradient in gradients)
        if any(count == 0 for count in stage_updates.values()) or not all(stage_gradients.values()):
            raise RuntimeError(f'Image stage training coverage failed: updates={stage_updates}, '
                               f'gradients={stage_gradients}')
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
            last_good=dict(step=args.steps, path=last_good.name),
            step_median_ms=statistics.median(sum(t.values()) for t in timings[2:] or timings),
            peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9,
            peak_reserved_gb=torch.cuda.max_memory_reserved()/1e9,
            total_parameters=sum(p.numel() for p in model.parameters()), checkpoint=checkpoint.name,
            image_stages=dict(total_parameters=len(initial_stages), updated_parameters=len(updated),
                              updates_by_stage=stage_updates,
                              finite_nonzero_gradient_by_stage=stage_gradients),
            timing_scope='fixed batch; excludes repeated loader/render', checkpoint_roundtrip=True)
    except NonFiniteTrainingError as exc:
        report.update(status='FAIL', error=str(exc), failure=exc.telemetry)
        raise
    except Exception as exc:
        report.update(status='BLOCKED' if str(exc).startswith('BLOCKED:') else 'FAIL', error=str(exc))
        raise
    finally:
        save_report(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
