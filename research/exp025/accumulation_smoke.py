"""Formal-path accumulation smoke: physical batch 4 x accumulation 12, AMP, resume.

`smoke.py` runs REFERENCE_BS=4, so it never exercises the shipped accumulation shape.
This state-machine smoke repeats one saved batch (or the online loader with --online) and
checks the counters and the LR a resumed run must continue -- optimizer step, scheduler
epoch, accumulation boundaries, AMP scale -- against an uninterrupted run.  Tensor
equality across sessions is not available on this GPU path, so it is judged against a
noise floor measured in the same process instead.  Engineering behaviour only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from pytorch_lightning.lite import LightningLite

from core.gdrn_modeling.models.GDRN_CAD import build_model_optimizer
from core.utils import my_checkpoint, solver_utils
from core.utils.my_checkpoint import MyCheckpointer
from .preflight import CONFIG, read_config
from .runtime import metadata, real_batch, save_report, seed_all


class _Lite(LightningLite):
    def run(self):
        pass


def update_count(optimizer):
    return sorted({int(state['step']) for state in optimizer.state.values() if 'step' in state})


def unwrapped_state(model):
    return {key.replace('_module.', ''): value for key, value in model.state_dict().items()}


def make_session(cfg, args, total_updates, resume):
    (args.output / 'checkpoints').mkdir(parents=True, exist_ok=True)  # as engine.do_train does
    lite = _Lite(accelerator='gpu', devices=1, precision=16 if args.amp else 32)
    model, optimizer = build_model_optimizer(cfg)
    model, wrapper = lite.setup(model, optimizer)
    state_optimizer = my_checkpoint.unwrap_optimizer_for_checkpoint(wrapper)
    scheduler = solver_utils.build_lr_scheduler(cfg, state_optimizer, total_iters=total_updates)
    scaler = getattr(lite._precision_plugin, 'scaler', None)
    checkpointables = dict(optimizer=state_optimizer, scheduler=scheduler)
    if scaler is not None:
        checkpointables['gradscaler'] = scaler
    checkpointer = MyCheckpointer(model, str(args.output / 'checkpoints'), **checkpointables)
    state = checkpointer.resume_or_load('', resume=resume)
    my_checkpoint.resync_wrapped_optimizer(wrapper)
    return dict(lite=lite, model=model, wrapper=wrapper, optimizer=state_optimizer,
                scheduler=scheduler, scaler=scaler, checkpointer=checkpointer, state=state)


def run_iterations(session, cfg, args, batch, start_iter, iterations, accumulate, iters_per_epoch,
                   max_iter, trace):
    # engine.do_train zeroes once before the loop and after every optimizer step
    session['wrapper'].zero_grad(set_to_none=True)
    for iteration in range(start_iter, start_iter + iterations):
        with torch.cuda.amp.autocast(enabled=args.amp):
            _, losses = session['model'](
                batch['roi_img'], roi_classes=batch['roi_cls'], gt_xyz=batch['roi_xyz'],
                gt_mask_visib=batch['roi_mask_visib'], do_loss=True)
            total = sum(losses.values())
        divisor = solver_utils.accumulation_window_size(iteration, accumulate, iters_per_epoch)
        session['lite'].backward(total / divisor)
        if not torch.isfinite(total):
            raise RuntimeError(f'Non-finite loss at iteration {iteration}')
        if solver_utils.should_optimizer_step(iteration, accumulate, iters_per_epoch, max_iter):
            # `.grad` still carries the AMP scaling here; an overflow would make the
            # scaler skip the step, which this smoke must never accept silently.
            accumulated = [p for p in session['model'].parameters() if p.grad is not None]
            if not all(torch.isfinite(p.grad).all() for p in accumulated):
                raise RuntimeError(f'Non-finite gradient at iteration {iteration}')
            before = session['optimizer'].param_groups[0]['lr']
            steps_before = update_count(session['optimizer'])
            scale_before = solver_utils.amp_scale(session['lite']._precision_plugin)
            session['wrapper'].step()
            session['wrapper'].zero_grad(set_to_none=True)
            scale_after = solver_utils.amp_scale(session['lite']._precision_plugin)
            # The engine's scheduler gate: a skipped update must not consume a tick.
            skipped = solver_utils.gradscaler_skipped_step(scale_before, scale_after)
            entry = dict(iteration=iteration, micro_step=iteration % iters_per_epoch + 1,
                         accumulation_divisor=divisor, lr=before, total_loss=float(total.detach()),
                         optimizer_steps=update_count(session['optimizer']),
                         scheduler_last_epoch_before=session['scheduler'].last_epoch,
                         updated=update_count(session['optimizer']) != steps_before,
                         amp_scale=scale_after, amp_scale_dropped=skipped)
            if not skipped:
                session['scheduler'].step()
            entry['scheduler_advanced'] = session['scheduler'].last_epoch > entry['scheduler_last_epoch_before']
            trace.append(entry)
    return trace


def max_delta(left, right, path='state', worst=None):
    """Largest absolute tensor difference between two state trees.

    Bit-exact comparison is not available here: the carried SDPA backward is atomic on
    CUDA, so two identical sessions already drift.  The smoke therefore reports deltas
    against a noise floor measured in the same process instead of asserting equality.
    """
    worst = worst or {'delta': 0.0, 'path': path}
    if isinstance(left, dict):
        assert left.keys() == right.keys(), (path, left.keys(), right.keys())
        for key in left:
            max_delta(left[key], right[key], f'{path}.{key}', worst)
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            max_delta(a, b, f'{path}[{index}]', worst)
    elif torch.is_tensor(left):
        delta = float((left.float() - right.float()).abs().max()) if left.numel() else 0.0
        if delta > worst['delta']:
            worst.update(delta=delta, path=path)
    else:
        assert left == right, (path, left, right)
    return worst


def state_deltas(reference, other):
    sections = {key: max_delta(reference[key], other[key], key)['delta'] for key in reference}
    return dict(sections, **{'worst': max_delta(reference, other)})


def final_state(session):
    state = dict(model=unwrapped_state(session['model']),
                 optimizer=session['optimizer'].state_dict(),
                 scheduler=session['scheduler'].state_dict(),
                 lr=session['optimizer'].param_groups[0]['lr'],
                 optimizer_steps=update_count(session['optimizer']))
    if session['scaler'] is not None:
        state['gradscaler'] = session['scaler'].state_dict()
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--load-batch', type=Path)
    parser.add_argument('--online', action='store_true', help='use the real loader instead of one saved batch')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--iters-per-epoch', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--split-epoch', type=int, default=2, help='epochs completed before the checkpoint')
    parser.add_argument('--amp', choices=('yes', 'no'), default='yes')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.amp = args.amp == 'yes'
    if args.iters_per_epoch < 1 or not 0 < args.split_epoch < args.epochs:
        parser.error('Require a positive iteration count and 0 < split epoch < epochs')
    if not args.online and not args.load_batch:
        parser.error('Require --load-batch or --online')
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = read_config(args.config, False, 'official_lmo')
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.WEIGHTS = ''  # build_model_optimizer already loads the configured initialization
    accumulate = solver_utils.get_accumulation_steps(cfg.SOLVER.REFERENCE_BS, cfg.SOLVER.IMS_PER_BATCH)
    total_updates = solver_utils.optimizer_updates_per_training(
        args.iters_per_epoch, args.epochs, accumulate)
    max_iter = args.iters_per_epoch * args.epochs
    report = dict(**metadata(cfg), run_id=args.output.name, status='RUNNING',
                  batch_size=args.batch_size, iters_per_epoch=args.iters_per_epoch,
                  epochs=args.epochs, split_epoch=args.split_epoch, accumulate_iter=accumulate,
                  total_optimizer_updates=total_updates, amp=args.amp,
                  batch_source='online' if args.online else str(args.load_batch),
                  interpretation='engineering state-machine smoke, not a performance result')
    save_report(args.output, report)
    try:
        if not torch.cuda.is_available() or not args.device.startswith('cuda'):
            raise RuntimeError('BLOCKED: accumulation smoke requires CUDA')
        torch.set_num_threads(4)
        # One repeated input: this checks the accumulation state machine, not generalization.
        batch = real_batch(cfg, args.device, args.batch_size, 'cpp',
                           load_batch=None if args.online else args.load_batch)

        split_iterations = args.split_epoch * args.iters_per_epoch

        # A: uninterrupted run over the full schedule.
        seed_all(42)
        reference = make_session(cfg, args, total_updates, resume=False)
        reference_trace = run_iterations(reference, cfg, args, batch, 0, max_iter, accumulate,
                                         args.iters_per_epoch, max_iter, [])
        reference_state = final_state(reference)
        report['continuous'] = reference_trace
        report['continuous_final'] = dict(lr=reference['optimizer'].param_groups[0]['lr'],
                                          optimizer_steps=update_count(reference['optimizer']),
                                          scheduler_last_epoch=reference['scheduler'].last_epoch)
        save_report(args.output, report)
        del reference
        torch.cuda.empty_cache()

        # B: run up to the split and checkpoint there.
        seed_all(42)
        split = make_session(cfg, args, total_updates, resume=False)
        split_trace = run_iterations(split, cfg, args, batch, 0, split_iterations,
                                     accumulate, args.iters_per_epoch, max_iter, [])
        split['checkpointer'].save('split', iteration=split_iterations - 1, epoch=args.split_epoch)
        split_state = final_state(split)
        report['split'] = split_trace
        report['split_saved'] = dict(lr=split['optimizer'].param_groups[0]['lr'],
                                     optimizer_steps=update_count(split['optimizer']),
                                     scheduler_last_epoch=split['scheduler'].last_epoch)
        del split
        torch.cuda.empty_cache()

        # B2: the same segment again, with no resume involved: this is the noise floor the
        # resume delta has to be judged against (CUDA SDPA backward is atomic).
        seed_all(42)
        repeat = make_session(cfg, args, total_updates, resume=False)
        run_iterations(repeat, cfg, args, batch, 0, split_iterations, accumulate,
                       args.iters_per_epoch, max_iter, [])
        report['run_to_run_noise_floor'] = state_deltas(split_state, final_state(repeat))
        del repeat
        torch.cuda.empty_cache()

        # C: resume from the split checkpoint and finish the schedule.
        seed_all(42)
        resumed = make_session(cfg, args, total_updates, resume=True)
        report['resume_start'] = dict(iteration=resumed['state'].get('iteration'),
                                      optimizer_steps=update_count(resumed['optimizer']),
                                      scheduler_last_epoch=resumed['scheduler'].last_epoch,
                                      lr=resumed['optimizer'].param_groups[0]['lr'])
        resumed_trace = run_iterations(resumed, cfg, args, batch, split_iterations,
                                       max_iter - split_iterations, accumulate,
                                       args.iters_per_epoch, max_iter, [])
        report['resumed'] = resumed_trace
        report['resumed_final'] = dict(lr=resumed['optimizer'].param_groups[0]['lr'],
                                       optimizer_steps=update_count(resumed['optimizer']),
                                       scheduler_last_epoch=resumed['scheduler'].last_epoch)
        resume_state = final_state(resumed)
        del resumed
        torch.cuda.empty_cache()

        # Exact, noise-free contract: the counters a broken resume resets, plus the LR the
        # scheduler produced for the updates that the resumed run performed.
        report['resume_state_delta'] = state_deltas(reference_state, resume_state)
        report['optimizer_step_boundaries'] = [entry['iteration'] for entry in reference_trace]
        report['updates_per_epoch'] = [
            sum(1 for entry in reference_trace if entry['iteration'] // args.iters_per_epoch == epoch)
            for epoch in range(args.epochs)]
        report['amp_scale_dropped'] = any(entry['amp_scale_dropped'] for entry in reference_trace)
        report['lr_trajectory_matches'] = (
            [entry['lr'] for entry in resumed_trace]
            == [entry['lr'] for entry in reference_trace[len(split_trace):]])
        report['optimizer_steps_match'] = (
            report['resumed_final']['optimizer_steps'] == report['continuous_final']['optimizer_steps']
            == [total_updates])
        report['scheduler_epoch_matches'] = (
            report['resumed_final']['scheduler_last_epoch']
            == report['continuous_final']['scheduler_last_epoch'] == total_updates)
        report['resume_matches_continuous'] = (
            report['resume_state_delta']['worst']['delta']
            <= max(3*report['run_to_run_noise_floor']['worst']['delta'], 1e-6))
        report['scheduler_advanced_only_on_updates'] = all(
            entry['scheduler_advanced'] == entry['updated'] for entry in reference_trace)
        for name in ('optimizer_steps_match', 'scheduler_epoch_matches', 'lr_trajectory_matches',
                     'resume_matches_continuous', 'scheduler_advanced_only_on_updates'):
            assert report[name], (name, report[name])
        assert not report['amp_scale_dropped'], 'AMP silently lowered the scale'
        report['status'] = 'PASS'
    except Exception as exc:
        report.update(status='BLOCKED' if str(exc).startswith('BLOCKED:') else 'FAIL', error=str(exc))
        raise
    finally:
        save_report(args.output, report)
    print(json.dumps({key: report[key] for key in
                      ('status', 'accumulate_iter', 'total_optimizer_updates', 'updates_per_epoch',
                       'optimizer_step_boundaries', 'continuous_final', 'split_saved', 'resume_start',
                       'resumed_final', 'optimizer_steps_match', 'scheduler_epoch_matches',
                       'lr_trajectory_matches', 'amp_scale_dropped', 'resume_state_delta',
                       'run_to_run_noise_floor', 'resume_matches_continuous',
                       'scheduler_advanced_only_on_updates') if key in report}, indent=2))


if __name__ == '__main__':
    main()
