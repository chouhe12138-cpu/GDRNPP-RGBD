"""Production-path AMP recovery: what the engine does when the GradScaler refuses a step.

`numerical_replay.py` replays the recorded overflow through a diagnostic GradScaler with
a fatal gradient guard.  This smoke drives the objects `main_gdrn.py` actually builds --
`LightningLite(precision=16)`, its precision plugin's GradScaler, the Lite optimizer
wrapper, the `_LiteModule` that enters fp16 autocast -- from the same full-arm step-160
last-good state, and follows the engine's own step / scheduler sequence.  Engineering
behaviour only: the fixed batch is repeated, so nothing here is a performance result.

The question the engine's correctness turns on: a skipped optimizer update must not
consume an LR-schedule tick.  `--unguarded-scheduler` restores the pre-fix loop to show
what the gate is protecting against.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from pytorch_lightning.lite import LightningLite

from core.gdrn_modeling.models.GDRN_CAD import build_model_optimizer
from core.utils import my_checkpoint, solver_utils
from .preflight import CONFIG, read_config
from .runtime import metadata, real_batch, restore_rng, save_report


class _Lite(LightningLite):
    def run(self):
        pass


def optimizer_steps(optimizer):
    """Sorted internal step counters of the optimizer that actually steps."""
    return sorted({int(state['step']) for state in optimizer.state.values() if 'step' in state})


def parameter_snapshot(model):
    """Clones of the trainable parameters, the only ones an optimizer step can move."""
    return {name: parameter.detach().clone()
            for name, parameter in model.named_parameters() if parameter.requires_grad}


def parameter_delta(snapshot, model):
    worst, where = 0.0, None
    for name, parameter in model.named_parameters():
        if name in snapshot:
            delta = float((parameter.detach() - snapshot[name]).abs().max()) if parameter.numel() else 0.0
            if delta > worst:
                worst, where = delta, name
    return dict(parameter_delta=worst, changed_parameter=where)


def make_session(cfg, args, checkpoint, updates_per_epoch, accumulate):
    """The production object graph, restored to the recorded step-160 training state."""
    (args.output / 'checkpoints').mkdir(parents=True, exist_ok=True)  # as engine.do_train does
    lite = _Lite(accelerator='gpu', devices=1, precision=args.precision,
                 plugins=solver_utils.amp_precision_plugins(cfg) if args.precision == 16 else None)
    model, optimizer = build_model_optimizer(cfg)
    model, wrapper = lite.setup(model, optimizer)
    state_optimizer = my_checkpoint.unwrap_optimizer_for_checkpoint(wrapper)
    total_updates = solver_utils.optimizer_updates_per_training(updates_per_epoch, args.epochs, accumulate)
    scheduler = solver_utils.build_lr_scheduler(cfg, state_optimizer, total_iters=total_updates)
    scaler = getattr(lite._precision_plugin, 'scaler', None)

    state = torch.load(checkpoint, map_location='cpu')
    model.module.load_state_dict(state['model'], strict=True)
    state_optimizer.load_state_dict(state['optimizer'])
    if scaler is not None and state.get('gradscaler') is not None:
        scaler.load_state_dict(state['gradscaler'])
    # The restored optimizer replaced its own state/param_groups: the wrapper used for
    # stepping must see the resumed objects, not the ones from construction.
    my_checkpoint.resync_wrapped_optimizer(wrapper)
    restore_rng(state['rng'])

    # Evidence that the Lite module really is the fp16 entry point the engine calls.
    observed = {}

    def note_precision(module, inputs):
        observed.update(autocast=bool(torch.is_autocast_enabled()), input_dtype=str(inputs[0].dtype))

    model.module.register_forward_pre_hook(note_precision)
    return dict(lite=lite, model=model, wrapper=wrapper, optimizer=state_optimizer, scheduler=scheduler,
                scaler=scaler, start_step=int(state['step']), source_step=int(state['step']),
                precision=args.precision, observed=observed)


def run_updates(session, batch, args, updates_per_epoch, accumulate, trace):
    """engine.do_train's loop shape: one zero_grad before it and after every step."""
    max_iter = args.steps + session['start_step']
    session['wrapper'].zero_grad(set_to_none=True)
    for iteration in range(session['start_step'], max_iter):
        # No autocast context here: `_LiteModule.forward` enters the plugin's precision
        # context (and casts the inputs) itself, exactly as engine.do_train calls it.
        _, losses = session['model'](batch['roi_img'], roi_classes=batch['roi_cls'], gt_xyz=batch['roi_xyz'],
                                     gt_mask_visib=batch['roi_mask_visib'], do_loss=True)
        total = sum(losses.values())
        divisor = solver_utils.accumulation_window_size(iteration, accumulate, updates_per_epoch)
        session['lite'].backward(total / divisor)
        if not solver_utils.should_optimizer_step(iteration, accumulate, updates_per_epoch, max_iter):
            continue

        # Read immediately around the step, as the engine does: nothing else touches the
        # scaler in between, and a decrease here can only mean the step was refused.
        scale_before = solver_utils.amp_scale(session['lite']._precision_plugin)
        step_before = optimizer_steps(session['optimizer'])
        scheduler_before = int(session['scheduler'].last_epoch)
        lr_before = float(session['optimizer'].param_groups[0]['lr'])
        before = parameter_snapshot(session['model'])
        session['wrapper'].step()  # production GradScaler path
        session['wrapper'].zero_grad(set_to_none=True)
        scale_after = solver_utils.amp_scale(session['lite']._precision_plugin)
        step_after = optimizer_steps(session['optimizer'])
        entry = dict(iteration=iteration, total_loss=float(total.detach()),
                     loss_finite=bool(torch.isfinite(total)), accumulation_divisor=divisor,
                     lr_before=lr_before, scale_before=scale_before, scale_after=scale_after,
                     optimizer_step_before=step_before, optimizer_step_after=step_after,
                     updated=step_after != step_before, scheduler_last_epoch_before=scheduler_before,
                     **parameter_delta(before, session['model']))
        assert entry['loss_finite'], entry  # engine.do_train asserts this too
        skipped = solver_utils.gradscaler_skipped_step(scale_before, scale_after)
        if args.unguarded_scheduler or not skipped:
            session['scheduler'].step()
        entry['scheduler_last_epoch_after'] = int(session['scheduler'].last_epoch)
        entry['scheduler_advanced'] = entry['scheduler_last_epoch_after'] > scheduler_before
        trace.append(entry)
    return trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--from-checkpoint', type=Path, required=True,
                        help='full-arm last-good state recorded by learnability.py')
    parser.add_argument('--load-batch', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=20, help='updates to attempt from the checkpoint')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--accumulation', type=int, default=1)
    parser.add_argument('--precision', type=int, choices=(16, 32), default=16)
    parser.add_argument('--unguarded-scheduler', action='store_true',
                        help='advance the scheduler unconditionally, i.e. the pre-fix loop')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.accumulation < 1:
        parser.error('Require positive steps, batch size and accumulation')
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = read_config(args.config, False, 'official_lmo')
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.WEIGHTS = ''
    updates_per_epoch = args.steps
    report = dict(**metadata(cfg), run_id=args.output.name, status='RUNNING', steps=args.steps,
                  batch_size=args.batch_size, source_batch=str(args.load_batch),
                  from_checkpoint=str(args.from_checkpoint), precision=args.precision,
                  accumulation=args.accumulation, scheduler='guarded by GradScaler skip'
                  if not args.unguarded_scheduler else 'advanced unconditionally (pre-fix control)',
                  interpretation='production-path AMP recovery probe, not a performance result')
    save_report(args.output, report)
    try:
        if not torch.cuda.is_available() or not args.device.startswith('cuda'):
            raise RuntimeError('BLOCKED: the production AMP path requires CUDA')
        torch.set_num_threads(4)
        batch = real_batch(cfg, args.device, args.batch_size, 'cpp', load_batch=args.load_batch)
        session = make_session(cfg, args, args.from_checkpoint, updates_per_epoch, args.accumulation)
        report.update(start_step=session['start_step'],
                      amp_scale=session['scaler'].get_scale() if session['scaler'] is not None else None,
                      forward_precision=session['observed'])
        trace = report['updates'] = []
        run_updates(session, batch, args, updates_per_epoch, args.accumulation, trace)
        report['forward_precision'] = session['observed']
        report['final'] = dict(optimizer_steps=optimizer_steps(session['optimizer']),
                               scheduler_last_epoch=int(session['scheduler'].last_epoch),
                               amp_scale=solver_utils.amp_scale(session['lite']._precision_plugin))
        trace = report['updates']
        report['overflow_steps'] = [entry['iteration'] for entry in trace if not entry['updated']]
        report['scheduler_mismatch_steps'] = [entry['iteration'] for entry in trace
                                              if entry['scheduler_advanced'] != entry['updated']]
        report['overflow_observed'] = bool(report['overflow_steps'])
        if report['overflow_observed']:
            first = next(entry for entry in trace if not entry['updated'])
            later = [entry for entry in trace if entry['iteration'] > first['iteration'] and entry['updated']]
            report['skip_event'] = first
            report['skip_scale_dropped'] = first['scale_after'] < first['scale_before']
            report['skip_left_parameters_untouched'] = first['parameter_delta'] == 0.0
            report['skip_left_optimizer_counter'] = first['optimizer_step_after'] == first['optimizer_step_before']
            report['process_continued'] = len(trace) == args.steps
            report['recovery'] = later[0] if later else None
            report['recovered_after_skip'] = bool(later) and later[0]['parameter_delta'] > 0
            assert report['skip_scale_dropped'] and report['skip_left_parameters_untouched'], first
            assert report['skip_left_optimizer_counter'] and report['process_continued'], first
        # The gate is the point of the run: in the pre-fix control a mismatch is the
        # expected outcome, so it is recorded rather than asserted.
        report['scheduler_gate_holds'] = not report['scheduler_mismatch_steps']
        if args.unguarded_scheduler:
            report['control_scheduler_mismatch'] = report['scheduler_mismatch_steps']
        else:
            assert report['scheduler_gate_holds'], report['scheduler_mismatch_steps']
        report['status'] = 'PASS'
    except Exception as exc:
        report.update(status='BLOCKED' if str(exc).startswith('BLOCKED:') else 'FAIL',
                      error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save_report(args.output, report)
    print(json.dumps({key: report[key] for key in
                      ('status', 'start_step', 'amp_scale', 'forward_precision', 'overflow_observed',
                       'overflow_steps', 'skip_event', 'recovery', 'recovered_after_skip',
                       'scheduler_mismatch_steps', 'final') if key in report}, indent=2))


if __name__ == '__main__':
    main()
