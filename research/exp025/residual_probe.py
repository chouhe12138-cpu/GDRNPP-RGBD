"""GT-T3-conditioned residual probe against the shipped residual head.

The shipped residual path regresses `(XYZ - anchor_T3) / radius_T3`, so the regression
target's meaning depends on a T3 identity.  The first round of this probe established
that the identity has to be an input: with the GT T3 CAD token concatenated, the
residual becomes learnable.  Residual V2 now feeds the *predicted* T3 distribution into
the shipped path instead, so `baseline_residual_only` measures that path.  The probe
keeps the GT-conditioned arm as the upper bound.  GT T3 never enters the shipped model
or inference: this script builds the alternative predictor itself and changes no
production code path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from core.gdrn_modeling.models.GDRN_CAD import build_model_optimizer
from core.gdrn_modeling.models.heads.hierarchical_cad_attention_head import bounded_residual, masked_mean
from .learnability import measure_prediction
from .preflight import CONFIG, read_config
from .runtime import (NonFiniteTrainingError, amp_step, load_last_good, metadata, real_batch,
                      save_last_good, save_report, seed_all)

ARMS = ('baseline_residual_only', 'gt_t3_conditioned')


class GTT3ConditionedResidual(nn.Module):
    """Diagnostic-only predictor: image token concatenated with the GT T3 CAD token."""

    def __init__(self, token_dim):
        super().__init__()
        self.token_dim = int(token_dim)
        self.net = nn.Sequential(nn.Linear(2*token_dim, token_dim), nn.GELU(), nn.Linear(token_dim, 3))

    def forward(self, image_tokens, t3_tokens, ids):
        """[B,P,D] image tokens, [B,512,D] T3 bank, [B,H,W] GT ids -> [B,P,3] raw residual."""
        b, p, d = image_tokens.shape
        selected = t3_tokens.gather(1, ids.reshape(b, -1)[..., None].expand(b, p, d))
        return self.net(torch.cat((image_tokens, selected), -1))


def branch_residual_loss(head, probe, tokens, banks, classes, xyz, mask, branch):
    """Probe residual conditioned on one symmetry branch, with that branch's own loss."""
    path, target, valid = head.targets(xyz, mask, classes, branch)
    raw = probe(tokens, banks[3], path[2])
    residual = bounded_residual(raw.transpose(1, 2).reshape(len(classes), 3, *xyz.shape[-2:]))
    value = masked_mean(F.smooth_l1_loss(residual.flatten(2).transpose(1, 2), target,
                                         beta=head.residual_beta, reduction='none').mean(-1), valid)
    return value, raw, residual


def probe_prediction(head, feature, classes, probe, xyz, mask, diagnostics=None):
    """Probe prediction dict; each residual is trained against its conditioning branch.

    The training objective is the guide's question -- given the GT T3 node, regress that
    node's residual -- so a candidate is selected by its own conditioned fit, exactly as
    the head selects a symmetry branch by the loss it is trained with.  Reported metrics
    come from `measure_prediction`, which re-scores the returned residual with the head's
    rule; for an object with two symmetry branches the two selections can differ, which
    the probe's own single-branch unit test pins down.
    """
    tokens, banks = head.encode(feature, classes)
    b, spatial = len(classes), xyz.shape[-2:]
    dense = lambda x: x.transpose(1, 2).reshape(b, -1, *spatial)
    losses, raws, residuals = [], [], []
    for branch in range(min(2, head.symmetry_transforms.shape[1])):
        value, raw, residual = branch_residual_loss(head, probe, tokens, banks, classes, xyz, mask, branch)
        losses.append(value)
        raws.append(raw)            # [B,P,3] pre-tanh predictor output
        residuals.append(residual)  # [B,3,H,W] bounded
    scores = torch.stack(losses, 1)
    allowed = torch.arange(scores.shape[1], device=classes.device)[None] < head.symmetry_counts[classes, None]
    index = torch.arange(b, device=classes.device)
    choice = scores.detach().masked_fill(~allowed, float('inf')).argmin(1)
    # One choice selects both views, so the recorded raw telemetry describes the branch
    # the returned residual came from.
    selected_raw = torch.stack(raws, 1)[index, choice]
    residual = torch.stack(residuals, 1)[index, choice]
    prediction = dict(t3_logits=dense(head.t3_classifier(tokens)),
                      mask_logit=dense(head.mask_predictor(tokens)), residual=residual)
    if diagnostics is not None:
        diagnostics.update(image_tokens=tokens.detach(), cad_tokens=banks[3].detach(),
                           t3_logits=prediction['t3_logits'].detach(),
                           mask_logit=prediction['mask_logit'].detach(),
                           raw_residual=dense(selected_raw).detach(), residual=residual.detach())
    return prediction, scores[index, choice].mean()


def probe_optimizer(model, probe, optimizer_cfg):
    """The parameters the conditioned arm trains: the head minus its shipped residual path."""
    parameters = [p for key, p in model.cad_attention_head.named_parameters()
                  if not key.startswith('residual_predictor') and p.requires_grad] + list(probe.parameters())
    return torch.optim.AdamW(parameters, lr=float(optimizer_cfg['lr']),
                             weight_decay=float(optimizer_cfg['weight_decay']),
                             betas=tuple(optimizer_cfg['betas']), eps=float(optimizer_cfg['eps']))


def arm_prediction(model, probe, batch, diagnostics=None):
    """One prediction for an arm; a probe is diagnostic-only and never joins the model."""
    if probe is None:
        return model.predict(batch['roi_img'], batch['roi_cls'], diagnostics=diagnostics)
    prediction, _ = probe_prediction(model.cad_attention_head, model.backbone_feature(batch['roi_img']),
                                     batch['roi_cls'], probe, batch['roi_xyz'], batch['roi_mask_visib'],
                                     diagnostics)
    return prediction


def load_conditioned_state(state, model, probe, optimizer):
    """Load a conditioned last-good checkpoint into an already built arm."""
    if state.get('probe') is None:
        raise ValueError('Conditioned last-good checkpoint carries no probe state')
    model.load_state_dict(state['model'], strict=True)
    probe.load_state_dict(state['probe'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    scaler = torch.cuda.amp.GradScaler()
    if state.get('gradscaler') is not None:
        scaler.load_state_dict(state['gradscaler'])
    return scaler


def restore_conditioned_arm(state, cfg, device, token_dim):
    """Rebuild model + probe + matched optimizer, then load a last-good checkpoint.

    The optimizer must be rebuilt from the same parameter set before its state dict is
    loaded, which is what makes this a real reload test of the probe state.
    """
    model, _ = build_model_optimizer(cfg)
    probe = GTT3ConditionedResidual(token_dim).to(device)
    optimizer = probe_optimizer(model, probe, cfg.SOLVER.OPTIMIZER_CFG)
    return (model, probe, optimizer, load_conditioned_state(state, model, probe, optimizer))


def roundtrip_check(cfg, args, batch, record, token_dim):
    """Reload the last saved conditioned checkpoint and re-measure the same step.

    Confirms the saved state is complete (probe parameters included) rather than only
    that the file exists: the rebuilt arm must reproduce the recorded metrics.
    """
    state = load_last_good(args.output / record['last_good']['path'])
    history = {entry['step']: entry for entry in record['steps']}
    step = int(state['step'])
    if step not in history:
        return dict(status='SKIPPED', reason=f'step {step} has no recorded metrics')
    model, probe, optimizer, scaler = restore_conditioned_arm(state, cfg, args.device, token_dim)
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast():
        prediction = arm_prediction(model, probe, batch, {})
        replayed = measure_prediction(model.cad_attention_head, batch, prediction, 0., {})
    reference = history[step]
    shared = [key for key, value in replayed.items()
              if isinstance(value, (int, float)) and isinstance(reference.get(key), (int, float))]
    deltas = {key: abs(replayed[key] - reference[key]) / (1 + abs(reference[key])) for key in shared}
    worst = max(deltas.values()) if deltas else 0.0
    return dict(status='MATCH' if worst <= 1e-4 else 'MISMATCH', step=step, metrics=len(shared),
                probe_parameters=sum(p.numel() for p in probe.parameters()),
                optimizer_step=update_step(optimizer), worst_relative_delta=worst,
                worst_metric=max(deltas, key=deltas.get) if deltas else None,
                reloaded_metrics=replayed)


def update_step(optimizer):
    values = {int(state['step']) for state in optimizer.state.values() if 'step' in state}
    return sorted(values)


def run_arm(name, model, optimizer, batch, args, report, probe=None, cfg=None):
    head = model.cad_attention_head
    classes = batch['roi_cls']
    scaler = torch.cuda.amp.GradScaler()

    def forward_prediction(diagnostics=None):
        return arm_prediction(model, probe, batch, diagnostics)

    record = dict(arm=name, steps=[], status='RUNNING')
    last_good = args.output / f'{name}_last_good.pth'
    with torch.no_grad():
        model.eval()
        diagnostics = {}
        with torch.cuda.amp.autocast():
            prediction = forward_prediction(diagnostics)
        record['steps'].append(dict(step=0, **measure_prediction(head, batch, prediction, 0., diagnostics)))
    save_last_good(last_good, model, optimizer, scaler, 0, dict(arm=name), probe=probe)
    record['last_good'] = dict(step=0, path=last_good.name)
    try:
        for step in range(1, args.steps + 1):
            report['current_step'] = step
            model.train()
            optimizer.zero_grad(set_to_none=True)
            diagnostics = {}
            with torch.cuda.amp.autocast():
                if probe is None:
                    prediction = model.predict(batch['roi_img'], classes, diagnostics=diagnostics)
                    losses, _ = head.loss(prediction, classes, batch['roi_xyz'], batch['roi_mask_visib'],
                                          route_weight=0.)
                else:
                    prediction, residual_loss = probe_prediction(
                        head, model.backbone_feature(batch['roi_img']), classes, probe,
                        batch['roi_xyz'], batch['roi_mask_visib'], diagnostics)
                    mask = batch['roi_mask_visib']
                    if mask.ndim == 3:
                        mask = mask[:, None]
                    losses = dict(loss_cad_residual=residual_loss,
                                  loss_cad_mask=F.binary_cross_entropy_with_logits(
                                      prediction['mask_logit'].float(), mask.float()))
                total = sum(losses.values())
            amp_step(model, optimizer, scaler, total, step=step, losses=losses,
                     diagnostics=diagnostics, head=head)
            if step % args.last_good_period == 0 and step < args.steps:
                save_last_good(last_good, model, optimizer, scaler, step, dict(arm=name), probe=probe)
                record['last_good'] = dict(step=step, path=last_good.name)
            if step % 20 == 0 or step == args.steps:
                model.eval()
                with torch.no_grad():
                    diagnostics = {}
                    with torch.cuda.amp.autocast():
                        prediction = forward_prediction(diagnostics)
                    record['steps'].append(dict(step=step, **measure_prediction(
                        head, batch, prediction, 0., diagnostics)))
                save_report(args.output, report)
                print(json.dumps(dict(arm=name, **record['steps'][-1])), flush=True)
        record['status'] = 'COMPLETE'
    except NonFiniteTrainingError as exc:
        record.update(status='FAIL', error=str(exc), failure=exc.telemetry)
    if probe is not None:
        record['roundtrip'] = roundtrip_check(cfg, args, batch, record, probe.token_dim)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--load-batch', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--last-good-period', type=int, default=20)
    parser.add_argument('--arms', default=','.join(ARMS))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    arms = tuple(args.arms.split(','))
    if args.steps < 1 or args.batch_size < 1 or args.last_good_period < 1 or not set(arms) <= set(ARMS):
        parser.error(f'Require positive steps/batch size/last-good period and arms within {ARMS}')
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = read_config(args.config, False, 'official_lmo')
    cfg.MODEL.DEVICE = args.device
    report = dict(**metadata(cfg), run_id=args.output.name, status='RUNNING', steps=args.steps,
                  batch_size=args.batch_size, source_batch=str(args.load_batch), arms={},
                  schedule='constant 3e-4, no formal warmup',
                  interpretation='fixed-batch residual diagnostic, not generalization')
    save_report(args.output, report)
    try:
        if not torch.cuda.is_available() or not args.device.startswith('cuda'):
            raise RuntimeError('BLOCKED: residual probe requires CUDA')
        torch.set_num_threads(4)
        batch = real_batch(cfg, args.device, args.batch_size, 'cpp', load_batch=args.load_batch)
        optim_cfg = cfg.SOLVER.OPTIMIZER_CFG
        for name in arms:
            report['current_arm'] = name
            save_report(args.output, report)
            seed_all(42)
            model, optimizer = build_model_optimizer(cfg)
            probe = None
            if name == 'gt_t3_conditioned':
                torch.manual_seed(42)
                probe = GTT3ConditionedResidual(
                    cfg.MODEL.POSE_NET.CAD_ATTENTION_HEAD.INIT_CFG['token_dim']).to(args.device)
                optimizer = probe_optimizer(model, probe, optim_cfg)
            report['arms'][name] = run_arm(name, model, optimizer, batch, args, report, probe, cfg)
            del model, optimizer, probe
            torch.cuda.empty_cache()
        report['status'] = 'COMPLETE'
    except Exception as exc:
        report.update(status='BLOCKED' if str(exc).startswith('BLOCKED:') else 'FAIL', error=str(exc))
        raise
    finally:
        save_report(args.output, report)
    print(json.dumps({name: arm['status'] for name, arm in report['arms'].items()}, indent=2))


if __name__ == '__main__':
    main()
