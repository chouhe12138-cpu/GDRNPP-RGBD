"""Short local LM13 AMP update and checkpoint round-trip; never formal training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from mmcv import Config

from core.gdrn_modeling.models.GDRN_CAD import build_model_optimizer
from research.exp025.preflight import audit_optimizer
from research.exp025.runtime import amp_step, real_batch, seed_all
from research.run_contract import validate_research_run_config

CONFIG = Path('configs/gdrn/lm/research/candidate_cad/smoke.py')


def mixed_real_render_batch(cfg, device, renderer_type):
    """Force both source domains through one online-XYZ batch_data call."""
    import ref
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from core.gdrn_modeling.datasets.data_loader_online import GDRN_Online_DatasetFromList
    from core.gdrn_modeling.engine.engine_utils import batch_data, get_renderer

    real_name, render_name = cfg.DATASETS.TRAIN
    real, render = DatasetCatalog.get(real_name), DatasetCatalog.get(render_name)
    records = [real[0], render[0], real[-1], render[-1]]
    dataset = GDRN_Online_DatasetFromList(cfg, split='train', lst=records,
                                          copy=False, serialize=False)
    raw = [dataset[index] for index in range(4)]
    meta = MetadataCatalog.get(real_name)
    renderer = get_renderer(cfg, ref.__dict__[meta.ref_key], meta.objs, gpu_id=0)
    try:
        batch = batch_data(cfg, raw, renderer=renderer, device=device, phase='train')
    finally:
        if hasattr(renderer, 'close'):
            renderer.close()
    if len(batch['roi_img']) != 4 or not (0 <= batch['roi_cls']).all() or not (batch['roi_cls'] < 13).all():
        raise RuntimeError('Mixed LM batch class/shape mismatch')
    return {key: batch[key] for key in ('roi_img', 'roi_cls', 'roi_xyz', 'roi_mask_visib')}


def run(cfg, output, *, renderer='cpp', steps=2, amp_scale=16384.0):
    validate_research_run_config(cfg, mode='smoke')
    if (cfg.RESEARCH_PROTOCOL.STAGE, cfg.DATASET_CONTEXT.KEY) != ('candidate', 'lm13'):
        raise ValueError('LM candidate config required')
    if steps < 2 or not torch.cuda.is_available():
        raise RuntimeError('At least two steps and local CUDA are required')
    output.mkdir(parents=True, exist_ok=False)
    seed_all(42)
    torch.set_num_threads(4)
    cfg.MODEL.DEVICE = 'cuda:0'
    cfg.MODEL.POSE_NET.XYZ_RENDERER = renderer
    model, optimizer = build_model_optimizer(cfg)
    groups = audit_optimizer(model, optimizer, cfg)
    # Check the normal loader path, then train on an explicit two-domain batch so
    # a random sampler cannot accidentally make this a one-domain-only smoke.
    real_batch(cfg, 'cuda:0', 4, renderer)
    batch = mixed_real_render_batch(cfg, 'cuda:0', renderer)
    model.train()
    initial_backbone = next(model.backbone.parameters()).detach().clone()
    initial_head = model.cad_attention_head.t3_classifier.weight.detach().clone()
    scaler = torch.cuda.amp.GradScaler(init_scale=amp_scale)
    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            _, losses = model(batch['roi_img'], roi_classes=batch['roi_cls'],
                              gt_xyz=batch['roi_xyz'], gt_mask_visib=batch['roi_mask_visib'],
                              do_loss=True)
            total = sum(losses.values())
        amp_step(model, optimizer, scaler, total, step=step + 1, losses=losses)
        history.append({key: float(value.detach()) for key, value in losses.items()})
    if torch.equal(initial_backbone, next(model.backbone.parameters())):
        raise RuntimeError('Backbone did not update')
    if torch.equal(initial_head, model.cad_attention_head.t3_classifier.weight):
        raise RuntimeError('CAD head did not update')
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast():
        before = model(batch['roi_img'], roi_classes=batch['roi_cls'], return_cad_debug=True)
    if not torch.isfinite(before['xyz_norm']).all():
        raise RuntimeError('Non-finite decoded XYZ')
    checkpoint = output / 'smoke_checkpoint.pth'
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scaler=scaler.state_dict(), steps=steps), checkpoint)
    restored = torch.load(checkpoint, map_location='cuda:0', weights_only=False)
    model.load_state_dict(restored['model'], strict=True)
    optimizer.load_state_dict(restored['optimizer'])
    scaler.load_state_dict(restored['scaler'])
    with torch.no_grad(), torch.cuda.amp.autocast():
        after = model(batch['roi_img'], roi_classes=batch['roi_cls'], return_cad_debug=True)
    torch.testing.assert_close(after['xyz_norm'], before['xyz_norm'], rtol=0, atol=0)
    result = dict(status='PASS', kind='lm13_candidate_cuda_smoke', renderer=renderer,
                  batch_size=4, steps=steps, amp_scale=amp_scale, amp_skipped_steps=0,
                  batch_domains=['lm_real', 'lm_imgn', 'lm_real', 'lm_imgn'],
                  optimizer_groups=groups, losses=history, checkpoint=str(checkpoint))
    (output / 'report.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--renderer', choices=('cpp', 'egl'), default='cpp')
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--amp-scale', type=float, default=16384.0)
    args = parser.parse_args()
    try:
        result = run(Config.fromfile(str(args.config)), args.output,
                     renderer=args.renderer, steps=args.steps, amp_scale=args.amp_scale)
    except Exception as exc:
        if args.output.is_dir():
            status = 'BLOCKED' if 'Bindless Textures not supported' in str(exc) else 'FAIL'
            (args.output / 'report.json').write_text(json.dumps(dict(
                status=status, kind='lm13_candidate_cuda_smoke', renderer=args.renderer,
                batch_size=4, steps=args.steps, error=str(exc)), indent=2))
        raise
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
