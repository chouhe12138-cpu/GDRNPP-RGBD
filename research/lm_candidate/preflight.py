"""Candidate LM13 CPU model and artifact preflight."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import torch
from mmcv import Config

from core.gdrn_modeling.models.GDRN_CAD import build_model_optimizer, dataset_context
from core.gdrn_modeling.cad.hierarchy import load_cad_hierarchy
from research.cad_hierarchy.diagnostics import hierarchy_sanity
from research.exp025.preflight import audit_optimizer, verify_imagenet_backbone
from research.run_contract import validate_research_run_config

CONFIG = Path('configs/gdrn/lm/research/candidate_cad/train_imagenet_full.py')


def run(cfg):
    if (cfg.RESEARCH_PROTOCOL.STAGE, cfg.DATASET_CONTEXT.KEY, cfg.TRAIN_PROFILE.NAME) != (
            'candidate', 'lm13', 'lm13_legacy_full'):
        raise ValueError('LM13 candidate identity mismatch')
    validate_research_run_config(cfg, mode='prepare')
    context = dataset_context(cfg)
    hierarchy = load_cad_hierarchy(context.hierarchy_path,
                                   expected_object_ids=context.object_ids, dataset_key=context.key)
    sanity = hierarchy_sanity({d: v for d, v in hierarchy.numpy_levels().items() if d <= 3},
                              context.object_ids)
    if sanity['result'] != 'PASS':
        raise RuntimeError(sanity)
    cfg.MODEL.DEVICE = 'cpu'
    torch.manual_seed(42)
    torch.set_num_threads(4)
    model, optimizer = build_model_optimizer(cfg)
    loaded = verify_imagenet_backbone(model, Path(cfg.MODEL.POSE_NET.BACKBONE.INIT_CFG.checkpoint_path))
    groups = audit_optimizer(model, optimizer, cfg)
    model.train()
    image = torch.randn(1, 3, 256, 256)
    classes = torch.zeros(1, dtype=torch.long)
    xyz = torch.full((1, 3, 64, 64), .5)
    mask = torch.ones(1, 1, 64, 64)
    _, losses = model(image, roi_classes=classes, gt_xyz=xyz,
                      gt_mask_visib=mask, do_loss=True)
    total = sum(losses.values())
    total.backward()
    if not torch.isfinite(total) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                       for p in model.parameters()):
        raise RuntimeError('Non-finite loss or gradient')
    if not any(p.grad is not None for p in model.backbone.parameters()):
        raise RuntimeError('Trainable ImageNet backbone has no gradients')
    model.eval()
    with torch.no_grad():
        decoded = model(image, roi_classes=classes, return_cad_debug=True)
    if not torch.isfinite(decoded['xyz_norm']).all():
        raise RuntimeError('Non-finite decoded XYZ')
    state = io.BytesIO()
    torch.save(model.state_dict(), state)
    state.seek(0)
    model.load_state_dict(torch.load(state, map_location='cpu', weights_only=True), strict=True)
    return dict(status='PASS', dataset=context.key, object_ids=list(context.object_ids),
                hierarchy_depth=hierarchy.depth, active_head_levels=[8, 64, 512],
                hierarchy_sha256=str(cfg.CAD_HIERARCHY_CONTRACT.SHA256),
                imagenet_tensors=loaded, optimizer_groups=groups,
                losses={key: float(value.detach()) for key, value in losses.items()},
                strict_state_dict_roundtrip=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    args = parser.parse_args()
    print(json.dumps(run(Config.fromfile(str(args.config))), indent=2))


if __name__ == '__main__':
    main()
