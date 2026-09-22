"""Local EXP026 artifact, optimizer, and CPU model contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from mmcv import Config

from core.gdrn_modeling.cad.hierarchy import load_cad_hierarchy
from core.gdrn_modeling.models.GDRN_CAD import build_model_optimizer, dataset_context
from research.cad_hierarchy.diagnostics import hierarchy_sanity
from research.exp025.preflight import audit_optimizer, verify_imagenet_backbone
from research.run_contract import validate_research_run_config
from .configuration import EXPERIMENT_ID, require_arm_hierarchy

ROOT = Path('configs/gdrn/lmo_pbr/research/exp026_residual_aligned_sampling_ablation')
ARMS = {'uniform_full': ROOT / 'train_uniform_full.py',
        'adaptive_l1_full': ROOT / 'train_adaptive_full.py'}


def read_config(arm):
    return Config.fromfile(str(ARMS[arm]))


def inspect_config(cfg):
    validate_research_run_config(cfg, mode='prepare')
    if cfg.TRAIN_PROTOCOL.NAME != 'exp026_lmo' or cfg.EXPERIMENT_ID != EXPERIMENT_ID:
        raise ValueError('Not an EXP026 config')
    if (cfg.BACKBONE_INIT, bool(cfg.TRAIN_BACKBONE), float(cfg.BACKBONE_LR_MULT)) != ('imagenet', True, 1.):
        raise ValueError('EXP026 requires ImageNet Full training at the head LR')
    if cfg.MODEL.WEIGHTS or cfg.MODEL.POSE_NET.CAD_ATTENTION_HEAD.INIT_CFG.residual_target_mode != 'predicted_route':
        raise ValueError('EXP026 requires fresh start and predicted-route residual')
    context = dataset_context(cfg)
    digest = require_arm_hierarchy(cfg, context)
    hierarchy = load_cad_hierarchy(context.hierarchy_path, expected_object_ids=context.object_ids,
                                   dataset_key=context.key)
    sanity = hierarchy_sanity(hierarchy.numpy_levels(), context.object_ids)
    if sanity['result'] != 'PASS':
        raise RuntimeError(sanity)
    return context, digest, sanity


def run_cpu(cfg):
    context, digest, sanity = inspect_config(cfg)
    cfg.MODEL.DEVICE = 'cpu'
    torch.manual_seed(42)
    torch.set_num_threads(4)
    model, optimizer = build_model_optimizer(cfg)
    loaded = verify_imagenet_backbone(model, cfg.MODEL.POSE_NET.BACKBONE.INIT_CFG.checkpoint_path)
    groups = audit_optimizer(model, optimizer, cfg)
    model.train()
    classes = torch.tensor([0])  # non-symmetric LM-O object
    prediction = model.predict(torch.randn(1, 3, 256, 256), classes)
    head = model.cad_attention_head
    ids = prediction['t3_logits'].detach().argmax(1).flatten(1)
    anchor = head.level3_anchors[classes[:, None], ids]
    radius = head.level3_radii[classes[:, None], ids]
    xyz = anchor + radius[..., None] * torch.tensor([.2, 0., 0.])
    xyz_norm = (xyz / head.extents[classes, None] + .5).transpose(1, 2).reshape(1, 3, 64, 64)
    mask = torch.ones(1, 1, 64, 64)
    losses, stats = head.loss(prediction, classes, xyz_norm, mask)
    total = sum(losses.values())
    total.backward()
    if not torch.isfinite(total) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise RuntimeError('EXP026 CPU loss/gradient non-finite')
    if not stats['cad_pred_route_residual_valid_points'] > 0:
        raise RuntimeError('EXP026 CPU residual has no valid supervision')
    if not head.residual_predictor.final.weight.grad.abs().sum() > 0:
        raise RuntimeError('EXP026 CPU residual predictor has no gradient')
    decoded = head.decode(prediction, classes)
    if not torch.isfinite(decoded).all():
        raise RuntimeError('EXP026 CPU decode non-finite')
    state = model.state_dict()
    model.load_state_dict(state, strict=True)
    return dict(status='PASS', arm=cfg.EXP026_ARM, hierarchy_sha256=digest,
                hierarchy_sanity=sanity, imagenet_backbone_tensors=loaded,
                optimizer_groups=groups, losses={k: float(v.detach()) for k, v in losses.items()},
                stats={k: float(v) for k, v in stats.items()}, checkpoint_roundtrip='strict_state_dict')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--arm', choices=tuple(ARMS))
    group.add_argument('--config', type=Path)
    args = parser.parse_args()
    print(json.dumps(run_cpu(Config.fromfile(str(args.config)) if args.config else read_config(args.arm)),
                     sort_keys=True))


if __name__ == '__main__':
    main()
