"""Fail-closed EXP027 configuration, hierarchy, and CPU model audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from mmcv import Config

from core.gdrn_modeling.models.GDRN_CAD import build_model_optimizer, dataset_context
from research.cad_hierarchy.contracts import require_exact_hierarchy
from research.cad_common.preflight import audit_optimizer, verify_imagenet_backbone
from research.run_contract import validate_research_run_config

ROOT = Path('configs/gdrn/lmo_pbr/research/exp027_multiscale_cad')
ARMS = {'A_multiscale_fpn': ROOT / 'train_a_multiscale_fpn.py',
        'B_cad_region_query': ROOT / 'train_b_cad_region_query.py'}
ARCHITECTURES = {'A_multiscale_fpn': 'multiscale_image_query',
                 'B_cad_region_query': 'hierarchical_cad_query'}
EXPERIMENT_ID = 'EXP-20260924-027-multiscale-cad-interaction'


def inspect_config(cfg):
    validate_research_run_config(cfg, mode='prepare', expected_experiment_id=EXPERIMENT_ID)
    arm = str(cfg.EXP027_ARM)
    if arm not in ARMS or cfg.TRAIN_PROTOCOL.NAME != 'exp027_lmo':
        raise ValueError('Unknown EXP027 arm or protocol')
    net = cfg.MODEL.POSE_NET
    if str(net.CAD_ATTENTION_HEAD.ARCHITECTURE) != ARCHITECTURES[arm]:
        raise ValueError('EXP027 architecture/arm mismatch')
    if tuple(net.BACKBONE.INIT_CFG.out_indices) != (0, 1, 2, 3):
        raise ValueError('EXP027 requires four ConvNeXt scales')
    if (cfg.BACKBONE_INIT, bool(cfg.TRAIN_BACKBONE), float(cfg.BACKBONE_LR_MULT)) != ('imagenet', True, 1.):
        raise ValueError('EXP027 requires ImageNet Full at the head LR')
    if cfg.MODEL.WEIGHTS or net.CAD_ATTENTION_HEAD.INIT_CFG.residual_target_mode != 'predicted_route':
        raise ValueError('EXP027 requires fresh predicted-route residual training')
    if (int(cfg.SEED), int(cfg.SOLVER.TOTAL_EPOCHS), int(cfg.SOLVER.IMS_PER_BATCH),
            int(cfg.SOLVER.REFERENCE_BS), int(cfg.TEST.EVAL_PERIOD)) != (42, 40, 48, 48, 5):
        raise ValueError('EXP027 matched 40-epoch protocol mismatch')
    if tuple(cfg.DATASETS.TRAIN) != ('lmo_pbr_train',) or tuple(cfg.DATASETS.TEST) != ('lmo_bop_test',):
        raise ValueError('EXP027 LM-O dataset mismatch')
    context = dataset_context(cfg)
    contract = cfg.CAD_HIERARCHY_CONTRACT
    if (contract.SHA256, contract.VARIANT, float(contract.LAMBDA_GEO)) != (
            '7ac75daa756ed7ae59463614737ccc46cd746173452cd1943a4a024d0817c631',
            'adaptive_512_l1', 1.):
        raise ValueError('EXP027 adaptive hierarchy identity mismatch')
    digest = require_exact_hierarchy(context.hierarchy_path, contract,
                                     dataset_key=context.key, object_ids=context.object_ids)
    return context, digest


def run_cpu(cfg):
    _, digest = inspect_config(cfg)
    cfg.MODEL.DEVICE = 'cpu'
    torch.manual_seed(42)
    torch.set_num_threads(4)
    model, optimizer = build_model_optimizer(cfg)
    loaded = verify_imagenet_backbone(model, cfg.MODEL.POSE_NET.BACKBONE.INIT_CFG.checkpoint_path)
    groups = audit_optimizer(model, optimizer, cfg)
    model.train()
    classes = torch.tensor([0])
    prediction = model.predict(torch.randn(1, 3, 256, 256), classes)
    head = model.cad_attention_head
    ids = prediction['t3_logits'].detach().argmax(1).flatten(1)
    anchor = head.level3_anchors[classes[:, None], ids]
    radius = head.level3_radii[classes[:, None], ids]
    xyz = anchor + radius[..., None] * torch.tensor([.2, 0., 0.])
    xyz_norm = (xyz / head.extents[classes, None] + .5).transpose(1, 2).reshape(1, 3, 64, 64)
    losses, stats = head.loss(prediction, classes, xyz_norm, torch.ones(1, 1, 64, 64))
    total = sum(losses.values())
    total.backward()
    if not torch.isfinite(total):
        raise RuntimeError('EXP027 CPU loss non-finite')
    required = ('laterals', 'lateral_alpha', 'cross_attention', 'residual_predictor', 'mask_predictor')
    if cfg.EXP027_ARM == 'B_cad_region_query':
        required += ('query_parents', 'pixel_projection', 'query_projection')
    else:
        required += ('t3_classifier',)
    coverage = {}
    for name in required:
        module = getattr(head, name)
        params = (module,) if isinstance(module, torch.nn.Parameter) else tuple(module.parameters())
        coverage[name] = any(p.grad is not None and bool(torch.isfinite(p.grad).all()) and
                             bool(p.grad.abs().sum() > 0) for p in params)
    if not all(coverage.values()):
        raise RuntimeError(f'EXP027 CPU gradient coverage failed: {coverage}')
    if not stats['cad_pred_route_residual_valid_points'] > 0:
        raise RuntimeError('EXP027 predicted-route residual has no valid points')
    model.load_state_dict(model.state_dict(), strict=True)
    return dict(status='PASS', arm=cfg.EXP027_ARM, hierarchy_sha256=digest,
                imagenet_backbone_tensors=loaded, optimizer_groups=groups,
                gradient_coverage=coverage, losses={k: float(v.detach()) for k, v in losses.items()},
                total_params=sum(p.numel() for p in model.parameters()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_cpu(Config.fromfile(str(args.config))), sort_keys=True))


if __name__ == '__main__':
    main()
