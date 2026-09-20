"""One initialization/freeze policy for config, preflight and diagnostic tools."""
from __future__ import annotations

import os

OFFICIAL_WEIGHTS = 'pretrained_models/lmo_pbr/model_final_wo_optim.pth'


def backbone_settings(train_backbone=False, backbone_init='official_lmo', lr_mult=.1):
    if backbone_init not in ('official_lmo', 'imagenet'):
        raise ValueError(f'Unknown BACKBONE_INIT: {backbone_init}')
    if not isinstance(train_backbone, bool) or lr_mult <= 0:
        raise ValueError('TRAIN_BACKBONE must be bool and LR multiplier positive')
    imagenet = backbone_init == 'imagenet'
    return dict(WEIGHTS='' if imagenet else OFFICIAL_WEIGHTS, POSE_NET=dict(BACKBONE=dict(
        FREEZE=not train_backbone, LR_MULT=lr_mult, INIT_CFG=dict(pretrained=False,
        checkpoint_path=os.environ.get('GDRN_CONVNEXT_BASE_WEIGHTS', '') if imagenet else ''))))


def set_mode(cfg, train_backbone, backbone_init):
    """Explicit tool override; normal mmcv --opts does not reexecute Python config."""
    cfg.TRAIN_BACKBONE, cfg.BACKBONE_INIT = bool(train_backbone), backbone_init
    settings = backbone_settings(bool(train_backbone), backbone_init, float(cfg.BACKBONE_LR_MULT))
    cfg.MODEL.WEIGHTS = settings['WEIGHTS']
    backbone = settings['POSE_NET']['BACKBONE']
    cfg.MODEL.POSE_NET.BACKBONE.FREEZE = backbone['FREEZE']
    cfg.MODEL.POSE_NET.BACKBONE.LR_MULT = backbone['LR_MULT']
    cfg.MODEL.POSE_NET.BACKBONE.INIT_CFG.update(backbone['INIT_CFG'])
    return cfg
