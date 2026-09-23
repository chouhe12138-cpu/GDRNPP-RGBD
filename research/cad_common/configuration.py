"""Shared CAD backbone configuration, without experiment artifact identity."""
from __future__ import annotations

import os


def backbone_settings(train_backbone=False, backbone_init='official_lmo', lr_mult=.1):
    if backbone_init not in ('official_lmo', 'imagenet'):
        raise ValueError(f'Unknown BACKBONE_INIT: {backbone_init}')
    if not isinstance(train_backbone, bool) or lr_mult <= 0:
        raise ValueError('TRAIN_BACKBONE must be bool and LR multiplier positive')
    imagenet = backbone_init == 'imagenet'
    # WEIGHTS stays empty: backbone initialization is not a full CAD checkpoint.
    return dict(WEIGHTS='', POSE_NET=dict(BACKBONE=dict(
        FREEZE=not train_backbone, LR_MULT=lr_mult, INIT_CFG=dict(pretrained=False,
        checkpoint_path=os.environ.get('GDRN_CONVNEXT_BASE_WEIGHTS', '') if imagenet else ''))))
