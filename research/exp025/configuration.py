"""One initialization policy, with two names that mean two different things.

`BACKBONE_INIT` initializes the backbone only (`official_lmo` from the recorded LMO
weights, `imagenet` from the ConvNeXt checkpoint).  `MODEL.WEIGHTS` names a *complete*
GDRN_CAD checkpoint -- the only thing a checkpoint can restore, because the head's
geometry buffers are non-persistent.  A fresh training run therefore starts from an
empty `MODEL.WEIGHTS`, and resume goes through the output directory as before.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from research.cad_common.configuration import backbone_settings

OFFICIAL_WEIGHTS = 'pretrained_models/lmo_pbr/model_final_wo_optim.pth'

# The head's geometry buffers are non-persistent, so a checkpoint is only reproducible
# together with the exact hierarchy artifact it was trained on.  Name, mode and level
# counts are not enough: a regenerated artifact could carry the same ones.
HIERARCHY_SHA256 = {
    'lmo': '02ce090949bc40b2732417fec23984f3f748431098c5f67c853839d10ff1a373',
    'lm13': '322cd3778f0325838675a7dc6bae1a4e1cf107bfa95e05c006dcee0836a66417',
}
CONSISTENT_V3_SHA256 = HIERARCHY_SHA256['lmo']


@lru_cache(maxsize=8)
def sha256_file(path):
    """Streaming SHA256, cached per resolved path (one read per process)."""
    digest = hashlib.sha256()
    with Path(path).resolve().open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require_consistent_v3(path):
    """Return the artifact's digest, or refuse to run on anything else."""
    if not Path(path).is_file():
        raise FileNotFoundError(f'EXP025 hierarchy artifact missing: {path}')
    actual = sha256_file(str(Path(path).resolve()))
    if actual != CONSISTENT_V3_SHA256:
        raise ValueError(f'EXP025 requires consistent_v3 {CONSISTENT_V3_SHA256}, '
                         f'got {actual} at {path}')
    return actual


def require_hierarchy(path, dataset_key):
    """Verify the dataset-specific immutable hierarchy used by EXP025."""
    if dataset_key not in HIERARCHY_SHA256:
        raise ValueError(f'EXP025 has no registered hierarchy digest for {dataset_key}')
    if not Path(path).is_file():
        raise FileNotFoundError(f'EXP025 hierarchy artifact missing: {path}')
    actual = sha256_file(str(Path(path).resolve()))
    expected = HIERARCHY_SHA256[dataset_key]
    if actual != expected:
        raise ValueError(f'EXP025 {dataset_key} hierarchy requires {expected}, got {actual} at {path}')
    return actual


def set_mode(cfg, train_backbone, backbone_init):
    """Switch the backbone init/freeze mode; never touches `cfg.MODEL.WEIGHTS`.

    Explicit tool override; normal mmcv --opts does not reexecute Python config.
    """
    cfg.TRAIN_BACKBONE, cfg.BACKBONE_INIT = bool(train_backbone), backbone_init
    settings = backbone_settings(bool(train_backbone), backbone_init, float(cfg.BACKBONE_LR_MULT))
    backbone = settings['POSE_NET']['BACKBONE']
    cfg.MODEL.POSE_NET.BACKBONE.FREEZE = backbone['FREEZE']
    cfg.MODEL.POSE_NET.BACKBONE.LR_MULT = backbone['LR_MULT']
    cfg.MODEL.POSE_NET.BACKBONE.INIT_CFG.update(backbone['INIT_CFG'])
    return cfg
