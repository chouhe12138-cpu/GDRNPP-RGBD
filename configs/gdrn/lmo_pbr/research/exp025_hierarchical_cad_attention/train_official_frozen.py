"""EXP025 arm F: original GDRNPP LM-O backbone, frozen on lab0."""
from research.exp025.configuration import backbone_settings

_base_ = ['./common.py']
EXP025_ARM = 'official_frozen'
TRAIN_BACKBONE = False
BACKBONE_INIT = 'official_lmo'
BACKBONE_LR_MULT = .1
MODEL = backbone_settings(TRAIN_BACKBONE, BACKBONE_INIT, BACKBONE_LR_MULT)
del backbone_settings
