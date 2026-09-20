"""EXP025 arm U: ImageNet ConvNeXt initialization, full training on lab1."""
from research.exp025.configuration import backbone_settings

_base_ = ['./common.py']
EXP025_ARM = 'imagenet_full'
TRAIN_BACKBONE = True
BACKBONE_INIT = 'imagenet'
BACKBONE_LR_MULT = .1
MODEL = backbone_settings(TRAIN_BACKBONE, BACKBONE_INIT, BACKBONE_LR_MULT)
del backbone_settings
