"""EXP025 arm U: ImageNet ConvNeXt initialization, full training on lab1."""
from research.exp025.configuration import backbone_settings

_base_ = ['./common.py']
EXP025_ARM = 'imagenet_full'
TRAIN_BACKBONE = True
BACKBONE_INIT = 'imagenet'
# The arm is a full fine-tune, and the official LM-O recipe this dataset is scored against
# trains its ImageNet ConvNeXt at the base LR (GDRN_double_mask.build_model_optimizer adds the
# backbone group at SOLVER.BASE_LR, no multiplier).  The .1 inherited from the LM13 GDR-Net
# protocol put the backbone at 3e-5, ~27x below that recipe with no formal run behind it.
BACKBONE_LR_MULT = 1.
MODEL = backbone_settings(TRAIN_BACKBONE, BACKBONE_INIT, BACKBONE_LR_MULT)
del backbone_settings
