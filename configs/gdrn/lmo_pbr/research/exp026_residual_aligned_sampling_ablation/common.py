"""EXP026 matched Full protocol; only the hierarchy differs between arms."""
from research.cad_common.configuration import backbone_settings

_base_ = ['../exp025_hierarchical_cad_attention/common.py']
EXPERIMENT_ID = 'EXP-20260922-026-residual-aligned-sampling-ablation'
TRAIN_BACKBONE = True
BACKBONE_INIT = 'imagenet'
BACKBONE_LR_MULT = 1.0
MODEL = backbone_settings(TRAIN_BACKBONE, BACKBONE_INIT, BACKBONE_LR_MULT)
MODEL['POSE_NET']['CAD_ATTENTION_HEAD'] = dict(INIT_CFG=dict(residual_target_mode='predicted_route'))
TRAIN_PROTOCOL = dict(NAME='exp026_lmo')
RESEARCH_PROTOCOL = dict(FORMAL_READY=True, SERVER_RELEASE_ALLOWED=True,
                         LOCAL_FORMAL_READY=True)
# On the matched local object-11 batch, adaptive_512_l1 at 32768 had a non-finite
# backbone gradient at step 4; both arms passed eight steps at 16384. The later
# server batch48 gate must independently confirm this shared initial scale.
SOLVER = dict(AMP=dict(INIT_SCALE=16384))
del backbone_settings
