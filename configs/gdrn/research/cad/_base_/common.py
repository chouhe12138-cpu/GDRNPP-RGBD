"""Dataset-neutral CAD model defaults.

The inherited GDRN base supplies framework defaults.  Every dataset/experiment
must set its own classes, data, hierarchy, optimizer, schedule, and evaluator.
"""

_base_ = ['../../../../_base_/gdrn_base.py']

MODEL = dict(PIXEL_MEAN=[0.0, 0.0, 0.0], POSE_NET=dict(
    NAME='GDRN_CAD', XYZ_ONLINE=True, XYZ_BP=True,
    BACKBONE=dict(INIT_CFG=dict(type='timm/convnext_base', pretrained=False,
                                in_chans=3, features_only=True, out_indices=(3,))),
    GEO_HEAD=dict(FREEZE=True, TRAIN_SUPERVISION=False,
                  INIT_CFG=dict(type='TopDownDoubleMaskXyzRegionHead', in_dim=1024),
                  XYZ_CLASS_AWARE=True, MASK_CLASS_AWARE=True, REGION_CLASS_AWARE=True),
    CAD_ATTENTION_HEAD=dict(ENABLED=True, INIT_CFG=dict(
        token_dim=256, num_heads=8, route_weight=1., residual_weight=1.,
        mask_weight=1., residual_beta=.1, residual_context_dim=64,
        residual_detach_route=True)),
    LOSS_CFG=dict(MASK_LOSS_TYPE='BCE')))
