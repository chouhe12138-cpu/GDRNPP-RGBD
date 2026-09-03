_base_ = ["../support_aware_rotation_residual/train.py"]

EXPERIMENT_ID = "EXP-20260903-017-b-detached-adapter-geometry"
OUTPUT_DIR = (
    "output/experiments/EXP-20260903-017-b-detached-adapter-geometry/"
    "RUN-SET-BY-LAUNCHER"
)

# The sole change from EXP017 is the head type.  It evaluates the same adapter
# on geometry_grid.detach(); EXP013A's own geometry-latent path remains intact.
MODEL = dict(
    POSE_NET=dict(
        PNP_NET=dict(
            INIT_CFG=dict(type="DetachedSupportAwareRotationResidualPnPNet")
        )
    )
)
