_base_ = ["./common.py"]

# EXP020-A (control): current XYZ/Mask/Region supervision only, reprojection
# loss disabled. Everything else is inherited from the shared protocol.
MODEL = dict(
    POSE_NET=dict(
        LOSS_CFG=dict(
            REPROJ_LW=0.0,
        )
    )
)
