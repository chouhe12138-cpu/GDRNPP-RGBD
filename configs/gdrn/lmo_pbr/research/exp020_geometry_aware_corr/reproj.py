_base_ = ["./common.py"]

# EXP020-B (treatment): identical to arm A except REPROJ_LW. The weight is the
# fixed nonzero value used by the first matched run; a data-driven calibration
# (gradient-magnitude scale check on a real smoke batch) may follow before the
# formal run, but the two arms must never differ in anything but REPROJ_LW.
MODEL = dict(
    POSE_NET=dict(
        LOSS_CFG=dict(
            REPROJ_LW=1.0,
        )
    )
)
