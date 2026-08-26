_base_ = ["../a_xyz_residual/train.py"]

EXPERIMENT_ID = "EXP-20260822-013-c-rt-decoupled-fusion"
OUTPUT_DIR = "output/experiments/EXP-20260822-013-c-rt-decoupled-fusion/RUN-SET-BY-LAUNCHER"
MODEL = dict(
    POSE_NET=dict(
        GEO_HEAD=dict(TRAIN_SUPERVISION=False),
        PNP_NET=dict(
            INIT_CFG=dict(
                type="RTDecoupledGeometryPnPNet",
                geometry_scale_r_init=0.1,
                geometry_scale_t_init=0.1,
            )
        ),
    )
)
