_base_ = ["../b_geometry_attention/train.py"]

EXPERIMENT_ID = "EXP-20260822-013-c-rt-decoupled-fusion"
OUTPUT_DIR = "output/experiments/EXP-20260822-013-c-rt-decoupled-fusion/RUN-SET-BY-LAUNCHER"
MODEL = dict(
    POSE_NET=dict(
        PNP_NET=dict(INIT_CFG=dict(type="RTDecoupledGeometryPnPNet"))
    )
)
