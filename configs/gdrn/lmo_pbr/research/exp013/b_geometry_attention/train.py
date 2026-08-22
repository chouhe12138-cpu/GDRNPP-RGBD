_base_ = ["../a_xyz_residual/train.py"]

EXPERIMENT_ID = "EXP-20260822-013-b-geometry-attention-residual"
OUTPUT_DIR = "output/experiments/EXP-20260822-013-b-geometry-attention-residual/RUN-SET-BY-LAUNCHER"
MODEL = dict(
    POSE_NET=dict(
        PNP_NET=dict(
            INIT_CFG=dict(
                type="GeometryAttentionResidualPnPNet",
                attention_scale_init=0.1,
            )
        )
    )
)
