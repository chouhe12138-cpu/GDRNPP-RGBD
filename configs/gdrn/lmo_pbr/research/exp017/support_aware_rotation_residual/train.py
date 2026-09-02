_base_ = ["../../exp013/a_xyz_residual/train.py"]

EXPERIMENT_ID = "EXP-20260902-017-support-aware-rotation-residual"
OUTPUT_DIR = (
    "output/experiments/EXP-20260902-017-support-aware-rotation-residual/"
    "RUN-SET-BY-LAUNCHER"
)

# The inherited config is EXP013A in full.  The only formal variable is this
# rotation-only residual adapter over A's existing Region-free 8x8 grid.
MODEL = dict(
    POSE_NET=dict(
        PNP_NET=dict(
            INIT_CFG=dict(
                type="SupportAwareRotationResidualPnPNet",
                adapter_token_channels=64,
                adapter_score_channels=32,
                alpha_r_init=1.0,
            )
        )
    )
)
