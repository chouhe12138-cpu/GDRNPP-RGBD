_base_ = ["../a_xyz_residual/train.py"]

EXPERIMENT_ID = "EXP-20260829-016-f-glm-pose-l-screening"
OUTPUT_DIR = "output/experiments/EXP-20260829-016-f-glm-pose-l-screening/RUN-SET-BY-LAUNCHER"

# EXP013F: GLM-Pose-L frozen screening. Relative to EXP013A the only changes
# are the head structure (M2 attention pooling replacing the flatten/FC late
# decoder, no geometry residual branch) and the M3 depth-statistic input;
# losses, solver, and every other protocol field are inherited unchanged.
# Renderer stays OFF (C pattern): frozen geometry supervision disabled means
# the engine never constructs a CPP or EGL renderer.
INPUT = dict(HEAD_DEPTH=True)
MODEL = dict(
    POSE_NET=dict(
        GEO_HEAD=dict(TRAIN_SUPERVISION=False),
        PNP_NET=dict(
            INIT_CFG=dict(
                type="GLMPoseLNet",
                use_depth_stats=True,
            )
        ),
    )
)
