_base_ = ["../../exp013/a_xyz_residual/train.py"]

EXPERIMENT_ID = "EXP-20260906-018-geometry-consistency-residual"
OUTPUT_DIR = "output/experiments/EXP-20260906-018-geometry-consistency-residual/RUN-SET-BY-LAUNCHER"

# A remains the raw initial pose head. This module runs AFTER camera-frame decode.
# Official upstream checkpoint / random A pose head initialization match A's run;
# do not warm-start from A E40 and silently give this experiment 40 extra epochs.
MODEL = dict(
    POSE_NET=dict(
        POSE_CORRECTOR=dict(
            ENABLED=True,
            INIT_CFG=dict(
                type="GeometryConsistencyCorrector",
                num_steps=1,
                support_threshold=0.5,
                residual_clip=2.0,
                max_rotation_deg=15.0,
                translation_extent_scale=0.15,
            ),
        )
    )
)
