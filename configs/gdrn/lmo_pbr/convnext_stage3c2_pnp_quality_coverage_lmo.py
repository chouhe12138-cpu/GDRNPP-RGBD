_base_ = ["./convnext_stage3c1_quality_coverage_lmo.py"]

OUTPUT_DIR = "output/stage3c/C2_joint"

RUN_ARTIFACTS = dict(
    STRUCTURED_LAYOUT=True,
    COMPACT_LOG=True,
    TENSORBOARD=False,
    SKIP_DUPLICATE_FINAL_EVAL=True,
)

MODEL = dict(
    POSE_NET=dict(
        PNP_NET=dict(FREEZE=False, LR_MULT=0.1),
        QUALITY_COVERAGE=dict(LR_MULT=1.0),
    ),
)

TRAIN = dict(PRINT_FREQ=500)
