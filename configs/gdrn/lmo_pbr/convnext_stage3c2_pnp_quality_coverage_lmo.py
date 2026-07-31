_base_ = ["./convnext_stage3c1_quality_coverage_lmo.py"]

OUTPUT_DIR = "output/EXP-20260731-006/pnp_quality_coverage_full"

MODEL = dict(
    POSE_NET=dict(
        PNP_NET=dict(FREEZE=False, LR_MULT=0.1),
        QUALITY_COVERAGE=dict(LR_MULT=1.0),
    ),
)
