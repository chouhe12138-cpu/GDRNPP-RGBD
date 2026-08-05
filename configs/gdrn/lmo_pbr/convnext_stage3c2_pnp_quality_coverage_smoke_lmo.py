_base_ = ["./convnext_stage3c2_pnp_quality_coverage_lmo.py"]

OUTPUT_DIR = "output/stage3c_smoke/C2_joint"

DATASETS = dict(
    TRAIN=("lmo_pbr_stage3_local_train",),
    TEST=(),
)

DATALOADER = dict(NUM_WORKERS=2)

SOLVER = dict(
    IMS_PER_BATCH=4,
    REFERENCE_BS=48,
    TOTAL_EPOCHS=1,
    CHECKPOINT_PERIOD=1,
    MAX_TO_KEEP=3,
)

TEST = dict(EVAL_PERIOD=0)
TRAIN = dict(PRINT_FREQ=100)
