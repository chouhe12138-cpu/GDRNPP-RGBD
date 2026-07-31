_base_ = ["./convnext_stage3c1_quality_coverage_lmo.py"]

OUTPUT_DIR = "output/EXP-20260731-006/quality_coverage_local"

DATASETS = dict(
    TRAIN=("lmo_pbr_stage3_local_train",),
    TEST=(),
    DET_FILES_TEST=(),
)

DATALOADER = dict(
    NUM_WORKERS=0,
    FILTER_VISIB_THR=0.3,
)

SOLVER = dict(
    IMS_PER_BATCH=4,
    REFERENCE_BS=48,
    TOTAL_EPOCHS=1,
    WARMUP_ITERS=20,
    CHECKPOINT_PERIOD=1,
    MAX_TO_KEEP=1,
    BEST_CHECKPOINT=dict(ENABLED=False),
)

TEST = dict(EVAL_PERIOD=0)
