_base_ = ["./convnext_stage3c1_official_gt_lmo.py"]

OUTPUT_DIR = "output/EXP-20260731-006/quality_coverage_full"
SEED = 20260731

MODEL = dict(
    POSE_NET=dict(
        BACKBONE=dict(FREEZE=True),
        GEO_HEAD=dict(FREEZE=True),
        PNP_NET=dict(FREEZE=True),
        QUALITY_COVERAGE=dict(
            ENABLED=True,
            FREEZE=False,
            LR_MULT=1.0,
            HIDDEN_DIM=32,
            MAX_RESIDUAL=0.25,
        ),
    ),
)

DATASETS = dict(
    TRAIN=("lmo_pbr_train",),
    TEST=("lmo_bop_test",),
    DET_FILES_TEST=(),
)

DATALOADER = dict(
    NUM_WORKERS=8,
    FILTER_VISIB_THR=0.3,
)

SOLVER = dict(
    IMS_PER_BATCH=48,
    REFERENCE_BS=48,
    TOTAL_EPOCHS=40,
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=8e-4, weight_decay=0.01),
    LR_SCHEDULER_NAME="flat_and_anneal",
    ANNEAL_METHOD="cosine",
    ANNEAL_POINT=0.72,
    WARMUP_FACTOR=0.001,
    WARMUP_ITERS=1000,
    CHECKPOINT_PERIOD=5,
    CHECKPOINT_BY_EPOCH=True,
    MAX_TO_KEEP=3,
    BEST_CHECKPOINT=dict(
        ENABLED=True,
        PRIMARY_METRIC="bop_ar",
        SECONDARY_METRIC="add_s_0.1d",
        PRIMARY_TIE_TOL=0.001,
    ),
)

TEST = dict(
    EVAL_PERIOD=5,
    TEST_BBOX_TYPE="gt",
    USE_PNP=False,
)
