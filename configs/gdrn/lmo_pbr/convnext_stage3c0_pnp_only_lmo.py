_base_ = ["./convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo.py"]

OUTPUT_DIR = "output/stage3c/B_patch_pnp"
SEED = 20260731

RUN_ARTIFACTS = dict(
    STRUCTURED_LAYOUT=True,
    COMPACT_LOG=True,
    TENSORBOARD=False,
    SKIP_DUPLICATE_FINAL_EVAL=True,
)

MODEL = dict(
    WEIGHTS="pretrained_models/lmo_pbr/model_final_wo_optim.pth",
    LOAD_DETS_TEST=False,
    POSE_NET=dict(
        BACKBONE=dict(
            FREEZE=True,
            INIT_CFG=dict(pretrained=False),
        ),
        GEO_HEAD=dict(FREEZE=True),
        PNP_NET=dict(FREEZE=False),
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
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=8e-5, weight_decay=0.01),
    LR_SCHEDULER_NAME="flat_and_anneal",
    ANNEAL_METHOD="cosine",
    ANNEAL_POINT=0.72,
    WARMUP_FACTOR=0.001,
    WARMUP_ITERS=200,
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

TEST = dict(EVAL_PERIOD=5, TEST_BBOX_TYPE="gt", USE_PNP=False)

TRAIN = dict(PRINT_FREQ=500)
