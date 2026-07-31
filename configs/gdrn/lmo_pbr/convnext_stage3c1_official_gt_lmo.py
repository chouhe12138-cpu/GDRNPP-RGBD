_base_ = ["./convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo.py"]

OUTPUT_DIR = "output/EXP-20260731-006/official_gt"
SEED = 20260731

MODEL = dict(
    WEIGHTS="pretrained_models/lmo_pbr/model_final_wo_optim.pth",
    LOAD_DETS_TEST=False,
    POSE_NET=dict(
        BACKBONE=dict(INIT_CFG=dict(pretrained=False)),
    ),
)

DATASETS = dict(
    TRAIN=("lmo_pbr_train",),
    TEST=("lmo_bop_test",),
    DET_FILES_TEST=(),
)

TEST = dict(
    EVAL_PERIOD=0,
    TEST_BBOX_TYPE="gt",
    USE_PNP=False,
)
