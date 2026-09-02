_base_ = ["../../convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo.py"]

MODEL = dict(
    WEIGHTS="pretrained_models/lmo_pbr/model_final_wo_optim.pth",
    LOAD_DETS_TEST=False,
    POSE_NET=dict(
        BACKBONE=dict(INIT_CFG=dict(pretrained=False)),
    ),
)

DATASETS = dict(
    TEST=("lmo_bop_test",),
    DET_FILES_TEST=(),
)

TEST = dict(
    TEST_BBOX_TYPE="gt",
    USE_PNP=False,
)

# BOP evaluation rendering is independent from the training-only XYZ renderer.
VAL = dict(RENDERER_TYPE="cpp")

RUN_ARTIFACTS = dict(
    STRUCTURED_LAYOUT=True,
    COMPACT_LOG=True,
    TENSORBOARD=False,
    SKIP_DUPLICATE_FINAL_EVAL=True,
)
