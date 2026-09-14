_base_ = ["../_base_/lmo_gt_eval.py"]

EXPERIMENT_ID = "EXP-20260914-021-global-guided-hierarchical-cad-correspondence"
OUTPUT_DIR = "output/experiments/EXP-20260914-021-global-guided-hierarchical-cad-correspondence/RUN-SET-BY-LAUNCHER"
SEED = 42
MODEL = dict(
    WEIGHTS="pretrained_models/lmo_pbr/model_final_wo_optim.pth",
    POSE_NET=dict(BACKBONE=dict(INIT_CFG=dict(pretrained=False))),
)
DATASETS = dict(TEST=("lmo_bop_test",), DET_FILES_TEST=())
TEST = dict(EVAL_PERIOD=0, TEST_BBOX_TYPE="gt", USE_PNP=True, PNP_TYPE="ransac_pnp")

