import os
from configs.gdrn.research.exp022_progressive_pcc.method import PCC_INIT_CFG

_base_ = ["../_base_/pbr40_screening.py"]

EXPERIMENT_ID = "EXP-20260916-022-progressive-pcc"
OUTPUT_DIR = "output/experiments/EXP-20260916-022-progressive-pcc/RUN-SET-BY-LAUNCHER"
SEED = 42
DATASET_CONTEXT = dict(
    KEY="lmo", CAD_REF_KEY="lm_full", BOP_DATASET="lmo",
    BOP_TARGETS_FILENAME="test_targets_bop19.json",
    REFERENCE_CONFIG="configs/gdrn/lmo_pbr/research/exp021_global_hierarchical_cad/a_official_eval.py",
    REFERENCE_CHECKPOINT="pretrained_models/lmo_pbr/model_final_wo_optim.pth",
)

DATALOADER = dict(NUM_WORKERS=16)
MODEL = dict(
    WEIGHTS="pretrained_models/lmo_pbr/model_final_wo_optim.pth",
    POSE_NET=dict(
        NAME="GDRN_PCC",
        XYZ_RENDERER="egl",
        BACKBONE=dict(FREEZE=True, INIT_CFG=dict(pretrained=False)),
        GEO_HEAD=dict(FREEZE=True, TRAIN_SUPERVISION=False),
        PCC_HEAD=dict(
            ENABLED=True,
            HIERARCHY_PATH=os.path.join(
                os.environ.get("GDRN_DATASET_CACHE_DIR", ".local/dataset_cache"),
                "exp022", "reused_v1.npz",
            ),
            INIT_CFG=PCC_INIT_CFG,
        ),
        LOSS_CFG=dict(MASK_LOSS_TYPE="BCE"),
    ),
)
SOLVER = dict(
    MAX_TO_KEEP=10,
    OPTIMIZER_CFG=dict(_delete_=True, type="AdamW", lr=3e-4,
                       weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8),
    AMP=dict(ENABLED=True),
    WARMUP_RATIO=0.04,
    WARMUP_FACTOR=0.001,
    WARMUP_METHOD="linear",
    LR_SCHEDULER_NAME="flat_and_anneal",
    ANNEAL_METHOD="cosine",
    TARGET_LR_FACTOR=0.01,
    BEST_CHECKPOINT=dict(ENABLED=False),
)
TEST = dict(TEST_BBOX_TYPE="gt", USE_PNP=True, PNP_TYPE="ransac_pnp")
del PCC_INIT_CFG
