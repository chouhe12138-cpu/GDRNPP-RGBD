import os

_base_ = ["../_base_/pbr40_screening.py"]

EXPERIMENT_ID = "EXP-20260916-022-progressive-pcc"
OUTPUT_DIR = "output/experiments/EXP-20260916-022-progressive-pcc/RUN-SET-BY-LAUNCHER"
SEED = 42

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
            INIT_CFG=dict(
                token_dim=256, beam_k=2,
                route_weight=1.0, residual_weight=1.0,
                mask_weight=1.0, residual_beta=0.1,
            ),
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
