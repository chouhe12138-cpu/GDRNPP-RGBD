import os

_base_ = ["../_base_/pbr40_screening.py"]

EXPERIMENT_ID = "EXP-20260914-021-global-guided-hierarchical-cad-correspondence"
OUTPUT_DIR = "output/experiments/EXP-20260914-021-global-guided-hierarchical-cad-correspondence/RUN-SET-BY-LAUNCHER"
SEED = 42

# Match the recent batch-48 formal/audit protocol.  Smoke configs explicitly
# override this to two workers for their batch-4 contract.
DATALOADER = dict(NUM_WORKERS=16)

_cache_root = os.environ.get("GDRN_DATASET_CACHE_DIR", ".local/dataset_cache")
_hierarchy_path = os.path.join(_cache_root, "exp021", "hierarchy_v1.npz")

MODEL = dict(
    WEIGHTS="pretrained_models/lmo_pbr/model_final_wo_optim.pth",
    POSE_NET=dict(
        XYZ_RENDERER="egl",
        USE_MTL=False,
        BACKBONE=dict(FREEZE=True, INIT_CFG=dict(pretrained=False)),
        GEO_HEAD=dict(FREEZE=True, TRAIN_SUPERVISION=False),
        PNP_NET=dict(FREEZE=True),
        QUALITY_COVERAGE=dict(ENABLED=False),
        CAD_HEAD=dict(
            ENABLED=True,
            FREEZE=False,
            TRAIN_SUPERVISION=True,
            LR_MULT=1.0,
            HIERARCHY_PATH=_hierarchy_path,
            INIT_CFG=dict(
                token_dim=256,
                num_heads=4,
                num_transformer_layers=2,
                ffn_dim=1024,
                coarse_loss_weight=0.25,
                fine_loss_weight=1.0,
                xyz_loss_weight=16.0,
                xyz_smooth_l1_beta=0.01,
                default_beam_k=4,
            ),
        ),
        LOSS_CFG=dict(
            XYZ_LW=0.0,
            MASK_LW=0.0,
            FULL_MASK_LW=0.0,
            REGION_LW=0.0,
            PM_LW=0.0,
            CENTROID_LW=0.0,
            Z_LW=0.0,
            ROT_LW=0.0,
            TRANS_LW=0.0,
            BIND_LW=0.0,
            REPROJ_LW=0.0,
        ),
    ),
)

SOLVER = dict(
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=8e-4, weight_decay=0.01),
    AMP=dict(ENABLED=True),
    WARMUP_ITERS=200,
    BEST_CHECKPOINT=dict(ENABLED=False),
)

# Periodic repository PnP is telemetry only. EXP021's decision uses the
# dedicated cross-checkpoint fixed-support evaluator.
TEST = dict(TEST_BBOX_TYPE="gt", USE_PNP=True, PNP_TYPE="ransac_pnp")
