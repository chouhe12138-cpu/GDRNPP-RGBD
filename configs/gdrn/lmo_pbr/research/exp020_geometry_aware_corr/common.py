_base_ = ["../_base_/pbr40_screening.py"]

EXPERIMENT_ID = "EXP-20260909-020-geometry-aware-correspondence-loss"
OUTPUT_DIR = (
    "output/experiments/EXP-20260909-020-geometry-aware-correspondence-loss/"
    "RUN-SET-BY-LAUNCHER"
)
SEED = 42

DATALOADER = dict(NUM_WORKERS=16)

# EXP020 formal protocol (shared by arms A/B).
#
# Scientific variable: a per-pixel GT-pose correspondence reprojection loss
# (LOSS_CFG.REPROJ_LW). The geometry producer is the only trainable module and
# is supervised only through geometry-level losses. Backbone and the official
# PNP_NET are frozen, and all pose-level losses are explicitly zeroed because
# freezing PNP_NET parameters does NOT stop a pose loss from back-propagating
# into coor_feat/geometry-head outputs.
MODEL = dict(
    WEIGHTS="pretrained_models/lmo_pbr/model_final_wo_optim.pth",
    POSE_NET=dict(
        # Online GT geometry is rendered for every training batch. EGL keeps
        # this path on GPU; the BOP evaluation renderer remains independently
        # fixed to cpp by the shared evaluation base.
        XYZ_RENDERER="egl",
        BACKBONE=dict(FREEZE=True, INIT_CFG=dict(pretrained=False)),
        GEO_HEAD=dict(FREEZE=False, TRAIN_SUPERVISION=True),
        PNP_NET=dict(FREEZE=True),
        QUALITY_COVERAGE=dict(ENABLED=False),
        LOSS_CFG=dict(
            # geometry supervision (kept)
            XYZ_LOSS_TYPE="L1",
            XYZ_LOSS_MASK_GT="visib",
            XYZ_LW=1.0,
            MASK_LOSS_TYPE="L1",
            MASK_LOSS_GT="trunc",
            MASK_LW=1.0,
            FULL_MASK_LOSS_TYPE="L1",
            FULL_MASK_LW=1.0,
            REGION_LOSS_TYPE="CE",
            REGION_LOSS_MASK_GT="visib",
            REGION_LW=1.0,
            # pose-level losses (isolated producer; gradient must not reach geometry)
            PM_LW=0.0,
            CENTROID_LW=0.0,
            Z_LW=0.0,
            ROT_LW=0.0,
            TRANS_LW=0.0,
            BIND_LW=0.0,
            # EXP020 reprojection loss knobs (read via .get with defaults)
            REPROJ_LOSS_TYPE="smooth_l1",
            REPROJ_SMOOTH_L1_BETA=1.0,
            REPROJ_LOSS_MASK_GT="visib",
            REPROJ_NORMALIZE_BY_RES=True,
            REPROJ_Z_EPS=1e-6,
        ),
    ),
)

# Optimizer/LR follow the most recent fine-tune-from-official-checkpoint
# protocol in this repository (EXP013A/E family and the PnP-only control:
# Ranger 8e-4, weight decay 0.01, warmup 200, batch 48, seed 42, 40 epochs).
# Both arms A/B use exactly these settings; only REPROJ_LW differs.
SOLVER = dict(
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=8e-4, weight_decay=0.01),
    WARMUP_ITERS=200,
    BEST_CHECKPOINT=dict(ENABLED=False),
)
