_base_ = ["../../_base_/pbr40_screening.py"]

EXPERIMENT_ID = "EXP-20260829-015-e-official-head-random"
OUTPUT_DIR = "output/experiments/EXP-20260829-015-e-official-head-random/RUN-SET-BY-LAUNCHER"
SEED = 42

DATALOADER = dict(NUM_WORKERS=16)

# EXP013E rebuilds the OFFICIAL ConvPnPNet head with random initialization
# under the frozen-feature protocol, so only the head architecture differs
# from EXP013A/B/C. Two consequences are load-bearing:
# 1. WEIGHTS points at the pnp-stripped derivative of the official checkpoint
#    (research/exp013/e_prep.py): the official pnp keys share their names with
#    this head, so loading the original file would overwrite the random init.
# 2. The head flags replicate the official configuration, NOT the EXP013
#    family: no mask multiplicative gating (MASK_ATTENTION="none"), GN+gelu,
#    flatten fc1. Renderer stays OFF (C pattern): frozen geometry supervision
#    disabled means the engine never constructs a CPP or EGL renderer.
MODEL = dict(
    WEIGHTS="pretrained_models/lmo_pbr/model_final_wo_optim_wo_pnp.pth",
    POSE_NET=dict(
        BACKBONE=dict(FREEZE=True, INIT_CFG=dict(pretrained=False)),
        GEO_HEAD=dict(FREEZE=True, TRAIN_SUPERVISION=False),
        PNP_NET=dict(
            FREEZE=False,
            INIT_CFG=dict(
                _delete_=True,
                type="ConvPnPNet",
                norm="GN",
                act="gelu",
                num_gn_groups=32,
                drop_prob=0.0,
                flat_op="flatten",
                denormalize_by_extent=True,
            ),
            WITH_2D_COORD=True,
            COORD_2D_TYPE="abs",
            REGION_ATTENTION=True,
            MASK_ATTENTION="none",
            ROT_TYPE="allo_rot6d",
            TRANS_TYPE="centroid_z",
        ),
        QUALITY_COVERAGE=dict(ENABLED=False),
    ),
)

SOLVER = dict(
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=8e-4, weight_decay=0.01),
    WARMUP_ITERS=200,
    BEST_CHECKPOINT=dict(ENABLED=False),
)
