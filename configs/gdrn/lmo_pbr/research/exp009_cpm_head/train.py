_base_ = ["../_base_/pbr40_screening.py"]

EXPERIMENT_ID = "EXP-20260809-009-cpm-head"
OUTPUT_DIR = (
    "output/experiments/EXP-20260809-009-cpm-head/RUN-SET-BY-LAUNCHER"
)
SEED = 42

DATALOADER = dict(NUM_WORKERS=16)

MODEL = dict(
    WEIGHTS="pretrained_models/lmo_pbr/model_final_wo_optim.pth",
    POSE_NET=dict(
        BACKBONE=dict(
            FREEZE=True,
            INIT_CFG=dict(pretrained=False),
        ),
        GEO_HEAD=dict(FREEZE=True),
        PNP_NET=dict(
            FREEZE=False,
            INIT_CFG=dict(
                _delete_=True,
                type="CorrespondenceAwareMomentPnPNet",
                hidden_dim=512,
                latent_dim=256,
                denormalize_by_extent=True,
                eps=1e-6,
                # Fixed before training by the 8192-sample audit on 2026-08-09.
                # Order: mu_X, mu_U, C_XX, C_UU, C_XU.
                moment_scales=(
                    0.053791501000523545,
                    0.8383049368858337,
                    0.00045910440967418244,
                    0.03372693955898269,
                    0.0008616717066615817,
                ),
                use_cross_covariance=True,
            ),
            WITH_2D_COORD=True,
            COORD_2D_TYPE="abs",
            REGION_ATTENTION=True,
            MASK_ATTENTION="mul",
            ROT_TYPE="allo_rot6d",
            TRANS_TYPE="centroid_z",
        ),
        QUALITY_COVERAGE=dict(ENABLED=False),
    ),
)

# Matched to the mandatory Stage 3C0/B control.  These are engineering
# parameters, not CPM method changes.
SOLVER = dict(
    OPTIMIZER_CFG=dict(
        _delete_=True,
        type="Ranger",
        lr=8e-5,
        weight_decay=0.01,
    ),
    WARMUP_ITERS=200,
    # Intermediate LM-O evaluations may be retained as diagnostics, but they
    # must never create the checkpoint used for the formal comparison.
    BEST_CHECKPOINT=dict(ENABLED=False),
)
