_base_ = ["../../_base_/pbr40_screening.py"]

EXPERIMENT_ID = "EXP-20260822-013-a-xyz-residual-bypass"
OUTPUT_DIR = "output/experiments/EXP-20260822-013-a-xyz-residual-bypass/RUN-SET-BY-LAUNCHER"
SEED = 42

DATALOADER = dict(NUM_WORKERS=16)

MODEL = dict(
    WEIGHTS="pretrained_models/lmo_pbr/model_final_wo_optim.pth",
    POSE_NET=dict(
        BACKBONE=dict(FREEZE=True, INIT_CFG=dict(pretrained=False)),
        GEO_HEAD=dict(FREEZE=True),
        PNP_NET=dict(
            FREEZE=False,
            INIT_CFG=dict(
                _delete_=True,
                type="XYZResidualBypassPnPNet",
                base_channels=64,
                mid_channels=96,
                high_channels=128,
                denormalize_by_extent=True,
                use_region_aux=True,
                region_aux_dim=16,
                coarse_grid_size=4,
                dropout=0.0,
                geometry_channels=(32, 48, 32),
                geometry_grid_size=8,
                geometry_scale_init=0.1,
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

SOLVER = dict(
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=8e-4, weight_decay=0.01),
    WARMUP_ITERS=200,
    BEST_CHECKPOINT=dict(ENABLED=False),
)
