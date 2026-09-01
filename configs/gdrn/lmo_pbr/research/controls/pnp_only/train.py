_base_ = ["../../_base_/pbr40_screening.py"]

EXPERIMENT_ID = "EXP-20260731-005-pnp-only-control"
OUTPUT_DIR = (
    "output/experiments/EXP-20260731-005-pnp-only-control/RUN-SET-BY-LAUNCHER"
)
SEED = 42

DATALOADER = dict(NUM_WORKERS=16)

# Long-term matched control for new pose-head experiments. It keeps the
# official pretrained ConvPnPNet, freezes backbone/geometry, and adapts only
# the PnP head under the current 40-epoch screening protocol. Exact historical
# EXP005 reproduction still uses the source commit recorded in its RECORD.
MODEL = dict(
    WEIGHTS="pretrained_models/lmo_pbr/model_final_wo_optim.pth",
    LOAD_DETS_TEST=False,
    POSE_NET=dict(
        BACKBONE=dict(FREEZE=True, INIT_CFG=dict(pretrained=False)),
        GEO_HEAD=dict(FREEZE=True, TRAIN_SUPERVISION=False),
        PNP_NET=dict(FREEZE=False),
        QUALITY_COVERAGE=dict(ENABLED=False),
    ),
)

SOLVER = dict(
    OPTIMIZER_CFG=dict(
        _delete_=True,
        type="Ranger",
        lr=8e-5,
        weight_decay=0.01,
    ),
    WARMUP_ITERS=200,
    BEST_CHECKPOINT=dict(ENABLED=False),
)
