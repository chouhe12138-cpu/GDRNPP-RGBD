_base_ = ["./lmo_gt_eval.py"]

DATASETS = dict(
    TRAIN=("lmo_pbr_train",),
)

DATALOADER = dict(
    NUM_WORKERS=8,
    FILTER_VISIB_THR=0.3,
)

# Pose-head screening freezes the geometry producer.  Its predictions are still
# consumed by the pose head, but rendered GT geometry and geometry losses are
# unnecessary.  Full-training experiments must explicitly override both fields.
MODEL = dict(
    POSE_NET=dict(
        XYZ_RENDERER=None,
        GEO_HEAD=dict(TRAIN_SUPERVISION=False),
    )
)

SOLVER = dict(
    IMS_PER_BATCH=48,
    REFERENCE_BS=48,
    TOTAL_EPOCHS=40,
    CHECKPOINT_PERIOD=5,
    CHECKPOINT_BY_EPOCH=True,
    MAX_TO_KEEP=3,
    BEST_CHECKPOINT=dict(
        ENABLED=True,
        PRIMARY_METRIC="bop_ar",
        SECONDARY_METRIC="add_s_0.1d",
        PRIMARY_TIE_TOL=0.001,
    ),
)

TEST = dict(EVAL_PERIOD=5)
TRAIN = dict(PRINT_FREQ=500)
