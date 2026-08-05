_base_ = ["./convnext_stage3c0_pnp_only_lmo.py"]

OUTPUT_DIR = "output/EXP-20260731-005/pnp_only_local"

RUN_ARTIFACTS = dict(
    _delete_=True,
    STRUCTURED_LAYOUT=False,
    COMPACT_LOG=False,
    TENSORBOARD=True,
    SKIP_DUPLICATE_FINAL_EVAL=False,
)

DATASETS = dict(
    TRAIN=("lmo_pbr_stage3_local_train",),
    TEST=(),
)

DATALOADER = dict(
    NUM_WORKERS=0,
    FILTER_VISIB_THR=0.3,
)

SOLVER = dict(
    IMS_PER_BATCH=4,
    REFERENCE_BS=48,
    TOTAL_EPOCHS=1,
    WARMUP_ITERS=20,
    CHECKPOINT_PERIOD=1,
)

TEST = dict(EVAL_PERIOD=0)

TRAIN = dict(PRINT_FREQ=20)
