_base_ = ["./train.py"]

# No XYZ_RENDERER override: frozen geometry supervision is disabled in
# train.py, so the engine never constructs a CPP or EGL renderer.

DATASETS = dict(TRAIN=("lmo_pbr_stage3_local_train",), TEST=())
DATALOADER = dict(NUM_WORKERS=2)
SOLVER = dict(
    IMS_PER_BATCH=4,
    REFERENCE_BS=48,
    TOTAL_EPOCHS=1,
    CHECKPOINT_PERIOD=1,
    MAX_TO_KEEP=1,
    BEST_CHECKPOINT=dict(ENABLED=False),
)
TEST = dict(EVAL_PERIOD=0)
