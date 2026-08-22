_base_ = ["./train.py"]

DATASETS = dict(TRAIN=("lmo_pbr_stage3_local_train",), TEST=())
DATALOADER = dict(NUM_WORKERS=16)
SOLVER = dict(
    IMS_PER_BATCH=48,
    REFERENCE_BS=48,
    TOTAL_EPOCHS=1,
    CHECKPOINT_PERIOD=1,
    MAX_TO_KEEP=1,
    BEST_CHECKPOINT=dict(ENABLED=False),
)
TEST = dict(EVAL_PERIOD=0)
