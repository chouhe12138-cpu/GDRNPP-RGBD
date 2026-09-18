_base_ = ["./train_lm13_pbr.py"]

DATASETS = dict(TRAIN=("lm_pbr_13_online_smoke",), TEST=())
DATALOADER = dict(NUM_WORKERS=0)
SOLVER = dict(IMS_PER_BATCH=1, REFERENCE_BS=1, TOTAL_EPOCHS=1,
              CHECKPOINT_PERIOD=1, BEST_CHECKPOINT=dict(ENABLED=False))
TEST = dict(EVAL_PERIOD=0)
