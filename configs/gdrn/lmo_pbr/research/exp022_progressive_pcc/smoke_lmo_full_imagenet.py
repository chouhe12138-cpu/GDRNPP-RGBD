"""One-epoch managed smoke for the LM-O ImageNet full-training arm."""

_base_ = ["./train_lmo_full_imagenet.py"]

DATALOADER = dict(NUM_WORKERS=2)
DATASETS = dict(TEST=())
SOLVER = dict(IMS_PER_BATCH=4, REFERENCE_BS=4, TOTAL_EPOCHS=1,
              CHECKPOINT_PERIOD=1, BEST_CHECKPOINT=dict(ENABLED=False))
TEST = dict(EVAL_PERIOD=0)
