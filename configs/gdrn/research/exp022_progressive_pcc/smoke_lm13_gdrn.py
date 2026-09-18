"""Short LM13 smoke on a few real LM images and a few DeepIM renders."""

_base_ = ["./train_lm13_gdrn.py"]

DATASETS = dict(
    TRAIN=("lm_13_train_smoke", "lm_imgn_13_train_1k_per_obj_smoke"),
    TEST=(),
)
DATALOADER = dict(NUM_WORKERS=0)
SOLVER = dict(IMS_PER_BATCH=4, REFERENCE_BS=4, TOTAL_EPOCHS=1,
              CHECKPOINT_PERIOD=1, BEST_CHECKPOINT=dict(ENABLED=False))
TEST = dict(EVAL_PERIOD=0)
