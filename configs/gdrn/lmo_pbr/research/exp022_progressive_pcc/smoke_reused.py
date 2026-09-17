_base_ = ["./train_reused.py"]

DATALOADER = dict(NUM_WORKERS=2)
DATASETS = dict(TEST=())
MODEL = dict(POSE_NET=dict(PCC_HEAD=dict(INIT_CFG=dict(
    token_dim=256, beam_k=2, num_heads=8,
    stage_attention=("global", "global", "window", "window"),
    window_size=8, shift_size=4, attention_dropout=0.0,
))))
SOLVER = dict(
    IMS_PER_BATCH=4, REFERENCE_BS=4, TOTAL_EPOCHS=1,
    CHECKPOINT_PERIOD=1, BEST_CHECKPOINT=dict(ENABLED=False),
)
TEST = dict(EVAL_PERIOD=0)
