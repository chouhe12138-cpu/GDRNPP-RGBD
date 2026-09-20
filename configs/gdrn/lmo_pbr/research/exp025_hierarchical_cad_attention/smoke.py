_base_ = ['./train.py']
DATALOADER = dict(NUM_WORKERS=0, PERSISTENT_WORKERS=False)
DATASETS = dict(TRAIN=('lmo_exp025_smoke',), TEST=())
DATA_CFG = dict(lmo_exp025_smoke='research/exp025/smoke_dataset.json')
SOLVER = dict(IMS_PER_BATCH=4, REFERENCE_BS=4, TOTAL_EPOCHS=1, CHECKPOINT_PERIOD=1)
TEST = dict(EVAL_PERIOD=0)
