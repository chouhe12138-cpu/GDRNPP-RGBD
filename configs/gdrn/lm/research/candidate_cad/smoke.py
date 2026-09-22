"""Short local LM13 data/model smoke; never a formal training entry."""
_base_ = ['./train_imagenet_full.py']

OUTPUT_DIR = 'output/candidates/lm13_cad/RUN-SET-BY-LOCAL-COMMAND-smoke'
DATASETS = dict(TRAIN=('lm_13_train_smoke', 'lm_imgn_13_train_1k_per_obj_smoke'), TEST=())
SOLVER = dict(TOTAL_EPOCHS=1, CHECKPOINT_PERIOD=1)
TEST = dict(EVAL_PERIOD=0)
