"""Small legacy LM evaluator probe; oracle poses, not model accuracy."""
_base_ = ['./train_imagenet_full.py']

OUTPUT_DIR = 'output/candidates/lm13_cad/RUN-SET-BY-LOCAL-COMMAND-eval-smoke'
DATASETS = dict(TEST=('lm_13_test_smoke',))
TEST = dict(EVAL_PERIOD=0)
