"""Historical LM13 prepared entry; use the neutral LM candidate path for new work.

This compatibility entry preserves its EXP025-era identity and locked status.
"""
_base_ = ['../../../lm/research/candidate_cad/train_imagenet_full.py']

EXPERIMENT_ID = 'EXP-20260920-025-hierarchical-cad-attention'
OUTPUT_DIR = 'output/experiments/EXP-20260920-025-hierarchical-cad-attention/RUN-SET-BY-LAUNCHER'
EXP025_ARM = 'lm13_imagenet_full_prepared'
TRAIN_PROTOCOL = dict(NAME='exp025_lm13')
RESEARCH_PROTOCOL = dict(STAGE='prepared')
