"""EXP026 arm G: ImageNet Full with three-level adaptive_512_l1."""
import os

_base_ = ['./common.py']
EXP026_ARM = 'adaptive_l1_full'
OUTPUT_DIR = 'output/experiments/EXP-20260922-026-residual-aligned-sampling-ablation/RUN-SET-BY-LAUNCHER-adaptive-l1'
MODEL = dict(POSE_NET=dict(CAD_ATTENTION_HEAD=dict(HIERARCHY_PATH=os.path.join(
    os.environ.get('GDRN_DATASET_CACHE_DIR', '.local/dataset_cache'), 'exp026',
    'RUN-20260921-ga-hfps-s20260919-a01', 'adaptive_512_l1.npz'))))
CAD_HIERARCHY_CONTRACT = dict(SHA256='7ac75daa756ed7ae59463614737ccc46cd746173452cd1943a4a024d0817c631',
                              MODE='geometry_adaptive', GENERATOR_VERSION=1, DEPTH=3,
                              LEVEL_COUNTS=(8, 64, 512), VARIANT='adaptive_512_l1', LAMBDA_GEO=1.0)
