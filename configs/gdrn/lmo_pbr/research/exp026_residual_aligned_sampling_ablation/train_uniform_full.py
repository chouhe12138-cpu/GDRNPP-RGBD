"""EXP026 arm U: ImageNet Full with three-level uniform_512."""
import os

_base_ = ['./common.py']
EXP026_ARM = 'uniform_full'
OUTPUT_DIR = 'output/experiments/EXP-20260922-026-residual-aligned-sampling-ablation/RUN-SET-BY-LAUNCHER-uniform'
MODEL = dict(POSE_NET=dict(CAD_ATTENTION_HEAD=dict(HIERARCHY_PATH=os.path.join(
    os.environ.get('GDRN_DATASET_CACHE_DIR', '.local/dataset_cache'), 'exp026',
    'RUN-20260921-ga-hfps-s20260919-a01', 'uniform_512.npz'))))
CAD_HIERARCHY_CONTRACT = dict(SHA256='35287beb3dc67b5f3cd376cc445f7727857aefa888dd11d639ab65b9193235c1',
                              MODE='geometry_adaptive', GENERATOR_VERSION=1, DEPTH=3,
                              LEVEL_COUNTS=(8, 64, 512), VARIANT='uniform_512', LAMBDA_GEO=0.0)
