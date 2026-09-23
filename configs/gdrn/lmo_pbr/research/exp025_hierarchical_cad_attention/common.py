"""LM-O EXP025 protocol. Arm configs own initialization and freeze state."""
import os

_base_ = ['../../../research/cad/_base_/common.py']
EXPERIMENT_ID = 'EXP-20260920-025-hierarchical-cad-attention'
OUTPUT_DIR = 'output/experiments/' + EXPERIMENT_ID + '/RUN-SET-BY-LAUNCHER'
SEED = 42
INPUT = dict(
    DZI_PAD_SCALE=1.5, COLOR_AUG_PROB=.8, COLOR_AUG_TYPE='code',
    COLOR_AUG_CODE=(
        'Sequential(['
        'Sometimes(0.5, CoarseDropout( p=0.2, size_percent=0.05) ),'
        'Sometimes(0.4, GaussianBlur((0., 3.))),'
        'Sometimes(0.3, pillike.EnhanceSharpness(factor=(0., 50.))),'
        'Sometimes(0.3, pillike.EnhanceContrast(factor=(0.2, 50.))),'
        'Sometimes(0.5, pillike.EnhanceBrightness(factor=(0.1, 6.))),'
        'Sometimes(0.3, pillike.EnhanceColor(factor=(0., 20.))),'
        'Sometimes(0.5, Add((-25, 25), per_channel=0.3)),'
        'Sometimes(0.3, Invert(0.2, per_channel=True)),'
        'Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),'
        'Sometimes(0.5, Multiply((0.6, 1.4))),'
        'Sometimes(0.1, AdditiveGaussianNoise(scale=10, per_channel=True)),'
        'Sometimes(0.5, iaa.contrast.LinearContrast((0.5, 2.2), per_channel=0.3)),'
        'Sometimes(0.5, Grayscale(alpha=(0.0, 1.0))),'
        '], random_order=True)'))
DATASETS = dict(TRAIN=('lmo_pbr_train',), TEST=('lmo_bop_test',), DET_FILES_TEST=())
DATASET_CONTEXT = dict(KEY='lmo', CAD_REF_KEY='lm_full', BOP_DATASET='lmo',
                       BOP_TARGETS_FILENAME='test_targets_bop19.json')
CAD_HIERARCHY_CONTRACT = dict(
    SHA256='02ce090949bc40b2732417fec23984f3f748431098c5f67c853839d10ff1a373',
    DATASET_KEY='lmo', MODE='consistent', GENERATOR_VERSION=3,
    DEPTH=4, LEVEL_COUNTS=(8, 64, 512, 4096))
MODEL = dict(BBOX_TYPE='AMODAL_CLIP', LOAD_DETS_TEST=False, POSE_NET=dict(
    NUM_CLASSES=8, XYZ_RENDERER='egl',
    GEO_HEAD=dict(NUM_REGIONS=64),
    PNP_NET=dict(INIT_CFG=dict(norm='GN', act='gelu'), REGION_ATTENTION=True,
                 WITH_2D_COORD=True, ROT_TYPE='allo_rot6d', TRANS_TYPE='centroid_z'),
    CAD_ATTENTION_HEAD=dict(ENABLED=True, HIERARCHY_PATH=os.path.join(
        os.environ.get('GDRN_DATASET_CACHE_DIR', '.local/dataset_cache'), 'exp025', 'consistent_v3.npz')),
    LOSS_CFG=dict(FULL_MASK_LOSS_TYPE='L1', FULL_MASK_LW=1., PM_LOSS_SYM=True,
                  CENTROID_LW=1., Z_LW=1.)))
DATALOADER = dict(NUM_WORKERS=16, FILTER_VISIB_THR=.3)
SOLVER = dict(IMS_PER_BATCH=48, REFERENCE_BS=48, TOTAL_EPOCHS=40, MAX_TO_KEEP=10,
    WEIGHT_DECAY=0.0,
    OPTIMIZER_CFG=dict(_delete_=True, type='AdamW', lr=3e-4, weight_decay=.01,
                      betas=(.9, .999), eps=1e-8),
    AMP=dict(ENABLED=True, INIT_SCALE=32768), WARMUP_RATIO=.04, WARMUP_FACTOR=.001,
    WARMUP_ITERS=1000, WARMUP_METHOD='linear',
    LR_SCHEDULER_NAME='flat_and_anneal', ANNEAL_METHOD='cosine', TARGET_LR_FACTOR=.01,
    BEST_CHECKPOINT=dict(ENABLED=False))
TEST = dict(TEST_BBOX_TYPE='gt', USE_PNP=True, PNP_TYPE='ransac_pnp', EVAL_PERIOD=5)
VAL = dict(DATASET_NAME='lmo', SCRIPT_PATH='lib/pysixd/scripts/eval_pose_results_more.py',
           TARGETS_FILENAME='test_targets_bop19.json', ERROR_TYPES='mspd,mssd,vsd,ad,reS,teS',
           RENDERER_TYPE='cpp', SPLIT='test', SPLIT_TYPE='', N_TOP=1, USE_BOP=True)
TRAIN = dict(PRINT_FREQ=500)
RUN_ARTIFACTS = dict(STRUCTURED_LAYOUT=True, COMPACT_LOG=True, TENSORBOARD=False,
                     SKIP_DUPLICATE_FINAL_EVAL=True)
TRAIN_PROTOCOL = dict(NAME='exp025_lmo', DATA_DOMAIN='syn_pbr')
RESEARCH_PROTOCOL = dict(SCHEDULE='configurable', FORMAL_READY=True)
