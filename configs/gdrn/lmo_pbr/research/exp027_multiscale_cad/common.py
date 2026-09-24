"""EXP027 matched LM-O protocol; architecture is chosen by each arm."""
import os

_base_ = ['../../../research/cad/_base_/common.py']
EXPERIMENT_ID = 'EXP-20260924-027-multiscale-cad-interaction'
OUTPUT_DIR = 'output/experiments/' + EXPERIMENT_ID + '/RUN-SET-BY-LAUNCHER'
SEED = 42
TRAIN_BACKBONE = True
BACKBONE_INIT = 'imagenet'
BACKBONE_LR_MULT = 1.0
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
    SHA256='7ac75daa756ed7ae59463614737ccc46cd746173452cd1943a4a024d0817c631',
    DATASET_KEY='lmo', MODE='geometry_adaptive', GENERATOR_VERSION=1,
    DEPTH=3, LEVEL_COUNTS=(8, 64, 512), VARIANT='adaptive_512_l1', LAMBDA_GEO=1.0)
MODEL = dict(WEIGHTS='', BBOX_TYPE='AMODAL_CLIP', LOAD_DETS_TEST=False, POSE_NET=dict(
    NUM_CLASSES=8, XYZ_RENDERER='egl',
    BACKBONE=dict(FREEZE=False, LR_MULT=1.0, INIT_CFG=dict(out_indices=(0, 1, 2, 3),
        checkpoint_path=os.environ.get('GDRN_CONVNEXT_BASE_WEIGHTS', ''))),
    GEO_HEAD=dict(NUM_REGIONS=64),
    PNP_NET=dict(INIT_CFG=dict(norm='GN', act='gelu'), REGION_ATTENTION=True,
                 WITH_2D_COORD=True, ROT_TYPE='allo_rot6d', TRANS_TYPE='centroid_z'),
    CAD_ATTENTION_HEAD=dict(ENABLED=True, HIERARCHY_PATH=os.path.join(
        os.environ.get('GDRN_DATASET_CACHE_DIR', '.local/dataset_cache'), 'exp026',
        'RUN-20260921-ga-hfps-s20260919-a01', 'adaptive_512_l1.npz'),
        INIT_CFG=dict(residual_target_mode='predicted_route',
                      backbone_channels=(128, 256, 512, 1024),
                      feature_resolutions=(64, 32, 16, 8),
                      pyramid_channels=(64, 128, 256, 512))),
    LOSS_CFG=dict(FULL_MASK_LOSS_TYPE='L1', FULL_MASK_LW=1., PM_LOSS_SYM=True,
                  CENTROID_LW=1., Z_LW=1.)))
DATALOADER = dict(NUM_WORKERS=16, FILTER_VISIB_THR=.3)
SOLVER = dict(IMS_PER_BATCH=48, REFERENCE_BS=48, TOTAL_EPOCHS=40, MAX_TO_KEEP=10,
    WEIGHT_DECAY=0.0,
    OPTIMIZER_CFG=dict(_delete_=True, type='AdamW', lr=3e-4, weight_decay=.01,
                      betas=(.9, .999), eps=1e-8),
    AMP=dict(ENABLED=True, INIT_SCALE=4096), WARMUP_RATIO=.04, WARMUP_FACTOR=.001,
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
TRAIN_PROTOCOL = dict(NAME='exp027_lmo', DATA_DOMAIN='syn_pbr')
RESEARCH_PROTOCOL = dict(SCHEDULE='configurable', FORMAL_READY=False,
                         SERVER_RELEASE_ALLOWED=False, LOCAL_FORMAL_READY=False)
