"""LM13 CAD candidate: historical real + ImageNet-render protocol.

Candidate only: no EXP number, server profile, or formal result.  The shared
CAD model is selected by this config, without dataset-specific model branches.
"""
import os

from research.cad_common.configuration import backbone_settings

_base_ = ['../../../research/cad/_base_/common.py']

EXPERIMENT_ID = 'LM13-CAD-CANDIDATE'
OUTPUT_DIR = 'output/candidates/lm13_cad/RUN-SET-BY-LOCAL-COMMAND'
SEED = 42
TRAIN_BACKBONE = True
BACKBONE_INIT = 'imagenet'
BACKBONE_LR_MULT = .1

DATASET_CONTEXT = dict(KEY='lm13', CAD_REF_KEY='lm_full', BOP_DATASET='lm',
                       BOP_TARGETS_FILENAME='test_targets_bop19.json')
DATASETS = dict(TRAIN=('lm_13_train_online', 'lm_imgn_13_train_1k_per_obj_online'),
                TEST=('lm_13_test_online',), DET_FILES_TEST=())
CAD_HIERARCHY_CONTRACT = dict(
    SHA256='322cd3778f0325838675a7dc6bae1a4e1cf107bfa95e05c006dcee0836a66417',
    DATASET_KEY='lm13', MODE='consistent', GENERATOR_VERSION=3,
    DEPTH=4, LEVEL_COUNTS=(8, 64, 512, 4096))

INPUT = dict(
    DZI_TYPE='uniform', DZI_PAD_SCALE=1.5, DZI_SCALE_RATIO=.25, DZI_SHIFT_RATIO=.25,
    BG_TYPE='VOC_table', BG_IMGS_ROOT='datasets/VOCdevkit/VOC2012/', NUM_BG_IMGS=10000,
    CHANGE_BG_PROB=.5, PBR_CHANGE_BG_PROB=.5, TRUNCATE_FG=False, COLOR_AUG_PROB=0.,
    COLOR_AUG_TYPE='code',
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
DATALOADER = dict(NUM_WORKERS=16, FILTER_VISIB_THR=0.)

MODEL = backbone_settings(TRAIN_BACKBONE, BACKBONE_INIT, BACKBONE_LR_MULT)
MODEL['LOAD_DETS_TEST'] = False
MODEL['BBOX_TYPE'] = 'AMODAL_CLIP'
MODEL['POSE_NET'].update(
    NUM_CLASSES=13, XYZ_RENDERER='egl',
    GEO_HEAD=dict(NUM_REGIONS=64),
    PNP_NET=dict(INIT_CFG=dict(norm='GN', act='gelu'), REGION_ATTENTION=True,
                 WITH_2D_COORD=True, ROT_TYPE='allo_rot6d', TRANS_TYPE='centroid_z'),
    LOSS_CFG=dict(FULL_MASK_LOSS_TYPE='L1', FULL_MASK_LW=1., PM_LOSS_SYM=True,
                  CENTROID_LW=1., Z_LW=1.),
    CAD_ATTENTION_HEAD=dict(HIERARCHY_PATH=os.path.join(
        os.environ.get('GDRN_DATASET_CACHE_DIR', '.local/dataset_cache'),
        'exp025', 'lm13', 'consistent_v3.npz'),
        INIT_CFG=dict(residual_target_mode='gt_route')))

SOLVER = dict(
    IMS_PER_BATCH=4, REFERENCE_BS=24, TOTAL_EPOCHS=160,
    WEIGHT_DECAY=0.0,
    CHECKPOINT_PERIOD=20, CHECKPOINT_BY_EPOCH=True, MAX_TO_KEEP=8,
    OPTIMIZER_CFG=dict(_delete_=True, type='Ranger', lr=1e-4, weight_decay=0),
    AMP=dict(_delete_=True, ENABLED=True), LR_SCHEDULER_NAME='flat_and_anneal',
    WARMUP_RATIO=None, WARMUP_FACTOR=.001, WARMUP_ITERS=1000, WARMUP_METHOD='linear',
    ANNEAL_METHOD='cosine', ANNEAL_POINT=.72, TARGET_LR_FACTOR=0.,
    BEST_CHECKPOINT=dict(ENABLED=False))

TEST = dict(EVAL_PERIOD=20, TEST_BBOX_TYPE='gt', USE_PNP=True, PNP_TYPE='ransac_pnp')
TRAIN_PROTOCOL = dict(NAME='cad_candidate', DATA_DOMAIN='real+imgn')
TRAIN = dict(PRINT_FREQ=500)
TRAIN_PROFILE = dict(NAME='lm13_legacy_full')
EVAL_PROTOCOL = dict(NAME='lm_legacy_diagnostic', BBOX_SOURCE='gt')
# USE_BOP=False selects GDRN_EvaluatorCustom, which reads GT from lm_13_test_online.
# The bb8 target filename is historical and is only passed to the BOP script path.
VAL = dict(DATASET_NAME='lm', SCRIPT_PATH='lib/pysixd/scripts/eval_pose_results_more.py',
           TARGETS_FILENAME='lm_test_targets_bb8.json', ERROR_TYPES='ad,rete,re,te,proj',
           SPLIT='test', SPLIT_TYPE='bb8', N_TOP=1, USE_BOP=False, RENDERER_TYPE='cpp')
RESEARCH_PROTOCOL = dict(SCHEDULE='configurable', STAGE='candidate', FORMAL_READY=False,
                         SERVER_RELEASE_ALLOWED=False)
RUN_ARTIFACTS = dict(STRUCTURED_LAYOUT=True, COMPACT_LOG=True, TENSORBOARD=False,
                     SKIP_DUPLICATE_FINAL_EVAL=True)

del backbone_settings
