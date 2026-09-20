"""Prepared LM13 arm: historical GDR-Net protocol with the EXP025 head.

This config is intentionally not accepted by the server launcher yet. LM-O finishes
first; LM13 must receive its own server resource gate and CUDA/EGL batch gate later.
"""
import os

from research.exp025.configuration import backbone_settings

_base_ = ['./common.py']

EXP025_ARM = 'lm13_imagenet_full_prepared'
TRAIN_BACKBONE = True
BACKBONE_INIT = 'imagenet'
BACKBONE_LR_MULT = .1

DATASET_CONTEXT = dict(KEY='lm13', CAD_REF_KEY='lm_full', BOP_DATASET='lm',
                       BOP_TARGETS_FILENAME='test_targets_bop19.json')
DATASETS = dict(
    TRAIN=('lm_13_train_online', 'lm_imgn_13_train_1k_per_obj_online'),
    TEST=('lm_13_test_online',), DET_FILES_TEST=())

INPUT = dict(
    DZI_TYPE='uniform', DZI_PAD_SCALE=1.5, DZI_SCALE_RATIO=.25, DZI_SHIFT_RATIO=.25,
    BG_TYPE='VOC_table', BG_IMGS_ROOT='datasets/VOCdevkit/VOC2012/', NUM_BG_IMGS=10000,
    CHANGE_BG_PROB=.5, PBR_CHANGE_BG_PROB=.5, TRUNCATE_FG=False, COLOR_AUG_PROB=0.)
DATALOADER = dict(NUM_WORKERS=16, FILTER_VISIB_THR=0.)

MODEL = backbone_settings(TRAIN_BACKBONE, BACKBONE_INIT, BACKBONE_LR_MULT)
MODEL['LOAD_DETS_TEST'] = False
MODEL['POSE_NET'].update(
    NUM_CLASSES=13,
    CAD_ATTENTION_HEAD=dict(HIERARCHY_PATH=os.path.join(
        os.environ.get('GDRN_DATASET_CACHE_DIR', '.local/dataset_cache'),
        'exp025', 'lm13', 'consistent_v3.npz')))

SOLVER = dict(
    IMS_PER_BATCH=4, REFERENCE_BS=24, TOTAL_EPOCHS=160,
    CHECKPOINT_PERIOD=20, CHECKPOINT_BY_EPOCH=True, MAX_TO_KEEP=8,
    OPTIMIZER_CFG=dict(_delete_=True, type='Ranger', lr=1e-4, weight_decay=0),
    AMP=dict(ENABLED=True), LR_SCHEDULER_NAME='flat_and_anneal',
    WARMUP_RATIO=None, WARMUP_FACTOR=.001, WARMUP_ITERS=1000, WARMUP_METHOD='linear',
    ANNEAL_METHOD='cosine', ANNEAL_POINT=.72, TARGET_LR_FACTOR=0.,
    BEST_CHECKPOINT=dict(ENABLED=False))

TEST = dict(EVAL_PERIOD=20, TEST_BBOX_TYPE='gt', USE_PNP=True, PNP_TYPE='ransac_pnp')
TRAIN_PROTOCOL = dict(NAME='exp025_lm13', DATA_DOMAIN='real+imgn')
EVAL_PROTOCOL = dict(NAME='lm_legacy_diagnostic', BBOX_SOURCE='gt')
VAL = dict(DATASET_NAME='lm', SCRIPT_PATH='lib/pysixd/scripts/eval_pose_results_more.py',
           TARGETS_FILENAME='lm_test_targets_bb8.json', ERROR_TYPES='ad,rete,re,te,proj',
           SPLIT='test', SPLIT_TYPE='bb8', N_TOP=1, USE_BOP=False, RENDERER_TYPE='cpp')
RESEARCH_PROTOCOL = dict(SCHEDULE='configurable', FORMAL_READY=False)

del backbone_settings
