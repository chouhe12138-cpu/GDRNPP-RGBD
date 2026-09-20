"""EXP025 single T3=512 distribution; edit these three independent controls."""
import os
from research.exp025.configuration import backbone_settings

TRAIN_BACKBONE = False
BACKBONE_INIT = 'official_lmo'
BACKBONE_LR_MULT = .1

_base_ = ['../_base_/pbr40_screening.py']
EXPERIMENT_ID = 'EXP-20260920-025-hierarchical-cad-attention'
OUTPUT_DIR = 'output/experiments/' + EXPERIMENT_ID + '/RUN-SET-BY-LAUNCHER'
SEED = 42
DATASET_CONTEXT = dict(KEY='lmo', CAD_REF_KEY='lm_full', BOP_DATASET='lmo',
                       BOP_TARGETS_FILENAME='test_targets_bop19.json')
MODEL = backbone_settings(TRAIN_BACKBONE, BACKBONE_INIT, BACKBONE_LR_MULT)
MODEL['POSE_NET'].update(dict(
    NAME='GDRN_CAD', XYZ_ONLINE=True, XYZ_RENDERER='egl', XYZ_BP=True,
    GEO_HEAD=dict(FREEZE=True, TRAIN_SUPERVISION=False),
    CAD_ATTENTION_HEAD=dict(ENABLED=True, HIERARCHY_PATH=os.path.join(
        os.environ.get('GDRN_DATASET_CACHE_DIR', '.local/dataset_cache'), 'exp022', 'consistent_v3.npz'),
        INIT_CFG=dict(token_dim=256, num_heads=8, route_weight=1., residual_weight=1.,
                      mask_weight=1., residual_beta=.1)),
    LOSS_CFG=dict(MASK_LOSS_TYPE='BCE')))
DATALOADER = dict(NUM_WORKERS=16)
SOLVER = dict(IMS_PER_BATCH=4, REFERENCE_BS=48, TOTAL_EPOCHS=40, MAX_TO_KEEP=10,
    OPTIMIZER_CFG=dict(_delete_=True, type='AdamW', lr=3e-4, weight_decay=.01,
                      betas=(.9, .999), eps=1e-8),
    AMP=dict(ENABLED=True), WARMUP_RATIO=.04, WARMUP_FACTOR=.001, WARMUP_METHOD='linear',
    LR_SCHEDULER_NAME='flat_and_anneal', ANNEAL_METHOD='cosine', TARGET_LR_FACTOR=.01,
    BEST_CHECKPOINT=dict(ENABLED=False))
TEST = dict(TEST_BBOX_TYPE='gt', USE_PNP=True, PNP_TYPE='ransac_pnp', EVAL_PERIOD=5)
TRAIN_PROTOCOL = dict(NAME='exp025_lmo', DATA_DOMAIN='syn_pbr')
RESEARCH_PROTOCOL = dict(SCHEDULE='configurable', FORMAL_READY=False)
del backbone_settings
