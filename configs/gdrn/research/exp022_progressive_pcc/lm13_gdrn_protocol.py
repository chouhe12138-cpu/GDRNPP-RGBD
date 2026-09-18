"""GDR-Net's LINEMOD training protocol: real LM images plus the DeepIM renders.

Follows GDR-Net's ``a6_cPnP_lm13``: 160 epochs at an effective batch of 24,
Ranger at ``1e-4`` without weight decay, 1000 warmup iterations, then a flat
phase that starts cosine annealing at 72% of training.  Colour augmentation is
off; the background policy is the per-domain one from ``common_base``.

This sits in the experiment directory rather than ``_base_/`` because mmcv
rejects duplicate keys between sibling bases: chaining ``common.py`` and a
``_base_`` protocol file together would make both inherit ``research_runtime``
and fail to load.
"""

_base_ = ["./common.py"]

INPUT = dict(
    DZI_TYPE="uniform",
    DZI_PAD_SCALE=1.5,
    DZI_SCALE_RATIO=0.25,
    DZI_SHIFT_RATIO=0.25,
    BG_TYPE="VOC_table",
    BG_IMGS_ROOT="datasets/VOCdevkit/VOC2012/",
    NUM_BG_IMGS=10000,
    # syn renders are always re-backgrounded; `img_type=real` uses
    # CHANGE_BG_PROB and `img_type=syn_pbr` uses PBR_CHANGE_BG_PROB
    CHANGE_BG_PROB=0.5,
    PBR_CHANGE_BG_PROB=0.5,
    TRUNCATE_FG=False,
    COLOR_AUG_PROB=0.0,
)

# GDR-Net keeps every annotated instance; the research default of 0.3 is a
# PBR-screening choice and would silently drop real LM annotations.
DATALOADER = dict(FILTER_VISIB_THR=0.0)

SOLVER = dict(
    IMS_PER_BATCH=4,  # physical batch; REFERENCE_BS accumulates to the nominal one
    REFERENCE_BS=24,
    TOTAL_EPOCHS=160,
    CHECKPOINT_PERIOD=10,
    CHECKPOINT_BY_EPOCH=True,
    MAX_TO_KEEP=10,
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=1e-4, weight_decay=0),
    AMP=dict(ENABLED=True),
    LR_SCHEDULER_NAME="flat_and_anneal",
    # NOTE: WARMUP_RATIO must stay unset here. solver_utils prefers it over
    # WARMUP_ITERS and would overwrite ANNEAL_POINT with that ratio, turning
    # the flat phase into a near-immediate cosine decay.
    WARMUP_RATIO=None,
    WARMUP_FACTOR=0.001,
    WARMUP_ITERS=1000,
    WARMUP_METHOD="linear",
    ANNEAL_METHOD="cosine",
    ANNEAL_POINT=0.72,
    # GDR-Net's LM13 flat_and_anneal decays to zero. The research runtime
    # default of 0.01 would leave the final LR at 1e-6, so override it here
    # rather than in the global base, which other experiments still inherit.
    TARGET_LR_FACTOR=0.0,
)
