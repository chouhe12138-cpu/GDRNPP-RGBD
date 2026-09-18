"""LM13 PBR protocol: the GDRNPP/BOP synthetic domain, not the GDR-Net LM one.

Kept as its own arm so the domain-randomisation experiment stays separate from
the real+render main protocol; its colour augmentation and background policy
are the PBR ones from ``common.py``.
"""

import os

_base_ = ["./common.py"]

EXPERIMENT_ID = "EXP-20260918-023-lm13-progressive-pcc-fulltrain"
OUTPUT_DIR = "output/experiments/EXP-20260918-023-lm13-progressive-pcc-fulltrain/RUN-SET-BY-LAUNCHER"
DATASET_CONTEXT = dict(KEY="lm13", CAD_REF_KEY="lm_full", BOP_DATASET="lm",
                       BOP_TARGETS_FILENAME="test_targets_bop19.json")
INPUT = dict(DZI_PAD_SCALE=1.5, COLOR_AUG_PROB=0.8, COLOR_AUG_TYPE="ROI10D",
             CHANGE_BG_PROB=0.5, PBR_CHANGE_BG_PROB=0.5)
DATASETS = dict(TRAIN=("lm_pbr_13_online_train",), TEST=("lm_bop_test_13",))
MODEL = dict(
    POSE_NET=dict(
        NUM_CLASSES=13,
        BACKBONE=dict(FREEZE=False, LR_MULT=0.1, INIT_CFG=dict(
            checkpoint_path=os.environ.get("GDRN_CONVNEXT_BASE_WEIGHTS", ""),
        )),
        PCC_HEAD=dict(HIERARCHY_PATH=os.path.join(
            os.environ.get("GDRN_DATASET_CACHE_DIR", ".local/dataset_cache"),
            "exp022", "lm13", "independent_v2.npz",
        )),
    ),
)
TRAIN_PROTOCOL = dict(NAME="lm13_pbr", DATA_DOMAIN="syn_pbr")
EVAL_PROTOCOL = dict(NAME="bop_official", BBOX_SOURCE="gt")
VAL = dict(DATASET_NAME="lm")
