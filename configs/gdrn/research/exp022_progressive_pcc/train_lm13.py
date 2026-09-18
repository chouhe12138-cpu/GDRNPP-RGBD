"""LM13 full backbone plus PCC preparation; no formal run is authorized yet."""

import os

_base_ = ["./common.py"]

EXPERIMENT_ID = "EXP-20260918-023-lm13-progressive-pcc-fulltrain"
OUTPUT_DIR = "output/experiments/EXP-20260918-023-lm13-progressive-pcc-fulltrain/RUN-SET-BY-LAUNCHER"
DATASET_CONTEXT = dict(KEY="lm13", CAD_REF_KEY="lm_full", BOP_DATASET="lm",
                       BOP_TARGETS_FILENAME="test_targets_bop19.json")
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
VAL = dict(DATASET_NAME="lm")
