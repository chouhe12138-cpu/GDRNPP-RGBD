"""T-LESS context reservation; requires data, hierarchy and symmetry support."""

import os

_base_ = ["./common.py"]

EXPERIMENT_ID = "EXP022-TLESS-NOT-ASSIGNED"
DATASET_CONTEXT = dict(KEY="tless", CAD_REF_KEY="tless", BOP_DATASET="tless",
                       BOP_TARGETS_FILENAME="test_targets_bop19.json")
DATASETS = dict(TRAIN=("tless_train_pbr",), TEST=("tless_bop_test_primesense",))
MODEL = dict(POSE_NET=dict(NUM_CLASSES=30, PCC_HEAD=dict(HIERARCHY_PATH=os.path.join(
    os.environ.get("GDRN_DATASET_CACHE_DIR", ".local/dataset_cache"),
    "exp022", "tless", "independent_v2.npz",
))))
VAL = dict(DATASET_NAME="tless")
