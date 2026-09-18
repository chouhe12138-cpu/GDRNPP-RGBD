"""LM13 BOP-official evaluation: the same training protocol, BOP test split."""

_base_ = ["./train_lm13_gdrn.py"]

DATASETS = dict(TEST=("lm_bop_test_13",))
# BBOX_SOURCE stays gt until the official Faster R-CNN boxes are available, so
# this is the BOP metric with a GT-box upper bound, not a detector result.
EVAL_PROTOCOL = dict(NAME="bop_official", BBOX_SOURCE="gt")
TEST = dict(TEST_BBOX_TYPE="gt", USE_PNP=True, PNP_TYPE="ransac_pnp")

VAL = dict(
    DATASET_NAME="lm",
    TARGETS_FILENAME="test_targets_bop19.json",
    ERROR_TYPES="mspd,mssd,vsd,ad,reS,teS",
    SPLIT="test",
    SPLIT_TYPE="",
    N_TOP=1,
    USE_BOP=True,
    RENDERER_TYPE="cpp",
)
