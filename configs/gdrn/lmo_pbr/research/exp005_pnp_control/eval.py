_base_ = ["./train.py"]

# The launch step must replace this with an explicitly indexed checkpoint.
MODEL = dict(WEIGHTS="REPLACE-WITH-INDEXED-CHECKPOINT")

DATASETS = dict(
    TEST=("lmo_bop_test",),
    DET_FILES_TEST=(),
)

TEST = dict(
    EVAL_PERIOD=0,
    TEST_BBOX_TYPE="gt",
    USE_PNP=False,
    AMP_TEST=False,
)
