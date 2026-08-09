_base_ = ["./convnext_cpm_head_local_lmo.py"]

# Local chain validation only.  This evaluates the fixed one-epoch smoke
# checkpoint and must not be reported as a formal CPM performance result.
OUTPUT_DIR = "output/cpm_head/local_eval"

RUN_ARTIFACTS = dict(
    _delete_=True,
    STRUCTURED_LAYOUT=True,
    COMPACT_LOG=True,
    TENSORBOARD=False,
    SKIP_DUPLICATE_FINAL_EVAL=True,
)

MODEL = dict(
    WEIGHTS="output/cpm_head/local_integration/model_0002047.pth",
    LOAD_DETS_TEST=False,
)

DATASETS = dict(
    TEST=("lmo_bop_test",),
    DET_FILES_TEST=(),
)

DATALOADER = dict(NUM_WORKERS=0)

TEST = dict(
    EVAL_PERIOD=0,
    TEST_BBOX_TYPE="gt",
    USE_PNP=False,
    AMP_TEST=False,
)
