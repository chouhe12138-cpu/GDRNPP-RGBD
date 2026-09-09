_base_ = ["./control.py"]

# Independent post-training BOP evaluation entry (direct head pose telemetry,
# GT box, no TEST.USE_PNP). The PRIMARY EXP020 conclusion is the matched
# classical PnP/RANSAC consumer (EXP019 protocol) run against the trained
# correspondence outputs; this config only provides the repository-standard
# BOP-AR / ADD(-S) checkpoint evaluation path and is not a substitute for it.
MODEL = dict(WEIGHTS="REPLACE-WITH-INDEXED-CHECKPOINT")
DATASETS = dict(TEST=("lmo_bop_test",), DET_FILES_TEST=())
TEST = dict(EVAL_PERIOD=0, TEST_BBOX_TYPE="gt", USE_PNP=False, AMP_TEST=False)
