"""LM13 main protocol: real LM images plus the DeepIM renders (GDR-Net LM)."""

import os

_base_ = ["./lm13_gdrn_protocol.py"]

EXPERIMENT_ID = "EXP-20260918-023-lm13-progressive-pcc-fulltrain"
OUTPUT_DIR = "output/experiments/EXP-20260918-023-lm13-progressive-pcc-fulltrain/RUN-SET-BY-LAUNCHER"
DATASET_CONTEXT = dict(KEY="lm13", CAD_REF_KEY="lm_full", BOP_DATASET="lm",
                       BOP_TARGETS_FILENAME="test_targets_bop19.json")
DATASETS = dict(
    TRAIN=(
        "lm_13_train_online",
        "lm_imgn_13_train_1k_per_obj_online",
    ),
    # `_online` is the same image/pose/bbox protocol with require_xyz=False, so
    # evaluation stops depending on a pre-generated xyz_crop the test set never
    # needs.
    TEST=("lm_13_test_online",),
    # The official Faster R-CNN boxes for LM are not available locally, so the
    # detector-bbox protocol cannot run yet; see TRAIN_PROTOCOL/EVAL_PROTOCOL.
    DET_FILES_TEST=(),
)
MODEL = dict(
    LOAD_DETS_TEST=False,
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
TEST = dict(
    # A full LM test pass runs RANSAC-PnP over ~13k images, and best-checkpoint
    # selection is off, so epoch 5 is too frequent for 160 epochs: every 20
    # epochs gives 8 checkpoints of interest instead of 32 full evaluations.
    # run_contract requires EVAL_PERIOD > 0 for a configurable formal run.
    EVAL_PERIOD=20,
    TEST_BBOX_TYPE="gt",
    USE_PNP=True,
    PNP_TYPE="ransac_pnp",
)

TRAIN_PROTOCOL = dict(NAME="lm13_gdrn", DATA_DOMAIN="real+imgn")
# BBOX_SOURCE=gt keeps this a diagnostic until the official detector boxes are
# downloaded; results are not comparable to detector-bbox LM numbers.
EVAL_PROTOCOL = dict(NAME="lm_legacy_diagnostic", BBOX_SOURCE="gt")

VAL = dict(
    DATASET_NAME="lm",
    SCRIPT_PATH="lib/pysixd/scripts/eval_pose_results_more.py",
    TARGETS_FILENAME="lm_test_targets_bb8.json",
    ERROR_TYPES="ad,rete,re,te,proj",
    SPLIT="test",
    SPLIT_TYPE="bb8",
    N_TOP=1,
    USE_BOP=False,
    RENDERER_TYPE="cpp",
)

# Formal runs stay locked at the research_runtime default (FORMAL_READY=False)
# until the training renderer is verified on the server.  The formal protocol
# needs XYZ_RENDERER="egl" for online geometry, which this machine cannot
# provide ("Bindless Textures not supported"), so the local checks cover
# everything except the renderer.  Enable the override below only after the
# EGL smoke, backward, optimizer step and checkpoint write pass there.
# RESEARCH_PROTOCOL = dict(SCHEDULE="configurable", FORMAL_READY=True)
