"""LM-O PCC full training from the ImageNet ConvNeXt-Base backbone."""

import os

_base_ = ["./train_reused.py"]

EXPERIMENT_ID = "EXP-20260919-024-lmo-progressive-pcc-fulltrain"
OUTPUT_DIR = "output/experiments/EXP-20260919-024-lmo-progressive-pcc-fulltrain/RUN-SET-BY-LAUNCHER"

MODEL = dict(
    WEIGHTS="",
    POSE_NET=dict(
        BACKBONE=dict(
            FREEZE=False,
            LR_MULT=0.1,
            INIT_CFG=dict(
                checkpoint_path=os.environ.get("GDRN_CONVNEXT_BASE_WEIGHTS", ""),
            ),
        ),
    ),
)

# Keep EXP022's LM-O PBR40/GT-box protocol and effective batch 48.  A physical
# batch of four uses the engine's existing 12-step gradient accumulation.
SOLVER = dict(IMS_PER_BATCH=4, REFERENCE_BS=48)
TRAIN_PROTOCOL = dict(NAME="lmo_full_imagenet", DATA_DOMAIN="syn_pbr")
RESEARCH_PROTOCOL = dict(SCHEDULE="configurable", FORMAL_READY=False)
