_base_ = ["../c_rt_decoupled/train.py"]

EXPERIMENT_ID = "EXP-20260827-014-d-fulltrain-imagenet"
OUTPUT_DIR = "output/experiments/EXP-20260827-014-d-fulltrain-imagenet/RUN-SET-BY-LAUNCHER"

# Full end-to-end: ImageNet-pretrained timm ConvNeXt-Base + random init heads.
XYZ_RENDERER = "egl"

MODEL = dict(
    WEIGHTS="",
    POSE_NET=dict(
        BACKBONE=dict(FREEZE=False, INIT_CFG=dict(pretrained=True)),
        GEO_HEAD=dict(FREEZE=False, TRAIN_SUPERVISION=True),
    ),
)

SOLVER = dict(WARMUP_ITERS=1000)
