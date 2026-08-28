_base_ = ["../c_rt_decoupled/train.py"]

EXPERIMENT_ID = "EXP-20260827-014-d-fulltrain-imagenet"
OUTPUT_DIR = "output/experiments/EXP-20260827-014-d-fulltrain-imagenet/RUN-SET-BY-LAUNCHER"

# Full end-to-end: ImageNet-pretrained timm ConvNeXt-Base + random init heads.
# The online XYZ renderer must be overridden inside POSE_NET: the engine only
# reads MODEL.POSE_NET.XYZ_RENDERER (engine_utils.get_renderer), so a top-level
# XYZ_RENDERER key is inert and silently leaves the inherited cpp value in place.
MODEL = dict(
    WEIGHTS="",
    POSE_NET=dict(
        XYZ_RENDERER="egl",
        BACKBONE=dict(FREEZE=False, INIT_CFG=dict(pretrained=True)),
        GEO_HEAD=dict(FREEZE=False, TRAIN_SUPERVISION=True),
    ),
)

SOLVER = dict(WARMUP_ITERS=1000)
