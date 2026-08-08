_base_ = ["../../_base_/pbr40_screening.py"]

# Copy this directory for a registered experiment before use. Formal configs
# must replace these placeholders and explicitly define trainable modules and
# optimizer settings.
EXPERIMENT_ID = "EXP-REPLACE-BEFORE-USE"
OUTPUT_DIR = "output/experiments/EXP-REPLACE-BEFORE-USE/RUN-SET-BY-LAUNCHER"
SEED = 0

MODEL = dict(
    POSE_NET=dict(
        BACKBONE=dict(FREEZE=True),
        GEO_HEAD=dict(FREEZE=True),
    ),
)
