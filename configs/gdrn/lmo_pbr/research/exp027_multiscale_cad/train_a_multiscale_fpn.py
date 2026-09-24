"""EXP027-A: image-query CAD interaction at four visual scales."""
_base_ = ['./common.py']
EXP027_ARM = 'A_multiscale_fpn'
OUTPUT_DIR = 'output/experiments/EXP-20260924-027-multiscale-cad-interaction/RUN-SET-BY-LAUNCHER-A'
MODEL = dict(POSE_NET=dict(CAD_ATTENTION_HEAD=dict(ARCHITECTURE='multiscale_image_query')))
