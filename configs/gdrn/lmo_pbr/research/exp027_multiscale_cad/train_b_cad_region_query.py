"""EXP027-B: fixed-identity hierarchical CAD region queries."""
_base_ = ['./common.py']
EXP027_ARM = 'B_cad_region_query'
OUTPUT_DIR = 'output/experiments/EXP-20260924-027-multiscale-cad-interaction/RUN-SET-BY-LAUNCHER-B'
MODEL = dict(POSE_NET=dict(CAD_ATTENTION_HEAD=dict(ARCHITECTURE='hierarchical_cad_query')))
