"""EXP027-A formal release on lab2 / physical GPU 2."""
_base_ = ['./train_a_multiscale_fpn.py']

OUTPUT_DIR = 'output/experiments/EXP-20260924-027-multiscale-cad-interaction/RUN-SET-BY-LAUNCHER-A-LAB2'
RESEARCH_PROTOCOL = dict(SERVER_RELEASE_ALLOWED=True, FORMAL_READY=True)
