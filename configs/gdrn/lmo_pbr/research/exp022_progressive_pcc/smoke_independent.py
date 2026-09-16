import os

_base_ = ["./smoke_reused.py"]

MODEL = dict(POSE_NET=dict(PCC_HEAD=dict(HIERARCHY_PATH=os.path.join(
    os.environ.get("GDRN_DATASET_CACHE_DIR", ".local/dataset_cache"),
    "exp022", "independent_v2.npz",
))))
