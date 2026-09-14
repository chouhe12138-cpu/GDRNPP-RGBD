_base_ = ["./common.py"]

MODEL = dict(POSE_NET=dict(CAD_HEAD=dict(INIT_CFG=dict(use_global_guidance=False))))

