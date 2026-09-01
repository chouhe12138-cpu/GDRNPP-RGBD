# EXP014-D — ImageNet Full End-to-End Training

- 状态：`PAUSED / FORMAL_A01_INVALIDATED / NO_ACTIVE_RETRAIN`
- 配置：`configs/gdrn/lmo_pbr/research/exp013/d_fulltrain/train.py`
- 服务器：lab1；seed：42

本地 full-model gate 和一轮 smoke 曾通过。服务器 formal a01 暴露 CPP renderer
覆盖 EGL 配置以及 OOM 问题，该次运行作废，不形成科学结果。EGL 修复和配置保留，
但当前没有重训授权；安全启动器和 RUNBOOK 不会自动恢复 D。
