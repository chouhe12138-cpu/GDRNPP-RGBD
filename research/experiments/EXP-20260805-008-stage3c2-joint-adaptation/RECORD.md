# EXP-20260805-008 — C2 联合适应

状态：`TRIGGERED — FORMAL_PENDING`

C2 从官方 LM-O checkpoint 独立初始化，冻结 backbone 和 geometry head，同时
训练原 Patch-PnP 与质量/覆盖模块。它与 B 使用同一数据、40 epoch、batch 48、
seed 和固定 Epoch 40 LM-O GT-box 评估协议。

本实验的主要因果比较是：

```text
C2 - B = 质量/覆盖模块在 Patch-PnP 同预算适应之外的额外贡献
```

截至最后服务器记录，C2 容器已经准备，流水线等待 GPU1；smoke、worker
benchmark 和正式训练是否已继续必须在服务器实时复查。

配置与当前复现链保持原位：

```text
configs/gdrn/lmo_pbr/convnext_stage3c2_pnp_quality_coverage_lmo.py
docker/l40/lab1_c2.sh
output/stage3c/C2_joint
```

本记录只补充实验身份，不授权修改或重新启动当前服务器任务。
