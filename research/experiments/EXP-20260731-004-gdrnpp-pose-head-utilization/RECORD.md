# EXP004 — Pose-Head XYZ Utilization

- 状态：`COMPLETE / PATCH_PNP_UNDERUTILIZATION`
- 协议：固定官方头与 support，把预测 XYZ 按 alpha 插值到 GT XYZ
- 范围：LM-O BOP19 1,445 targets、无训练

RANSAC 的 ADD(-S) 随 alpha 从 0 到 1 单调由 53.841% 提高到 99.377%，而
官方 Patch-PnP 从 50.242% 下降到 49.550%，BOP AR 同样没有改善。该实验支持
“correspondence 信息没有被直接头稳定消费”，不支持把 oracle/RANSAC 当最终方法。
