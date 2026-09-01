# EXP001 — Pose Aggregation Diagnostic

- 状态：`COMPLETE / FAIL`
- 协议：官方 checkpoint、LM-O BOP19 1,445 targets、GT bbox、无训练
- 结论：未找到稳定优于官方 Patch-PnP 的固定几何聚合方案。

未过滤 RANSAC-EPnP 的 BOP AR 提高约 0.573 个百分点，但只有 5/8 个物体
非负，未通过预注册的稳定性门槛。结果只说明存在物体依赖的正信号，不授权把
RANSAC 作为部署路径。
