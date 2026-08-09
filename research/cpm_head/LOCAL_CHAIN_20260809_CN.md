# CPM 本地完整链路验收（2026-08-09）

结论：`PASS_LOCAL_ENGINEERING_ONLY`。CPM 已通过实现、数值、短训、保存/重载、
推理、标准 evaluator、BOP/ADD(-S) 产物索引和 diagnostic smoke；这不构成方法
有效或无效的科学结论。

关键事实：

- 8192 样本 moment audit 全部有限；仅 36/524288 个 Region 同时为低 coverage
  和 `n_eff < 2`，未增加低支持裁剪；
- raw moment P95 尺度比约 1826，已在训练前冻结非学习 P95-abs 缩放；
- 1 epoch 共 2048 micro-batch、171 optimizer update，loss 全程有限；
- 375 个 official 共享张量逐位不变，9 个 CPM state tensor 严格重载；
- checkpoint SHA256：
  `e3d431c40eaf6354c5d21d18cb3d06534524ffbfedb9648be1c329492f7774a3`；
- 8-target × 19-condition diagnostic 无意外非有限值，模型状态不变；除 14 个
  公共条件外，已实际执行 XYZ α=0.25/0.50/0.75、coverage-only 和 CXU-null；
- 标准 LM-O GT-box evaluator 的 BOP 和 ADD(-S) 均能唯一索引。Epoch 1 指标为
  0，仅说明未充分训练的 smoke checkpoint 性能很差，不用于研究判断。

最终扩展诊断因本轮操作系统阻断本地 GPU 访问而使用 CPU/FP32 完成；此前
CUDA 构建、短训、保存/重载、推理和 14-condition smoke 均已通过。机器可读
详情和证据哈希见 `LOCAL_CHAIN_20260809.json`。C2 完成归档、B 决策、确定
Git commit、Docker image 和服务器实时检查之前，不启动 CPM formal。
