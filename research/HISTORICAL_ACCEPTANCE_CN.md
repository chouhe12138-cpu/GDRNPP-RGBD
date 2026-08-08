# 历史实验验收总表

历史原始记录与产物保持只读；本表只记录重新验证结果。

| 实验 | 状态 | 验收范围 |
|---|---|---|
| EXP-20260729-000-official-gdrnpp-baseline | PASS_WITH_NOTE | 官方 checkpoint、GT-box config 与 EXP007 共享完整基线产物 |
| EXP-20260730-001-gdrnpp-pose-aggregation-diagnostic | PASS | 历史 full 产物、官方 checkpoint、BOP/ADD(-S) 与 FAIL 结论 |
| EXP-20260731-002-gdrnpp-causal-oracle | PASS_WITH_NOTE | 历史 full 产物、Stage 1 基线复现、Oracle 指标与 PASS_XYZ_GEOMETRY 结论 |
| EXP-20260731-003-pbr-validation-calibration | PASS_WITH_NOTE | 固定 PBR split、calibration300 产物、ADD(-S) 模式与泄漏边界 |
| EXP-20260731-004-gdrnpp-pose-head-utilization | PASS | 历史 full 产物、alpha=0 基线、Patch-PnP/RANSAC 曲线与结论 |
| EXP-20260731-005-pnp-only-control | PENDING_EXTERNAL | 本地 PnP-only pilot、checkpoint 与参数隔离；B formal 保持未启动 |
| EXP-20260731-006-quality-coverage | PENDING_EXTERNAL | 本地 C1 pilot 与 formal 配置；Epoch 40 checkpoint、评估产物等待服务器只读核验 |
| EXP-20260804-007-pose-head-information-flow | PENDING_EXTERNAL | official/full 完整产物；C1/B/C2 同协议诊断保持待完成 |
| EXP-20260805-008-stage3c2-joint-adaptation | PENDING_EXTERNAL | C2 frozen config、官方初始化权重与运行身份；Epoch 40 结果等待现有服务器运行结束 |
