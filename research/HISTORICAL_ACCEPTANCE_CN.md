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
| EXP-20260731-006-quality-coverage | PASS_WITH_NOTE | C1 Epoch 40 checkpoint、严格加载、参数隔离、日志与最终 BOP/ADD(-S) 已用外部只读副本复验；raw evaluator 目录未复制 |
| EXP-20260804-007-pose-head-information-flow | PENDING_EXTERNAL | official/full 完整产物；C1 formal checkpoint 的 8-target smoke 已通过，C1/B/C2 同协议 full 仍待完成 |
| EXP-20260805-008-stage3c2-joint-adaptation | RUNNING_SNAPSHOT | C2 身份和 Epoch 5–30 周期指标已核对；下载日志到 Epoch 33，当前 BOP 最佳为 Epoch 25，最终结果仍待完成 |

追加复验不覆盖首次 `ACCEPTANCE.json`。C1、EXP007 C1 smoke 和 C2 的本次证据
分别见各实验目录下的 `REVALIDATION_20260808.json` 与
`REVALIDATION_20260808_CN.md`。
