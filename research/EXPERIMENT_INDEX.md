# 实验索引

本页由各实验目录中的 `EXPERIMENT.json` 生成；详细结果以 RECORD 和原始产物为准。

| 实验 ID | 阶段 | 角色 | 状态 | 标题 |
|---|---|---|---|---|
| EXP-20260729-000-official-gdrnpp-baseline | Baseline | official baseline | COMPLETE | 官方 GDRNPP LM-O GT-box 基线 |
| EXP-20260730-001-gdrnpp-pose-aggregation-diagnostic | Stage 1 | diagnostic | COMPLETE | GDRNPP 位姿聚合诊断 |
| EXP-20260731-002-gdrnpp-causal-oracle | Stage 2 | diagnostic | COMPLETE | GDRNPP 因果 Oracle 诊断 |
| EXP-20260731-003-pbr-validation-calibration | Stage 3A | calibration | COMPLETE | LM-PBR 验证基础设施与校准 |
| EXP-20260731-004-gdrnpp-pose-head-utilization | Stage 3B | diagnostic | COMPLETE | 冻结 Patch-PnP 的 XYZ 利用诊断 |
| EXP-20260731-005-pnp-only-control | Stage 3C0 / B | B | COMPLETE | Patch-PnP-only 适应对照 |
| EXP-20260731-006-quality-coverage | Stage 3C1 | C1 | COMPLETE | 质量/覆盖残差注意力 C1 |
| EXP-20260804-007-pose-head-information-flow | Stage 3C mechanism diagnostic | official/C1/B/C2 diagnostic | RUNNING | Patch-PnP 姿态头信息流全量统计诊断 |
| EXP-20260805-008-stage3c2-joint-adaptation | Stage 3C2 | C2 | COMPLETE | Patch-PnP 与质量/覆盖模块联合适应 C2 |
| EXP-20260809-009-cpm-head | Stage 4 / CPM | CPM | COMPLETE | Correspondence-Aware Moment Pose Head（CPM-Head） |
| EXP-20260816-010-cpm-official-lr-control | Stage 4 / CPM optimization control | CPM-LR-control | FAILED | CPM-Head 官方量级学习率受控实验 |
| EXP-20260817-011-cpm-xyz-region-consistency-diagnostic | Stage 4 / CPM mechanism diagnostic | diagnostic | COMPLETE | CPM XYZ–Region 一致性固定权重诊断 |
| EXP-20260817-012-hierarchical-correspondence-head | Stage 4E / correspondence-preserving pose head | hierarchical-correspondence-head | COMPLETE | 层级密集 Correspondence Pose Head |
| EXP-20260822-013-a-xyz-residual-bypass | Stage 4F / geometry-path ablation | xyz-residual-bypass | COMPLETE | XYZ-ROI2D 独立几何残差支路 |
| EXP-20260822-013-b-geometry-attention-residual | Stage 4F / local geometry attention | geometry-attention-residual | COMPLETE | 局部几何注意力残差支路 |
| EXP-20260822-013-c-rt-decoupled-fusion | Stage 4F / conditional R-t decoupling | rt-decoupled-fusion | AUTHORIZED | Rotation/Translation 专用特征融合 |
| EXP-20260827-014-d-fulltrain-imagenet | Stage 4F / full end-to-end training | d-fulltrain-imagenet | AUTHORIZED | ImageNet 预训练 ConvNeXt 全量端到端训练 |
| EXP-20260829-016-f-glm-pose-l-screening | Stage 4H / GLM-Pose-L head screening | glm-pose-l-screening | AUTHORIZED | GLM-Pose-L 头筛选(M2 注意力池化 + M3 深度统计) |
| EXP-20260829-015-e-official-head-random | Stage 4H / official-head random-init control | official-head-random | AUTHORIZED | 官方头随机初始化冻结对照 |
