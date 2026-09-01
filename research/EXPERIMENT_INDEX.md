# 实验索引

本页是实验状态的唯一中央索引。详细协议、结果和结论只写在对应 `RECORD.md`。

| 实验 | 状态 | 结论 | 记录 |
|---|---|---|---|
| EXP000 官方基线 | COMPLETE | LM-O GT-box 官方基线 | [RECORD](experiments/EXP-20260729-000-official-gdrnpp-baseline/RECORD.md) |
| EXP001 Pose Aggregation | COMPLETE | FAIL，RANSAC 信号不稳定 | [RECORD](experiments/EXP-20260730-001-gdrnpp-pose-aggregation-diagnostic/RECORD.md) |
| EXP002 Causal Oracle | COMPLETE | PASS，XYZ geometry 是主因 | [RECORD](experiments/EXP-20260731-002-gdrnpp-causal-oracle/RECORD.md) |
| EXP003 PBR Calibration | COMPLETE | CALIBRATION_MISMATCH | [RECORD](experiments/EXP-20260731-003-pbr-validation-calibration/RECORD.md) |
| EXP004 Head Utilization | COMPLETE | PATCH_PNP_UNDERUTILIZATION | [RECORD](experiments/EXP-20260731-004-gdrnpp-pose-head-utilization/RECORD.md) |
| EXP005 PnP-only | COMPLETE | E40 matched control | [RECORD](experiments/EXP-20260731-005-pnp-only-control/RECORD.md) |
| EXP006 Quality/Coverage | COMPLETE | C1_SCREEN_FAIL | [RECORD](experiments/EXP-20260731-006-quality-coverage/RECORD.md) |
| EXP007 Information Flow | COMPLETE | Region 主导、XYZ 改善未稳定传递 | [RECORD](experiments/EXP-20260804-007-pose-head-information-flow/RECORD.md) |
| EXP008 Joint Adaptation | COMPLETE | C2_SCREEN_FAIL | [RECORD](experiments/EXP-20260805-008-stage3c2-joint-adaptation/RECORD.md) |
| EXP009 CPM Head | COMPLETE | CPM_SCREEN_FAIL | [RECORD](experiments/EXP-20260809-009-cpm-head/RECORD.md) |
| EXP010 CPM LR Control | FAILED | formal 约 E27 崩溃，不重试 | [RECORD](experiments/EXP-20260816-010-cpm-official-lr-control/RECORD.md) |
| EXP011 XYZ–Region | COMPLETE | MISMATCH_IMPORTANT | [RECORD](experiments/EXP-20260817-011-cpm-xyz-region-consistency-diagnostic/RECORD.md) |
| EXP012 Hierarchical Head | COMPLETE | E40 稳定平台 | [RECORD](experiments/EXP-20260817-012-hierarchical-correspondence-head/RECORD.md) |
| EXP013A XYZ Residual | COMPLETE / PASS | 三项 E40 gate 全过 | [RECORD](experiments/EXP-20260822-013-a-xyz-residual-bypass/RECORD.md) |
| EXP013B Geometry Attention | COMPLETE / FAIL | attention 有效，严格 BOP gate 未过 | [RECORD](experiments/EXP-20260822-013-b-geometry-attention-residual/RECORD.md) |
| EXP013C R/t Decoupled | COMPLETE / FAIL | rotation supported，整体筛选失败 | [RECORD](experiments/EXP-20260822-013-c-rt-decoupled-fusion/RECORD.md) |
| EXP014-D Full Train | PAUSED | a01 无效，无活动重训授权 | [RECORD](experiments/EXP-20260827-014-d-fulltrain-imagenet/RECORD.md) |
| EXP013E Official Random | COMPLETE | 部分支持 M2/M3 并重 | [RECORD](experiments/EXP-20260829-015-e-official-head-random/RECORD.md) |
| EXP013F GLM-Pose-L | COMPLETE / FAIL | 2/4 gate，边缘失败 | [RECORD](experiments/EXP-20260829-016-f-glm-pose-l-screening/RECORD.md) |
