# 实验索引

本页只提供实验目录。当前进展和下一步见 [STATUS_CN.md](STATUS_CN.md)；详细协议、结果、
结论和证据边界只写在对应 `RECORD.md`。退出 HEAD 的历史入口按 RECORD 中的 source commit
使用独立 worktree 恢复。
LM13 ImageNet full 后续臂已完成配置、hierarchy 与本地 CUDA smoke，保持 server-disabled，
待 LM-O 完成后再独立开放。

| 实验 | 状态 | 结论 | 记录 |
|---|---|---|---|
| EXP027 Full E15 checkpoint 诊断 | DIAGNOSTIC_COMPLETE | 保持预测 T3 route 的 oracle 残差在 122 ROI 上将 ADD(-S)0.1d 成功从 57 提至 105；主瓶颈指向 residual/精细 XYZ 解码，非 E40 正式结论 | [RECORD](experiments/EXP-20260922-027-exp025-e15-full-diagnostic/RECORD.md) |
| EXP026 GA-HFPS 三层采样 | LOCAL_OFFLINE_COMPLETE / TRAINING_NOT_STARTED | 三层 λ1 离线工程 gate 通过；复杂区域改善但平滑区域及物体 12 连通性有代价，仅建议后续 matched 训练验证 | [RECORD](experiments/EXP-20260921-026-geometry-adaptive-hfps/RECORD.md) |
| EXP025 统一 T3 CAD attention | ACTIVE / FORMAL_IN_PROGRESS | Frozen 已评估到 E20、Full 已评估到 E25；其余固定点未交付 | [RECORD](experiments/EXP-20260920-025-hierarchical-cad-attention/RECORD.md) |
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
| EXP014-D Full Train | CLOSED / NOT_RESUMED | a01 无效；旧入口已退出 HEAD | [RECORD](experiments/EXP-20260827-014-d-fulltrain-imagenet/RECORD.md) |
| EXP013E Official Random | COMPLETE | 部分支持 M2/M3 并重 | [RECORD](experiments/EXP-20260829-015-e-official-head-random/RECORD.md) |
| EXP013F GLM-Pose-L | COMPLETE / FAIL | 2/4 gate，边缘失败 | [RECORD](experiments/EXP-20260829-016-f-glm-pose-l-screening/RECORD.md) |
| EXP017 Rotation Residual | COMPLETE / FAIL | E40 3/5 gate，rotation/BOP 未过 | [RECORD](experiments/EXP-20260902-017-support-aware-rotation-residual/RECORD.md) |
| EXP017-B Detached Adapter Geometry | COMPLETE / NO_OVERALL_GAIN | E40 rotation/BOP 略升，ADD 下降 | [RECORD](experiments/EXP-20260903-017-b-detached-adapter-geometry/RECORD.md) |
| EXP018 Geometry-Consistency Residual | COMPLETE / MARGINAL_GAIN / CLOSED | E40 四项均略高于 EXP013A，rotation 增幅最大；BOP +0.002346 未达到设计阶段建议的 +0.003，单 seed 证据不足以支持继续投入 | [RECORD](experiments/EXP-20260906-018-geometry-consistency-residual/RECORD.md) |
| EXP019 EPro Geometry Utilization | COMPLETE / DIAGNOSTIC_ONLY | Gate A/B 通过、历史复现 5/6 越界；原 evaluator STOP 与后续用户机制通过 review 并列保留 | [RECORD](experiments/EXP-20260907-019-epro-geometry-utilization/RECORD.md) |
| EXP020 Geometry-Aware Correspondence Loss | CLOSED / FORMAL_RESULTS_PARTIAL | A/B E5–E40 direct-pose 已记录；matched PnP 与 exit code 未生成，用户结束补证 | [RECORD](experiments/EXP-20260909-020-geometry-aware-correspondence-loss/RECORD.md) |
| EXP021 Global-Guided Hierarchical CAD Correspondence | CLOSED / FORMAL_RESULTS_PARTIAL | B/C E5–E40 direct-pose 已记录；exit code 与 fixed-support matched PnP 未生成 | [RECORD](experiments/EXP-20260914-021-global-guided-hierarchical-cad-correspondence/RECORD.md) |
| EXP022 Progressive Hierarchical CAD PCC | SUPERSEDED / FORMAL_NOT_STARTED | 本地工程验证保留；服务器 EGL、formal 与 matched PnP 未运行 | [RECORD](experiments/EXP-20260916-022-progressive-pcc/RECORD.md) |
| EXP023 LM13 Progressive PCC Fulltrain | SUPERSEDED / FORMAL_NOT_STARTED | 本地协议与接线证据保留；服务器 EGL 与 formal 未运行 | [RECORD](experiments/EXP-20260918-023-lm13-progressive-pcc-fulltrain/RECORD.md) |
| EXP024 LM-O Progressive PCC ImageNet Fulltrain | SUPERSEDED / FORMAL_NOT_STARTED | lab1 batch4 EGL smoke 通过，但 formal 未启动；组合策略转入 EXP025 | [RECORD](experiments/EXP-20260919-024-lmo-progressive-pcc-fulltrain/RECORD.md) |
