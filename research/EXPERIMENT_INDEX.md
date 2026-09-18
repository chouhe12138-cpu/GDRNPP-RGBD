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
| EXP017 Rotation Residual | COMPLETE / FAIL | E40 3/5 gate，rotation/BOP 未过 | [RECORD](experiments/EXP-20260902-017-support-aware-rotation-residual/RECORD.md) |
| EXP017-B Detached Adapter Geometry | COMPLETE / NO_OVERALL_GAIN | E40 rotation/BOP 略升，ADD 下降 | [RECORD](experiments/EXP-20260903-017-b-detached-adapter-geometry/RECORD.md) |
| EXP018 Geometry-Consistency Residual | COMPLETE / MARGINAL_GAIN / CLOSED | E40 四项均略高于 EXP013A，rotation 增幅最大；BOP +0.002346 未达到设计阶段建议的 +0.003，单 seed 证据不足以支持继续投入 | [RECORD](experiments/EXP-20260906-018-geometry-consistency-residual/RECORD.md) |
| EXP019 EPro Geometry Utilization | RUN_COMPLETE | 1,445 targets；原 Gate A/B 通过，历史复现 5/6 越界；原 evaluator decision 为 PROTOCOL_REPRODUCTION_FAILED_STOP（保留）；2026-09-08 用户 review 判定机制通过，后续 gate 改用相对阈值 | [RECORD](experiments/EXP-20260907-019-epro-geometry-utilization/RECORD.md) |
| EXP020 Geometry-Aware Correspondence Loss | CLOSED / E40_DIRECT_POSE_RECORDED / MATCHED_PNP_NOT_RUN / EXIT_CODE_UNCONFIRMED | 用户决定结束后续补证；A/B E5–E40 direct-pose 已记录，正式 matched PnP 未运行，历史证据缺口保留 | [RECORD](experiments/EXP-20260909-020-geometry-aware-correspondence-loss/RECORD.md) |
| EXP021 Global-Guided Hierarchical CAD Correspondence | FORMAL_BC_E15_AVAILABLE / TRAINING_EXIT_UNCONFIRMED / MATCHED_PNP_PENDING | B/C source `effc99b` 的 E5/E10/E15 常规评估及 checkpoint 保存日志已同步；完整 formal、fixed-support matched PnP 和资源 gate 尚未提供 | [RECORD](experiments/EXP-20260914-021-global-guided-hierarchical-cad-correspondence/RECORD.md) |
| EXP022 Progressive Hierarchical CAD PCC | STAGE1_REFACTORED / LOCAL_CPP_SMOKE_PASS / SERVER_EGL_PENDING | 四级 image attention 与局部 Q/K/V matcher 已通过本地测试、CPU preflight 和 CUDA+CPP 固定 batch 对比；正式训练与完整 matched PnP 尚未运行 | [RECORD](experiments/EXP-20260916-022-progressive-pcc/RECORD.md) |
| EXP023 LM13 Progressive PCC Fulltrain | SERVER_INTEGRATION_ADDED / LAUNCHER_CLOSURE_ADDED / EGL_SMOKE_PENDING / FORMAL_NOT_STARTED | 协议重整、训练前收口与服务器集成完成：三协议 + 两 loader + preflight 硬校验 + `lm_imgn` 只读 mount + ConvNeXt env + profile-aware runtime gate + `server_preflight`；launcher 收口再修 `create`/`check_host` 目录职责、`lm13_real_only` 的 `lm_imgn` 依赖、未知 `TRAIN_PROTOCOL.NAME` fail-closed 与 hierarchy 的 container-safe 准备。本机数据/CPP smoke/真实入口 1 epoch + checkpoint 通过，`285 passed`。本机 EGL 不可用；**服务器侧的挂载、容器 env 与 EGL smoke 未验证**，`FORMAL_READY` 仍为 `False` | [RECORD](experiments/EXP-20260918-023-lm13-progressive-pcc-fulltrain/RECORD.md) |
| EXP024 LM-O Progressive PCC ImageNet Fulltrain | SERVER_EGL_SMOKE_PASS / FORMAL_UNLOCKED / FORMAL_NOT_STARTED | 在 EXP022 LM-O 协议上增加独立全主干训练臂：ImageNet ConvNeXt 初始化、物理 batch4 累积至 effective batch48、40 epoch；lab1 CPU preflight 与 EGL/AMP 真实 batch4 smoke 通过，无跳步，formal 配置已解锁，等待第二段 release 与正式启动。与冻结臂同时改变初始化和训练范围，差值不作单因素解释 | [RECORD](experiments/EXP-20260919-024-lmo-progressive-pcc-fulltrain/RECORD.md) |
