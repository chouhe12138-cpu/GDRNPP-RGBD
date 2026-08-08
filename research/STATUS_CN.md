# 当前研究状态

最后核对：2026-08-08（本地工作区）；服务器动态状态最后记录于 2026-08-06，
使用前必须重新检查。

## 项目做到哪里

```text
Stage 1:   COMPLETE — FAIL
Stage 2:   COMPLETE — PASS（XYZ GEOMETRY）
Stage 3A:  COMPLETE — CALIBRATION_MISMATCH
Stage 3B:  COMPLETE — PATCH_PNP_UNDERUTILIZATION
Stage 3C0: LOCAL PILOT PASS — B 正式对照暂不启动
Stage 3C1: FORMAL COMPLETE — C1_SCREEN_FAIL
Stage 3C2: TRIGGERED/RUNNING_RECHECK_REQUIRED — 保持原运行链
```

Stage 2 证明预测 XYZ 几何是主要因果瓶颈。Stage 3B 进一步确认，官方冻结
Patch-PnP 无法把更准确的 XYZ 稳定转化为更好的直接 `R,t`。C1 的质量/覆盖
残差注意力训练稳定，但固定 Epoch 40 和最佳 Epoch 5 都未通过 LM-O 筛选门槛。

官方姿态头的 EXP-20260804-007 全量信息流诊断已经完成：Region 是主导输入，
XYZ 响应不稳定，ROI 2D 的空间对应在全连接层前后几乎消失。C1/B/C2 固定
Epoch 40 的同协议横向诊断尚未全部完成。

## 当前任务

1. 保持 C2 原环境和原运行链，重新只读核验实时状态并等待 Epoch 40。
2. 暂不启动 B formal；C2 Epoch 40 结果后由用户决定是否取消 B。
3. 补齐 C1 formal checkpoint/评估与 EXP007 cross-model 的服务器只读验收证据。
4. C2 结束后完成 checkpoint、BOP、ADD(-S)、日志和 diagnostic 归档。

## 已验证的本地状态

- 历史验收实现及证据绑定 commit：`4edab641cbe0aa43e9220d92d4f785a2b920cb31`。
- 研究相关测试：`100 passed`（含实验管理与历史验收基础设施）。
- B 和 C2 正式 config preflight：PASS。
- 官方初始化权重 SHA-256：
  `bafa869d4e6c00410517ecb1add59f234ed1642e47fabcf3aa6e0e8a1b498a8c`。

## 已知记录问题

- Docker 历史构建记录中的 `43 passed` 是当时快照；当前本地完整基线为 `100 passed`。
- EXP-000～008 已纳入统一 experiment 索引；其中 EXP-008 是已触发但尚待实时
  核验运行状态的 C2。
- C1 正式 raw checkpoint/日志主要保存在服务器或外部 Windows 路径，本地仓库
  只有记录、哈希和部分 pilot 产物。
- 服务器状态具有时效性，本文不替代实时检查。

## 关键入口

- Stage 3C 总览：`research/STAGE_03C_EXPERIMENT_OVERVIEW.md`
- C1 记录：`research/experiments/EXP-20260731-006-quality-coverage/RECORD.md`
- 姿态头诊断：`research/POSE_HEAD_DIAGNOSTIC_HANDOFF_CN.md`
- 服务器状态：`research/SERVER_RUNTIME_STATUS_CN.md`
- 运行手册：`research/RUNBOOK_CN.md`
- 历史验收：`research/HISTORICAL_ACCEPTANCE_CN.md`
