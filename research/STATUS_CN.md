# 当前研究状态

最后核对：2026-08-08（本地工作区）；服务器动态状态最后记录于 2026-08-06，
使用前必须重新检查。

## 项目做到哪里

```text
Stage 1:   COMPLETE — FAIL
Stage 2:   COMPLETE — PASS（XYZ GEOMETRY）
Stage 3A:  COMPLETE — CALIBRATION_MISMATCH
Stage 3B:  COMPLETE — PATCH_PNP_UNDERUTILIZATION
Stage 3C0: LOCAL PILOT PASS — B 正式对照待完成
Stage 3C1: FORMAL COMPLETE — C1_SCREEN_FAIL
Stage 3C2: TRIGGERED — C2 正式实验运行门待完成
```

Stage 2 证明预测 XYZ 几何是主要因果瓶颈。Stage 3B 进一步确认，官方冻结
Patch-PnP 无法把更准确的 XYZ 稳定转化为更好的直接 `R,t`。C1 的质量/覆盖
残差注意力训练稳定，但固定 Epoch 40 和最佳 Epoch 5 都未通过 LM-O 筛选门槛。

官方姿态头的 EXP-20260804-007 全量信息流诊断已经完成：Region 是主导输入，
XYZ 响应不稳定，ROI 2D 的空间对应在全连接层前后几乎消失。C1/B/C2 固定
Epoch 40 的同协议横向诊断尚未全部完成。

## 当前任务

1. 重新核验服务器 GPU、后台等待器、容器和 B/C2 benchmark 状态。
2. 保持当前 B/C2 镜像和复现链不变，完成各自 smoke、隔离验证和 worker gate。
3. 运行 B（Patch-PnP-only）与 C2（联合适应）的单 seed、40 epoch 正式实验。
4. 对 C1/B/C2 固定 Epoch 40 checkpoint 完成同协议信息流诊断。
5. 用 `C2 - B` 区分“姿态头需联合适应”与“质量/覆盖模块本身无效”。

## 已验证的本地状态

- Git HEAD：`25fb5e3cd2c8dd10681f539eef399c40468130db`。
- 研究相关测试：`98 passed`（含新增实验管理基础设施）。
- B 和 C2 正式 config preflight：PASS。
- 官方初始化权重 SHA-256：
  `bafa869d4e6c00410517ecb1add59f234ed1642e47fabcf3aa6e0e8a1b498a8c`。

## 已知记录问题

- Docker 历史构建记录中的 `43 passed` 是当时快照；当前本地完整基线为 `98 passed`。
- EXP-001～008 已纳入统一 experiment 索引；其中 EXP-008 是已触发但尚待实时
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
