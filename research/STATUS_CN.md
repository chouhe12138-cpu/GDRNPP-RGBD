# 当前研究状态

最后核对：2026-08-10（本地工作区及用户提供的 lab0/lab1 只读资料）；C2
仍在旧 `lab1_chx` 中运行，服务器实时状态在实际启动前仍需重新检查。

## 项目做到哪里

```text
Stage 1:   COMPLETE — FAIL
Stage 2:   COMPLETE — PASS（XYZ GEOMETRY）
Stage 3A:  COMPLETE — CALIBRATION_MISMATCH
Stage 3B:  COMPLETE — PATCH_PNP_UNDERUTILIZATION
Stage 3C0: LOCAL PILOT PASS — EXP005/B FORMAL AUTHORIZED
Stage 3C1: FORMAL COMPLETE — C1_SCREEN_FAIL
Stage 3C2: RUNNING（HANDOFF SNAPSHOT）— 保持原运行链
Stage 4:   CPM LOCAL ENGINEERING PASS — AUTHORIZED AFTER C2 ARCHIVE
```

Stage 2 证明预测 XYZ 几何是主要因果瓶颈。Stage 3B 进一步确认，官方冻结
Patch-PnP 无法把更准确的 XYZ 稳定转化为更好的直接 `R,t`。C1 的质量/覆盖
残差注意力训练稳定，但固定 Epoch 40 和最佳 Epoch 5 都未通过 LM-O 筛选门槛。

官方姿态头的 EXP-20260804-007 全量信息流诊断已经完成：Region 是主导输入，
XYZ 响应不稳定，ROI 2D 的空间对应在全连接层前后几乎消失。C1/B/C2 固定
Epoch 40 的同协议横向诊断尚未全部完成。

## 当前任务

1. 将当前本地实现形成干净 Git commit 并推送 Gitee，由 lab0 构建唯一镜像。
2. lab0 先使用 `lab0_chx` 启动 seed `42` 的 EXP005/B managed formal。
3. 保持 C2 原环境和旧 `lab1_chx` 运行链，等待 Epoch 40 并完成结果封存。
4. C1 formal checkpoint、日志和最终指标已完成外部只读复验；EXP007 的 C1
   8-target smoke 已通过，1,445-target full 与 B/C2 横向诊断仍待后续安排。
5. C2 封存并释放容器名后，以相同 commit/image 重建 `lab1_chx`，再启动
   seed `42` 的 EXP009。
6. EXP005/EXP009 使用统一入口、非覆盖 run 目录、精简日志和结构化指标。

## 已验证的本地状态

- 历史验收实现及证据绑定 commit：`4edab641cbe0aa43e9220d92d4f785a2b920cb31`。
- 研究相关测试：`138 passed`（含 CPM、受管运行、实验管理与历史验收基础设施）。
- B 和 C2 正式 config preflight：PASS。
- 官方初始化权重 SHA-256：
  `bafa869d4e6c00410517ecb1add59f234ed1642e47fabcf3aa6e0e8a1b498a8c`。
- C1 Epoch 40 checkpoint SHA-256 已由外部副本复核；严格加载 PASS，官方
  392 个张量逐位不变，最终 BOP AR `0.6897416378`、ADD(-S) 宏平均 `0.5057`。
- C2 commit `24a1d7aa7159cfc8113f4cfe8c11bbd2ce3ba290` 与 Docker image 身份已由
  交接包核对；Epoch 5–30 的 BOP 已结构化记录，当前 BOP 最佳为 Epoch 25
  `0.6927381776`，但仍未达到 `+0.50 pp` 筛选门槛。
- EXP007 使用 C1 formal checkpoint 的 CPU/FP32 diagnostic smoke：8/8 targets、
  14/14 条件、模型状态不变，质量门 PASS；该临时 smoke 不替代 formal full。
- EXP009 CPM 使用 8192-sample moment audit 固定非学习尺度，参数量 `822,281`；
  1-epoch local chain 的构建、迁移、forward/backward、保存/严格重载、标准
  evaluator 及 8-target × 19-condition diagnostic 均通过。该结果仅为工程验收，
  不构成方法有效性结论。

## 已知记录问题

- Docker 历史构建记录中的 `43 passed` 是当时快照；当前本地研究测试基线为
  `138 passed`。
- EXP-000～009 已纳入统一 experiment 索引；EXP-008 正在运行，EXP005 已获
  formal 授权，EXP009 为 C2 封存后生效的条件授权。
- C1 正式 raw checkpoint/日志保存在外部 Windows 路径；仓库只跟踪复验记录
  和哈希，不复制大文件。完整 raw evaluator 目录仍未复制。
- C1/C2 历史自动 ADD(-S) 查找器曾不兼容 evaluator 的冒号目录/文件名；当前
  通用 loader 已兼容两种分隔符，原始 `null` 与 traceback 仍保留、不覆盖。
- 2026-08-09 最终扩展 diagnostic 时，本地 GPU 被操作系统阻断；CPU/FP32
  19-condition smoke 与 formal-config CPU preflight 通过，CUDA preflight 待
  GPU 恢复后重跑。此前本地 CUDA 短训、推理和 14-condition smoke 已通过。
- 服务器状态具有时效性，本文不替代实时检查。

## 关键入口

- Stage 3C 总览：`research/STAGE_03C_EXPERIMENT_OVERVIEW.md`
- C1 记录：`research/experiments/EXP-20260731-006-quality-coverage/RECORD.md`
- 姿态头诊断：`research/POSE_HEAD_DIAGNOSTIC_HANDOFF_CN.md`
- CPM 实验记录：`research/experiments/EXP-20260809-009-cpm-head/RECORD.md`
- CPM 本地验收：`research/cpm_head/LOCAL_CHAIN_20260809_CN.md`
- 服务器状态：`research/SERVER_RUNTIME_STATUS_CN.md`
- 运行手册：`research/RUNBOOK_CN.md`
- 历史验收：`research/HISTORICAL_ACCEPTANCE_CN.md`
