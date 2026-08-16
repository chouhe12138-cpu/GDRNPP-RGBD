# 当前研究状态

最后核对：2026-08-16。EXP005/B 已完成固定 Epoch 40；EXP009/CPM 仍待固定
Epoch 40 收尾；EXP010 已获准在 lab0 使用新的确定 Git release 启动。

## 项目做到哪里

```text
Stage 1:   COMPLETE — FAIL
Stage 2:   COMPLETE — PASS（XYZ GEOMETRY）
Stage 3A:  COMPLETE — CALIBRATION_MISMATCH
Stage 3B:  COMPLETE — PATCH_PNP_UNDERUTILIZATION
Stage 3C0: EXP005/B FORMAL COMPLETE
Stage 3C1: FORMAL COMPLETE — C1_SCREEN_FAIL
Stage 3C2: FORMAL COMPLETE — C2_SCREEN_FAIL
Stage 4:   EXP009/CPM FORMAL RUNNING
Stage 4C:  EXP010/CPM-LR CONTROL AUTHORIZED
```

Stage 2 证明预测 XYZ 几何是主要因果瓶颈。Stage 3B 进一步确认，官方冻结
Patch-PnP 无法把更准确的 XYZ 稳定转化为更好的直接 `R,t`。C1 的质量/覆盖
残差注意力训练稳定，但固定 Epoch 40 和最佳 Epoch 5 都未通过 LM-O 筛选门槛。

官方姿态头的 EXP-20260804-007 全量信息流诊断已经完成：Region 是主导输入，
XYZ 响应不稳定，ROI 2D 的空间对应在全连接层前后几乎消失。C1/B/C2 固定
Epoch 40 的同协议横向诊断尚未全部完成。

## 当前任务

1. EXP005/B 的正式 run `RUN-20260811-063606-formal-s42-a01` 已完成；EXP009
   继续绑定旧 release `652d7fd9d38f8ea5cea0c5a98cc9477b66623180`。EXP010
   使用新 release 在 lab0 启动，不修改旧 release 或 output。
2. 两者共用稳定 environment image
   `sha256:f3055cb660032bbb4c1b7cfd9b1840a6c98359d0562a3a4f0601f7238f7291ee`；
   image build-source 与实验 source commit 独立记录，普通 Python/config 更新
   不触发镜像重建。
3. C2 Epoch 40 已完成并最小归档；Epoch 40 checkpoint、完整日志和 BOP 已完成
   哈希核验，历史 ADD(-S) 未生成并明确记录为缺失。
4. C1 formal checkpoint、日志和最终指标已完成外部只读复验；EXP007 的 C1
   8-target smoke 已通过，1,445-target full 与 B/C2 横向诊断仍待后续安排。
5. 两个 `b39f680...` smoke 失败 run 继续保留为基础设施证据；`dcf6d57...`
   重建失败发生在容器进程启动前，没有产生实验 run。二者均不计入模型结果。
6. `652d7fd...` 下两实验的 gate、smoke 和 audit 已通过；正式结果必须使用固定
   Epoch 40，禁止根据 LM-O test 中间 checkpoint 选模。
7. EXP005/EXP009 使用统一入口、非覆盖 run 目录、精简日志和结构化指标。
8. EXP005/EXP009 的 formal/audit/eval worker 固定为 `16`，smoke 为 `2`；不改
   历史 B/C2 配置。
9. EXP010 已注册并授权为 EXP009 的严格学习率控制：从官方 checkpoint 重新
   开始，唯一科学变量是 `lr 8e-5 → 8e-4`。EXP009 与 EXP010 可以并行；最终
   比较等待两者固定 Epoch 40 评估和诊断完成。

## 已验证的本地状态

- 历史验收实现及证据绑定 commit：`4edab641cbe0aa43e9220d92d4f785a2b920cb31`。
- 正式运行源码 commit：`652d7fd9d38f8ea5cea0c5a98cc9477b66623180`。
- 研究相关测试：`147 passed`（含 source/environment 解耦、CPM、受管运行、实验
  管理与历史验收基础设施）。
- B 和 C2 正式 config preflight：PASS。
- 官方初始化权重 SHA-256：
  `bafa869d4e6c00410517ecb1add59f234ed1642e47fabcf3aa6e0e8a1b498a8c`。
- C1 Epoch 40 checkpoint SHA-256 已由外部副本复核；严格加载 PASS，官方
  392 个张量逐位不变，最终 BOP AR `0.6897416378`、ADD(-S) 宏平均 `0.5057`。
- C2 commit `24a1d7aa7159cfc8113f4cfe8c11bbd2ce3ba290` 与历史 Docker image 身份已
  核对；BOP 最佳为 Epoch 35 `0.6935201845`，固定 Epoch 40 为
  `0.6930057670`，均未达到 `+0.50 pp` 筛选门槛。Epoch 40 checkpoint SHA-256
  为 `44d69b5cf97829d0a982bba6c6644b4fe2f41a3c88908fecb6ebb0deb905b270`。
- EXP007 使用 C1 formal checkpoint 的 CPU/FP32 diagnostic smoke：8/8 targets、
  14/14 条件、模型状态不变，质量门 PASS；该临时 smoke 不替代 formal full。
- EXP009 CPM 使用 8192-sample moment audit 固定非学习尺度，参数量 `822,281`；
  1-epoch local chain 的构建、迁移、forward/backward、保存/严格重载、标准
  evaluator 及 8-target × 19-condition diagnostic 均通过。该结果仅为工程验收，
  不构成方法有效性结论。
- `652d7fd...` 服务器有效运行：EXP005 smoke
  `RUN-20260811-061212-smoke-s42-a01`、audit
  `RUN-20260811-062719-audit-s42-a01`；EXP009 smoke
  `RUN-20260811-061226-smoke-s42-a01`、audit
  `RUN-20260811-062736-audit-s42-a01`。四个 run 均由用户回传为完成，smoke
  均生成并记录 `checkpoints/model_epoch_001.pth`。

## 已知记录问题

- EXP005 固定 Epoch 40 的 BOP AR 为 `0.6919123414`、ADD(-S) 为
  `0.5065743945`。外部 checkpoint 副本仍在下载，完成后再核验哈希和严格加载；
  下载中的临时文件不解释为 checkpoint 损坏。
- Docker 历史构建记录中的 `43 passed` 是当时快照；当前本地研究测试基线为
  `147 passed`。
- EXP005/EXP009 的 2026-08-11 首次 managed smoke 都因只读 dataset cache
  基础设施错误无效；训练入口吞异常后 postprocess 报缺少
  `model_epoch_001.pth`。两次 run 无 checkpoint/指标，不属于科学失败，且不可
  覆盖或删除。
- EXP-000～009 已纳入统一 experiment 索引；EXP008 已完成且筛选失败，EXP005
  和 EXP009 已通过正式运行前 gate 并由用户确认为 formal 运行中。精确 formal
  `run_id` 尚未回传，不应从启动时间推测。
- C1 正式 raw checkpoint/日志保存在外部 Windows 路径；仓库只跟踪复验记录
  和哈希，不复制大文件。完整 raw evaluator 目录仍未复制。
- C1/C2 历史自动 ADD(-S) 查找器曾不兼容 evaluator 的冒号目录/文件名；C2
  Epoch 35/40 下载证据中没有 ADD(-S) 宏平均或逐物体结果，保持为缺失，不用
  BOP `ad` 替代。
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
