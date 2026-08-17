# 当前研究状态

最后核对：2026-08-18。本页只汇总已经由记录、日志、checkpoint 或哈希支持的
当前事实；服务器实时状态必须重新执行只读检查。

## 代码状态

```text
branch:      main
HEAD:        29580f65abfeb7625bab252011c19399325b0fa2
origin/main: 29580f65abfeb7625bab252011c19399325b0fa2
worktree:    dirty（EXP011 记录更新与 EXP012 待提交实现；全部保留）
```

EXP011 CPM XYZ–Region 一致性固定权重诊断已完成。失败的早期 QC run 均保留；
最终 full a02 在 RTX 4060 上完成 1,445 targets × 10 conditions 和 BOP19，QC
PASS，预注册结论为 `MISMATCH_IMPORTANT`。

EXP012 层级密集 correspondence head 已完成本地 CPU/CUDA preflight 和严格 checkpoint
roundtrip，工程 gate PASS；当前仅为 `AUTHORIZED`，没有服务器 run 或性能结果。

## 阶段状态

```text
Stage 1:   COMPLETE — FAIL
Stage 2:   COMPLETE — PASS_XYZ_GEOMETRY
Stage 3A:  COMPLETE — CALIBRATION_MISMATCH
Stage 3B:  COMPLETE — PATCH_PNP_UNDERUTILIZATION
Stage 3C0: EXP005/B COMPLETE
Stage 3C1: C1 COMPLETE — C1_SCREEN_FAIL
Stage 3C2: C2 COMPLETE — C2_SCREEN_FAIL
Stage 4:   EXP009 FIXED EPOCH 40 COMPLETE — CPM_SCREEN_FAIL
Stage 4C:  EXP010 AUTHORIZED — ACCESS PASS，NO FORMAL RUN RECORDED
Stage 4D:  EXP011 COMPLETE — MISMATCH_IMPORTANT
Stage 4E:  EXP012 AUTHORIZED — LOCAL PREFLIGHT PASS，NO SERVER RUN
```

## 当前实验事实

### EXP005/B

- formal run：`RUN-20260811-063606-formal-s42-a01`；固定 Epoch 40 完成。
- BOP AR `0.6919123414`；ADD(-S)@0.1d macro-object `0.5065743945`。
- 外部权重 `E:\\6D姿态估计\\EXP-005\\model_epoch_040.pth`，大小
  `519078612` bytes，SHA-256
  `39c0128526f68cf9c4f7a1780ff095e71be8cdada186136eb8699d99d68d009e`。
- 本地读取为 Epoch 40 / iteration `255919`，包含 optimizer、scheduler 和
  392 个模型张量。没有服务器端原文件 SHA，未记录两端哈希一致。

### EXP009/CPM

- formal run：`RUN-20260811-063626-formal-s42-a01`。
- 早期训练/恢复日志记录过 Epoch 38 / iteration `243123/255920` 和
  `CUDA error: unspecified launch failure`；这些中断证据继续保留。
- 后续外部固定 Epoch 40 权重
  `E:\\6D姿态估计\\EXP-009\\model_epoch_040.pth` 已核验，大小
  `387752090` bytes，SHA-256
  `d447569bf7a1034bb57f38c90ef25bbaac8f1bb7ef3b9d74ef9db75eb32f040d`。
- Epoch 40 权重可读，记录 Epoch 40 / iteration `255919`，包含 optimizer、
  scheduler 和 384 个模型张量。
- 新下载的完整 `console.log` 记录 Epoch 40 / iteration `255919/255920`、
  checkpoint 保存和 `FINAL_EVAL_REUSED periodic_epoch=40`；SHA-256 为
  `258be3940b53012abb5099ee4582a75923df306e2bba994917d82502e26547e0`。
- 固定 Epoch 40 BOP AR `0.5983921569`；训练 `EVAL_SUMMARY` 的
  ADD(-S)@0.1d `0.3806228374` 实际是 target-micro recall，早期记录曾误标为
  macro-object。由 console 中八个逐物体 recall 等权计算的 macro-object 为
  `0.3768665461`；逐物体非负 `2/8`。该语义修正不改变三项 gate 均失败和
  `CPM_SCREEN_FAIL` 结论。
- Epoch 35：BOP AR `0.5994232987`、ADD(-S) `0.3861591696`，仅保留为中间结果。
- Epoch 30 checkpoint SHA-256 为
  `d5fabd8ad3f2be5ecf3fcc52a18386d151732f7593a0daa2ca22181c0add5ce0`；
  BOP AR `0.5994625144`。已有预览诊断不指定造成响应的根因。

### EXP010/CPM 学习率控制

- 状态为 `AUTHORIZED`；尚无 formal run 身份。
- lab0 access 回传为 PASS；source commit 为
  `29580f65abfeb7625bab252011c19399325b0fa2`。
- environment image ID 为
  `sha256:f3055cb660032bbb4c1b7cfd9b1840a6c98359d0562a3a4f0601f7238f7291ee`。
- 当前没有 create、gate、smoke、audit 或 formal 启动完成的回传证据。

### EXP011/CPM XYZ–Region 一致性诊断

- 最终 run：`RUN-20260817-023144-full-s20260817-a02`；固定 EXP009 E40
  checkpoint，CUDA FP32，1,445 targets × 10 conditions，QC PASS。
- Pred Region 下 GT-XYZ effect：BOP AR `-0.2894763552`、macro ADD(-S)
  `-0.2796477506`；GT Region 下分别为 `-0.1381130334`、`-0.1024982420`。
- interaction：BOP `+0.1513633218`、macro ADD(-S) `+0.1771495085`；rescue
  ratio 分别为 `0.5229`、`0.6335`，且 ADD interaction 为正的物体为 `8/8`。
- 预注册 decision：`MISMATCH_IMPORTANT`。该结论支持 XYZ–Region 不一致是
  GT-XYZ oracle 恶化的重要污染因素；GT Region 在 Pred XYZ 下本身降低绝对
  性能，因此它不是可直接部署的性能改进，也未证明不一致是唯一根因。

### EXP012/层级密集 Correspondence Pose Head

- 状态为 `AUTHORIZED`；当前没有 lab1 access、create、gate 或 formal run。
- 用户已明确授权服务器 gate PASS 后直接 formal；EXP012 不要求 smoke/audit48。
- 冻结 backbone/geometry head，只训练 868,746 参数的层级 correspondence head；
  保留逐像素 metric XYZ↔absolute ROI2D，先局部编码，再聚合 fine/mid 摘要与
  high-level 4×4 空间网格。
- Region 只作为零启动辅助残差，不定义 grouping/pooling；这属于待验证设计，
  不能记为已经消除 EXP011 mismatch。
- 本地 CPU/CUDA full-model forward/backward、optimizer step、official shared-state migration 和
  strict checkpoint roundtrip 均 PASS。它们只证明工程链可执行，不证明性能。

## 其他已完成事实

- C1 固定 Epoch 40：BOP AR `0.6897416378`、ADD(-S) macro-object `0.5057`，
  结论 `C1_SCREEN_FAIL`。
- C2 固定 Epoch 40：BOP AR `0.6930057670`，结论 `C2_SCREEN_FAIL`；历史
  ADD(-S) 未生成，保持缺失。
- EXP007 official Patch-PnP full 已完成；C1/B/C2 固定 checkpoint 的同协议
  full 尚未全部完成。
- EXP005、EXP009 的首次 managed smoke 是无效基础设施 run，均无 checkpoint
  或科学指标；原目录保留，不计入模型结果。

## 关键入口

- 研究计划：`research/RESEARCH_PLAN.md`
- Stage 3C 总览：`research/STAGE_03C_EXPERIMENT_OVERVIEW.md`
- EXP005：`research/experiments/EXP-20260731-005-pnp-only-control/RECORD.md`
- EXP009：`research/experiments/EXP-20260809-009-cpm-head/RECORD.md`
- EXP010：`research/experiments/EXP-20260816-010-cpm-official-lr-control/RECORD.md`
- EXP011：`research/experiments/EXP-20260817-011-cpm-xyz-region-consistency-diagnostic/RECORD.md`
- EXP012：`research/experiments/EXP-20260817-012-hierarchical-correspondence-head/RECORD.md`
- 姿态头诊断：`research/POSE_HEAD_DIAGNOSTIC_HANDOFF_CN.md`
- 服务器快照：`research/SERVER_RUNTIME_STATUS_CN.md`
- 运行手册：`research/RUNBOOK_CN.md`
