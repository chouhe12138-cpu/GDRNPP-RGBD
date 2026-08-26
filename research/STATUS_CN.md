# 当前研究状态

最后核对：2026-08-26。本页只汇总已经由记录、日志、checkpoint、正式评估、哈希或 QC 支持的当前事实；服务器实时状态必须重新执行只读检查。

Git branch、HEAD、远端跟踪和 worktree 属于运行时事实，不在本页固化。需要核对时直接执行 `git status --short --branch`、`git rev-parse HEAD` 和 `git rev-parse --abbrev-ref --symbolic-full-name @{u}`。

EXP011 CPM XYZ–Region 一致性固定权重诊断已完成。失败的早期 QC run 均保留；最终 full a02 在 RTX 4060 上完成 1,445 targets × 10 conditions 和 BOP19，QC PASS，预注册结论为 `MISMATCH_IMPORTANT`。

EXP012 已完整训练并评估到 E40。E15 曾出现主要集中于 rotation 的临时退化，但 E20 后恢复，E30–E40 进入稳定平台；EXP013 固定使用 E40 结果作为比较基准。

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
Stage 4E:  EXP012 COMPLETE — E40 STABLE PLATEAU RECORDED
Stage 4F:  EXP013 A COMPLETE/PASS；B COMPLETE/STRICT BOP GATE FAIL；C A-BASED REVISION AUTHORIZED/FORMAL NOT STARTED
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

- 状态：`COMPLETE / EPOCH_040`。
- 正式实现 source commit：`2ca752b3f091292172044209f7c8651280d377bd`。
- 40 epochs 已完整训练并每 5 epoch 评估；此前“E15 后停止”的旧记录已由 E20–E40 正式结果纠正。
- 冻结 backbone/geometry head，只训练 868,746 参数的 `HierarchicalCorrespondencePnPNet`；保留逐像素 metric XYZ↔absolute ROI2D，先局部编码，再聚合 fine/mid 摘要与 high-level 4×4 空间网格。
- Region 只作为零启动辅助残差，不参与 grouping/pooling；后期仍使用共享 2208→256→256 pose representation 输出 allo rot6d 与 centroid-z translation。
- 本地 CPU/CUDA full-model forward/backward、optimizer step、official shared-state migration 和 strict checkpoint roundtrip 均 PASS。这些只证明工程链可执行，不证明性能。

正式评估：

| Epoch | BOP AR | ADD(-S)@0.1d | AR_reS | AR_teS |
|---:|---:|---:|---:|---:|
| 5 | 0.642973 | 0.489273 | 0.345790 | 0.783391 |
| 10 | 0.645972 | 0.473356 | 0.428835 | 0.773472 |
| 15 | 0.544083 | 0.377163 | 0.124798 | 0.798847 |
| 20 | 0.644362 | 0.428374 | — | — |
| 25 | 0.671084 | 0.474048 | — | — |
| 30 | 0.678768 | 0.493426 | — | — |
| 35 | 0.679061 | 0.492042 | — | — |
| 40 | **0.678800** | **0.494118** | **0.491349** | **0.791926** |

- E15 pose 泛化曾严重下降，主要变化发生在 rotation；E20 后恢复，E30–E40 的 BOP AR 稳定在约 `0.679`。因此 E15 诊断只描述该训练阶段，不能再解释为不可恢复的最终崩塌。
- Region×0 在 E5/E10/E15 都造成大幅退化：BOP AR 分别约为 `0.642784→0.353449`、`0.646028→0.316653`、`0.544002→0.251550`。这说明 EXP012 对 Region 输入有强依赖，但依赖强度没有在 E15 突然跃升，不能据此断言 Region 是 E15 崩塌唯一原因。
- Pred XYZ→GT XYZ 的 Three-Path alpha 诊断中，fixed Region 与 synced Region 都没有稳定 pose rescue；E5/E10 最终 Three-Path summary 和 E15 Three-Path 均 QC PASS。
- E10↔E15 checkpoint interpolation endpoints QC PASS。BOP AR `0.646009→0.636141→0.600161→0.574572→0.544127`，AR_reS 同样单调下降；AR_teS 不呈相同行为。该证据支持参数连线上存在连续的 rotation-specific 泛化恶化方向，但不等同于真实训练轨迹。
- Activation drift QC=`FAIL`，只保留为线索，不作为主要诊断证据。

EXP012 的结论边界：

- 已被实验直接支持：EXP012 在 E15 有 rotation-specific 临时退化，随后恢复至 E40 稳定平台；Region 是强依赖输入；E5/E10/E15 阶段改善 XYZ 或同步 Region 没有稳定 rescue learned pose。
- 合理但尚未证实：R/t 共享后期 latent 可能造成不同优化需求之间的干扰；learned pose decoder 可能形成了对 Region 或训练分布的 shortcut。
- 尚未证明：“共享 R/t latent 导致崩塌”“Region 是唯一根因”“joint R/t 本身错误”“PnP 必须成为最终部署结构”。

EXP012 E40 冻结基准为 BOP AR `0.678800`、ADD(-S) `0.494118`、AR_reS `0.491349`、AR_teS `0.791926`。

## 其他已完成事实

### EXP013 E40 最终事实

- A E40：BOP AR `0.683956`、ADD(-S) `0.510727`、AR_reS `0.498039`、AR_teS `0.797693`；相对 EXP012 的增量为 `+0.005156 / +0.016609`，逐物体非负 `5/8`，三项 gate 全部通过。
- B E40：BOP AR `0.683691`、ADD(-S) `0.514187`、AR_reS `0.498039`、AR_teS `0.801153`；ADD 和逐物体 `5/8` 通过，但 BOP 增量 `+0.004891` 比冻结门槛少 `0.000109`，因此严格 gate 失败。
- B 与 A 的 BOP 差 `-0.000265` 位于 `±0.001` 区间，ADD 高 `+0.003460`，故 attention 按 B−A 规则有效；原 B-based C 因 B 未通过全部 EXP012 gate 而停止。
- 2026-08-26 在任何 C formal run 之前，用户明确提供 A-based R/t decoupling 修订方案。revision 2 已完成本地 CPU/CUDA preflight 与 1 epoch、2048 iteration 无 renderer 真实数据 smoke；工程 gate PASS。用户随后明确要求提交、生成 bundle 并准备在 lab0 训练，因此 C 已授权，但 formal 尚未启动，必须使用新的确定 commit 建立只读 release。
- A/B E40 的本地小样本结构诊断均保持权重哈希不变。结果显示几何残差、Region 和空间布局都被实际使用；R/t 共享梯度存在弱冲突或明显尺度失衡。该机制证据不替代 formal gate。

- C1 固定 Epoch 40：BOP AR `0.6897416378`、ADD(-S) macro-object `0.5057`，
  结论 `C1_SCREEN_FAIL`。
- C2 固定 Epoch 40：BOP AR `0.6930057670`，结论 `C2_SCREEN_FAIL`；历史
  ADD(-S) 未生成，保持缺失。
- EXP007 official Patch-PnP full 已完成；C1/B/C2 固定 checkpoint 的同协议
  full 尚未全部完成。
- EXP005、EXP009 的首次 managed smoke 是无效基础设施 run，均无 checkpoint
  或科学指标；原目录保留，不计入模型结果。

## 关键入口

- 实验索引：`research/EXPERIMENT_INDEX.md`
- 研究计划：`research/RESEARCH_PLAN.md`
- Stage 3C 总览：`research/STAGE_03C_EXPERIMENT_OVERVIEW.md`
- EXP005：`research/experiments/EXP-20260731-005-pnp-only-control/RECORD.md`
- EXP009：`research/experiments/EXP-20260809-009-cpm-head/RECORD.md`
- EXP010：`research/experiments/EXP-20260816-010-cpm-official-lr-control/RECORD.md`
- EXP011：`research/experiments/EXP-20260817-011-cpm-xyz-region-consistency-diagnostic/RECORD.md`
- EXP012：`research/experiments/EXP-20260817-012-hierarchical-correspondence-head/RECORD.md`
- 姿态头诊断快照：`research/POSE_HEAD_DIAGNOSTIC_HANDOFF_CN.md`
- 服务器快照：`research/SERVER_RUNTIME_STATUS_CN.md`
- 运行手册：`research/RUNBOOK_CN.md`
