# 当前研究状态

最后核对：2026-08-19。本页只汇总已经由记录、日志、checkpoint、正式评估或 QC 支持的当前事实；服务器实时状态必须重新执行只读检查。

Git branch、HEAD、远端跟踪和 worktree 属于运行时事实，不在本页固化。需要核对时直接执行 `git status --short --branch`、`git rev-parse HEAD` 和 `git rev-parse --abbrev-ref --symbolic-full-name @{u}`。

EXP012 已完成 E5/E10/E15 正式评估和后续无重训练诊断。E15 出现严重且主要集中于 rotation 的后期泛化退化，因此 EXP012 路线关闭，不继续按原计划跑到 E40。当前进入 EXP013：从完整信息流和计算成本重新设计 correspondence-guided 6D pose architecture。

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
Stage 4C:  EXP010 — historical record unchanged by this sync
Stage 4D:  EXP011 COMPLETE — MISMATCH_IMPORTANT
Stage 4E:  EXP012 COMPLETE — CLOSED AFTER E15 ROTATION-SPECIFIC GENERALIZATION COLLAPSE
Stage 5:   EXP013 PLANNED — CORRESPONDENCE-GUIDED ARCHITECTURE SCREENING, NO FORMAL 40E
```

## EXP012 当前事实

- 正式实现 source commit：`2ca752b3f091292172044209f7c8651280d377bd`；后续两次 main 提交仅为文档更新，不改变 EXP012 模型代码。
- E5：BOP AR `0.642973`，ADD(-S)@0.1d `0.489273`，AR_reS `0.345790`，AR_teS `0.783391`。
- E10：BOP AR `0.645972`，ADD(-S)@0.1d `0.473356`，AR_reS `0.428835`，AR_teS `0.773472`。
- E15：BOP AR `0.544083`，ADD(-S)@0.1d `0.377163`，AR_reS `0.124798`，AR_teS `0.798847`。
- 训练损失 E5→E10→E15 持续下降，但 E15 pose 泛化严重下降；主要变化发生在 rotation，而 translation 指标没有同步崩塌。
- Region×0 在 E5/E10/E15 都造成大幅退化，说明 EXP012 对 Region 输入有强依赖；但依赖强度没有在 E15 突然跃升，不能据此断言 Region 是 E15 崩塌唯一原因。
- Pred XYZ→GT XYZ 的 Three-Path alpha 诊断中，fixed Region 与 synced Region 都没有稳定 pose rescue；E5/E10 最终 Three-Path summary 和 E15 Three-Path 均 QC PASS。
- E10↔E15 checkpoint interpolation endpoints QC PASS。BOP AR `0.646009→0.636141→0.600161→0.574572→0.544127`，AR_reS 同样单调下降；AR_teS 不呈相同行为。该证据支持参数连线上存在连续的 rotation-specific 泛化恶化方向，但不等同于真实训练轨迹。
- Activation drift QC=`FAIL`，只保留为线索，不作为主要诊断证据。

## EXP012 结论边界

已被实验直接支持的是：correspondence 中存在强 pose 信息；EXP012 learned decoder 对 Region 依赖强；提高 XYZ 不能稳定改善 learned pose；E15 后期退化主要集中于 rotation。尚未证明的是“共享 R/t latent 导致崩塌”“Region 是唯一根因”“joint R/t 本身错误”或“PnP 必须成为最终部署结构”。

因此 EXP012 关闭的含义是停止继续修补该 hierarchical learned pose head，不是否定 correspondence-guided pose estimation。

## EXP013 当前方向

EXP013 只保留一条中心信息流：

```text
RGB -> lightweight visual features -> 2D-3D correspondence -> pose
```

当前重点比较两条体系化候选，并尽可能共享同一 correspondence frontend：

- A：`Light Backbone -> correspondence + reliability/coverage -> weighted PnP -> R,t`。作为显式几何 reference；是否部署由真实 accuracy-latency Pareto 决定，不预设 PnP/RANSAC 快或慢。
- B：`Light Backbone -> correspondence -> rotation-specific / translation-specific representations -> direct R,t + geometric consistency`。Rotation 主要使用 normalized/canonical/internal geometry；Translation 主要使用 absolute 2D、ROI/bbox scale、K、object extent 等 metric cues。解耦发生在信息表示层，不只是拆最后两个 FC；R/t 仍允许有明确几何含义的有限耦合。

Backbone、Geometry/Correspondence representation、Region、Mask、Pose decoder 均重新审视。Region、Full Mask 与 GDRNPP 原 pose head 不默认继承；任何新模块必须说明它解决 correspondence→pose 链上的哪个明确问题。

## EXP013 执行边界

当前只开放 architecture screening，不授权 40-epoch 全量 PBR。优先 E1/E3/E5 或等价小规模实验，同时记录 BOP AR、AR_reS、AR_teS、ADD(-S)、逐物体、参数量、FLOPs/MACs、peak VRAM、GPU forward latency 和 end-to-end latency。

A 好/B 差：优先定位 learned mapping；A/B 都差：先修 correspondence frontend；A 差/B 好：排查 direct branch 的额外上下文是合理线索还是 shortcut；A/B 都好：再按 accuracy-efficiency Pareto 与后期稳定性选择。

## 关键入口

- 实验索引：`research/EXPERIMENT_INDEX.md`
- EXP012：`research/experiments/EXP-20260817-012-hierarchical-correspondence-head/RECORD.md`
- EXP013：`research/experiments/EXP-20260819-013-correspondence-guided-architecture/RECORD.md`
- 研究计划：`research/RESEARCH_PLAN.md`
- 姿态头诊断快照：`research/POSE_HEAD_DIAGNOSTIC_HANDOFF_CN.md`
- 服务器快照：`research/SERVER_RUNTIME_STATUS_CN.md`
- 运行手册：`research/RUNBOOK_CN.md`
