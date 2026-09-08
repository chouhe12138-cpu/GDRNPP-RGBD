# 当前研究状态

最后核对：2026-09-08。

## 当前结论

- EXP012 已完成 40 epoch，E40 BOP AR `0.678800`、ADD(-S) `0.494118`、
  AR_reS `0.491349`、AR_teS `0.791926`，是 EXP013 的固定比较基准。
- EXP013A 通过相对 EXP012 的正式门槛；B 的 ADD 改善但严格 BOP 门槛差
  `0.000109`；C 支持 R/t 解耦改善 rotation，但总体 ADD 门槛失败。
- EXP013E 官方头随机初始化对照完成：BOP `0.688581`、reS `0.535409`，对
  “结构可读出与预训练继承并重”提供部分支持。
- EXP013F GLM-Pose-L 完成：BOP `0.684129`、reS `0.515802`，四项门槛通过
  2 项，结论为边缘 `SCREEN_FAIL`。
- EXP017 canonical formal 已完成 E40：BOP `0.681709`、ADD `0.512111`、
  reS `0.494348`、teS `0.800461`。预注册五项门槛通过三项，rotation/BOP 未过，
  结论 `SCREEN_FAIL`。
- EXP017-B 已完成 E40：BOP `0.683686`、ADD `0.498270`、reS `0.503114`、
  teS `0.800692`。相对 EXP017 的 rotation/BOP 略升，但 ADD 下降 `0.013841`，
  结论 `NO_OVERALL_GAIN`。原 B 记录未独立列出数值 gate，参照母实验门槛仅达 1/5。
- EXP014-D 的 formal a01 因渲染器覆盖事故和 OOM 作废。EGL 修复保留，实验
  当前 `PAUSED`，没有重训授权。
- EXP019 full run `RUN-20260907-182144-full-s20260730` 已完成 1,445 targets；
  原始 Gate A/B 均通过，历史复现 6 点中 5 点越界（原容差 0.001）。原 evaluator
  decision `PROTOCOL_REPRODUCTION_FAILED_STOP` 保留为历史输出。2026-09-08 用户
  review 判定**机制通过**：EPro-PnP 稳定消费逐步改善的 XYZ；漂移折算 2–6 个
  target（0.14%–0.83% 相对），属正常运行差异。后续 gate 改用相对阈值
  （±3%–±5%），见 [DECISIONS](DECISIONS.md) 与
  [review](notes/20260908-solver-in-the-loop-review.md)。

## 当前代码边界

- 保留上游 GDRNPP、EXP012、EXP013 A–F、暂停的 D、EXP017、EXP018、EXP019
  和当前结构诊断。
- EXP001–011 的科学结论保留在 RECORD；旧专用执行代码和 managed experiment
  审计框架已退出当前工作树，可通过 Git 历史恢复。
- lightweight framework 已由 EXP013F 在 lab0/lab1 双机 smoke 验证，EXP005
  matched control smoke 也完成；先前 launcher/cache/native 问题均为基础设施迁移
  失败，不属于科学结果。
- EXP017/EXP017-B 已完成，保留实现和正式记录。按用户要求删除了本轮新增的本地
  E40 adapter 开关诊断脚本和产物；该诊断不混入正式训练结果。
- 任何新增服务器 run 必须先由用户明确选择实验和配置，再使用
  `docker/l40/experiment.sh`；本次没有服务器操作。

## 下一步

EXP017 与 EXP017-B 已以 E40 收口。用户已指定 EXP018：EXP013A initial pose 后增加一次
Geometry-Consistency Residual correction；实现与 CPU preflight 已通过，当前等待用户
手动真实数据 smoke。详见 [EXP018 RECORD](experiments/EXP-20260906-018-geometry-consistency-residual/RECORD.md)
与 [实现/命令](exp018/README.md)。该集成当时已获本地提交并推送 GitHub 授权；里程碑标签为
`exp018-integration-cpu-pass`，不代表真实 smoke 或正式性能 gate 通过。没有服务器操作；
smoke 收口和正式 gate 确认后才进入发布/训练流程。不恢复 D，不自动增加 seed。

用户随后指定 EXP019：在历史 EXP004 的 fixed support 与 XYZ alpha sweep 上增加
uniform-weight EPro-PnP consumer。full run `RUN-20260907-182144-full-s20260730`
（commit `69e0e8a`）已完成全量诊断，不再是待执行任务。原始检查与 decision 如上，
2026-09-08 用户 review 已判定机制通过。详见
[EXP019 RECORD](experiments/EXP-20260907-019-epro-geometry-utilization/RECORD.md)
与 [执行说明](exp019/README.md)。

用户 2026-09-08 确定新研究主线：用可微 EPro-PnP 作为显式几何后端，把最终姿态监督
反传约束 Geometry/Correspondence Head（XYZ、ROI2D、Mask、Region、Reliability），
使对应关系学习以“对求解器有用”为目标；EPro-PnP 不是主要创新点。当前尚未安排具体
实验与服务器操作。工程前置项是把 EPro-PnP 与 GDRNPP 放到同进程以支持梯度反传
（EXP019 当前因顶层 `lib` 包名冲突运行在独立 spawn 子进程）。文献对照、重合度分析与
gate 口径见 [solver-in-the-loop review](notes/20260908-solver-in-the-loop-review.md)；
EPRO-GDR（arXiv 2409.11819）已在方法骨架上占位，本项目新意须落在机制归因、
对应关系级指标与跨域。
