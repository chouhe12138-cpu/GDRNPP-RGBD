# 当前研究状态

最后核对：2026-09-06。

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

## 当前代码边界

- 保留上游 GDRNPP、EXP012、EXP013 A–F、暂停的 D、EXP017 和当前结构诊断。
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
与 [实现/命令](exp018/README.md)。用户已授权本地提交并推送 GitHub；集成里程碑标签为
`exp018-integration-cpu-pass`，不代表真实 smoke 或正式性能 gate 通过。没有服务器操作；
smoke 收口和正式 gate 确认后才进入发布/训练流程。不恢复 D，不自动增加 seed。
