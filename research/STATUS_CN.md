# 当前研究状态

最后核对：2026-09-03。

## 当前结论

- EXP012 已完成 40 epoch，E40 BOP AR `0.678800`、ADD(-S) `0.494118`、
  AR_reS `0.491349`、AR_teS `0.791926`，是 EXP013 的固定比较基准。
- EXP013A 通过相对 EXP012 的正式门槛；B 的 ADD 改善但严格 BOP 门槛差
  `0.000109`；C 支持 R/t 解耦改善 rotation，但总体 ADD 门槛失败。
- EXP013E 官方头随机初始化对照完成：BOP `0.688581`、reS `0.535409`，对
  “结构可读出与预训练继承并重”提供部分支持。
- EXP013F GLM-Pose-L 完成：BOP `0.684129`、reS `0.515802`，四项门槛通过
  2 项，结论为边缘 `SCREEN_FAIL`。
- EXP017 canonical formal 正在运行，E10 已评估：BOP `0.642787`、ADD `0.491349`、
  reS `0.419608`、teS `0.770473`，E40 仍是唯一决策点。代码/autograd 复核确认 adapter
  rotation 梯度会额外进入 translation 共用的 geometry encoder；E10 8 样本只读诊断中
  该增量梯度范数非零且不可忽略。已按严格单变量准备 EXP017-B detach 候选并通过本地
  tests/preflight，但未获 smoke 或训练授权，不影响当前 EXP017 formal。
- EXP014-D 的 formal a01 因渲染器覆盖事故和 OOM 作废。EGL 修复保留，实验
  当前 `PAUSED`，没有重训授权。

## 当前代码边界

- 保留上游 GDRNPP、EXP012、EXP013 A–F、暂停的 D、EXP017 和当前结构诊断。
- EXP001–011 的科学结论保留在 RECORD；旧专用执行代码和 managed experiment
  审计框架已退出当前工作树，可通过 Git 历史恢复。
- lightweight framework 已由 EXP013F 在 lab0/lab1 双机 smoke 验证，EXP005
  matched control smoke 也完成；先前 launcher/cache/native 问题均为基础设施迁移
  失败，不属于科学结果。
- 当前 EXP017 canonical formal 由用户在服务器运行；不得停止或修改。任何新增 run 必须
  先由用户明确选择实验和配置，再使用 `docker/l40/experiment.sh`。

## 下一步

等待 EXP017 E40，不因 E5/E10 中间趋势停止或改动当前 formal。EXP017-B 当前只作为
graph-isolation 候选保留；若之后另行授权，最小下一步是本地真实小批次 smoke，不直接
启动 screening/formal。不自动恢复 D，也不因单次边缘差距自动增加 seed。
