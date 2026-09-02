# 当前研究状态

最后核对：2026-09-02。

## 当前结论

- EXP012 已完成 40 epoch，E40 BOP AR `0.678800`、ADD(-S) `0.494118`、
  AR_reS `0.491349`、AR_teS `0.791926`，是 EXP013 的固定比较基准。
- EXP013A 通过相对 EXP012 的正式门槛；B 的 ADD 改善但严格 BOP 门槛差
  `0.000109`；C 支持 R/t 解耦改善 rotation，但总体 ADD 门槛失败。
- EXP013E 官方头随机初始化对照完成：BOP `0.688581`、reS `0.535409`，对
  “结构可读出与预训练继承并重”提供部分支持。
- EXP013F GLM-Pose-L 完成：BOP `0.684129`、reS `0.515802`，四项门槛通过
  2 项，结论为边缘 `SCREEN_FAIL`。
- EXP017 已按 EXP013A 主体实现 rotation-only spatial residual adapter；新增参数
  `13,000`。运行框架复核已将冻结 geometry 的训练 renderer 关闭，同时保留 CPP BOP
  evaluation，并补齐 clean-tree/protocol/metadata/checkpoint-order gate。修复前 formal
  已停止并保留；source `6b4d412` 的服务器 smoke 以 `exit_code=0` 完成，用户已授权
  从同一 release 启动 canonical formal。
- EXP014-D 的 formal a01 因渲染器覆盖事故和 OOM 作废。EGL 修复保留，实验
  当前 `PAUSED`，没有重训授权。

## 当前代码边界

- 保留上游 GDRNPP、EXP012、EXP013 A–F、暂停的 D、EXP017 和当前结构诊断。
- EXP001–011 的科学结论保留在 RECORD；旧专用执行代码和 managed experiment
  审计框架已退出当前工作树，可通过 Git 历史恢复。
- lightweight framework 已由 EXP013F 在 lab0/lab1 双机 smoke 验证，EXP005
  matched control smoke 也完成；先前 launcher/cache/native 问题均为基础设施迁移
  失败，不属于科学结果。
- 当前没有活动服务器训练。新运行必须先由用户明确选择实验和配置，再使用
  `docker/l40/experiment.sh`。

## 下一步

EXP017 本地门禁已经满足，实现已提交；下一步是生成并转移 release bundle，用户完成
服务器只读检查并确认 GPU 后，再单独请求 formal 训练授权。不自动恢复 D，也不因单次
边缘差距自动增加 seed。完整数值与结论边界见实验索引中的 RECORD。
