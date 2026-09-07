# EXP019 — EPro-PnP Geometry Utilization Diagnostic

## 问题与协议

在历史 EXP004 的 fixed shared support 与 XYZ alpha intervention 下，EPro-PnP 是否
能像 matched RANSAC 一样建立稳定的 `better geometry -> better pose` 关系？

- experiment_id：`EXP-20260907-019-epro-geometry-utilization`。
- 状态：`IMPLEMENTED / ENGINEERING_PREFLIGHT_PASS / AWAITING_USER_SMOKE`；2026-09-07。
- official checkpoint；LM-O BOP19 1,445 targets；GT bbox；无训练。
- alpha `[0,.25,.5,.75,1]`；Patch/RANSAC/EPro 同 producer 与同 xyz_alpha。
- RANSAC/EPro 使用从 Git 历史恢复的完全相同 fixed support。
- EPro 主条件为 uniform w2d；没有 learned reliability、EXP018、BPnP 或 sweep。
- 配置：`configs/gdrn/lmo_pbr/research/exp019/epro_diagnostic.py`。
- 实现与命令：`research/exp019/README.md`。
- EPro 上游 zip SHA-256：`65bfd7639f2c1975fbe397a16ac482168d2f96a6d9239e6317166a2d0215d37b`
  （GitHub archive 标识 `6803aeba9fcac755067062cc5fdd40b761fc3fed`）。
- 交接包 zip SHA-256：`94be97f302527810f0316b4475fc9c34fb589496bd31abe733c7ddb4565fe50a`；
  包内 `SHA256SUMS.txt` 全部通过。
- run_id、正式指标、逐物体结果：尚未生成。

## 预注册 gate

先要求历史复现绝对差不超过 `0.001`：Patch/RANSAC alpha=0 与 RANSAC alpha=1 的
ADD/BOP。随后：

- alpha=1 EPro BOP AR ≥ 0.95 且 ADD(-S) ≥ 0.95；
- Spearman(alpha, EPro ADD/BOP) 均 ≥ 0.90；
- 相对 matched RANSAC 的 ADD/BOP recovery 均 ≥ 0.50。

失败则 `STOP_DO_NOT_TRAIN`，先检查 metric XYZ、ROI2D、K、resize、support 与 pose
convention；不自动训练或调参。

## 当前证据边界

Conda `pytorch22`、CPU、无真实样本：

- 交接包全部 SHA-256 通过；从 Git `d702030` 恢复 exact support/intervention/RANSAC 语义。
- 新增单测 5 项通过；连同 pose_structure pure utils 与 EXP018 回归共 `30 passed`。
- synthetic EPro solve 通过：rotation error `0°`，translation error
  `8.929934e-8 m`。
- official checkpoint hash、配置、LM-O registrar、GT-box loader 与模型构建通过；模型为
  `GDRN_DoubleMask`，EXP018 corrector 不存在。
- runner/evaluator CLI、launcher preflight 和 `git diff --check` 通过。

尚未读取真实诊断样本、执行 32-target smoke、完整推理或 BOP evaluation。上述工程证据
不构成科学结果。
