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
- 初始新增单测 5 项通过；连同 pose_structure pure utils 与 EXP018 回归共 `30 passed`。
- synthetic EPro solve 通过：rotation error `0°`，translation error
  `8.929934e-8 m`。
- official checkpoint hash、配置、LM-O registrar、GT-box loader 与模型构建通过；模型为
  `GDRN_DoubleMask`，EXP018 corrector 不存在。
- runner/evaluator CLI、launcher preflight 和 `git diff --check` 通过。

尚未读取真实诊断样本、执行 32-target smoke、完整推理或 BOP evaluation。上述工程证据
不构成科学结果。

## Dense-output 接口修复

用户第一次手动 smoke 在首个 forward 后、任何 target 完成前失败：当前 test forward
默认只返回 `rot/trans`，旧 EXP004 代码假设 `coor_x/y/z/mask/region` 总在输出中，触发
`KeyError: 'coor_x'`。run_id 未从截图中取得；该失败不进入科学结论。

修复限定在 EXP019 config adapter：设置 `TEST.SAVE_RESULTS_ONLY=True` 以使用当前模型
已有的 dense-output 接口，同时保持 `TEST.USE_PNP=False`，不改变网络姿态、不触发
evaluator-side PnP。新增回归断言后 EXP019 `6 passed`。
修复后的 EXP019、pose_structure pure utils 与 EXP018 相关回归合计 `31 passed`；
`git diff --check` 通过。

Agent 仅执行一个真实 LM-O target 的 CPU wiring check（临时路径
`/tmp/exp019-fix-xJSXWJ/run`，不是 smoke/formal）：5 个 alpha 均完成，Patch/RANSAC/EPro
failure 均为 0，support 308 点，GT-XYZ 最大重投影误差 `0.00545669 px`；alpha=0
Patch 重入最大 R 差 `1.78813934e-7`、t 差 0。该检查只证明报错链路已修复。

## GT-XYZ 检查口径修复

用户第二次手动 smoke 在 `2/942/0` 读取阶段失败，当前实现报告 GT-XYZ 最大重投影
误差 `0.7689 px`，尚未完成 32 targets，故仍不进入科学结论。根因不是求解器或数据
越界：该 LM-O 原始旋转标注的 determinant 为 `1.0136865`、正交误差约 `0.00987`；
EXP019 的直接矩阵投影遗漏了历史 EXP004 使用的 OpenCV Rodrigues 转换。

修复恢复 EXP004 的完整检查口径：GT XYZ 仍按原始 `R` 生成，检查在全部 GT-visible
有效深度点上用 `cv2.Rodrigues` 与 `cv2.projectPoints`，阈值恢复为历史预注册的
`<0.5 px`。对原始 `2/942/0` 的 9,761 个可见有效深度像素进行只读复核，历史口径
最大误差为 `0.385464 px`，而错误的直接矩阵口径为 `0.773077 px`。新增该非正交
LM-O 标注的回归测试。修复后 EXP019、pose_structure 与 EXP018 相关回归合计
`37 passed`；EPro 合成求解及官方模型构建 preflight 通过。未执行新的 smoke 或正式实验。
