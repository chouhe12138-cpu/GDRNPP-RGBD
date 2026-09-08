# EXP019 — EPro-PnP Geometry Utilization Diagnostic

## Protocol：问题与原协议

在历史 EXP004 的 fixed shared support 与 XYZ alpha intervention 下，EPro-PnP 是否
能像 matched RANSAC 一样建立稳定的 `better geometry -> better pose` 关系？

- experiment_id：`EXP-20260907-019-epro-geometry-utilization`。
- 运行状态：`RUN_COMPLETE`；full run 于 2026-09-07 完成，2026-09-08 记录。
  原 evaluator decision：`PROTOCOL_REPRODUCTION_FAILED_STOP`；本次未重新裁决。
- official checkpoint；LM-O BOP19 1,445 targets；GT bbox；无训练。
- alpha `[0,.25,.5,.75,1]`；Patch/RANSAC/EPro 同 producer 与同 xyz_alpha。
- RANSAC/EPro 使用从 Git 历史恢复的完全相同 fixed support。
- EPro 主条件为 uniform w2d；没有 learned reliability、EXP018、BPnP 或 sweep。
- 声明配置：`configs/gdrn/lmo_pbr/research/exp019/epro_diagnostic.py`；runner 未读取
  此文件。实际诊断参数来自 `research/diagnostics/exp019_epro/config.py` 的
  `ExperimentConfig()`，seed 为 `20260730`；模型配置与 checkpoint 见下方运行信息。
- 实现与命令：[EXP019 说明](../../exp019/README.md)。
- EPro 上游 zip SHA-256：`65bfd7639f2c1975fbe397a16ac482168d2f96a6d9239e6317166a2d0215d37b`
  （GitHub archive 标识 `6803aeba9fcac755067062cc5fdd40b761fc3fed`）。
- 交接包 zip SHA-256：`94be97f302527810f0316b4475fc9c34fb589496bd31abe733c7ddb4565fe50a`；
  包内 `SHA256SUMS.txt` 全部通过。
- full run：`RUN-20260907-182144-full-s20260730`（source commit `69e0e8a`，1,445 targets，7,225 行，无训练）。

### 预注册 gate（保持原样）

先要求历史复现绝对差不超过 `0.001`：Patch/RANSAC alpha=0 与 RANSAC alpha=1 的
ADD/BOP。随后：

- alpha=1 EPro BOP AR ≥ 0.95 且 ADD(-S) ≥ 0.95；
- Spearman(alpha, EPro ADD/BOP) 均 ≥ 0.90；
- 相对 matched RANSAC 的 ADD/BOP recovery 均 ≥ 0.50。

失败则 `STOP_DO_NOT_TRAIN`，先检查 metric XYZ、ROI2D、K、resize、support 与 pose
convention；不自动训练或调参。

## Runs / Provenance：运行与证据来源

以下五次运行依据本地原始 `run_metadata.json` 核对。所有 run 目录均位于仓库根目录
下的 `output/experiments/EXP-20260907-019-epro-geometry-utilization/<run_id>/`；
该路径为 Git 忽略的本机外置产物，不保证其他 checkout 已包含这些文件。

| run_id | source commit | metadata 状态 | 完成 targets | 运行事实 |
|---|---|---|---:|---|
| RUN-20260907-175516-smoke-s20260730 | 530faa2 | FAILED | 0 | `KeyError: 'coor_x'`，limit=32 |
| RUN-20260907-180450-smoke-s20260730 | 0d3257f | COMPLETE | 32 | 160 行，limit=32 |
| RUN-20260907-180525-full-s20260730 | 0d3257f | FAILED | 1,203 | `2/942/0: GT XYZ reprojection 0.7689px`，limit=null |
| RUN-20260907-182019-smoke-s20260730 | 69e0e8a | COMPLETE | 32 | 160 行，limit=32 |
| RUN-20260907-182144-full-s20260730 | 69e0e8a | COMPLETE | 1,445 | 7,225 行，limit=null |

五份 metadata 的 `gdrn_config` 均指向本机仓库下的
`configs/gdrn/lmo_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo.py`，
checkpoint 均为 `pretrained_models/lmo_pbr/model_final_wo_optim.pth`（official checkpoint）。
本诊断未训练，不生成新的 epoch checkpoint；官方 checkpoint epoch 未由 metadata 提供。
CLI 当前默认模型配置是 `_base_/lmo_gt_eval.py`，不能用当前默认值替代上述实际运行记录。

full run 的四份紧凑原始 JSON 原样保存于本实验 `evidence/`，未改字段或精度：

- [run_metadata.json](evidence/RUN-20260907-182144-full-s20260730/run_metadata.json)：来源、运行状态与计数。
- [summary.json](evidence/RUN-20260907-182144-full-s20260730/summary.json)：全部 alpha/consumer 的聚合指标及逐物体 ADD。
- [reproduction_report.json](evidence/RUN-20260907-182144-full-s20260730/reproduction_report.json)：六点历史值、本轮值、差值与原容差检查。
- [gate_report.json](evidence/RUN-20260907-182144-full-s20260730/gate_report.json)：原始 Gate A/B、派生量与 evaluator decision。

完整 `poses.jsonl`、`bop_results/` 与 `bop_eval/` 保持在上述 full run 目录，未复制入 Git。
失败 run 和 smoke 的 metadata 仍在各自外置目录；表中事实不替代原文件。

2026-09-08 勘误：旧文字将 `0.7689 px` 失败描述为“第二次 smoke、尚未完成 32 targets”。
原始 metadata 显示它属于 `RUN-20260907-180525-full-s20260730`，已完成 1,203 targets；
其前后分别有一次完成的 32-target smoke。首个 `coor_x` 失败 run_id 也已由 metadata 补齐。
本次仅更正文档身份、时序与产物定位，不改原始运行输出。

## Observed：运行事实与原始程序输出

### 历史初始 CPU 工程验证

Conda `pytorch22`、CPU、无真实样本：

- 交接包全部 SHA-256 通过；从 Git `d702030` 恢复 exact support/intervention/RANSAC 语义。
- 初始新增单测 5 项通过；连同 pose_structure pure utils 与 EXP018 回归共 `30 passed`。
- synthetic EPro solve 通过：rotation error `0°`，translation error
  `8.929934e-8 m`。
- official checkpoint hash、配置、LM-O registrar、GT-box loader 与模型构建通过；模型为
  `GDRN_DoubleMask`，EXP018 corrector 不存在。
- runner/evaluator CLI、launcher preflight 和 `git diff --check` 通过。

该初始阶段尚未读取真实诊断样本、执行 32-target smoke、完整推理或 BOP evaluation；
后续运行见上表。上述历史工程证据不构成科学结果，本次整理未重跑验证。

### Dense-output 修复阶段的工程证据

首次手动 smoke `RUN-20260907-175516-smoke-s20260730` 在任何 target 完成前失败，
metadata 记录 `KeyError: 'coor_x'`；该失败不进入科学结论。

修复限定在 EXP019 config adapter：设置 `TEST.SAVE_RESULTS_ONLY=True` 以使用当前模型
已有的 dense-output 接口，同时保持 `TEST.USE_PNP=False`，不改变网络姿态、不触发
evaluator-side PnP。新增回归断言后 EXP019 `6 passed`。
修复后的 EXP019、pose_structure pure utils 与 EXP018 相关回归合计 `31 passed`；
`git diff --check` 通过。

Agent 仅执行一个真实 LM-O target 的 CPU wiring check（临时路径
`/tmp/exp019-fix-xJSXWJ/run`，不是 smoke/formal）：5 个 alpha 均完成，Patch/RANSAC/EPro
failure 均为 0，support 308 点，GT-XYZ 最大重投影误差 `0.00545669 px`；alpha=0
Patch 重入最大 R 差 `1.78813934e-7`、t 差 0。该检查只证明报错链路已修复。

### GT-XYZ 检查修复阶段的工程证据

failed full run `RUN-20260907-180525-full-s20260730` 已完成 1,203 targets，
随后在 `2/942/0` 读取阶段报告 GT-XYZ 重投影误差 `0.7689 px`；未完成全量诊断，
不进入正式科学结论。历史检查记录该 LM-O 原始旋转标注的 determinant 为
`1.0136865`、正交误差约 `0.00987`，原因解释见 Interpretation。

修复恢复 EXP004 的完整检查口径：GT XYZ 仍按原始 `R` 生成，检查在全部 GT-visible
有效深度点上用 `cv2.Rodrigues` 与 `cv2.projectPoints`，阈值恢复为历史预注册的
`<0.5 px`。对原始 `2/942/0` 的 9,761 个可见有效深度像素进行只读复核，历史口径
最大误差为 `0.385464 px`，而错误的直接矩阵口径为 `0.773077 px`。新增该非正交
LM-O 标注的回归测试。修复后 EXP019、pose_structure 与 EXP018 相关回归合计
`37 passed`；EPro 合成求解及官方模型构建 preflight 通过。这些是当时修复验证记录；
之后完成的 smoke 和 full run 见运行表，不把阶段性“尚未运行”描述当作当前状态。

### 完整 full run（RUN-20260907-182144-full-s20260730）

- source commit `69e0e8aea1f54cb4514666a6a2f6e397b8ad78d4`，branch `EPro-GDRNPP研究`，
  worktree clean；cuda:0；`limit=null`，完成 1,445/1,445 targets，7,225 行。
- 求解失败：Patch 0；RANSAC 55；EPro 55（RANSAC/EPro 均为同一 11 个 target × 5 α，
  全部位于 scene 2）。GT-XYZ 最大重投影误差 `0.421903 px`（历史阈值 `<0.5 px`）。
- 指标与原始报告均随 run 落盘，紧凑原样副本见上方 evidence 链接。

### 汇总指标（ADD(-S) / BOP AR，均按原聚合口径）

| α | Patch ADD | Patch BOP | RANSAC ADD | RANSAC BOP | EPro ADD | EPro BOP |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.5066 | 0.6904 | 0.5398 | 0.6909 | 0.5433 | 0.7009 |
| 0.25 | 0.5003 | 0.6899 | 0.6166 | 0.7176 | 0.6221 | 0.7298 |
| 0.50 | 0.4990 | 0.6897 | 0.7398 | 0.7831 | 0.7550 | 0.7899 |
| 0.75 | 0.4962 | 0.6898 | 0.8519 | 0.8539 | 0.8595 | 0.8647 |
| 1.00 | 0.4997 | 0.6837 | 0.9924 | 0.9924 | 0.9924 | 0.9924 |

reS/teS：Patch α0→1 `0.5306→0.5276 / 0.8023→0.8044`；RANSAC `0.5444→0.9924 /
0.8076→0.9924`；EPro `0.5661→0.9924 / 0.8136→0.9924`。

### EPro 逐物体 ADD(-S)

| α | ape | can | cat | driller | duck | eggbox | glue | holepuncher |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.514 | 0.834 | 0.538 | 0.890 | 0.206 | 0.372 | 0.707 | 0.280 |
| 0.25 | 0.697 | 0.900 | 0.632 | 0.955 | 0.439 | 0.106 | 0.821 | 0.430 |
| 0.50 | 0.840 | 0.940 | 0.784 | 0.960 | 0.761 | 0.133 | 0.893 | 0.725 |
| 0.75 | 0.954 | 0.975 | 0.883 | 0.970 | 0.917 | 0.222 | 0.957 | 0.985 |
| 1.00 | 0.994 | 1.000 | 0.994 | 0.980 | 1.000 | 0.978 | 0.993 | 1.000 |

### 原始 Gate / evaluator 输出

- Gate A（α=1 EPro ADD/BOP ≥ 0.95）：`0.9924 / 0.9924` → 通过。
- Gate B（Spearman ≥ 0.90；recovery ≥ 0.50）：`rho = 1.0 / 1.0`；
  ADD/BOP recovery = `0.992 / 0.967` → 通过。
- 历史复现 gate（EXP004 六点绝对差 ≤ 0.001）：`reproduction_report.json` 状态 `FAIL`，
  6 点中 5 点越界，仅 Patch α=0 BOP（+0.00019）通过；逐点有符号差见 Derived。
- 因此整体 decision 被覆盖为 **`PROTOCOL_REPRODUCTION_FAILED_STOP`**
  （evaluate.py 复现 gate 失败时强制覆盖，即使 Gate A/B 通过）。

## Derived：派生量与历史复现对照

以下量由原始 JSON 中的指标直接计算，未新增评估或改变聚合口径：

- `delta = metric(alpha=1) - metric(alpha=0)`：EPro ADD/BOP 为
  `0.449135 / 0.291474`；matched RANSAC 为 `0.452595 / 0.301439`。
- `recovery = delta_EPro / delta_RANSAC`：ADD/BOP 为 `0.992355 / 0.966941`。
- 原报告 Spearman(alpha, EPro ADD/BOP) 均约为 `1.0`，只描述本次五点聚合曲线。
- 下表 `差 = 本轮 - 历史`，检查使用 `abs(差) ≤ 0.001`；数值为 recall 比例，
  不是百分数。展示值有舍入，检查以原 JSON 精度为准。

| 对照点 | 本轮 | 历史 | 差 | ≤0.001？ |
|---|---:|---:|---:|---|
| Patch α=0 ADD | 0.5066 | 0.50242 | +0.00415 | ❌ |
| Patch α=0 BOP | 0.6904 | 0.69021 | +0.00019 | ✅ |
| RANSAC α=0 ADD | 0.5398 | 0.53841 | +0.00138 | ❌ |
| RANSAC α=0 BOP | 0.6909 | 0.69255 | −0.00160 | ❌ |
| RANSAC α=1 ADD | 0.9924 | 0.99377 | −0.00138 | ❌ |
| RANSAC α=1 BOP | 0.9924 | 0.99377 | −0.00138 | ❌ |

## Evidence Boundary：证据边界

全量运行完成、Gate A/B 通过和历史复现 gate 失败是不同事实，均予保留。该 run
使用一次固定协议和五个 alpha；没有新增跨 run/seed 稳定性证据。原始记录中的工程
修复原因不解释 full run 与历史指标的偏差；此次整理也未完成偏差归因或可比性 review。
不能仅凭越界判定机制失败，也不能据此把漂移认定为无害或宣布正式机制结论成立。

## Interpretation / Review：历史解释与后续复核

- 历史 dense-output 修复解释：当时 test forward 默认只返回 `rot/trans`，旧调用假设
  `coor_x/y/z/mask/region` 总在输出中，导致首个 smoke 报错。
- 历史 GT-XYZ 修复解释：直接矩阵投影遗漏了 EXP004 使用的 OpenCV Rodrigues
  转换；原记录将该报错归于检查口径，而非求解器或数据越界。本次保留该工程解释，
  不将它扩展为历史复现偏差的原因。
- 后续研究 review 尚未完成。本次只按原始 metadata/JSON 修正客观记录并区分层级，
  不加入新的科学解释或结论，不修改历史 gate。

## Decision：保留原处理，不重新裁决

原 evaluator decision 保持 `PROTOCOL_REPRODUCTION_FAILED_STOP`，Gate A/B 的 true
输出同时保留。原记录按预注册协议作出的“不进入正式科学结论、不触发训练”处理不变；
程序输出与研究判断分层，不表示本次已经完成新的研究裁决或获得新实验授权。

## 2026-09-08 用户 Review 决策（并列记录，不覆盖原始输出）

用户 review 判定 EXP019 为**机制通过**（Decision，用户）：α-sweep 证明 EPro-PnP 能
稳定消费逐步改善的 XYZ（α 0→1，EPro ADD `0.5433→0.9924`、BOP `0.7009→0.9924`），
同一数据中官方 Patch-PnP 对同样改善不响应（ADD `0.5066→0.4997`、BOP
`0.6904→0.6837`）；历史复现 6 点偏差折算为 2–6 个 target（0.14%–0.83% 相对），低于
该指标的二项噪声（SE 约 1.8%–2.6% 相对），视为正常运行差异，不否定机制。

- 原 evaluator decision `PROTOCOL_REPRODUCTION_FAILED_STOP` 与其 gate 输出保持原样，
  作为 Observed 保留；本节是与原始输出并列的研究判断，不改写历史 gate。
- 用户同时决定：后续 gate 不再使用 0.001 这类绝对数值阈值，改为按指标尺度设定的
  相对浮动（如 ±3%–±5%）；接近饱和的指标仍用绝对阈值。
- 机制推理、文献证据与口径建议见
  [solver-in-the-loop review](../../notes/20260908-solver-in-the-loop-review.md)。
