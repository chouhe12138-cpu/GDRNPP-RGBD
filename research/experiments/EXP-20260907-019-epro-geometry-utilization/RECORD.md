# EXP019 — EPro-PnP Geometry Utilization Diagnostic

## 问题与协议

在历史 EXP004 的 fixed shared support 与 XYZ alpha intervention 下，EPro-PnP 是否
能像 matched RANSAC 一样建立稳定的 `better geometry -> better pose` 关系？

- experiment_id：`EXP-20260907-019-epro-geometry-utilization`。
- 状态：`RUN_COMPLETE / PROTOCOL_REPRODUCTION_FAILED_STOP`；full run 于 2026-09-07 完成，2026-09-08 记录。
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
- full run：`RUN-20260907-182144-full-s20260730`（source commit `69e0e8a`，1,445 targets，7,225 行，无训练）。

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

## 正式 full run（RUN-20260907-182144-full-s20260730）

### 运行记录

- source commit `69e0e8aea1f54cb4514666a6a2f6e397b8ad78d4`，branch `EPro-GDRNPP研究`，
  worktree clean；cuda:0；`limit=null`，完成 1,445/1,445 targets，7,225 行。
- 求解失败：Patch 0；RANSAC 55；EPro 55（RANSAC/EPro 均为同一 11 个 target × 5 α，
  全部位于 scene 2）。GT-XYZ 最大重投影误差 `0.421903 px`（历史阈值 `<0.5 px`）。
- 正式指标、reproduction report、gate report 见本目录 `summary.json`、
  `reproduction_report.json`、`gate_report.json`，均随 run 落盘。

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

### Gate 判定

- Gate A（α=1 EPro ADD/BOP ≥ 0.95）：`0.9924 / 0.9924` → 通过。
- Gate B（Spearman ≥ 0.90；recovery ≥ 0.50）：`rho = 1.0 / 1.0`；
  ADD/BOP recovery = `0.992 / 0.967` → 通过。
- 历史复现 gate（EXP004 六点绝对差 ≤ 0.001）：`reproduction_report.json` 状态 `FAIL`，
  6 点中 5 点越界，仅 Patch α=0 BOP（+0.00019）通过；逐点绝对差见下表。
- 因此整体 decision 被覆盖为 **`PROTOCOL_REPRODUCTION_FAILED_STOP`**
  （evaluate.py 复现 gate 失败时强制覆盖，即使 Gate A/B 通过）。

### 历史复现逐点对照（本轮 − 历史 EXP004）

| 对照点 | 本轮 | 历史 | 差 | ≤0.001？ |
|---|---:|---:|---:|---|
| Patch α=0 ADD | 0.5066 | 0.50242 | +0.00415 | ❌ |
| Patch α=0 BOP | 0.6904 | 0.69021 | +0.00019 | ✅ |
| RANSAC α=0 ADD | 0.5398 | 0.53841 | +0.00138 | ❌ |
| RANSAC α=0 BOP | 0.6909 | 0.69255 | −0.00160 | ❌ |
| RANSAC α=1 ADD | 0.9924 | 0.99377 | −0.00138 | ❌ |
| RANSAC α=1 BOP | 0.9924 | 0.99377 | −0.00138 | ❌ |

### 证据边界

该 run 只记录上述运行事实与 gate 数值。因历史复现 gate 未过，按预注册协议本 run
不进入正式科学结论；此处不做机制归因或训练建议。
