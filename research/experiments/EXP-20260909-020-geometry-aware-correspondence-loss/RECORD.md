# EXP020 — Geometry-Aware Correspondence Reprojection Supervision

## Status

- lifecycle：`CLOSED`
- execution：`FORMAL_RESULTS_PARTIAL`
- direct-pose E5–E40：已记录
- fixed-support matched PnP：未运行
- A/B exit code：未确认
- 2026-09-16 用户决定结束补证，不追加实验或机制结论。

## Research question

在 continuous normalized XYZ 上增加由 GT pose、object extent 与 crop-resized camera matrix
构造的 per-pixel reprojection loss，能否产生更真实、更适合普通 PnP 的 correspondence？
本实验只改变 producer supervision，不引入 EPro-PnP、differentiable PnP、confidence head
或新 pose head。

## Protocol

- experiment ID：`EXP-20260909-020-geometry-aware-correspondence-loss`。
- base `ed6409c`；实现 `64e9098`；review fix `e5ad38b`；formal source
  `c2a7723d5cf468de97ef8aa6f860c4f474203f30`。
- 唯一变量：A `LOSS_CFG.REPROJ_LW=0.0`，B `REPROJ_LW=1.0`。
- official checkpoint、seed42、40 epoch、batch48、Ranger 8e-4、wd .01、warmup200。
- backbone/PNP_NET frozen，GEO_HEAD trainable；保留 XYZ/mask/full-mask/region loss；直接
  pose losses 为 0。训练使用 EGL，BOP evaluation 使用 CPP。
- A run：`RUN-20260909-124125-formal-s42-a01`；B run：
  `RUN-20260909-124137-formal-s42-a01`。
- 两臂镜像 `gdrnpp-research:torch220-cu121-sm89-c0be1ade7ea9`，image revision `c0be1ade7ea9`。

主评价原定为 fixed-support matched RANSAC-PnP：reference checkpoint 生成一次
`S_fixed = reference_pred_visible ∩ gt_visible ∩ valid_depth`，冻结 support、2D 点、K、seed、
3 px threshold 与 100 iterations，A/B 只替换预测 XYZ。仓库 direct-pose evaluator 仅作
telemetry，不能代替该主评价。

## Canonical runs and engineering disposition

| run /阶段 | status | scientific use | disposition |
|---|---|---|---|
| CPP renderer A/B smoke (`698a8fe`) | COMPLETE | 无 | protocol-invalid；训练 renderer 应为 EGL |
| B EGL smoke `RUN-20260909-113749-smoke-s42-a01` | FAILED | 无 | 只读 repo 下 `.cache` 路径错误 |
| A EGL smoke `RUN-20260909-120624-smoke-s42-a01` | FAILED | 无 | nested loader 未透传 cache path |
| 修复后 A/B smoke | PASS | integrity | EGL/CUDA/loss/optimizer 链路通过 |
| A/B formal runs | 到达 E40 | partial formal telemetry | 无 exit code；matched PnP 未运行 |

cache 问题通过把 renderer 模型加载 cwd 临时切到可写 XDG cache 解决；launcher idle gate
改为拒绝容器内除 `sleep infinity` 外的进程。它们是工程修复，不改变 A/B 科学变量。

formal 前检查包括 loss 数学/梯度、A/B config、strict checkpoint、only-GEO_HEAD trainable、
GPU real smoke 和 matched evaluator identity。单 batch 梯度标定为
`g_xyz=4.9539`、`g_reproj_raw=0.3695`、ratio `0.0746`，因此保持 `REPROJ_LW=1.0`；未做
lambda sweep。这些只证明链路和量级。

## Observed direct-pose results

两份日志均到 epoch40 iteration `255919/255920`，并记录保存 `model_epoch_040.pth`；日志未见
Traceback/OOM/ERROR，但未提供 exit code，也未核验服务器 checkpoint 文件内容。

| Epoch | arm | BOP AR | ADD(-S)0.1d | AR_reS | AR_teS |
|---:|---|---:|---:|---:|---:|
| 5 | A | 0.563426 | 0.255363 | 0.433679 | 0.634141 |
| 5 | B | 0.567089 | 0.278893 | 0.449366 | 0.638754 |
| 10 | A | 0.518987 | 0.253979 | 0.419839 | 0.571857 |
| 10 | B | 0.521398 | 0.253979 | 0.412918 | 0.573010 |
| 15 | A | 0.497206 | 0.238062 | 0.370934 | 0.546251 |
| 15 | B | 0.495430 | 0.222837 | 未核实 | 未核实 |
| 20 | A | 0.481451 | 0.215225 | 0.348558 | 0.528028 |
| 20 | B | 0.475649 | 0.193080 | 未核实 | 未核实 |
| 25 | A | 0.481423 | 0.215225 | 0.355248 | 0.522722 |
| 25 | B | 0.486946 | 0.215225 | 未核实 | 未核实 |
| 30 | A | 0.482198 | 0.217301 | 0.322953 | 0.529181 |
| 30 | B | 0.478404 | 0.204152 | 0.324798 | 0.523645 |
| 35 | A | 0.464734 | 0.164014 | 0.317878 | 0.506574 |
| 35 | B | 0.465008 | 0.154325 | 0.326413 | 0.502653 |
| 40 | A | 0.460300 | 0.159170 | 0.326182 | 0.495502 |
| 40 | B | 0.462563 | 0.152941 | 0.325952 | 0.494118 |

B 的 E15/E20/E25 score 文件名与日志 epoch 不一致：三份文件的 BOP AR 分别与日志 E20/E25/E30
相符。表中 BOP/ADD 使用带明确 epoch 的日志；无法可信归属的 reS/teS 保持“未核实”，不重映射。

### Required per-object ADD(-S)0.1d points

| object | A E5 | B E5 | A E40 | B E40 |
|---|---:|---:|---:|---:|
| ape | 0.125714 | 0.068571 | 0.017143 | 0.045714 |
| can | 0.301508 | 0.346734 | 0.145729 | 0.130653 |
| cat | 0.210526 | 0.216374 | 0.134503 | 0.134503 |
| driller | 0.345000 | 0.355000 | 0.150000 | 0.115000 |
| duck | 0.033333 | 0.027778 | 0.038889 | 0.061111 |
| eggbox | 0.405556 | 0.477778 | 0.338889 | 0.333333 |
| glue | 0.314286 | 0.342857 | 0.164286 | 0.157143 |
| holepuncher | 0.295000 | 0.375000 | 0.270000 | 0.240000 |

E5 是两臂 BOP/ADD 的最高固定评价点，但只作事后描述，不用于改写预注册的 E40 比较。
其他 epoch 逐物体 ADD 保存在各 arm 的 `eval_summary.jsonl`。

## Derived / interpretation

E40 的 B−A：BOP `+0.002263`（+0.49%）、ADD `−0.006228`（−3.91%）、reS
`−0.000231`（−0.07%）、teS `−0.001384`（−0.28%）。direct-pose 指标整体随训练下降，
B 没有形成四项一致收益。

这只描述更新 geometry 与冻结 Patch-PnP 的组合表现；没有 fixed-support matched PnP，不能判断
correspondence 是否更真实或更适合 classical PnP。单 seed 也不支持跨 run 稳定性结论。

## Decision

不依据 direct-pose telemetry 宣布机制通过或失败。原本计划补充 exit code、明确 E40 checkpoint
和 matched PnP；2026-09-16 用户决定结束 EXP020，故这些缺口永久保留，不用 EXP021 或其他
近似指标填补，也不新增训练或 seed。

## Evidence map

- A：[`evidence/RUN-20260909-124125-formal-s42-a01/`](evidence/RUN-20260909-124125-formal-s42-a01/)
- B：[`evidence/RUN-20260909-124137-formal-s42-a01/`](evidence/RUN-20260909-124137-formal-s42-a01/)

每个目录保留 E5–E40 八份原始 BOP score JSON 和从 console 提取的 `eval_summary.jsonl`。
完整日志、checkpoint 与大型 evaluator 输出外置于 `E:\6D姿态估计\EXP020\实验A` 和 `实验B`。

## Known gaps / boundaries

- fixed-support matched PnP/RANSAC 未运行；identity/perturbed smoke 不可替代。
- A/B run exit code、服务器 E40 checkpoint 文件与 B E15/E20/E25 reS/teS 归属未确认。
- 没有跨 seed 证据；direct-pose telemetry 不是 producer 机制主判据。
