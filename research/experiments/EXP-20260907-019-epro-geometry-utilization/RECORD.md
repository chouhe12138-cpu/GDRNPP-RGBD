# EXP019 — EPro-PnP Geometry Utilization Diagnostic

## Status

- lifecycle：`COMPLETE`
- execution：`DIAGNOSTIC_ONLY`
- full run：`RUN-20260907-182144-full-s20260730`
- source：`69e0e8aea1f54cb4514666a6a2f6e397b8ad78d4`
- 原 evaluator decision：`PROTOCOL_REPRODUCTION_FAILED_STOP`
- 2026-09-08 用户 review：机制通过；与原 evaluator decision 并列，不覆盖原输出。

## Research question and protocol

在 EXP004 的 fixed shared support 与 XYZ alpha intervention 下，EPro-PnP 是否能建立稳定的
`better geometry → better pose` 响应？本实验不训练模型。

- official LM-O checkpoint、BOP19 1,445 targets、GT bbox。
- alpha `[0,.25,.5,.75,1]`；Patch/RANSAC/EPro 使用相同 producer 与 xyz_alpha。
- RANSAC/EPro 使用从历史恢复的同一 fixed support；EPro 使用 uniform w2d。
- 实际参数来自 `research/diagnostics/exp019_epro/config.py`，seed `20260730`。
- full run 完成 1,445 targets、7,225 行；Patch 失败 0，RANSAC/EPro 各失败 55
  （同一 11 个 target × 5 alpha）。

原预注册 gate：先要求 EXP004 六个复现点绝对差 ≤0.001，再要求 alpha=1 EPro
ADD/BOP ≥0.95、Spearman ≥0.90、相对 matched RANSAC recovery ≥0.50。复现失败时原协议
要求 `STOP_DO_NOT_TRAIN`。

## Canonical runs

| run_id | source | status | targets | scientific use |
|---|---|---|---:|---|
| `RUN-20260907-175516-smoke-s20260730` | `530faa2` | FAILED | 0 | dense-output wiring failure，不进入结论 |
| `RUN-20260907-180450-smoke-s20260730` | `0d3257f` | COMPLETE | 32 | smoke only |
| `RUN-20260907-180525-full-s20260730` | `0d3257f` | FAILED | 1,203 | GT-XYZ 检查口径失败，不进入科学结论 |
| `RUN-20260907-182019-smoke-s20260730` | `69e0e8a` | COMPLETE | 32 | 修复后 smoke |
| `RUN-20260907-182144-full-s20260730` | `69e0e8a` | COMPLETE | 1,445 | canonical diagnostic |

前两类工程问题分别是 dense output 未返回 `coor_x`，以及对非严格正交 LM-O 标注直接使用
矩阵投影。最终实现恢复 EXP004 的 OpenCV Rodrigues 检查口径；canonical full run 的 GT-XYZ
最大重投影误差为 0.421903 px，低于原 0.5 px 阈值。修复流水不作为机制证据。

## Observed results

### 聚合指标

| α | Patch ADD | Patch BOP | RANSAC ADD | RANSAC BOP | EPro ADD | EPro BOP |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.5066 | 0.6904 | 0.5398 | 0.6909 | 0.5433 | 0.7009 |
| 0.25 | 0.5003 | 0.6899 | 0.6166 | 0.7176 | 0.6221 | 0.7298 |
| 0.50 | 0.4990 | 0.6897 | 0.7398 | 0.7831 | 0.7550 | 0.7899 |
| 0.75 | 0.4962 | 0.6898 | 0.8519 | 0.8539 | 0.8595 | 0.8647 |
| 1.00 | 0.4997 | 0.6837 | 0.9924 | 0.9924 | 0.9924 | 0.9924 |

reS/teS：Patch alpha0→1 为 `0.5306→0.5276 / 0.8023→0.8044`；RANSAC 为
`0.5444→0.9924 / 0.8076→0.9924`；EPro 为 `0.5661→0.9924 / 0.8136→0.9924`。

### EPro 逐物体 ADD(-S)

| α | ape | can | cat | driller | duck | eggbox | glue | holepuncher |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.514 | 0.834 | 0.538 | 0.890 | 0.206 | 0.372 | 0.707 | 0.280 |
| 0.25 | 0.697 | 0.900 | 0.632 | 0.955 | 0.439 | 0.106 | 0.821 | 0.430 |
| 0.50 | 0.840 | 0.940 | 0.784 | 0.960 | 0.761 | 0.133 | 0.893 | 0.725 |
| 0.75 | 0.954 | 0.975 | 0.883 | 0.970 | 0.917 | 0.222 | 0.957 | 0.985 |
| 1.00 | 0.994 | 1.000 | 0.994 | 0.980 | 1.000 | 0.978 | 0.993 | 1.000 |

### 原始 gate / evaluator 输出

- Gate A：alpha=1 EPro ADD/BOP `0.9924/0.9924`，PASS。
- Gate B：Spearman ADD/BOP 均约 1.0；ADD/BOP recovery `0.992355/0.966941`，PASS。
- reproduction gate：六点中 5 点超过绝对差 0.001，FAIL。
- evaluator 因 reproduction gate 失败将总 decision 覆盖为
  `PROTOCOL_REPRODUCTION_FAILED_STOP`；该程序输出保持原样。

复现差值（本轮−历史）：Patch alpha0 ADD `+0.00415`、BOP `+0.00019`；RANSAC alpha0
ADD `+0.00138`、BOP `−0.00160`；RANSAC alpha1 ADD/BOP 均 `−0.00138`。

## Derived / interpretation

- EPro alpha0→1 的 ADD/BOP 增量为 `0.449135/0.291474`；matched RANSAC 为
  `0.452595/0.301439`，recovery 为 `0.992355/0.966941`。
- 单次固定协议和五点 alpha 曲线不证明跨 run/seed 稳定性。
- 工程修复原因不能解释 full run 与历史复现值的偏差；原 evaluator 的复现失败和 Gate A/B
  通过是不同 Observed，不能互相覆盖。

## Decisions

原协议处理保持不变：evaluator decision 仍是 `PROTOCOL_REPRODUCTION_FAILED_STOP`，未因此触发
训练或调参。

2026-09-08 用户 review 另行判定“机制通过”：EPro 随 XYZ 改善由 ADD/BOP
`0.5433/0.7009` 上升到 `0.9924/0.9924`，而 Patch 对同一干预无正响应。用户认为六点复现
偏差相当于少量 target 波动，不否定机制。后续 gate 改用与指标尺度匹配的相对容差
（通常 ±3%–±5%；近饱和指标仍用绝对阈值）。该 review 不改写原 evaluator 输出。

## Evidence map

canonical full run 的紧凑原始文件：

- [run metadata](evidence/RUN-20260907-182144-full-s20260730/run_metadata.json)
- [summary](evidence/RUN-20260907-182144-full-s20260730/summary.json)
- [reproduction report](evidence/RUN-20260907-182144-full-s20260730/reproduction_report.json)
- [gate report](evidence/RUN-20260907-182144-full-s20260730/gate_report.json)

完整 `poses.jsonl`、BOP outputs/eval、smoke 与失败 run metadata 保留在本地外置 run 目录。
solver-in-the-loop review 的有效决策与证据边界已在上文完整摘要；旧独立说明页已退出 HEAD。

## Known gaps / boundaries

- 没有跨 seed 或重复 full run。
- 未完成历史复现偏差的独立归因；用户 review 是研究决策，不是新观察。
- 本实验只证明 solver 对受控 correspondence 改善的响应，不给出新模型 formal 精度。
