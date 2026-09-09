# EXP018 — Geometry-Consistency Residual Pose Head

## 研究问题与协议

在 EXP013A initial camera-frame pose 后，用 predicted metric XYZ–ROI2D 在当前 pose
下的显式 2D reprojection residual 驱动一次 Region-free、support-masked、轻量
6DoF correction，能否比 EXP013A 获得稳定的整体姿态收益？实现细节见
[实现说明](../../exp018/README.md)。

- experiment_id：`EXP-20260906-018-geometry-consistency-residual`。
- formal run_id：`RUN-20260906-115421-formal-s42-a01`。
- source commit：`3c9f5596d93bda54d84c4f19107e1b098291d221`；配置：
  `configs/gdrn/lmo_pbr/research/exp018/gcr_pose/train.py`。
- seed 42；LM-PBR；40 epoch；batch 48；Ranger lr `8e-4`、wd `0.01`、warmup 200；
  GT-box LM-O BOP19 evaluation，无 PnP/depth refine。
- official checkpoint 提供冻结 producer；EXP013A pose head 随机初始化；训练
  `pnp_net.*` 与新增的 13,831 参数 `pose_corrector.*`，不从 A E40 继续训练。
- 固定比较基准为 EXP013A E40：BOP AR `0.683956`、ADD(-S) `0.510727`、
  AR_reS `0.498039`、AR_teS `0.797693`。
- 运行镜像：`gdrnpp-research:torch220-cu121-sm89-c0be1ade7ea9`；训练日志未见
  traceback/OOM，最大显存记录约 5,250 MiB。

## 正式结果（Observed）

checkpoint 均为 `model_epoch_NNN.pth`。BOP AR、AR_reS、AR_teS 来自各 epoch 的
`scores_bop19_*epoch.json`；ADD(-S)0.1d 与逐物体 recall 来自同一运行日志中的
`EVAL_SUMMARY`。原始汇总与控制台日志位于用户保留目录
`E:\6D姿态估计\EXP018`。

| Epoch | BOP AR | ADD(-S)0.1d | AR_reS | AR_teS |
|---:|---:|---:|---:|---:|
| 5 | 0.618136 | 0.413149 | 0.365398 | 0.747174 |
| 10 | 0.640457 | 0.448443 | 0.433449 | 0.764475 |
| 15 | 0.656325 | 0.474048 | 0.441984 | 0.781546 |
| 20 | 0.652895 | 0.491349 | 0.402307 | 0.790542 |
| 25 | 0.670985 | 0.477509 | 0.471280 | 0.788235 |
| 30 | 0.677506 | 0.494118 | 0.492503 | 0.794464 |
| 35 | 0.677769 | 0.483045 | 0.498962 | 0.789850 |
| **40** | **0.686302** | **0.516263** | **0.517647** | **0.800461** |

E40 是全部四项聚合指标的最佳预定评估点。E40 ADD(-S)0.1d 逐物体 recall：ape
`0.520000`、can `0.793970`、cat `0.438596`、driller `0.825000`、duck
`0.072222`、eggbox `0.388889`、glue `0.750000`、holepuncher `0.345000`。

## 派生比较与简要分析（Derived / Interpretation）

E40 相对固定 EXP013A E40 的变化为：

| 指标 | EXP013A E40 | EXP018 E40 | 差值 | 相对变化 |
|---|---:|---:|---:|---:|
| BOP AR | 0.683956 | 0.686302 | +0.002346 | +0.34% |
| ADD(-S)0.1d | 0.510727 | 0.516263 | +0.005536 | +1.08% |
| AR_reS | 0.498039 | 0.517647 | +0.019608 | +3.94% |
| AR_teS | 0.797693 | 0.800461 | +0.002768 | +0.35% |

计算口径为 `EXP018 E40 - EXP013A E40`，相对变化再除以 EXP013A E40。四项最终
指标同方向改善，rotation 指标增幅最大，说明一次 pose-conditioned correction
可能提供了有限的旋转修正能力。曲线存在明显波动：例如 BOP 在 E15→E20、ADD 在
E20→E25 和 E30→E35 回落，直到 E40 才同时达到最佳，因此证据只支持单个 seed 的
最终小幅增益，不证明跨 run 稳定性或明确的 correspondence 利用机制。

设计阶段只明确建议 BOP 至少 `+0.003`；实际为 `+0.002346`，差 `0.000654`。
ADD/reS/teS 的非劣数值阈值与逐物体判据当时没有完整预注册，因此不事后把本结果
包装成严格 gate pass/fail。结合 40 epoch 成本、额外 correction 分支和较小的整体
effect size，当前证据不足以支持继续扩展这条路线。

## 工程证据与边界

- 本地真实数据 smoke `RUN-20260906-183345-local-smoke-s42-11264` 为 PASS：2 个
  optimizer steps、13,831 correction 参数、base init/checkpoint round-trip 精确等价、
  backbone/geometry 权重不变、真实 test mapper forward 成功。它只证明执行链。
- 实现阶段新增 21 项测试，连同回归共 `120 passed`；实际模型 CPU preflight 通过。
- formal 只运行 seed 42，未做 matched control 重跑、多 seed 或机制干预；不能从
  聚合指标推断每个对象、遮挡强度或跨域上的稳定收益。

## Decision（2026-09-09）

状态：`COMPLETE / MARGINAL_GAIN / CLOSED`。

保留 EXP018 实现与正式结果作为一次轻量 post-decode geometry correction 的历史证据；
不追加 smoke、formal、多 seed、诊断或结构扩展，不作为当前开发主线。后续资源转入
EXP020 correspondence supervision + matched classical PnP/RANSAC。
