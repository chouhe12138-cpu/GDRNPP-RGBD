# EXP017 — Support-aware Rotation-only Spatial Residual Adapter

## 协议与状态

- 状态：`IMPLEMENTED / LOCAL_PREFLIGHT_PASS / LOCAL_REAL_SMOKE_PASS / FORMAL_NOT_RUN`
- 问题：完整保留 EXP013A 时，只给 raw rotation 增加 Region-free、position-aware、
  support-masked spatial residual，能否提高 rotation 而不损害 translation/ADD？
- 唯一变量：EXP013A 8×8 geometry grid 上的 rotation-only adapter。
- experiment ID：`EXP-20260902-017-support-aware-rotation-residual`
- 配置：
  `configs/gdrn/lmo_pbr/research/exp017/support_aware_rotation_residual/train.py`
- 基础源码：`5d284946defad56d11c28b40d7a903ae632aa5a5`；当前实现未提交，正式运行前必须
  先由用户授权提交并记录最终 source commit。
- formal 协议与 A matched：seed 42、LM-PBR train、LM-O BOP19 GT-box、batch 48、
  40 epoch、Ranger lr `8e-4`、weight decay `0.01`、warmup 200、冻结 backbone/geometry，
  E40 唯一决策点；不从 A E40 warm-start pose head。

## 固定结构

- A 主体、Region-free geometry encoder、translation、loss/solver/data 全部不变。
- 输入仅为 A 的 `G[B,32,8,8]` 与原 visible support；不读取 Region、main feature、
  depth stats。
- 32→64 token projection + learned `[1,64,64]` absolute position embedding。
- `Linear(64,32)→GELU→Linear(32,1)` learned pooling；support 只作二值有效性 mask，
  不使用 `log(S)` 或 occupancy soft bias；无效权重为 0，有效权重归一化，全零报错。
- pooled 64-D → LayerNorm → `Linear(64,64)→GELU→Linear(64,6)`。
- `raw_r = raw_r_A + alpha_r * delta_r`；`raw_t = raw_t_A`。
- `alpha_r=1.0` 可学习；delta 最后一层 weight/bias 全零，因此初始 delta、R 差精确为 0，
  T 与 A bitwise equal。
- A head `1,432,987` 参数；adapter `13,000`；EXP017 head `1,445,987`，低于新增
  `15,000` 硬门槛。

## 本地验证

### 测试

2026-09-02 在 Conda `pytorch22`：

- 最终完整相关测试：`84 passed`（research 公共、EXP012、EXP013、EXP017、
  pose_structure）。

覆盖 config/注册、参数预算、A state 迁移、初始 R/T exact、R/t 梯度隔离、首步输出层
梯度、非零训练后 projection/pooling/position 梯度、Region 不变量、translation 不变量、
support 外污染、masked softmax、partial/zero support、batch 1/多 batch、CPU、checkpoint
round-trip，以及 learned/uniform/position/token intervention。

### CPU preflight

- 状态：`PASS`
- A / EXP017 / adapter 参数：`1,432,987 / 1,445,987 / 13,000`
- official checkpoint shared-state 迁移通过；仅 `pnp_net` trainable。
- 初始 delta exact zero、raw R value-exact、raw T bitwise equal。
- translation-only adapter gradient zero；rotation output gradient 非零；一次非零更新后
  adapter internal gradients 非零。
- full-model forward/backward/optimizer step 通过；loss `1.4615242e-07`（合成输入，
  只作数值门禁）。

### 本地真实 LM-PBR smoke

- 成功 run：`RUN-20260902-195750-local-smoke-s42-a02`；RTX 4060 Laptop；batch 2；
  `lmo_pbr_stage3_local_train`；3 个受限 optimizer step，不是 epoch/formal。
- translation-only 1 step 未改变 adapter；随后 2 个 rotation-inclusive full-loss step。
- full loss：`1.850037 → 1.825658`；输出/loss/梯度有限。
- backbone/geometry 冻结且未更新；A pose head 更新 88 个 parameter tensors；adapter
  更新 14 个 parameter tensors；第二步 projection/pooling/position 真实梯度非零。
- checkpoint strict round-trip value-exact。
- `alpha_r=0.99998397`；pooling entropy `4.1588831`；valid support mass `1.0`；
  raw_r baseline RMS `0.1110750`；rotation residual RMS `0.1087917`。这些只验证链路，
  不作性能或结构选择证据。

失败 run `RUN-20260902-195620-local-smoke-s42-a01` 的训练计算正常，但 smoke 使用
PyTorch `_version` 判断 Ranger `.data` 参数更新而产生假阴性；改为逐值快照后 a02
通过。a01 不进入科学结论。

## 预注册 formal gate

冻结基准为 EXP013A E40：BOP `0.683956`、ADD `0.510727`、AR_reS `0.498039`、
AR_teS `0.797693`。EXP017 E40 必须同时满足：

| 指标 | 门槛 | 相对 A |
|---|---:|---:|
| AR_reS | ≥ `0.513039` | ≥ `+0.015000` |
| BOP AR | ≥ `0.686956` | ≥ `+0.003000` |
| ADD(-S) target-micro | ≥ `0.505727` | 下降不超过 `0.005000` |
| AR_teS | ≥ `0.794693` | 下降不超过 `0.003000` |
| 逐物体 ADD 非负 | ≥ `5/8` | 覆盖约束 |

五项全过才为 `PASS`。若仅 rotation 通过，记为
`ROTATION_SUPPORTED / SCREEN_FAIL`。边缘差距不自动增加 seed。

## 当前结论与边界

实现、单测、CPU preflight 和本地真实小批次 smoke 已证明代码链路满足请求的结构和
隔离约束，可以进入“请求 formal 训练授权”阶段；这不等于 formal 已获授权或方法有效。
在用户明确授权前，不提交、不制作服务器 release、不启动 40 epoch 训练。
