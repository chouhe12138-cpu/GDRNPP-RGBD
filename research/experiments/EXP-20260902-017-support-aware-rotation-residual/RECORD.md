# EXP017 — Support-aware Rotation-only Spatial Residual Adapter

## 协议与状态

- 状态：`COMPLETE / SCREEN_FAIL_3_OF_5_GATES`；2026-09-06 核对正式结果。
- 问题：完整保留 EXP013A 时，只给 raw rotation 增加 Region-free、position-aware、
  support-masked spatial residual，能否提高 rotation 而不损害 translation/ADD？
- 唯一变量：EXP013A 8×8 geometry grid 上的 rotation-only adapter。
- experiment ID：`EXP-20260902-017-support-aware-rotation-residual`
- 配置：
  `configs/gdrn/lmo_pbr/research/exp017/support_aware_rotation_residual/train.py`
- 基础源码：`5d284946defad56d11c28b40d7a903ae632aa5a5`；EXP017 实现 commit：
  `6815761`。canonical formal source：`6b4d412ed2518086efafe2c24190dea60b2baf25`。
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

### 服务器 smoke

- `RUN-20260902-130841-smoke-s42-a01`：`INFRA_FAILED`。source `3cfbceb94252`，
  image `gdrnpp-research:torch220-cu121-sm89-c0be1ade7ea9`。
- runtime/config gate 通过后，在模型构建前的 online renderer 初始化失败：容器可从
  image `PYTHONPATH` 导入 `bop_renderer`，但 launcher 未向只读 release 容器注入
  `BOP_RENDERER_PATH=/opt/bop_renderer/build`；repo mount 又遮蔽了 image 内
  `/workspace/gdrnpp/bop_renderer` symlink。
- 该 run 没有构建模型、没有 optimizer step，不进入科学结论。修复限定为 launcher
  显式环境变量和对应 runtime gate，不修改 EXP017、训练协议或镜像。

### 运行框架复核

2026-09-02 复核发现 EXP017 继承的 EXP013A 配置只冻结 geometry head，却没有显式将
`GEO_HEAD.TRAIN_SUPERVISION` 设为 `False`，因此默认值 `True` 仍触发训练阶段 CPP
renderer 和无梯度用途的 geometry loss。BOP evaluation 使用独立的
`VAL.RENDERER_TYPE`，不应随训练 renderer 一起关闭。

修复后共享 pose-head screening 配置为：训练 supervision `False`、训练 renderer
`None`、BOP evaluation renderer `cpp`。完整相关测试 `96 passed`，CPU preflight
`PASS`，去除训练 renderer 后的真实 LM-PBR CPU smoke `PASS`；本轮 WSL CUDA 不可见，
没有重复本地 GPU smoke。

formal `RUN-20260902-133725-formal-s42-a01` 从修复前 source `33f285b` 启动。该 run 的
pose-head 梯度不因冻结 geometry loss 改变，但不具备新的完整 metadata、clean-tree/
protocol gate 和 checkpoint-before-evaluation 保证，因此标记为
`SUPERSEDED_STOPPED`，不作为修复后的 canonical formal；旧输出保留。

修复后服务器 smoke `RUN-20260902-152755-smoke-s42-a01` 使用 source `6b4d412`、
`smoke.py`、LM-PBR local train、batch 4、1 epoch（2048 iterations），`exit_code=0`；
生成 `run_metadata.json`、meta/train/checkpoints 结构并正常结束。该 run 证明新的
clean-tree/config/renderer/mount gate 与训练链路可用，不作为性能结论。用户已明确授权
从同一 clean release 启动 canonical formal。

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

## Canonical formal 完整结果

- run：`RUN-20260902-154756-formal-s42-a01`；source `6b4d412ed251`；seed 42。
- checkpoint：`model_epoch_040.pth`，epoch 40 / iteration 255919；日志确认保存、
  全部训练完成及 `FINAL_EVAL_REUSED periodic_epoch=40`。
- 外置证据：`E:\6D姿态估计\EXP-014\` 的 console.log、八个 scores_bop19_*epoch.json
  与 E40 checkpoint。目录名不是实验 ID；其中旧 F 诊断文件不属于本次 formal。
- ADD 来自日志 `EVAL_SUMMARY.add_s_0.1d`，采用 1,445 targets 的 micro 口径；
  BOP/reS/teS 来自对应 epoch JSON，BOP 已与日志交叉核对。不用 AR_ad 替代 ADD。
- 对共享 geometry 优化耦合的代码与 E10 autograd 复核见
  `research/exp017/EXP017B_DECISION.md`。

| Epoch | BOP AR | ADD(-S) target-micro | AR_reS | AR_teS |
|---:|---:|---:|---:|---:|
| 5 | 0.639732 | 0.501730 | 0.351557 | 0.781084 |
| 10 | 0.642787 | 0.491349 | 0.419608 | 0.770473 |
| 15 | 0.659130 | 0.488581 | 0.450750 | 0.780854 |
| 20 | 0.649087 | 0.469204 | 0.433449 | 0.763091 |
| 25 | 0.658579 | 0.486505 | 0.457209 | 0.778777 |
| 30 | 0.669520 | 0.514879 | 0.457439 | 0.795617 |
| 35 | 0.671518 | 0.492042 | 0.491580 | 0.787774 |
| 40 | 0.681709 | 0.512111 | 0.494348 | 0.800461 |

E40 是 BOP/reS/teS 最高点，E30 是 ADD 最高点；仅描述轨迹，不替换 E40 决策点。
保留最终/gate 点与 ADD 最佳点的逐物体 ADD：

| Epoch | ape | can | cat | driller | duck | eggbox | glue | holepuncher |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 0.514286 | 0.829146 | 0.502924 | 0.830000 | 0.194444 | 0.272222 | 0.750000 | 0.240000 |
| 40 | 0.474286 | 0.809045 | 0.526316 | 0.805000 | 0.077778 | 0.350000 | 0.757143 | 0.310000 |

### E40 gate 判定

| 指标 | E40 | 门槛 | 判定 |
|---|---:|---:|---|
| AR_reS | 0.494348 | ≥0.513039 | FAIL |
| BOP AR | 0.681709 | ≥0.686956 | FAIL |
| ADD(-S) target-micro | 0.512111 | ≥0.505727 | PASS |
| AR_teS | 0.800461 | ≥0.794693 | PASS |
| 相对 A 逐物体 ADD 非负 | 5/8 | ≥5/8 | PASS |

相对 A，cat、driller、duck、glue、holepuncher 的 ADD 改善，ape、can、eggbox 下降。
五项通过三项；rotation 门槛未过，不能标为 ROTATION_SUPPORTED。

## 当前结论与边界

canonical formal 已完成，结论为 SCREEN_FAIL；没有获得预期的 rotation/BOP 收益。
EXP017-B 已完成独立 matched formal，见其 RECORD；不追加 seed 或自动启动新训练。
按用户要求，2026-09-05 新增的本地 E40 adapter 开关诊断脚本及产物于 2026-09-06
删除；本记录只采用服务器正式训练/评估结果，不将已删除诊断的本地指标混入正式结论。
