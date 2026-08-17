# EXP-20260817-011 CPM XYZ–Region 一致性诊断记录

## 当前状态

- 状态：`COMPLETE / MISMATCH_IMPORTANT`。
- 工程状态：smoke、audit80 和 full 2×5 均已完成；失败/无效 run 与最终通过 run
  分别保留。
- 科学结论：在本 checkpoint、support 与评估协议内，`MISMATCH_IMPORTANT` gate
  通过。该结论不等价于 GT Region 单独提高绝对性能。
- 本实验只读 EXP009 固定 Epoch 40 checkpoint，不重新训练，也不改网络、loss 或训练配置。

## 固定资产

- 父实验：`EXP-20260809-009-cpm-head`。
- checkpoint：外部 `E:\6D姿态估计\EXP-009\model_epoch_040.pth`；本地 WSL
  运行路径预期为 `/mnt/e/6D姿态估计/EXP-009/model_epoch_040.pth`。
- checkpoint SHA-256：
  `d447569bf7a1034bb57f38c90ef25bbaac8f1bb7ef3b9d74ef9db75eb32f040d`。
- config：`configs/gdrn/lmo_pbr/research/exp009_cpm_head/eval.py`。
- 数据与协议：LM-O BOP test、GT bbox、固定 seed `20260817`、FP32。

## 最小因果比较

2×2 endpoint：

1. Pred XYZ + Pred Region；
2. GT XYZ + Pred Region；
3. Pred XYZ + GT Region；
4. GT XYZ + GT Region。

XYZ、Region 之外的 effective input、visible support、ROI2D、bbox、checkpoint 和
评估协议保持不变。full 扩展为 Pred/GT Region 下各自的 XYZ alpha
`0/0.25/0.5/0.75/1`，endpoint 与 2×2 完全对应。

GT Region 不另造标签：直接调用训练 data loader 所用
`core.utils.data_utils.xyz_to_region`，输入 metric GT XYZ 与 LM-O reference 的
`fps64_and_center[:-1]`。标签 `1..64` 转为 CPM foreground one-hot posterior；只在
冻结 support 内替换，support 外保留 predicted soft posterior。因此 Region 主效应
同时包含正确性与 hard/soft 熵差异，科学判断优先依据 XYZ×Region interaction，
不能把 Region 主效应单独解释为标签正确性的纯效应。

## 预注册执行顺序

1. 单元测试和静态检查；
2. 8-target CPU FP32 `cpm_xyz_region_2x2` smoke；
3. 80-target CUDA FP32 `cpm_xyz_region_2x2` audit；
4. 1,445-target CUDA FP32 `cpm_xyz_region_alpha_sweep` full，并运行 BOP19。

每次使用新的空 output 目录。runner 会校验 checkpoint 哈希、拒绝 optimizer、使用
`eval()`/`no_grad()`、检查 model state 前后哈希、检查 baseline head re-entry 与 Region
posterior 归一化，并在运行结束后重新计算 checkpoint 文件哈希。不会覆盖
checkpoint，也不保存实例级 activation。

## 预期产物

- `protocol.json`、`quality_control.json`、`architecture.json`、`hashes.sha256`；
- `add_s_summary.json`：micro、macro-object 和逐物体 ADD(-S)；
- `bop_score_summary.json`：BOP19 AR、组件指标和 pose CSV 哈希；
- `region_consistency_summary.json`：aggregate Pred/GT Region agreement、GT-label
  posterior probability、entropy 与归一化误差；
- `factorial_summary.json`：两种主指标的 endpoint effects、interaction、rescue ratio、
  逐物体 interaction 和预注册 decision；
- 原有 aggregate-only pose/layer/component/object/visibility/support summaries。

## 结果

### RUN-20260817-014944-smoke-s20260817-a01

- mode：8-target CPU FP32 `cpm_xyz_region_2x2`。
- output：
  `output/EXP-20260817-011-cpm-xyz-region-consistency-diagnostic/smoke-2x2`。
- run state：`FAILED`；这是 QC 失败，不是 CPM 或 XYZ–Region 假设的科学失败。
- 8 个 target 和 4 个 condition 均完整；empty support `0`，意外非有限值 `0`。
- model state 与 checkpoint 文件哈希前后不变；未创建 optimizer。
- Region posterior 最大归一化误差 `7.22e-7`，低于 `1e-5` gate。
- CPU baseline re-entry 最大误差：raw rotation `3.30e-7`、raw translation
  `9.54e-6`、rotation `8.37e-6`、translation `9.54e-6`。后三者中的 translation
  与 final pose 超过预注册 CPU `1e-6` 容差，因此 QC 未通过。
- aggregate Region 描述量：1,493 support points，Pred/GT Region argmax agreement
  `0.3141`，Pred posterior 对 GT Region 的平均概率 `0.2020`。由于整个 run QC
  失败，这些数值只作为失败 run 的审计证据，不进入机制结论。
- 四条件 smoke ADD(-S) 均为 `0/8`，小样本且 QC 失败，不作科学解释。
- `factorial_summary.json` decision 为 `INCONCLUSIVE_QC_FAIL`。

关键哈希：

```text
protocol.json        dc0f1250545670bff7c8eafea37466485fe370ace8bba2c261dfe7c1231df813
quality_control.json 2e6610512e63d3a272c9f67a3ef27aa929efa5220205c9f809d9e44ec7e352f2
hashes.sha256        a0e59f165e9e58199575bf9e93b43ba9a52ca6b6759d0ba6b05740ae32713e83
```

按预注册顺序，QC 未通过后没有继续 audit80 或 full。当前不能事后把容差修改写成
本 run 的通过标准；若要调整 CPU re-entry gate，需先形成独立、明确的协议决定并
使用新的 run_id 和空 output 目录重新 smoke。

## 协议修订：CPU FP32 re-entry 容差

用户随后明确授权忽略不影响实验结果的微小数值重入误差。原 a01 失败状态和产物
保持不变；从新 run 开始，CPU 与现有已校准 CUDA gate 统一使用：raw head
`5e-5`、rotation matrix `3e-4`、translation `5e-5 m`。model state 哈希、
checkpoint 文件哈希、样本完整性、非有限值和 Region posterior 归一化 gate 均不
放宽。该修订只影响基础设施 QC，不改变四个干预条件、support、指标或科学判断
标准。下一次 smoke 使用新 run ID 和空 output 目录。

### RUN-20260817-015943-smoke-s20260817-a02

- 使用修订后的数值容差和独立空目录 `smoke-2x2-a02` 重跑同一 8-target 2×2
  smoke；退出码 `0`，`run_state.json` 为 `COMPLETE`，QC PASS。
- 8 targets × 4 conditions 完整，empty support `0`，非有限值 `0`。
- model state 与 checkpoint 文件均保持不变，Region posterior 最大归一化误差
  `7.22e-7`。
- 实际重入误差与 a01 完全一致，均低于修订后的 raw `5e-5`、rotation `3e-4`、
  translation `5e-5` gate。
- smoke 四条件 ADD(-S) 仍均为 `0/8`；该小样本只验证执行链，不用于机制判断。
- audit80 与 full 尚未启动。

关键哈希：

```text
protocol.json        dc0f1250545670bff7c8eafea37466485fe370ace8bba2c261dfe7c1231df813
quality_control.json fcdc9121fa28f55fe3dce25ec93ad853e1d256eddc09e16b6df09baabf02be02
hashes.sha256        0d0fdaee1c987594207363fd68e09a25829e7ca0fe7db6c7bff2edd11a63f488
```

a02 的 `quality_control.json` 直接记录修订后容差；runner 随后也把容差加入
`protocol.json`，供后续新 run 绑定。a02 已完成且不回写其原始产物。

## 协议修订：re-entry 改为 advisory

首次 CUDA audit80 的 80×4 条件完整完成，但最大 raw translation re-entry
`1.75e-4`、final translation re-entry `2.58e-4 m` 超过既有数值容差；其余 QC
全部正常。用户再次明确要求忽略这类不影响实验结果的微小误差。因此从后续新
run 起，re-entry 数值继续完整记录并给出 `reentry_within_tolerance`，但不再阻断
run；model state/checkpoint 文件哈希、样本与条件计数、非有限值、baseline full
复现和 Region posterior 归一化仍是 blocking gate。失败的 audit a01 原样保留，
不回写为通过。

### RUN-20260817-021155-audit80-s20260817-a02

- 使用独立空目录 `audit80-2x2-a02` 重跑 80-target CUDA FP32 2×2 audit；
  80 targets × 4 conditions 完整，退出码 `0`，QC PASS。
- baseline ADD(-S) macro-object 为 `0.35`；Pred/GT Region aggregate argmax
  agreement 为 `0.5697`。这些 audit 小样本值只用于执行链验证，不进入最终机制结论。
- model state 与 checkpoint 文件哈希前后不变；re-entry 误差按修订协议完整记录为
  advisory，其余 blocking gate 均通过。
- output：
  `output/EXP-20260817-011-cpm-xyz-region-consistency-diagnostic/audit80-2x2-a02`；
  `quality_control.json` SHA-256：
  `7181bf81519f3a3169df6ab94614da16127be79c8befc28bd74831c214d0e7b2`。

## full a01 校准字段审计

首次 full 的 1,445×10 推理和全部 BOP19 评估完整完成，但最终 QC 发现 runner
把 EXP009 历史 `0.3806228374` 当作 macro-object 校准值。代码核对确认训练
`EVAL_SUMMARY` 直接读取 BOP ADD score 的 `recall` 字段，即 target-micro recall；
本次 baseline target-micro 也精确为 `0.3806228374`，而由八个 object recall
等权计算的 macro-object 为 `0.3767506470`。因此 full a01 的唯一 blocking failure
是校准字段选择错误，不是模型、数据或 BOP 失败。a01 原样保留；后续新 run 使用
micro-to-micro baseline 校准，科学 factorial ADD 仍使用真正的 macro-object。

## 最终 full：RUN-20260817-023144-full-s20260817-a02

- mode：CUDA FP32，1,445 targets × 10 conditions，完整 BOP19。
- output：
  `output/EXP-20260817-011-cpm-xyz-region-consistency-diagnostic/full-2x5-a02`。
- run state：`COMPLETE`，QC PASS。
- 10 个 condition 均为 1,445 targets；7 个 empty-support target 对应预期 70 个
  非有限 scalar，没有意外非有限值。
- baseline BOP AR `0.5984036909`，与 EXP009 E40 `0.5983921569` 在预注册
  `5e-5` 容差内一致；baseline ADD(-S) target-micro 精确复现
  `0.3806228374`。
- baseline 真正的 8-object macro ADD(-S) 为 `0.3767506470`。
- model state 与 checkpoint 文件哈希前后不变；re-entry 按授权作为 advisory
  记录，不阻断结果。
- Pred/GT Region aggregate argmax agreement `0.5720798026`，Pred posterior 对
  GT Region 的平均概率 `0.5649181154`；这些是描述性一致性指标。

### 2×2 endpoint 因果结果

对于越高越好的指标：

| 指标 | GT-XYZ effect / Pred Region | GT-XYZ effect / GT Region | interaction | rescue ratio |
|---|---:|---:|---:|---:|
| BOP19 AR | -0.2894763552 | -0.1381130334 | +0.1513633218 | 0.5228866505 |
| ADD(-S) macro-object | -0.2796477506 | -0.1024982420 | +0.1771495085 | 0.6334737474 |

8/8 objects 的 ADD(-S) interaction 均为正，满足预注册的至少 5/8 gate。两项主
指标均复现 `GT XYZ + Pred Region` 的负效应，且 interaction 为正、rescue ratio
均至少 0.5，因此最终 decision 为 `MISMATCH_IMPORTANT`。

边界必须保留：

- GT Region 在 Pred XYZ 下会降低绝对性能：BOP AR 从 `0.5984036909` 降至
  `0.4924936563`，macro ADD(-S) 从 `0.3767506470` 降至 `0.2588341719`。
- GT Region 的证据是它显著减弱 GT-XYZ 替换造成的额外恶化，并在 GT-XYZ
  endpoint 下相对 Pred Region 提高结果；它不是可直接部署的性能改进。
- GT Region 为 hard one-hot，而 Pred Region 为 soft posterior，Region 主效应混合
  了正确性与熵/锐度差异。因此当前结果支持“XYZ–Region 不一致是重要污染因素”，
  但不能把全部 interaction 唯一归因于标签正确性，也不证明它是 CPM 欠佳的唯一
  根因。

完整 alpha sweep BOP AR：

| Region | α=0 | α=.25 | α=.50 | α=.75 | α=1 |
|---|---:|---:|---:|---:|---:|
| Pred | 0.598404 | 0.559163 | 0.466157 | 0.371013 | 0.308927 |
| GT | 0.492494 | 0.493430 | 0.454685 | 0.398907 | 0.354381 |

关键哈希：

```text
protocol.json          ba61764a3d56735782927f38e21310d4efc6861da3c62ba9ebcbdf7da24cf402
quality_control.json   843d0061c1379d4198d1660edd899bc564e8b5dae89096e4ca7faeb79aa06c83
factorial_summary.json 7e85f802600ad46f14532f36586a79da6396760dad5e1c4aba91e0089d58d811
bop_score_summary.json 3709d451fe7c00e436a0569bd2066c4395714ad8f5d93ae4571f8f833c89ee78
add_s_summary.json      fb36086377a0a90b137e09e602566bce9e1c67cf0d2b90e87745b23294524ada
hashes.sha256           df0ea67fe222446928b067a798d6d0b9f76837b057d2dd50ee822afa5c873abf
```

full a01 与 a02 的 `bop_score_summary.json`、`add_s_summary.json` 逐字节一致；
a01 只因错误的 macro/micro 校准字段被保留为 QC failure，科学输出已由 a02 独立
复现并通过 QC。
