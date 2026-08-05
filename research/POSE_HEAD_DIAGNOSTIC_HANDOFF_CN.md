# Patch-PnP 姿态头全量统计排查交接

最后更新：2026-08-04

状态：

```text
OFFICIAL_FULL_COMPLETE — C1_EPOCH_40_TRANSFER_VERIFIED
```

本文件记录客观排查协议、运行状态和最终结果。研究推断与候选结构在全量排查
完成后另存到 `/mnt/e/6D姿态估计的研究`，不提前把假设写成结论。

## 开始阅读

工作区：

```text
/home/wsluser/GDRNPP-RGBD
```

开始工作前先运行 `git status`，保留全部未提交内容。当前 C1 正式训练必须完成
预注册的 40 轮，本诊断不得停止、替换或干扰该训练。

诊断代码：

```text
research/pose_head_diagnostic/
```

原始运行产物：

```text
output/EXP-20260804-007-pose-head-information-flow/
```

## 研究问题

全量统计排查回答：

1. XYZ、ROI 2D 和 Region 分别对冻结姿态头产生多大影响；
2. 信息在输入、卷积、Flatten、fc1、fc2 或最终输出的哪个位置衰减；
3. 更准确的 XYZ 为什么不能稳定转化为更准确的姿态；
4. 缺陷更符合输入失衡、空间对应丢失、输出分支共享还是训练目标捷径；
5. official、C1、B 和 C2 的信息消费方式是否发生可解释的改变。

本诊断不训练模型，不把传统 PnP 或固定步优化作为最终方法。

## 冻结协议

```text
Experiment: EXP-20260804-007-pose-head-information-flow
Dataset:    LM-O BOP19
Targets:    1,445
BBox:       GT
Seed:       20260804
Modes:      smoke / audit80 / full
Precision:  deterministic FP32 on CPU/GPU
Updates:    0
Optimizer:  none
```

官方 checkpoint SHA-256：

```text
bafa869d4e6c00410517ecb1add59f234ed1642e47fabcf3aa6e0e8a1b498a8c
```

C1、B 和 C2 必须使用各自固定 Epoch 40 checkpoint，并在运行结果中记录实际
SHA-256。不得使用 LM-O 测试结果选择中间 checkpoint。

## 干预条件

固定支持为：

```text
预测可见支持 ∩ GT 可见区域 ∩ 有效深度
```

十四种条件：

```text
baseline
GT-X / GT-Y / GT-Z / GT-XY / GT-XYZ
XYZ / ROI 2D / Region 支持内空间置换
Region 空间均值化
边界 GT-XYZ / 等量内部 GT-XYZ
高误差 25% GT-XYZ / 等量随机 GT-XYZ
```

支持外保持原值。空间置换使用逐目标确定性排列。边界/内部和高误差/随机对照
修改点数必须一致。

## 统计输出

程序运行时可以在内存暂存逐目标标量，但不写出逐实例激活或逐实例层响应表。

保存的汇总层级：

```text
总体
8 个物体类别
4 个可见率区间
对称/非对称
支持点数量四分位
```

每项汇总包含样本数、有限值数、均值、标准差、中位数、5/25/75/95 分位数、
95% bootstrap 置信区间以及改善/恶化比例。

正式输出文件：

```text
protocol.json
architecture.json
overall_condition_summary.csv
layer_response_summary.csv
pose_component_summary.csv
per_object_summary.csv
visibility_summary.csv
symmetry_summary.csv
support_quartile_summary.csv
condition_summary.json
bop_score_summary.json
quality_control.json
hashes.sha256
run_state.json
```

BOP 工具所需的逐目标姿态 CSV 只保存在 Git 忽略的 `output/` 中，不进入本
交接文档，也不作为逐实例研究材料。

## 执行状态

| 阶段 | 状态 | 结果 |
|---|---|---|
| 聚合工具单元测试 | PASS | 9 tests passed |
| 8 目标 smoke | PASS | 8/8 targets；14/14 conditions；QC PASS |
| 80 目标 audit | PASS | 80/80 targets；14/14 conditions；QC PASS |
| official 全量 | PASS | 1,445/1,445；14组BOP19；QC PASS |
| C1 Epoch 40 全量 | 权重已转移并通过SHA-256核验，等待诊断full | 固定Epoch 40指标已核验 |
| B Epoch 40 全量 | 等待 B 完成 | — |
| C2 Epoch 40 全量 | 等待 C2 完成 | — |

执行顺序不可跳过：

```text
tests
  -> official smoke
  -> official audit80
  -> official full
  -> 当前完成
  -> C1/B/C2 各自固定 Epoch 40 权重 full
  -> 横向统计分析
```

### official smoke 运行记录

运行环境为本机 WSL、`conda pytorch22`、CPU，固定种子 `20260804`。第一次运行
在完成推理后的聚合阶段暴露指标字段契约错误：聚合器请求
`rotation_error_gt_deg` / `translation_error_gt_mm`，而统一
`pose_metrics()` 接口返回 `rotation_error_deg` / `translation_error_mm`。
修复字段名并增加模式契约回归测试后，8 项测试和同协议重跑均通过。第一次失败
未写出任何正式产物，成功重跑沿用原输出目录。

质量控制：

```text
processed targets:                 8 / 8
conditions:                        14 / 14, each 8 records
empty support targets:             0
non-finite scalar count:           0
max baseline rotation re-entry:    2.682209014892578e-07
max baseline translation re-entry: 2.384185791015625e-07
re-entry tolerance:                1e-06
model state unchanged:             true
optimizer created:                 false
instance-level features persisted: false
run state:                         COMPLETE
```

产物：

```text
output/EXP-20260804-007-pose-head-information-flow/official/smoke/
hashes.sha256 SHA-256:
7b12e92c2b84d66d3f85601f315dd911e78799f0c216879632311ac844895f6b
```

`hashes.sha256` 记录的 13 个文件均已独立复算通过，没有逐实例激活或逐实例层
响应文件。smoke 只验证工具链和质量门，不形成机制结论。

### CUDA、精度与 full 验收记录

本机环境必须先执行 `conda activate pytorch22`。受限执行环境内看不到 GPU，
但在获批的设备访问下确认：

```text
GPU:              NVIDIA GeForce RTX 4060 Laptop GPU
VRAM:             8,188 MiB
PyTorch:          2.2.0
CUDA build:       12.1
CUDA tensor test: PASS
```

第一次 CUDA audit 使用共享工具默认 AMP，baseline raw 重入误差达到约
`1e-3`，未通过严格门。探针证明差值等于 FP16 量化步长。正式协议因此冻结为
确定性 FP32，与 C1/B/C2 评估协议一致。CUDA FP32 的实测数值底噪和正式门为：

```text
raw Patch-PnP tolerance:       5e-05
final rotation tolerance:      3e-04
final translation tolerance:   5e-05
```

full 首次完成全部推理和 BOP 后，旧质量门错误地混用了训练日志的8物体macro
ADD、诊断逐目标micro ADD以及精确BOP相等要求。原失败QC保存在：

```text
quality_control_precalibration.json
run_state_precalibration.json
```

重新验收在写入前复核了原始 `hashes.sha256`、14个姿态CSV和14个BOP分数。
最终质量控制：

```text
processed targets:                  1,445 / 1,445
conditions:                         14 / 14, each 1,445 records
empty support targets:              7
expected non-finite scalar count:   98 = 7 * 14
unexpected non-finite scalar count: 0
max raw rotation re-entry:          4.443526268005371e-05
max raw translation re-entry:       9.059906005859375e-06
max final rotation re-entry:        2.3484230041503906e-04
max final translation re-entry:     1.049041748046875e-05
model state unchanged:              true
optimizer created:                  false
instance-level features persisted:  false
run state:                          COMPLETE
```

最终产物共510 MiB，`hashes.sha256`记录的3,571个文件全部独立复算通过：

```text
output/EXP-20260804-007-pose-head-information-flow/official/full/
hashes.sha256 SHA-256:
fdabeb5b16f514a86e061204b50550dffbdfacb250c3bd51fc1109995ee4f969
```

## 总体统计结果

正数改善量表示误差减小，负数表示恶化。ADD为诊断内部1,445目标micro
recall，不是训练日志中8物体macro平均。

| 条件 | BOP AR (%) | ΔBOP (pp) | ADD(-S) (%) | ΔADD (pp) | 旋转改善 (°) | 平移改善 (mm) |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 69.0399 | — | 50.5190 | — | — | — |
| GT-X | 69.0099 | -0.0300 | 50.5882 | +0.0692 | -0.1090 | +0.4454 |
| GT-Y | 69.0293 | -0.0106 | 50.4498 | -0.0692 | +0.1714 | -0.0983 |
| GT-Z | 68.7121 | -0.3278 | 49.2042 | -1.3149 | -0.0228 | -0.3140 |
| GT-XY | 68.5931 | -0.4468 | 50.9343 | +0.4152 | -0.5031 | +0.4454 |
| GT-XYZ | 68.3709 | -0.6690 | 49.6194 | -0.8997 | -0.5359 | +0.0895 |
| 打乱XYZ | 66.2992 | -2.7407 | 51.4187 | +0.8997 | -1.6396 | -0.4598 |
| 打乱ROI 2D | 69.0201 | -0.0198 | 50.5882 | +0.0692 | -0.0159 | +0.0354 |
| 打乱Region | 58.5790 | -10.4609 | 28.2353 | -22.2837 | -1.8420 | -10.6356 |
| Region均值化 | 53.5885 | -15.4514 | 21.9377 | -28.5813 | -2.4239 | -16.4456 |
| GT-XYZ边界 | 68.8602 | -0.1797 | 49.7578 | -0.7612 | +0.0867 | -0.2063 |
| GT-XYZ等量内部 | 69.1091 | +0.0692 | 49.8270 | -0.6920 | +0.1129 | +0.0502 |
| GT-XYZ高误差25% | 69.0913 | +0.0514 | 49.8270 | -0.6920 | +0.1834 | +0.1119 |
| GT-XYZ等量随机 | 68.9707 | -0.0692 | 50.0346 | -0.4844 | +0.1405 | -0.0869 |

## 8 个物体类别统计

只记录聚合ADD(-S)百分比：

| 物体 | baseline | GT-XYZ | 打乱ROI 2D | 打乱Region | Region均值化 |
|---|---:|---:|---:|---:|---:|
| ape | 49.14 | 48.57 | 49.14 | 7.43 | 5.14 |
| can | 81.91 | 76.88 | 81.91 | 24.62 | 14.57 |
| cat | 47.37 | 41.52 | 46.78 | 12.87 | 11.70 |
| driller | 82.50 | 82.50 | 83.00 | 41.50 | 21.00 |
| duck | 8.89 | 7.78 | 10.00 | 21.67 | 8.89 |
| eggbox | 38.89 | 41.67 | 38.33 | 40.56 | 47.22 |
| glue | 75.00 | 76.43 | 75.71 | 44.29 | 48.57 |
| holepuncher | 22.00 | 23.50 | 21.50 | 33.50 | 24.00 |

## 可见率与支持规模统计

下表为相对baseline的ADD变化（pp）。Region破坏的总体伤害主要集中在中高
可见率和中等支持规模，但物体间存在异质性。

| 分组 | baseline ADD (%) | GT-XYZ | 打乱Region | 高误差25% | 等量随机 |
|---|---:|---:|---:|---:|---:|
| 可见率 <0.1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 可见率 0.1–0.3 | 9.43 | 0.00 | +1.89 | 0.00 | 0.00 |
| 可见率 0.3–0.5 | 34.06 | -2.17 | -5.07 | -1.45 | -1.45 |
| 可见率 >=0.5 | 54.63 | -0.81 | -25.46 | -0.64 | -0.40 |
| 支持Q1 | 46.96 | +0.28 | -12.15 | 0.00 | +0.55 |
| 支持Q2 | 71.43 | -1.92 | -41.76 | -1.37 | -0.82 |
| 支持Q3 | 56.15 | -3.07 | -37.99 | -2.23 | -0.56 |
| 支持Q4 | 27.42 | +1.11 | +2.77 | +0.83 | -1.11 |

## 逐层响应统计

表内是平均 relative-L2：

| 条件 | 输入 | Conv1 | Conv3 | fc1 | fc2 | raw旋转 | raw平移 | 最终姿态 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GT-XYZ | .01135 | .05004 | .04877 | .01417 | .01277 | .08959 | .00243 | .04435 |
| 打乱XYZ | .02610 | .10916 | .12247 | .02767 | .02379 | .11271 | .01041 | .04260 |
| 打乱ROI 2D | .02717 | .03356 | .01830 | .00155 | .00106 | .00342 | .00058 | .00213 |
| 打乱Region | .52645 | .65035 | .30001 | .04760 | .03954 | .09154 | .02663 | .05740 |
| Region均值化 | .37226 | .61808 | .34573 | .06137 | .05325 | .12162 | .03593 | .06416 |
| GT-XYZ边界 | .00764 | .02742 | .02033 | .00530 | .00431 | .02630 | .00100 | .01075 |
| GT-XYZ等量内部 | .00474 | .02209 | .02148 | .00508 | .00409 | .02591 | .00101 | .00971 |
| GT-XYZ高误差 | .00923 | .03564 | .02840 | .00602 | .00493 | .03508 | .00136 | .01350 |
| GT-XYZ等量随机 | .00551 | .01624 | .01302 | .00336 | .00271 | .01845 | .00066 | .00729 |

## official / C1 / B / C2 对比

待各固定 Epoch 40 权重可用后填写。

## 客观结论

official全量可以确认：

- official Patch-PnP并非完全忽略XYZ：GT-XYZ使最终姿态产生明显响应；但完整
  GT-XYZ同时降低BOP `0.6690 pp`和ADD `0.8997 pp`，说明更准确XYZ没有被稳定
  转化为更准确姿态；
- Region是姿态头的主导输入。打乱或均值化Region分别降低BOP `10.4609 pp`
  和`15.4514 pp`，影响远大于XYZ和ROI 2D；
- ROI 2D空间置换仅改变BOP `-0.0198 pp`，逐层响应在fc1/fc2后几乎消失，
  表明official姿态头对其空间对应利用很弱；
- XYZ和Region变化在卷积阶段被放大，进入fc1/fc2后明显压缩。raw旋转对XYZ
  的相对响应远高于raw平移，信息消费具有明显分支不平衡；
- 高误差区域相对随机区域产生更强逐层响应和略高BOP，但ADD仍下降，不能作为
  已验证改进方法；
- official机制结论已经完成，C1/B/C2是否改变这种信息消费模式仍待对应固定
  Epoch 40权重的同协议full。

```text
OFFICIAL_MECHANISM = REGION_DOMINANT_XYZ_UNSTABLE
CROSS_MODEL_COMPARISON = PENDING
```

## 复现命令

正式阶段先运行：

```bash
PYTHONPATH=/home/wsluser/GDRNPP-RGBD \
PYTHONPYCACHEPREFIX=/tmp/gdrnpp-pycache \
conda run -n pytorch22 python -m pytest -q \
  -o cache_dir=/tmp/gdrnpp-pytest-cache \
  research/pose_head_diagnostic/tests
```

然后执行 official smoke：

```bash
PYTHONPATH=/home/wsluser/GDRNPP-RGBD \
PYTHONPYCACHEPREFIX=/tmp/gdrnpp-pycache \
conda run -n pytorch22 python \
  -m research.pose_head_diagnostic.run_statistical_diagnostic \
  --mode smoke \
  --model-role official \
  --config-file configs/gdrn/lmo_pbr/convnext_stage3c1_official_gt_lmo.py \
  --weights pretrained_models/lmo_pbr/model_final_wo_optim.pth \
  --device cpu \
  --seed 20260804 \
  --output-dir output/EXP-20260804-007-pose-head-information-flow/official/smoke
```

CUDA运行必须先执行 `conda activate pytorch22`，并使用获批的GPU设备访问。
其他模型角色必须等待对应固定 Epoch 40 checkpoint，并记录实际 SHA-256。

## 下一步动作

1. 使用已核验的C1固定Epoch 40权重运行同协议full；
2. 分别完成GPU0/B和GPU1/C2的容器、数据访问及smoke运行门；
3. 两道运行门通过后，在GPU0/GPU1启动B和C2正式训练；
4. 对B、C2固定Epoch 40权重运行同协议full；
5. 完成模型角色横向比较；
6. 将事实、推断、替代解释和候选修改写入：

```text
/mnt/e/6D姿态估计的研究/
Patch-PnP姿态头全量统计排查与改进备选_2026-08-04.md
```

该外部分析文件在全量排查结束前不得提前形成最终结论。
