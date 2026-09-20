# EXP-20260920-025-hierarchical-cad-attention

## 问题、协议与当前决策

2026-09-20：检验不使用 GT-parent feature routing 的层级 CAD attention，能否通过
统一 T3=512 分布及 T1/T2 marginal 监督学习 dense correspondence。
不是 EXP022 的严格单变量对照：token 交互、层数与路由设计均改变。
设计以 [README](../../exp025/README.md) 和用户最后修订为准，交接包中的独立
c1/c2/c3 head 不采用。历史 EXP022 模型、loss 与 artifact 未改；共享 context
只加向后兼容的可选 hierarchy_path 参数。

配置：`configs/gdrn/lmo_pbr/research/exp025_hierarchical_cad_attention/train.py`。
seed42；LM-O PBR/GT-box；正式候选40epoch、effective batch48（4×12）、AMP；
官方初始化冻结为默认，另支持解冻及 ImageNet。T1/T2/T3 NLL 各1，残差与mask各1。
正式预定 E5/E10/E15/E20/E25/E30/E35/E40 指标均**未生成**，best 关闭。
本轮所有运行均为 diagnostic，不能用于正式精度、跨 seed 或服务器吞吐结论。

源码基点 `dbc1535925e092bfbec47a8b32e05f78c4687cc2`（第二轮工作树另基于
`f2f735f`）；第二轮改动已提交。环境 Conda `pytorch22`、torch2.2、RTX4060 Laptop。
本机产物统一在 `output/diagnostics/<run_id>/`；固定 batch
`.local/exp025/fixed_batch4_a01.pt` 来自 frozen_a01 的在线 CPP 渲染。
hierarchy `.local/dataset_cache/exp022/consistent_v3.npz`；不复制数据或权重入 Git。

Decision：**NUMERICAL_REVIEW 已完成 / FORMAL_NOT_STARTED**。第二轮按修复指导包
`CAD_GDRNPP_EXP025_FIX_GUIDE_f2f735f` 定位并修复了 optimizer resume 边界，把
non-finite gradient 定位为 AMP 缩放后的 fp16 梯度上溢，并完成 residual 条件化对照；
模型、loss、层级与协议未改，`FORMAL_READY=False` 不变。是否据此修改训练策略
（放宽 AMP guard 或降低初始 scale）与正式 residual 头（T3 conditioning 与/或零初始化）
仍待用户确认。

## 预注册检查的作用

- Integrity：概率和、nested 标签、FP32 target、加载/冻结/optimizer 覆盖、真实
  batch loss/梯度有限、checkpoint 往返；失败不宣称该执行链通过。
- Mechanism：同一 batch/初始参数的 residual-only 与 full 两臂，各200步，
  每20步记录 NLL/accuracy、GT-path 和 predicted-path XYZ mm；观察响应关系，
  不设事后下降阈值，不把单 batch 视为泛化证明。
- Advisory：固定 batch 显存/耗时，不含重复加载/渲染；不是正式吞吐 gate。
- 服务器 EGL 与正式 E5–E40 未执行，本机 CPP 不替代它们。

## Observed：工程验证

CPU preflight：`exp025_cpu_a01`（official frozen）、
`exp025_cpu_unfrozen_a01`（official trainable）、
`exp025_cpu_imagenet_a01`（ImageNet trainable）均 PASS。
官方340个 backbone tensors 精确加载，ImageNet340张量核对；模型95,683,588参数，
backbone87,564,416、head8,119,172。head lr3e-4，解冻 backbone lr3e-5。
ImageNet 本地来源 `/home/wsluser/.cache/torch/hub/checkpoints/convnext_base_1k_224_ema.pth`。

| run_id | 模式 | 观察 |
|---|---|---|
| exp025_amp_frozen_a01 | official frozen、在线CPP、batch4×8step | PASS，0跳步；冻结张量不变；checkpoint往返一致 |
| exp025_amp_unfrozen_a01 | official trainable、相同保存batch4×8step | PASS，0跳步；主干和分类器更新；checkpoint往返一致 |
| exp025_amp_imagenet_a01 | ImageNet trainable、相同batch4、计划8step | FAIL：Non-finite gradient；旧报告未保存失败步号和部分history，不猜测具体步号 |
| exp025_main_smoke_a01 | main_gdrn、独立8-image split、CPP/AMP | 退出0；展开55 ROI，13iter；model_epoch_001.pth，iteration12 |
| exp025_main_resume_a01 | 新目录指向前一checkpoint、--resume、TOTAL_EPOCHS=2 | 退出0；iteration25，model_epoch_002.pth；optimizer完整恢复异常，不能标全链PASS |

前两项 smoke 在补齐整个 symmetry target 的 autocast-disable 保护前执行；
nested traversal 和 marginal 此前已是 FP32。保护后单测、ImageNet检查、fixed-batch
诊断与 main smoke 使用新路径。前两项保留为当时工程观察，不伪称最终版本全模式通过。
smoke checkpoint 均为 `smoke_checkpoint.pth`（iteration7，无正式epoch意义）。
冻结/解冻固定 batch 整步中位数120.600/244.884ms，峰值allocated1.440/2.982GB，
reserved1.705/3.093GB，计时排除前2步、无逐步loader/renderer。

测试：`pytest -q research` 为 **338 passed / 3 failed**。三项均为既有
`research/exp022/tests/test_lm_protocol.py` 向只读 LM models 目录写cache失败，
未借此扩大修改历史loader。EXP025+launcher子集最终 **74 passed**；
涵盖极端logits概率归一、T3 CE等价、三层梯度、nested NumPy对照、marginal标签、
argmax反例、对称分支、空support、AMP target、冻结范围及mock服务器资源门。
服务器 Docker/GPU/EGL 未实际操作。

续跑checkpoint审计：原/续跑的optimizer step均为13，scheduler last_epoch为13/26，
GradScaler growth_tracker为13/26、scale均65536，196个model张量发生变化。
Observed：权重继续更新，但optimizer步数没有累计到26；不能宣称完整optimizer恢复。
当前 Lightning Lite wrapper 的 `state_dict()` 委托内部optimizer，而继承的
`load_state_dict()` 没有同样委托，是需要下一轮核查的共享框架边界；本轮不擅自
修改会影响历史实验的公共训练框架。后续正式训练前必须处理并重测。

原始紧凑报告副本位于本目录 `evidence/<run_id>.json`，逐字来自上述4个 AMP/fixed
运行目录的 `report.json`，保留原 FAIL 与缺失字段；checkpoint与完整日志仍外置。

## Observed：固定 batch 可学习性

`run_id=exp025_fixed200_a01`，official frozen，同一真实batch4、seed42、
两臂分别重新初始化，constant lr3e-4，无正式warmup。两臂step0指标相同。
mask有效点3829；误差口径为各实例有效像素均值再跨实例平均，所选对称分支与各臂loss一致。
概率分层 accuracy 使用 marginal argmax；predicted-path XYZ 始终用 T3 argmax。

| 指标 | step0（共同） | residual-only step200 | full step160（最后完整记录） |
|---|---:|---:|---:|
| residual SmoothL1 | 0.418296 | 0.540440 | 0.126886 |
| 三层原始 NLL 和 | 12.634731 | 13.651350（权重0） | 0.079639 |
| T3 accuracy | 0.000795 | 0.000591 | 0.999592 |
| GT-path XYZ mm | 13.154466 | 15.869346 | 4.904890 |
| predicted-path XYZ mm | 82.075043 | 71.363358 | 4.904985 |

residual-only200步完成，checkpoint `residual_only_diagnostic.pth`，iteration199；
full 在160步之后、下一次180步报告前遇到 **Non-finite gradient** 并退出1。
没有 full 最终checkpoint、200步指标或完整两臂PASS；旧runner未记录失败精确步号。
原始报告保留 FAIL，不改写成成功。后续runner补记current_step、AMP scale与首批异常参数，
并在失败时保留已完成history，属于可观测性补强，不改变网络、loss或检查强度。

## Derived / Interpretation：问题定位边界

full 的 route NLL 与XYZ下降支持这个固定batch上的分类信号可学习，但不支持数值稳定性
或泛化通过。residual-only 的残差未改善；不能只据 full 改善宣称独立残差模块通过。
读取 residual-only step200 checkpoint 的只读前向审计：raw residual 范围
[-12.578125, 8.0546875]，全部ROI/像素/坐标中 `abs(raw)>9` 比例0.0623983，
FP32 `abs(tanh(raw))==1` 比例0.0622559。这证明部分饱和，但还未证明它是平台唯一原因。

ImageNet同初始权重与batch的只读backward对照：AMP scale65536、AMP scale1024、
FP32 scale1均无非有限参数梯度；loss分别13.526480、13.526480、13.527170。
故不能把8-step失败直接归因于初始scale，也未证明降低scale可修复后续异常。

## Observed：optimizer resume 边界修复（2026-09-20）

指导包 `CAD_GDRNPP_EXP025_FIX_GUIDE_f2f735f` 的 P0-A。先在本机确认边界，未按伪代码猜属性：
`pytorch_lightning 1.6.4` 的 `_LiteOptimizer.__init__` 把 `__dict__` 复制自被包装
optimizer 并把 `__class__` 换成 `type("Lite"+cls, (_LiteOptimizer, cls), {})`，因此
本机实例类型为 `LiteAdamW`、`isinstance(..., torch.optim.Optimizer)` 为 **True**；
`state_dict()` 委托 `self._strategy.optimizer_state(self.optimizer)`（保存正确），
而 `load_state_dict` 沿 MRO 落到 `torch.optim.Optimizer.load_state_dict`，只更新
wrapper 自己的 `__dict__`，真正 step 的底层 optimizer 不被恢复。

实验（`/tmp/probe_lite.py`，本机 CPU）：3 次 step 后 wrapper 与 inner 的 step 同为 3；
对 wrapper 调 `load_state_dict` 后 wrapper step 恢复为 3、**inner state 为空**，
且 `wrapper.param_groups is not inner.param_groups`；参数对象身份保持一致。

修复（训练仍用 wrapper，checkpoint/调度器绑定底层 optimizer）：

- `core/utils/my_checkpoint.py`：新增 `unwrap_optimizer_for_checkpoint`（普通 optimizer
  原样返回；Lite 类 wrapper 返回经校验的底层 optimizer；未知或候选不唯一时 fail-closed；
  校验 wrapper 与 inner 的 `param_groups` Parameter 身份一致）与
  `resync_wrapped_optimizer`（load 之后把 wrapper 的 `state`/`param_groups` 重新指回
  inner 的对象，使 LR 记录与调度器写入不再脱钩）。`MyCheckpointer.__init__` 对
  `optimizer` checkpointable 也做同一 unwrap。
- `core/gdrn_modeling/engine/engine.py`：`state_optimizer` 用于建 scheduler、注册
  checkpointer、`consolidate_state_dict` 与 lr 日志；`resume_or_load` 之后调用
  `resync_wrapped_optimizer`。

确定性 contract（`research/tests/test_lite_optimizer_resume.py`，CPU，
LightningLite + MyCheckpointer + 真实 `flat_and_anneal` 调度器）：连续 26 次 update 与
13+save+resume+13 的 model/optimizer/scheduler `state_dict()` 逐张量相等，AdamW step 均为
26，`exp_avg/exp_avg_sq` 恢复。同一 fixture 下按修复前路径（wrapper 注册、scheduler 建在
wrapper 上）重放：inner optimizer step 停在 **13**，训练实际使用的 LR 3e-07 与记录的
3e-06 不一致（另一变体停在 resume 前的 2.2575e-04），权重与连续训练最大差
**1.42e-3**。该测试还固定了 engine 的接线顺序，防止回退。

历史 `exp025_main_resume_a01` 审计（optimizer step 13、scheduler 26、growth_tracker 26）
与上述边界一致：full-load 路径下 scheduler/scaler 恢复了，只有训练 optimizer 没有。
本修复影响所有经 `engine.do_train` 的训练，不只 EXP025。真实 `main_gdrn` resume 复验见下节。

真实入口复验（`exp025_main_smoke_a03`，在线 CPP renderer、AMP、`REFERENCE_BS=48` →
accumulation 12；该 smoke split 展开 55 ROI、13 iteration/epoch，所以每 epoch 在
iteration 11 与 12 各更新一次）：

| 运行 | checkpoint | iteration | optimizer step | scheduler last_epoch | growth_tracker |
|---|---|---:|---:|---:|---:|
| 第一段 TOTAL_EPOCHS=1 | `model_epoch_001.pth` | 12 | **2** | 2 | 2 |
| 第二段 `--resume` TOTAL_EPOCHS=2 | `model_epoch_002.pth` | 25 | **4** | 4 | 4 |

续跑在上一次的基础上继续累计（2→4），与 scheduler（2→4）一致；修复前同一流程下
optimizer step 会重新从本段计数（历史 run 正是 13 对 26）。

## Observed：non-finite gradient 定位（2026-09-20）

指导包 P0-B。所有 step 改经 `runtime.amp_step`，失败抛出 `NonFiniteTrainingError`，
携带 step、AMP scale、全部 loss 分量、token/logit/raw residual 统计与逐模块梯度范数
与元素最大值；每个 arm 每 20 步写 `model/optimizer/scaler/step/RNG` 的 last-good
checkpoint。`numerical_replay` 从 last-good 状态重放 A 当前 AMP / B 低 scale AMP / C FP32。

| 运行 | 结果 |
|---|---|
| `exp025_amp_imagenet_a02` ImageNet trainable 8 step | FAIL step **3**，scale 65536，唯一非有限参数 `cad_attention_head.mask_predictor.weight`；同 step 各 loss 有限（t1 1.880/t2 3.740/t3 6.062/res 0.542/mask 0.648），token norm 均值 23.0，raw residual abs max 6.26、无饱和 |
| `exp025_amp_frozen_a02` official frozen 8 step（在线 CPP） | PASS，8/8 无跳步；峰值 allocated 1.465 GB；step 中位数 606.0 ms |
| `exp025_fixed200_a02` residual-only 200 步 | COMPLETE，与 a01 逐点一致（见下节） |
| `exp025_fixed200_a02` full 200 步 | FAIL step **167**，scale **65536**（此前从未 drop），非有限参数 `decoder.3.block.0.weight`、`decoder.3.block.3.weight`、`residual_predictor.weight`；同 step 各 loss 有限（t3 1.002/res 0.148/mask 0.044），raw residual abs max 0.844、无饱和 |

`numerical_replay`：`exp025_replay_a01` 从 full 臂 step160 last-good 重放 40 步。

| 臂 | scale | 结果 | 最后有限步总 loss |
|---|---:|---|---:|
| A current AMP | 65536 | NON_FINITE，首个坏步 **167**（复现原失败），坏参数含 `decoder.2.block.3.weight` | 0.556632 @166 |
| B low-scale AMP | 1024 | FINITE 到 200 | 0.347895 @200 |
| C FP32 | — | FINITE 到 200 | 0.348405 @200 |

scale sweep（同一起点、同一 batch、40 步、单臂）：65536 与 32768 均在 step 167 非有限；
**16384 / 8192 / 4096 / 1024 全部 FINITE 到 200**，末步总 loss 0.349006 / 0.347070 /
0.347895（16384、4096、1024）；FP32 0.348405。该窗口内未逐元素记录的对比见
`exp025_replay_scale32768`：step 166 的最大 **未缩放**梯度元素为 `residual_predictor`
0.754、`decoder` 0.708，其他模块 ≤0.122；同时该批固定轨迹在 step 163–166 出现总 loss
从 0.210 升到 0.558 的瞬态，低 scale 下随后回落到约 0.35。

Derived：失败不是 loss/激活发散——失败步的 loss、logits、token norm、raw residual 全部有限；
它是 **AMP 缩放后的 fp16 梯度上溢**：`GradScaler` 以 `scale` 放大 loss 后反传，任何真实梯度
元素超过 `65504/scale` 就会在 fp16 中变成 ±inf（本窗内 `scale=32768` 失败、
`scale=16384` 通过，故该瞬态的梯度元素量级在 **(2.0, 4.0]**，与 step166 观测到的 0.75
同阶、方向一致）。当前训练循环把任何非有限梯度视为致命错误（`amp_step` 与原代码的
`NonFiniteTrainingError`/`RuntimeError`），因此 GradScaler 本应"减半 scale 并跳过该步"
的正常恢复路径被拦截。**未**加 gradient clipping、未关 AMP、未全局降 LR、未改
`SET_NAN_GRAD_TO_ZERO`，也未改 loss/结构/初始化；是否放宽该 guard 属训练策略决定，
留给用户确认。

## Observed：固定 batch 可学习性（a02 复现）与 residual 条件化诊断（2026-09-20）

`exp025_fixed200_a02` 用同一固定 batch、seed42、官方冻结主干复跑两臂，残差臂与 a01 逐点
一致（step0 residual `0.418296`、GT-path XYZ `13.154466` mm、predicted-path
`82.075` mm），可复现 a01 记录。该臂每 20 步读数（residual SmoothL1 / GT-path XYZ mm /
raw residual abs max / `abs(tanh(raw))==1` 比例）：

| step | 0 | 20 | 60 | 100 | 140 | 200 |
|---|---:|---:|---:|---:|---:|---:|
| residual | 0.418296 | 0.540440 | 0.540440 | 0.540440 | 0.540440 | 0.540440 |
| GT-path XYZ mm | 13.1545 | 15.8693 | 15.8693 | 15.8693 | 15.8693 | 15.8693 |
| raw abs max | 1.894 | 12.148 | 12.945 | 12.805 | 12.359 | 12.195 |
| tanh 饱和比例 | 0 | 0.0948 | 0.0731 | 0.0637 | 0.0567 | 0.0535 |

同一批次的 `oracle_error_mm`（bounded residual 表示上限误差）= **0.0003 mm**，即
clip 到单位球几乎不损失表达能力；而 raw 幅度在 20 步内升到 ±12 并保持 5–9% 的
`tanh` 精确饱和，随后 residual loss 在 9 位小数上不再变化（180 步内只在两个相邻
fp32 值间抖动），GT-path XYZ 也停在 15.86934–35 mm。full 臂同批次的 0/20/…/160 读数为
residual `0.418296→0.493256→…→0.115741`、T3 边际 accuracy `0.000795→0.999393`、
GT-path XYZ `13.1545→4.5741` mm，raw abs max 全程 ≤4.3 且无饱和。

`residual_probe`（诊断专用，GT-T3 conditioning 不进入正式模型/推理）：

| 臂 | run | step0 residual / GT-path mm | step200 residual / GT-path mm | raw abs max @200 | 饱和 @200 |
|---|---|---|---:|---:|---:|
| baseline（正式 residual 路径，随机初始化） | `exp025_probe200_a01`/`a03` | 0.418296 / 13.1545 | 0.540440 / 15.8693 | 12.20 | 0.0535 |
| zero-init（同路径，最后一层零初始化） | `exp025_probe200_a03` | 0.147614 / 5.6881 | 0.125804 / 4.9398 | 0.60 | 0.0000 |
| GT-T3-conditioned | `exp025_probe200_a02` | 0.219496 / 7.9437 | **0.056706 / 2.6669** | 0.89 | 0.0000 |

| step | 0 | 20 | 60 | 120 | 200 |
|---|---:|---:|---:|---:|---:|
| zero-init residual | 0.147614 | 0.144009 | 0.139954 | 0.135792 | 0.125804 |
| zero-init GT-path mm | 5.6881 | 5.6154 | 5.4929 | 5.3360 | 4.9398 |
| conditioned residual | 0.219496 | — | — | 0.092793 | 0.056706 |

评分规则对照（`exp025_probe200_a04`，同一 batch/seed/200 步，仅改 conditioned 臂的选支方式）：
把候选 residual 的"上报值"改成 head 规则下的 `min over targets`，再按该值选支，则
conditioned 臂**不再下降**（residual 0.219496→0.245296、GT-path 7.9437→8.6508 mm），
zero-init 臂与 baseline 臂不受影响（0.147614→0.129779 / 0.540440）。原因是这个
`min over targets` 目标会把预测拉向"当前最近的那个分支的目标"，而不再拟合被条件化的
那个 T3 节点的目标，与"给定 GT T3 节点能否回归该节点 residual"的问题不一致。最终采用
上一段的规则（每个候选按自己的条件分支目标训练与选支），a04 作为该选择的反面对照保留。
两臂的**度量**始终由同一个 `measure_prediction`（head 规则）给出，所以 baseline 与
conditioned 的比较口径一致。

Derived（诊断口径，单 batch、seed42、200 步、常数 lr3e-4、route weight 0、含 mask loss）：

- baseline 臂的随机初始化在 step0 就比"直接预测 T3 anchor"（zero-init 的 step0，
  0.1476）更差，前 20 步更把 raw residual 推到 ±12 并产生 5–9% 的 `tanh` 精确饱和，
  此后 loss 在 9 位小数上不动；即随机初始化的回归头落进饱和区，梯度约等于 0。
- 同一网络、同一 loss、同一数据下，最后一层零初始化后 residual 单调下降且完全无饱和，
  说明 residual regression 本身可学，平台不是"不可学"。
- 显式的 GT-T3 conditioning 收益更大（200 步 0.0567 / 2.67 mm），说明正式残差路径缺的是
  T3 身份的显式输入。
- 三臂训练参数集不完全相同（conditioned 臂以 probe 的 `Linear(2D,D)+GELU+Linear(D,3)`
  替换 `residual_predictor`；zero-init 臂只改初始化），属条件对照而非严格单变量消融。
  GT T3 仅用于该诊断的输入，未进入 loss、正式 forward 或推理。
- **是否据此修改正式 residual 头（加入预测 T3 的 soft conditioning 与/或零初始化）尚未决定**，
  本轮未改任何正式模型、loss 或配置。

## Observed：formal-path accumulation smoke（2026-09-20）

`exp025_accum_a03`，官方冻结、AMP、同一保存 batch 重复输入，`REFERENCE_BS=48` →
`accumulate_iter=12`（physical 4），25 iteration/epoch × 5 epoch、15 次 optimizer update。
结果为 **PASS**：

| 项 | 值 |
|---|---|
| accumulation 边界 | iteration `[11, 23, 24, 36, 48, 49, 61, 73, 74, 86, 98, 99, 111, 123, 124]`（每 epoch 3 次，含最后一个 1-micro-step 的尾部窗口） |
| updates/epoch | `[3, 3, 3, 3, 3]` |
| 连续训练终点 | optimizer step `[15]`，scheduler `last_epoch 15`，lr `3.0e-06` |
| 50 iteration 处存档 | optimizer step `[6]`，scheduler `last_epoch 6` |
| resume 起点读回 | iteration 49，optimizer step `[6]`，scheduler `last_epoch 6`，lr 与存档一致 |
| resume 终点 | optimizer step `[15]`（不是 9），scheduler `last_epoch 15` |
| LR 轨迹 | 与连续训练逐 update 相等 |
| AMP | 未发生 silent skip / scale 下降 |

**噪声底**：同一段 50 iteration 用两个互不 resume 的 session 各跑一次，权重最大差
**1.56e-3**（最差张量 `geometry.global_readers.2.attention.in_proj_bias`）；resume 与连续
训练的最大差 **2.17e-3**（最差张量 `cross_attention.2.attention.in_proj_bias`），
两者同阶。也就是说本机 CUDA+AMP 路径本身不能逐位复现：两次完全相同的运行在数次 update
后即可差到 1e-3 量级，且差值集中在 attention 的 `in_proj_bias`。因此该 smoke 的硬判据
是**无噪声的整数量**（optimizer step、scheduler epoch、accumulation 边界、LR 序列、
AMP 无跳步），张量比较只作为"不超过实测噪声底的 3 倍"的辅助判据。
噪声底的具体来源（SDPA backward 的 atomic 归约在 fp16 大 token norm 下放大）本轮**未**
隔离验证，只能确认它不是 resume 引入的；也不改变 fixed-batch 读数的可复现性
（a01 与 a02 的逐点均值一致到 6–9 位小数）。

## 证据文件

`evidence/` 下：`exp025_amp_imagenet_a02.json`（ImageNet AMP 失败 + 逐步 telemetry）、
`exp025_amp_frozen_a02/a03.json`（其中 a03 的计时已排除 last-good 写入：8/8 无跳步、
整步中位数 149.9 ms、峰值 allocated/reserved 1.465/1.726 GB）、
`exp025_fixed200_a02.json`（instrumented 两臂）、
`exp025_replay_a01.json`（AMP/低scale/FP32 matched 重放）、
`exp025_amp_scale_sweep_a01.json`（derived：scale 扫描汇总）、
`exp025_probe200_a01/a02/a03/a04.json`（residual 三臂与评分规则对照；a01 的 conditioned
臂因诊断脚本自身 bug 中止，`status=FAIL`，其 baseline 臂有效且与 a03/a04 逐点一致）、
`exp025_accum_a03.json`（formal-path smoke）、`exp025_main_resume_a03.json`
（derived：真实入口 resume 的 checkpoint 审计）。checkpoint 与完整日志仍外置。

## 下一步（待用户确认）

1. **训练策略**：是否允许 GradScaler 的正常恢复（非有限梯度时降 scale 并跳过该步），
   或把初始 scale 调低到 16384 及以下。两者都会改变现有"非有限即致命"的 guard 语义，
   未经确认不实施。
2. **正式 residual 头**：诊断支持两条独立改动——按预测 T3 分布做 soft conditioning
   （`T3 probability -> soft expected CAD token -> concat(image token, soft CAD token)`，
   不用 GT teacher forcing），以及最后一层零/小初始化。是否修改、以及是否作为单变量
   对照分臂，由用户决定；本轮未动正式路径。
3. **噪声底**：如需跨 run 逐张量比较（例如严格单变量消融），需要先隔离本机
   CUDA+AMP 不可复现的来源，或固定 `torch.use_deterministic_algorithms` 后重测。
4. 正式训练、服务器 EGL、E5–E40 与完整 BOP/matched-PnP 评价均未执行；
   `FORMAL_READY` 保持 `False`。
