# EXP-20260920-025-hierarchical-cad-attention

## 问题、协议与当前决策

2026-09-20：检验不使用 GT-parent feature routing 的层级 CAD attention，能否通过
统一 T3=512 分布及 T1/T2 marginal 监督学习 dense correspondence。
不是 EXP022 的严格单变量对照：token 交互、层数与路由设计均改变。
设计以 [README](../../exp025/README.md) 和用户最后修订为准，交接包中的独立
c1/c2/c3 head 不采用。历史 EXP022 模型、loss 与 artifact 未改；共享 context
只加向后兼容的可选 hierarchy_path 参数。

配置：`configs/gdrn/lmo_pbr/research/exp025_hierarchical_cad_attention/train.py`。
seed42；LM-O PBR/GT-box；正式候选40epoch、真实 batch48（accumulate=1）、AMP；
官方初始化冻结为默认，另支持解冻及 ImageNet。T1/T2/T3 NLL 各1，残差与mask各1。
正式预定 E5/E10/E15/E20/E25/E30/E35/E40 指标均**未生成**，best 关闭。
本轮所有运行均为 diagnostic，不能用于正式精度、跨 seed 或服务器吞吐结论。

源码基点 `dbc1535925e092bfbec47a8b32e05f78c4687cc2`（第二轮工作树另基于
`f2f735f`）；第二轮改动已提交。环境 Conda `pytorch22`、torch2.2、RTX4060 Laptop。
本机产物统一在 `output/diagnostics/<run_id>/`；固定 batch
`.local/exp025/fixed_batch4_a01.pt` 来自 frozen_a01 的在线 CPP 渲染。
hierarchy `.local/dataset_cache/exp022/consistent_v3.npz`；不复制数据或权重入 Git。

Decision：**STRUCTURE_REFACTORED / LOCAL_SMOKE_PASS / FORMAL_NOT_STARTED**。
2026-09-20 按交接包 v2 重构 image 分支（四级 Image-SA 写回并进入下一级 transition）并把
formal 恢复为真实 batch48；旧固定 batch 结果为重构前结构的历史诊断证据（见下节）。
`FORMAL_READY=False`，服务器真实 batch48 与 EGL gate 未执行。此前一轮的
NUMERICAL_REVIEW 结论（optimizer resume 修复、AMP 上溢定位、residual 条件化）
建立在重构前结构上，其结论边界见对应小节。第二轮按修复指导包
`CAD_GDRNPP_EXP025_FIX_GUIDE_f2f735f` 定位并修复了 optimizer resume 边界，把
non-finite gradient 定位为 AMP 缩放后的 fp16 梯度上溢，并完成 residual 条件化对照；
模型、loss、层级与协议未改，`FORMAL_READY=False` 不变。是否据此修改训练策略
（放宽 AMP guard 或降低初始 scale）与正式 residual 头（T3 conditioning 与/或零初始化）
仍待用户确认。

## Observed：Image 分支结构重构与 formal batch 修正（2026-09-20，交接包 v2）

按 `CAD_EXP025_local_agent_pack_v2` 修正两处偏离。**本段之后的所有 EXP025 固定 batch
结果（Residual V1/V2、AMP scale sweep、route/T3 accuracy、residual probe、生产路径
AMP recovery、CPU/CUDA smoke 参数值）都属于重构前的 Image 分支**，保留为历史诊断证据，
不删除，也不当作新结构的验证。

**原实现偏离**：image 分支当时是 `Conv2d(1024→512)` + 三个 `CADStageTransition` 的纯
上采样链，自注意力只在整条链之后对 64×64 执行一次 window+shift，即每个 stage 的 SA 都
不影响后续 Spatial Transition 的输入（算了 SA token，transition 仍吃原 feature）。

**当前实现**：`input_adapter(1024→512)` 之后是四个 `ImageSAStage`（8×8 global、
16×16 global、32×32 window+shift、64×64 window+shift），每级把 attention 结果 residual
写回 feature，写回后的 feature 才进入下一级 `CADStageTransition`；最后一级返回
tokens `[B,4096,256]`，再依次对 T0/T1/T2/T3 做 cross-attention，之后供 t3_classifier /
Residual V2 / mask。CAD 分支、loss、target、decode、Residual V2 条件化与 detach、
mask BCE、hierarchy 与 SHA 契约均未改。末级 stage 不回写（`to_feature=None`），
因为它的 feature 没有下游消费者，保留一个无梯度的投影只会增加死参数；global stage
用 `AttentionBlock(tokens, tokens)` 调用，否则 `context=None` 会让它的 context norm
成为无梯度死参数。

| 项目 | 重构前 | 重构后 |
|---|---|---|
| image 分支 | transition 链 + 末端一次 SA(64×64) | 4 级 SA 交错 3 个 transition |
| head 参数 | 8,217,796 | **10,787,652**（preflight 实测） |
| 模型总参数 | 95,683,588 | **98,352,068**（preflight 实测） |
| formal batch | IMS_PER_BATCH=4、REFERENCE_BS=48（本机 4×12 形状） | **48/48**，accumulate=1 |

**formal batch**：原来的 formal 配置直接带着本机显存受限的 4×12 形状；现在 `train.py`
是真实 batch48（每 iteration 一次真实 optimizer update），本机形状改由 `smoke.py`
（4/48）与诊断脚本显式携带，`accumulation_smoke` 新增 `--batch-size/--reference-bs`
且不再从 formal 继承。engine 的梯度累计未删除。

**本轮验证（Observed；全部为本地 CPU/CUDA，不是正式结果）**：

- `pytest -q research/exp025/tests` **49 passed**（新增 `test_image_branch.py` 6 项）；
  `pytest -q research` **377 passed**；`research/tests/test_lite_optimizer_resume.py` 含在
  后者中。
- `python -m research.exp025.preflight` **PASS**，`hierarchy_sha_match=true`，
  head 参数 10,787,652。
- 四级 SA 的结构契约：shape/token 数（8×8→64×64、末级 4096×256）、每级调用一次、
  扰动任一级 SA 参数都会改变下一级输入与下一级 transition 的输入、CA 顺序
  T0/T1/T2/T3（1/8/64/512 tokens）、每级 SA 参数都有梯度（`test_image_branch.py`）。
- `real_smoke`（CUDA、batch4、8 步、CPP、official frozen）：初始 scale
  32768/16384/8192 均 **PASS**（无跳步、checkpoint 往返一致、90/90 个 image stage
  参数更新）；65536 在 step1 失败于 `mask_predictor.weight`。
- `amp_boundary_probe`（新工具）：同 seed42、同 batch、同 backbone 初始化的一步
  FP32 对照。重构前结构 `mask_predictor` 未缩放梯度元素最大值 0.707
  （×65536 = 46,355 < 65,504，通过）；重构后 1.156（×65536 = 75,746 > 65,504，上溢）。
  两结构的 FP32 与 AMP loss 逐项一致、其余模块梯度都有限且低约一个量级，因此这是
  **缩放后的 fp16 边界**，不是 loss/结构发散；重构后 Mask 头的初始梯度约为旧结构的
  1.6×。
- `learnability` 60 步（batch4、scale16384、LOCAL_SMOKE）：两臂 loss 有限，full 臂
  T3 NLL 6.40→0.94、T3 marginal accuracy 0.001→0.732、GT-path XYZ 5.63→5.30 mm、
  predicted-path XYZ 74.2→6.2 mm、无 tanh 饱和、AMP 无跳步。只作工程可训练性检查，
  不与旧结构数值比较（旧数值属于重构前结构，且步数/形状不同）。
- `accumulation_smoke`（4/48 → accumulate 12，短 schedule）：**PASS**，
  optimizer step 边界、scheduler epoch、LR 轨迹与连续训练逐项一致、无 AMP 跳步。
- 真实入口 `main_gdrn.py` + `smoke.py`（本机 CPP、AMP、8-image/55-ROI split）：退出码 0，
  13 iteration/epoch、2 次 optimizer update（iteration 11、12 的 accumulation 边界与
  截断窗口）、`scheduler.last_epoch=2`、checkpoint `model_epoch_001.pth`；GradScaler 在
  这 2 次 update 中保持 65536、无跳过。**对照**：单步（accumulate=1）的 `real_smoke` 在
  65536 第 1 步就上溢。两者的差别是 accumulation 的 divisor 把每个 micro-batch 的缩放
  fp16 梯度缩小 12 倍，因此 formal 的 48/48（accumulate=1）比本机 4×12 诊断更贴近 fp16
  边界——batch48 下的 AMP 行为必须由服务器实测，不能用本机 4×12 结果替代。
- 服务器真实 batch48 + EGL gate **未执行**（本机无 EGL），`FORMAL_READY` 保持 `False`，
  未启动 40 epoch 正式训练。

Derived：重构后 mask 头未缩放梯度元素最大值 1.156 与 fp16 上限的关系决定了 65536
必然上溢、32768 不会（阈值在两者之间），与旧结构在同一量级（旧结构历史 trained 状态的
scale sweep 为 16384 通过、32768 在 step167 失败）。因此没有依据按旧结果固定 formal 的
初始 scale；AMP 在真实 batch48 下的行为待服务器实测。

## 预注册检查的作用

- Integrity：概率和、nested 标签、FP32 target、加载/冻结/optimizer 覆盖、真实
  batch loss/梯度有限、checkpoint 往返；失败不宣称该执行链通过。
- Mechanism：同一 batch/初始参数的 residual-only 与 full 两臂，各200步，
  每20步记录 NLL/accuracy、GT-path 和 predicted-path XYZ mm；观察响应关系，
  不设事后下降阈值，不把单 batch 视为泛化证明。
- Advisory：固定 batch 显存/耗时，不含重复加载/渲染；不是正式吞吐 gate。
- 服务器 EGL 与正式 E5–E40 未执行，本机 CPP 不替代它们。

## Observed：工程验证

> 以下各节记录的是 **重构前** Image 分支（transition 链 + 末端一次 64×64 SA）与
> formal 4×12 形状下的历史诊断，run_id 与数值原样保留；它们不构成新结构的验证。

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
它是 **AMP 缩放后的 fp16 反向传播上溢**：`GradScaler` 以 `scale` 放大 loss 后反传，fp16
反向链路上任一量（最终 parameter gradient 或中间激活梯度）超过 fp16 上限 65504 就会变成
±inf。scale 扫描给出的是一个**量级窗口**：`scale=32768` 失败、`scale=16384` 通过，等价于
溢出量在缩放后跨过 65504、未缩放时约在 (2.0, 4.0] 量级。这里说法保持为窗口而非定值：
scale=32768 记录到的最终（未缩放）梯度元素只有 0.754（`residual_predictor`）与 0.708
（`decoder`），若溢出发生在中间量上，最终梯度不必落进该区间；本轮未逐层记录 fp16
中间量，因此不能断言"最终 parameter gradient 必然在 (2.0, 4.0]"。当前诊断循环把任何
非有限梯度视为致命错误（`runtime.amp_step` 的 `NonFiniteTrainingError`），因此
GradScaler 本应"减半 scale 并跳过该步"的正常恢复路径被它拦截——**这是诊断脚本语义，
不是 engine 语义**，见下一节的生产路径验证。**未**加 gradient clipping、未关 AMP、
未全局降 LR、未改 `SET_NAN_GRAD_TO_ZERO`，也未改 loss/结构/初始化。

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

## Observed：生产路径 AMP recovery 与 scheduler gate（2026-09-20）

上一节的"非有限即致命"是 **诊断脚本** `runtime.amp_step` 的语义；`engine.do_train` 没有这个
guard，它把 `lite.backward` 与 Lite optimizer wrapper 交给 precision plugin 的 `GradScaler`。
`amp_recovery_smoke` 因此用生产对象复跑：`LightningLite(precision=16)` + `lite.setup` 得到的
`_LiteModule`/Lite wrapper + plugin 的 `GradScaler`，从同一 `full_last_good.pth`（step160）继续。

| 运行 | 结果 |
|---|---|
| `exp025_amp_recovery_RECHECK`（新 gate，20 update） | **PASS**；`forward_precision={autocast: true, input_dtype: float16}` |
| `exp025_amp_recovery_unguarded_a01`（`--unguarded-scheduler`，修复前循环） | PASS，但 `scheduler_mismatch_steps=[166]` |

skip 事件（两臂都发生在同一处）：

| 项 | 值 |
|---|---|
| iteration | 166（自 step160 起的第 7 次 update，即历史记录的 step 167） |
| loss | 1.138556（有限） |
| scale | 65536 → **32768** |
| 内部 optimizer step | 166 → **166**（不增加） |
| 参数 delta | **0.0**（无参数更新） |
| 进程 | 继续；下一 update（167）optimizer step 167、参数 delta 4.86e-4、scheduler 6→7 |

对照：去掉 gate 后同一位置 scheduler 仍然前进（`last_epoch` 6→7），终点 optimizer step **179**
而 `scheduler.last_epoch` **20**——LR 计划白吃了一个 tick；加上 gate 后终点是 19，与真实
update 数一致。因此 `engine.do_train` 现在读 `solver_utils.amp_scale(self._precision_plugin)`
夹住 `optimizer.step()`，用 `solver_utils.gradscaler_skipped_step` 判断这次 update 是否真的
落到参数上，只有真的落到才 `scheduler.step()`。回归测试见
`research/tests/test_lite_optimizer_resume.py`（含一个用真实 Lite wrapper + 交替拒绝的 stub
plugin 驱动的行为测试：6 次调用里 3 次被拒 → scheduler 只前进 3；去掉 gate 则前进 6）。

附注：`_LiteModule.forward` 会自动进入 precision plugin 的 fp16 autocast 并把浮点输入 cast 成
fp16，确认了**生产路径确实是 fp16 autocast 推理 + 缩放反传**；诊断脚本用的是"fp32 输入 +
autocast"的近似，两者不完全相同（这解释了为何本 smoke 的失败位置与 `learnability` 相差 0
步，但 loss 数值不完全一致）。

## Observed：Residual V2 实现与 fixed-batch 复验（2026-09-20）

正式残差路径按诊断结论改为：T3 logits → FP32 softmax → detach 概率 → 对投影到 64 维的 T3
CAD token 求期望（`[B,P,512]×[B,512,64]`）→ 与 image token 拼接 → `Linear(320,256)+GELU+
Linear(256,3)`（**最后一层权重与 bias 零初始化**）→ bounded residual。不使用 GT T3、不做
top-k、loss 定义（SmoothL1 beta .1、权重 1）与 route target 不变、train/infer 同一 forward。

固定 batch 复验（同 batch、seed42、官方冻结主干、常数 lr3e-4）。residual-only 臂（route
weight 0）在**同一 AMP scale 65536** 下 V2 走完 200 步（V1 同臂走完 200 步）：

| step | 0 | 20 | 100 | 160 | 200 |
|---|---:|---:|---:|---:|---:|
| V1 residual | 0.418296 | 0.540440 | 0.540440 | 0.540440 | 0.540440 |
| **V2 residual** | 0.147610 | 0.143590 | 0.136480 | 0.127300 | **0.123810** |
| V2 GT-path XYZ mm | 5.6881 | 5.6089 | 5.3768 | 5.0193 | **4.8974** |
| V2 raw abs max | 0.000 | 0.130 | 0.424 | 0.603 | 0.645 |
| V2 tanh 饱和 | 0 | 0 | 0 | 0 | **0** |

V2 的 step0（0.147610 / 5.6881 mm）与第一轮 zero-init 对照（0.147614 / 5.6881）一致，说明
零初始化按预期生效、初始残差恰为 T3 anchor。

full 臂在**同一 scale 65536** 下两代都跑到 step100（V2 在 step104 因同一 fp16 上溢中止，
见下），逐点对照：

| step | 0 | 20 | 40 | 60 | 80 | 100 |
|---|---:|---:|---:|---:|---:|---:|
| V1 route sum | 12.6347 | 4.8162 | 2.0633 | 1.1244 | 0.5665 | 0.4468 |
| V2 route sum | 12.6347 | 4.8464 | 2.0293 | 0.9828 | 0.5397 | **0.2625** |
| V1 T3 NLL | 6.4077 | 3.4146 | 1.5548 | 0.8537 | 0.4502 | 0.3465 |
| V2 T3 NLL | 6.4077 | 3.4279 | 1.5402 | 0.7675 | 0.4310 | **0.2146** |
| V1 residual | 0.41830 | 0.49326 | 0.29642 | 0.19233 | 0.14569 | 0.13951 |
| V2 residual | 0.15346 | 0.14639 | 0.14054 | 0.13387 | 0.12438 | **0.11149** |
| V1 GT-path mm | 13.1545 | 15.1232 | 10.2734 | 6.9053 | 5.5040 | 5.3125 |
| V2 GT-path mm | 5.7389 | 5.5993 | 5.4272 | 5.2290 | 4.9123 | **4.4830** |

route 侧两边在 step20 相差 +0.6%、step40 起 V2 更好（step100 好 41%），**没有出现
"residual 改善但 route 崩坏"**。V2 full 臂在 65536 于 step104 出现与 V1 同类的 fp16 上溢
（`decoder.3.block.0.weight`，loss 全有限），因此用 `--amp-scale 16384`（scale 扫描已确认
该档通过）把 200 步跑完：

| V2 full arm @16384 | 0 | 40 | 80 | 120 | 160 | 200 |
|---|---:|---:|---:|---:|---:|---:|
| residual | 0.15346 | 0.14054 | 0.12438 | 0.11493 | 0.08608 | **0.06995** |
| GT-path mm | 5.7389 | 5.4272 | 4.9123 | 4.5832 | 3.6145 | **3.0799** |
| predicted-path mm | 83.4377 | 7.4437 | 5.2137 | 4.9379 | 3.6158 | **3.0794** |
| T1/T2/T3 acc | .109/.000/.001 | .979/.485/.586 | .995/.918/.893 | .988/.974/.860 | 1./1./.998 | **1./1./.9998** |
| route sum | 12.6347 | 2.0293 | 0.5627 | 0.5892 | 0.0801 | **0.0290** |
| raw abs max / 饱和 | 0.000 / 0 | 0.295 / 0 | 0.469 / 0 | 0.693 / 0 | 0.711 / 0 | **0.812 / 0** |
| context token norm（均值） | 4.42 | 6.54 | 10.68 | 18.57 | 18.91 | 19.01 |
| AMP scale | 16384 | 16384 | 16384 | 16384 | 16384 | 16384（无跳步） |

跨 scale 的读数需要分开看：V2@16384 在 step160 的 route 0.0801 / T3 NLL 0.0677 比
V1@65536 的 0.0751 / 0.0614 略差（+6.7% / +10%），而 residual（0.0861 对 0.1157）与
GT-path XYZ（3.61 对 4.57 mm）明显更好。匹配对照（同 scale）以 step0–100 那两行为准。

Gate B 对照：predicted soft-T3 conditioning 已接入正式 head ✔；概率默认 detach ✔
（配置 `residual_detach_route=True`）；context 投影 64 维 ✔；最后一层 zero-init ✔；
正式 forward/推理无 GT T3 ✔；loss 定义未变 ✔；200 步 route 未崩 ✔；raw 不再随机饱和 ✔。

**兼容性**：V2 改了 `residual_predictor` 的子模块结构（`weight/bias` → `context_projection/
fuse/final`），因此 c0facbd 及之前产生的 last-good/diagnostic checkpoint 不能加载进新 head
（`require_full_checkpoint` 的 key 检查也会因此拒绝旧格式），这些文件只作为历史证据保留。

## Observed：residual probe telemetry 修复与 conditioned 臂重跑（2026-09-20）

诊断脚本自身有两个缺陷：conditioned 臂把 **bounded** residual 当成 `raw_residual` 记录
（因此该臂此前的"raw abs max / 饱和"不能作为 raw predictor 证据），且 conditioned 臂的
last-good 不含 probe 参数。已修：每个 symmetry 分支同时保留 raw 与 bounded，用同一个
`choice` 选支；last-good 追加 `probe_state_dict`；跑完重建 model+probe+optimizer 重载做
roundtrip 校验。重跑 `exp025_probe_conditioned_RECHECK`（只跑 conditioned 臂）：

| 项 | 值 |
|---|---|
| 训练轨迹 | step0 0.219496 / 7.9437 mm → step200 **0.057928 / 2.7128 mm**（与 a02 的 0.056706 / 2.6669 同值域，确认修复未改动训练路径） |
| raw_residual（修复后） | abs max 1.524、`abs(tanh)==1` 比例 **0**（修复前记录的是 bounded 值 0.893） |
| roundtrip | `status=MATCH`，step 180，20 个指标最大相对差 **0.0**，probe 参数 132099，optimizer 内部 step `[180]` |

`zero_init_residual` 臂已随 V2 成为正式初始化，不再作为独立诊断臂；其第一轮结果保留在
`exp025_probe200_a03/a04.json`。

## Observed：初始化/checkpoint 语义与 hierarchy SHA（2026-09-20）

- `BACKBONE_INIT` 与 `MODEL.WEIGHTS` 拆开：`backbone_settings()` 不再填 legacy
  `MODEL.WEIGHTS`，fresh train 为 `""`（`resume_or_load("")` 走到 "Initializing model from
  scratch"），`set_mode()` 不再改写用户显式给出的 `MODEL.WEIGHTS`；主干初始化仍由
  `load_official_backbone`（official_lmo）或 `BACKBONE.INIT_CFG.checkpoint_path`（imagenet）
  负责。
- `--eval-only` / `TEST.SAVE_RESULTS_ONLY` 现在 fail-closed：`require_full_checkpoint` 要求
  checkpoint 的 model keys 同时含 `backbone.*`、`cad_attention_head.t3_classifier.weight`、
  `cad_attention_head.residual_predictor.final.weight`、`cad_attention_head.mask_predictor.weight`，
  否则报错而不是用随机 head 打分；resume 的 `last_checkpoint` 机制未改。
- hierarchy 身份固定为 `consistent_v3.npz` 的 SHA256
  `02ce090949bc40b2732417fec23984f3f748431098c5f67c853839d10ff1a373`：
  `research/exp025/configuration.py` 保存该常量 + streaming hash（`lru_cache`），
  `GDRN_CAD.dataset_context()`、preflight 与所有报告的 `metadata()` 都校验；本机实测
  `hierarchy_sha_match=true`（preflight PASS）。

## 验证与回归（2026-09-20 收口；重构前结构）

| 命令 | 结果 |
|---|---|
| `pytest -q research/exp025/tests` | **42 passed**（原 39：+3 V2 单测文件新增 8 项，减去已合并的 5 项旧用例重排） |
| `pytest -q research/tests/test_lite_optimizer_resume.py` | **8 passed**（新增 scheduler gate 行为测试与 helper 单测） |
| `pytest -q research` | **370 passed**（上一轮记录 354；历史 `338 passed / 3 failed` 的三项 exp022 `test_lm_protocol` 在本机未复现） |
| `python -m research.exp025.preflight` | **PASS**，`hierarchy_sha_match=true`，head 参数 8,217,796 |
| CPU preflight（official_lmo/unfrozen、imagenet/trainable） | PASS（见证据文件） |
| `amp_recovery_smoke`（CUDA，生产对象） | **PASS** + 修复前对照（见本节上文） |
| `fixed200_v2_a01` / `fixed200_v2_scale16384_a01`（CUDA） | 前者两臂 @65536（full 在 step104 中止），后者 full 臂 200 步完成 |
| `probe_conditioned_RECHECK`（CUDA） | COMPLETE，roundtrip MATCH |
| `accum_RECHECK`（CUDA） | **PASS**，`scheduler_advanced_only_on_updates=true`，resume 与连续训练 LR 轨迹逐 update 相等，无 AMP 跳步 |
| `cpp_RECHECK`（CUDA，在线 CPP） | **PASS**，8/8 步、无跳步、checkpoint roundtrip true、整步中位数 152.4 ms、峰值 allocated/reserved 1.523/1.783 GB |
| 服务器 EGL smoke | **未执行**（仅由用户在服务器运行） |

Gate A（production AMP）✔；Gate B（Residual V2）✔；Gate C（checkpoint 语义）✔；
Gate D（hierarchy identity）✔；Gate E 的服务器 EGL 一项未执行，因此 `FORMAL_READY` 保持
`False`，未启动 40 epoch 正式训练。

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
（derived：真实入口 resume 的 checkpoint 审计）。本轮新增：`exp025_amp_recovery_recheck.json`
（生产路径 AMP recovery，新 gate）、`exp025_amp_recovery_unguarded_a01.json`（修复前循环对照）、
`exp025_fixed200_v2_a01.json`（V2 两臂 @65536，full 臂在 step104 中止）、
`exp025_fixed200_v2_scale16384_a01.json`（V2 full 臂 @16384 走完 200 步）、
`exp025_probe_conditioned_recheck.json`（conditioned 臂 telemetry 修复后重跑 + roundtrip）、
`exp025_cpp_recheck.json`（CPP real smoke）、`exp025_accum_recheck.json`（带 scheduler gate 的
accumulation smoke）。checkpoint 与完整日志仍外置。

重构轮新增（均对应重构后结构）：`exp025_refactor_boundary_a01.json` 与
`exp025_refactor_boundary_old_a01.json`（一步 FP32/AMP 边界探针，后者在 `eb2fdb2` 的
detached worktree 中用同一脚本、同一 batch 运行）、`exp025_refactor_smoke_scale32768_a01.json`
（最高通过 scale 的 8 步 smoke）、`exp025_refactor_smoke_a01.json`（65536 在 step1 失败及
其 telemetry）、`exp025_refactor_fixed60_a01.json`（60 步固定 batch 可训练性 smoke）、
`exp025_refactor_accum_a01.json`（4/48 本地形状的状态机 smoke）、
`exp025_refactor_main_smoke_a02.json`（derived：真实入口 1 epoch 的 checkpoint 审计）。

## 下一步（待用户确认）

重构后（2026-09-20 交接包 v2）：

1. **服务器真实 batch48 与 EGL gate 未执行**：需在服务器用新 release 跑 `exp025_lmo`
   profile 的 preflight、EGL 真实 batch48 前反向、optimizer update、checkpoint
   save/resume 与峰值显存，并实测 batch48 下的 AMP/GradScaler 行为；通过前
   `FORMAL_READY` 保持 `False`。
2. **AMP 初始 scale**：本机边界是 32768 通过、65536 在 step1 上溢（Mask 头）；生产
   engine 已按 GradScaler 语义跳过并降 scale，是否把正式初始 scale 固定为某个值属训练
   策略，未经确认不实施。
3. **机制复验**：新结构的 route/residual 响应需要重新做 fixed-batch 诊断（必要时再到
   E5–E40）；重构前的 fixed-batch 数值与结论不能迁移到新结构。

以下 3 条为重构前结构遗留的问题，保留原文：

1. **诊断脚本的 guard 语义**：生产 engine 已按 GradScaler 的真实语义运行（跳过 + 降 scale +
   不推进 scheduler）；`runtime.amp_step` 仍然是"非有限即致命"的诊断口径，用于把失败钉在
   具体 step/参数上。是否让诊断脚本也接受 GradScaler 的正常恢复（或固定用 16384），由用户
   决定；本轮未改诊断语义，也未在正式配置里改 AMP scale。
2. **V2 的 65536 上溢**：V2 full 臂在 scale 65536 于 step104 出现与 V1 同类的 fp16 上溢
   （不同步数、同一机制，`decoder` 侧）。是否把正式初始 scale 调低到 16384 及以下属训练策略，
   未经确认不实施。
3. **噪声底**：如需跨 run 逐张量比较（例如严格单变量消融），需要先隔离本机
   CUDA+AMP 不可复现的来源，或固定 `torch.use_deterministic_algorithms` 后重测。
4. 正式训练、服务器 EGL、E5–E40 与完整 BOP/matched-PnP 评价均未执行；
   `FORMAL_READY` 保持 `False`。
