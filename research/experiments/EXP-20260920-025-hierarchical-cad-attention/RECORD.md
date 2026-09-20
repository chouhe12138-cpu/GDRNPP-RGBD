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

源码基点 `dbc1535925e092bfbec47a8b32e05f78c4687cc2` + 本轮未提交工作树；
并非一个已提交 EXP025 release。环境 Conda `pytorch22`、torch2.2、RTX4060 Laptop。
本机产物统一在 `output/diagnostics/<run_id>/`；固定 batch
`.local/exp025/fixed_batch4_a01.pt` 来自 frozen_a01 的在线 CPP 渲染。
hierarchy `.local/dataset_cache/exp022/consistent_v3.npz`；不复制数据或权重入 Git。

Decision：实现已接线，**NUMERICAL_REVIEW / FORMAL_NOT_STARTED**。
`FORMAL_READY=False`；不因为 smoke 通过而忽略后续学习诊断问题。

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

## 下一步（待用户确认）

建议先对失败点做诊断复现（记录精确步号、参数与AMP scale，比较同状态的FP32
backward），再做残差小幅/零初始化的matched对照；这会改变初始化实验条件，
本轮尚未实施。保持现有设计与权重，不擅自新增终端LayerNorm、改损失权重或关闭AMP。
正式训练、完整BOP/matched-PnP评价与服务器部署尚未执行。
