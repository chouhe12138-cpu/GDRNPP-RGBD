# EXP025：统一 T3 分类与层级 CAD attention

实现以用户最新的统一 512 分类要求为准，替代交接包中三个独立 8-way head；
不改写 EXP022 的 PCC、GT-parent / beam 推理或历史实验结果。
实验事实与诊断结论只记在 [RECORD](../experiments/EXP-20260920-025-hierarchical-cad-attention/RECORD.md)。

## 模型与监督契约

- ConvNeXt 输出 `[B,1024,8,8]`，`1×1 Conv` adapter→512；随后是**四级 Image-SA
  stage**：8×8 global、16×16 global、32×32 window8+shift4、64×64 window8+shift4。
  每级先投影到 256 维做 self-attention，再把结果 residual 写回 feature，**写回后的
  feature 才进入下一级 Spatial Transition**（`1×1 Conv/GN/GELU + 3×3 Conv/GN/GELU`
  bilinear 上采样，得 256@16、128@32、64@64）；末级 SA 的 tokens `[B,4096,256]` 直接
  作为后续 query，其 feature 无下游消费者因此不回写。PreNorm、残差 FFN ratio2、
  8 heads、dropout0。CAD 分支在 Image 解码阶段完全独立，只在最终 tokens 汇合。
- 最终 tokens 依次读取 T0/T1/T2/T3（CA 顺序固定，bank 为 1/8/64/512 tokens）。
- 固定 `consistent_v3.npz` 只取 T1=8、T2=64、T3=512；T4 不进入模型。
  共享 10 维几何 descriptor MLP，T1 全局 attention，T2/T3 为 parent broadcast
  后的 sibling attention；不反向更新 parent。learned T0 顺序读取三层；
  图像读取单个 T0 时 attention 等价于全局广播，不宣称其具有空间选择性。
- 唯一分类输出为 `[B,512,64,64]` T3 logits。FP32 `log_softmax` 后，
  连续每 8 个 child 用 `logsumexp` 得到 T2，再同样得到 T1。
  三层 NLL **相加、不取平均**，各权重 1。无 Hamming loss、binary code、
  hard routing 或 teacher forcing。GT 仅用于 loss 与显式标记的诊断 decode。
- GT 在 FP32 中沿各 parent 的 8 个 child 做 nested 最近邻；全局标签满足
  `T2_id=T3_id//8`、`T1_id=T3_id//64`，不是分别全局最近邻。
  最多两个 SE(3) 对称等价分支按整实例 route+residual loss 选取同一分支。
- residual 为 `tanh` 后投影到单位球，SmoothL1 beta=.1，权重1；可见 mask
  BCE 权重1。推理只用 `argmax(T3 logits)`，取 leaf anchor/radius 加 bounded
  residual 解码 XYZ，不使用 marginal argmax（两者的祖先可能不同）。
- Residual V2：残差回归目标 `(XYZ-anchor_T3)/radius_T3` 依赖 T3 identity，因此预测器
  额外读取 **预测** T3 分布——`logits → FP32 softmax → detach` 概率对投影到 64 维的 T3
  CAD token 求期望（`[B,P,512]×[B,512,64]`），与 image token 拼接后过
  `Linear(D+64,D)+GELU+Linear(D,3)`；最后一层权重与 bias 零初始化，初始残差恰为 0，
  即从 T3 anchor 出发。不用 GT id、不做 top-k、train/infer 同一 forward；detach 使
  residual loss 不反向改写已稳定的 T3 分类器（该分支只经 image token 影响它）。
- 几何 buffer 非持久化，checkpoint 之外必须保留相同 hierarchy artifact：LM-O
  `consistent_v3.npz` SHA256 `02ce0909…1a373`，LM13 SHA256 `322cd377…6417`；
  `dataset_context`/preflight/每个 report 都按数据集校验并记录摘要（同一进程只 hash 一次）。
- `BACKBONE_INIT` 只负责主干初始化；`MODEL.WEIGHTS` 只表示**完整 GDRN_CAD checkpoint**，
  因此 fresh train 为 `""`，resume 仍走 output 目录 + `--resume`。`--eval-only`（或
  `SAVE_RESULTS_ONLY`）在缺少完整 checkpoint（t3_classifier / residual predictor /
  mask predictor / backbone 任一缺失）时 fail-closed，不会用随机 head 打分。

## 配置与入口

正式配置拆成两个明确 arm：`train_official_frozen.py` 用于 lab0，加载原 GDRNPP LM-O
主干并冻结；`train_imagenet_full.py` 用于 lab1，从 ImageNet ConvNeXt 初始化并训练完整
主干。共享协议在 `common.py`。两臂同时改变初始化和训练范围，只比较组合策略。
诊断工具默认读取所给配置；ImageNet 通过
`GDRN_CONVNEXT_BASE_WEIGHTS` 指定本机权重，不自动下载。

`train_lm13_imagenet_full.py` 是 LM-O 之后的预备 arm。它使用 13 类 LM real + DeepIM
renders、ImageNet ConvNeXt 全量训练、Ranger 1e-4（backbone LR×0.1）、effective batch24
（4×6 accumulation）、160 epoch、1000-step warmup、前72% flat 后 cosine 到0，并每20
epoch checkpoint/eval。对应 hierarchy 位于
`.local/dataset_cache/exp025/lm13/consistent_v3.npz`。本地 CPU preflight 和 batch4
CUDA+CPP/AMP 三步 smoke 已通过；launcher 当前明确拒绝 `exp025_lm13`，待 LM-O 完成后再补
服务器资源门和 EGL gate。

LM-O PBR40/GT-box，seed42，AdamW lr3e-4、wd.01、betas(.9,.999)、eps1e-8；
解冻主干与头同为 3e-4（`BACKBONE_LR_MULT=1.`：官方 LM-O 配方对本干不加乘子，
而继承自 LM13 协议的 .1 让主干只有 3e-5、无实验证据支持——见 RECORD 的 Decision）。
**formal 是真实 batch48（`IMS_PER_BATCH=48`、`REFERENCE_BS=48`，
accumulate=1，每 iteration 一次真实 optimizer update）**；AMP 显式开启；40epoch，
warmup4%，cosine 到初始 lr 的 .01；E5–E40 定点评价，best checkpoint 关闭。
`FORMAL_READY=True`，两臂共享 `SOLVER.AMP.INIT_SCALE=32768`。本机显存受限的 4×12 形状
只存在于 `smoke.py`（4/48）与诊断脚本的显式参数里，不改 formal；engine 的梯度累计保留给
这些本地路径。`smoke.py` 使用独立有界 8-image split，不能用于正式实验。

从仓库根目录执行（每次使用新的 output 名称）：

```bash
source /home/wsluser/miniconda3/etc/profile.d/conda.sh
conda activate pytorch22
python -m pytest -q research/exp025/tests
python -m research.exp025.preflight
python -m research.exp025.real_smoke --output output/diagnostics/exp025_smoke_NEW --save-batch .local/exp025/batch_NEW.pt
python -m research.exp025.amp_boundary_probe --output output/diagnostics/exp025_boundary_NEW --load-batch .local/exp025/batch_NEW.pt
python -m research.exp025.learnability --output output/diagnostics/exp025_fixed_NEW \
    --load-batch .local/exp025/batch_NEW.pt --amp-scale 16384
python -m research.exp025.accumulation_smoke --output output/diagnostics/exp025_accum_NEW --load-batch .local/exp025/batch_NEW.pt
```

`real_smoke` 默认 CUDA、CPP、batch4、8 steps；可用 `--train-backbone yes`、
`--backbone-init imagenet`、`--renderer egl`、`--load-batch`、`--amp-scale`（初始
GradScaler scale，生产默认 65536）。它同时校验四级 Image-SA 的参数确实被更新。
`amp_boundary_probe` 从同一 seed/batch 各跑一步 FP32 与指定 scale 的 AMP，报告每个
子模块未缩放梯度元素最大值、`scale × abs_max` 与 fp16 上限 65504 的关系，用来判断
非有限梯度是缩放后的 fp16 边界还是 loss/结构发散；在旧 commit 的 worktree 中运行同一
命令即可做跨结构对照。
`learnability` 固定官方冻结主干、同一 batch/seed/初始参数，分别执行 residual+mask
与完整 loss 各200步，常数 lr，每20步记录；GT-path 仅用于隔离残差诊断，不是训练路由。
不预设通用下降阈值，不用 fixed-batch 数值冒充泛化性能。
**本地诊断不要用默认的 65536 起步**：当前结构在本机 batch4、无 accumulation 的单步路径上
已在 65536 首步溢出（`amp_boundary_probe` 测得是 Mask 头缩放后的 fp16 边界），示例因此显式
给 `--amp-scale 16384`。默认值本身仍是 65536，因为生产入口按 torch/Lite 的默认起步，
只有在 `SOLVER.AMP.INIT_SCALE` 被显式设置时才改变；已测的通过/溢出是**本地固定 batch、
当前初始状态**的结果，不能外推到 formal batch48。

## AMP 初始 scale 配置（`SOLVER.AMP.INIT_SCALE`）

`main_gdrn` 把 `SOLVER.AMP.INIT_SCALE` 交给 `LightningLite` 的 precision plugin
（`solver_utils.amp_precision_plugins`）：设置后 Lite 使用该值创建 GradScaler，未设置时
返回 `None`，所有历史实验继续使用 Lite 自己的默认 scaler（65536）。只改初始值——动态
growth/backoff、scaler 的 checkpoint 保存/恢复、以及 GradScaler 跳步时 scheduler 不推进的
逻辑都不变。要求 `SOLVER.AMP.ENABLED=True` 且值 ≥ 1，否则 fail-closed。
2026-09-20 服务器 gate 后，两个正式配置共享 `SOLVER.AMP.INIT_SCALE=32768`——65536 在
两臂同一步、同一张量（Mask 头）缩放后溢出，32768 两臂均 PASS，按"共同通过的最高值"固定。
本地 `real_smoke`/`learnability`/`amp_boundary_probe` 的 `--amp-scale`
默认跟随该配置（未设置时仍是 65536）。

## 历史证据边界

2026-09-20 重构了 image 分支（此前是 transition 链 + 末端一次 64×64 SA，SA 结果不进入
后续 transition），并把 formal 从本机 4×12 形状改回真实 batch48。**重构前记录的
Residual V1/V2、AMP scale sweep、route/T3 accuracy、residual probe、生产路径 AMP
recovery 以及当时的 CPU/CUDA smoke 参数值都属于旧结构**，在 RECORD 中原样保留并已标注，
不能当作新结构的验证或机制证据；新结构的机制复验从零开始。

## 数值诊断工具

- 所有 step 都经过 `runtime.amp_step`：失败时抛出 `NonFiniteTrainingError`，携带该步的
  step、AMP scale、全部 loss 分量、图像/CAD token 范数、T3 logits、mask logit、raw
  residual 分布与逐模块梯度范数，`learnability`/`real_smoke` 把它写进 `report.json` 的
  `failure` 字段并保留已完成 history。每个 arm 每 `--last-good-period`（默认20）步写
  `model/optimizer/scaler/step/RNG` 到 `<arm>_last_good.pth`。
- `accumulation_smoke` 覆盖本地 4×12 形状（`--batch-size 4 --reference-bs 48`，**不再从
  formal 继承**）的状态机：
  optimizer step 只在 accumulation 边界发生、scheduler 只在真实 update 上推进、AMP 无跳步、
  save+resume 后与不间断训练在**无噪声计数与 LR 轨迹上逐项相等**。CUDA+AMP 同一配置两次运行
  本就有约 1e-3 量级 run-to-run 噪声（carried SDPA backward 在 CUDA 上是 atomic），
  因此张量差只相对同进程内测得的 noise floor 判断（阈值 3×noise floor），不做逐位相等断言；
  硬判据是 optimizer step / scheduler epoch / LR 序列 / accumulation 边界 / scaler。
  它复用 `solver_utils` 的 `accumulation_window_size`/`should_optimizer_step` 与
  `MyCheckpointer`。
- 重构前使用的 numerical replay、residual probe 与专用 AMP recovery runner 已在结论固定后
  退出 HEAD；其代码可从 RECORD 所列 source commit 恢复，原始紧凑报告继续随 RECORD 保存。

Optimizer 恢复边界属共享框架修复（`core/utils/my_checkpoint.py` +
`engine.do_train`）：LightningLite wrapper 继承 `torch.optim.Optimizer.load_state_dict`，
只恢复 wrapper 自己的 `state`/`param_groups`，真正 step 的底层 optimizer 不被更新，
resume 后 scheduler 的 LR 写入也会与训练 optimizer 脱钩。回归测试见
`research/tests/test_lite_optimizer_resume.py`。

服务器仅由用户按 RUNBOOK bundle/release/Docker 流程运行；`exp025_lmo` profile 根据
arm 和初始化来源检查权重、EXP025 hierarchy、PBR/LM-O/VOC，并执行 CPU preflight。
launcher 强制 `official_frozen→lab0`、`imagenet_full→lab1`，历史 profile 已退出 HEAD。

## 验证分层

- Integrity：概率归一、marginal/nested 标签一致、AMP target FP32、有限梯度、
  optimizer 覆盖、冻结范围、真实 batch 与 checkpoint 往返。
- Mechanism：固定批次 route/residual 响应、GT-path 与 predicted-path XYZ 误差；
  单 batch/seed 不支持跨 run 稳定或正式精度结论。
- Advisory：本机显存/耗时；固定 batch 计时不含逐步数据加载/渲染。
- 正式评估和服务器 EGL 独立验收；未生成时明确保留缺口。
