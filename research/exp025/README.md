# EXP025：统一 T3 分类与层级 CAD attention

实现以用户最新的统一 512 分类要求为准，替代交接包中三个独立 8-way head；
不改写 EXP022 的 PCC、GT-parent / beam 推理或历史实验结果。
实验事实与诊断结论只记在 [RECORD](../experiments/EXP-20260920-025-hierarchical-cad-attention/RECORD.md)。

## 模型与监督契约

- ConvNeXt 输出 `[B,1024,8,8]`，adapter→512；三次 bilinear 上采样与
  `1×1 Conv/GN/GELU + 3×3 Conv/GN/GELU` 得到 256@16、128@32、64@64。
- 投影到 256 维；64×64 图像 token 先做 window8、shift4 self-attention，
  再依次读取 T0/T1/T2/T3。PreNorm、残差 FFN ratio2、8 heads、dropout0。
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
- 几何 buffer 非持久化，checkpoint 之外必须保留相同 hierarchy artifact。

## 配置与入口

唯一训练配置：`configs/gdrn/lmo_pbr/research/exp025_hierarchical_cad_attention/train.py`。
直接编辑顶部 `TRAIN_BACKBONE` / `BACKBONE_INIT` / `BACKBONE_LR_MULT`；
初始化来源与冻结状态独立，支持 official_lmo / imagenet × frozen / trainable。
mmcv 的 `--opts` 不会重新执行 Python 派生字段，不要只用它覆盖顶部控制变量。
诊断工具的同名 CLI 会同步所有派生字段。ImageNet 通过
`GDRN_CONVNEXT_BASE_WEIGHTS` 指定本机权重，不自动下载。

LM-O PBR40/GT-box，seed42，AdamW lr3e-4、wd.01、betas(.9,.999)、eps1e-8；
解冻主干 lr 乘 .1。物理 batch4、累积12、effective48；AMP 显式开启；40epoch，
warmup4%，cosine 到初始 lr 的 .01；E5–E40 定点评价，best checkpoint 关闭。
`FORMAL_READY=False`。`smoke.py` 使用独立有界 8-image split，不能用于正式实验。

从仓库根目录执行（每次使用新的 output 名称）：

```bash
source /home/wsluser/miniconda3/etc/profile.d/conda.sh
conda activate pytorch22
python -m pytest -q research/exp025/tests
python -m research.exp025.preflight
python -m research.exp025.real_smoke --output output/diagnostics/exp025_smoke_NEW --save-batch .local/exp025/batch_NEW.pt
python -m research.exp025.learnability --output output/diagnostics/exp025_fixed_NEW --load-batch .local/exp025/batch_NEW.pt
python -m research.exp025.numerical_replay --output output/diagnostics/exp025_replay_NEW \
    --from-checkpoint output/diagnostics/exp025_fixed_NEW/full_last_good.pth \
    --load-batch .local/exp025/batch_NEW.pt
python -m research.exp025.residual_probe --output output/diagnostics/exp025_probe_NEW --load-batch .local/exp025/batch_NEW.pt
python -m research.exp025.accumulation_smoke --output output/diagnostics/exp025_accum_NEW --load-batch .local/exp025/batch_NEW.pt
```

`real_smoke` 默认 CUDA、CPP、batch4、8 steps；可用 `--train-backbone yes`、
`--backbone-init imagenet`、`--renderer egl`、`--load-batch`。
`learnability` 固定官方冻结主干、同一 batch/seed/初始参数，分别执行 residual+mask
与完整 loss 各200步，常数 lr，每20步记录；GT-path 仅用于隔离残差诊断，不是训练路由。
不预设通用下降阈值，不用 fixed-batch 数值冒充泛化性能。

## 数值诊断工具

- 所有 step 都经过 `runtime.amp_step`：失败时抛出 `NonFiniteTrainingError`，携带该步的
  step、AMP scale、全部 loss 分量、图像/CAD token 范数、T3 logits、mask logit、raw
  residual 分布与逐模块梯度范数，`learnability`/`real_smoke` 把它写进 `report.json` 的
  `failure` 字段并保留已完成 history。每个 arm 每 `--last-good-period`（默认20）步写
  `model/optimizer/scaler/step/RNG` 到 `<arm>_last_good.pth`。
- `numerical_replay` 从某个 last-good state 出发，用同一 batch 跑 A 当前 AMP /
  B 指定低 scale AMP（`--scale`）/ C FP32（`amp_step(..., scaler=None)`）三臂，报告每臂
  首个非有限步、首个坏参数与该步梯度范数；只做重放，不改 loss/结构/协议。
- `residual_probe` 有三个诊断臂（`--arms` 可选）：`baseline_residual_only`（正式残差
  路径）、`zero_init_residual`（最后一层零初始化的同路径对照）、`gt_t3_conditioned`
  （image token 与 GT T3 CAD token 拼接后过 `Linear(2D,D)+GELU+Linear(D,3)`）。三臂
  同 batch/seed/steps/lr 且 route weight 0，用于判断 residual 是否可学、以及是否缺
  T3 conditioning；GT T3 不进入正式模型、损失或推理。
- `accumulation_smoke` 覆盖正式形状（physical batch4、effective48、accumulate12）的状态机：
  optimizer step 只在 accumulation 边界发生、scheduler 与 update 一一对应、AMP 无跳步、
  save+resume 后与不间断训练逐张量一致。它复用 `solver_utils` 的
  `accumulation_window_size`/`should_optimizer_step` 与 `MyCheckpointer`。

Optimizer 恢复边界属共享框架修复（`core/utils/my_checkpoint.py` +
`engine.do_train`）：LightningLite wrapper 继承 `torch.optim.Optimizer.load_state_dict`，
只恢复 wrapper 自己的 `state`/`param_groups`，真正 step 的底层 optimizer 不被更新，
resume 后 scheduler 的 LR 写入也会与训练 optimizer 脱钩。回归测试见
`research/tests/test_lite_optimizer_resume.py`。

服务器仅由用户按 RUNBOOK bundle/release/Docker 流程运行；新增 `exp025_lmo`
profile 根据初始化来源检查权重、consistent hierarchy、PBR/LM-O/VOC，并执行 CPU
preflight。launcher 本地 mock 通过不代表服务器 EGL、挂载或正式训练通过。

## 验证分层

- Integrity：概率归一、marginal/nested 标签一致、AMP target FP32、有限梯度、
  optimizer 覆盖、冻结范围、真实 batch 与 checkpoint 往返。
- Mechanism：固定批次 route/residual 响应、GT-path 与 predicted-path XYZ 误差；
  单 batch/seed 不支持跨 run 稳定或正式精度结论。
- Advisory：本机显存/耗时；固定 batch 计时不含逐步数据加载/渲染。
- 正式评估和服务器 EGL 独立验收；未生成时明确保留缺口。
