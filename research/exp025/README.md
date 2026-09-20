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
- Residual V2：残差回归目标 `(XYZ-anchor_T3)/radius_T3` 依赖 T3 identity，因此预测器
  额外读取 **预测** T3 分布——`logits → FP32 softmax → detach` 概率对投影到 64 维的 T3
  CAD token 求期望（`[B,P,512]×[B,512,64]`），与 image token 拼接后过
  `Linear(D+64,D)+GELU+Linear(D,3)`；最后一层权重与 bias 零初始化，初始残差恰为 0，
  即从 T3 anchor 出发。不用 GT id、不做 top-k、train/infer 同一 forward；detach 使
  residual loss 不反向改写已稳定的 T3 分类器（该分支只经 image token 影响它）。
- 几何 buffer 非持久化，checkpoint 之外必须保留相同 hierarchy artifact：EXP025 固定
  `consistent_v3.npz` SHA256 `02ce0909…1a373`，`dataset_context`/preflight/每个 report
  都校验并记录该摘要（同一进程只 hash 一次）。
- `BACKBONE_INIT` 只负责主干初始化；`MODEL.WEIGHTS` 只表示**完整 GDRN_CAD checkpoint**，
  因此 fresh train 为 `""`，resume 仍走 output 目录 + `--resume`。`--eval-only`（或
  `SAVE_RESULTS_ONLY`）在缺少完整 checkpoint（t3_classifier / residual predictor /
  mask predictor / backbone 任一缺失）时 fail-closed，不会用随机 head 打分。

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
- `residual_probe` 有两个诊断臂（`--arms` 可选）：`baseline_residual_only`（正式残差
  路径）与 `gt_t3_conditioned`（image token 与 GT T3 CAD token 拼接后过
  `Linear(2D,D)+GELU+Linear(D,3)`）。两臂同 batch/seed/steps/lr 且 route weight 0，
  用于判断 residual 是否可学、以及是否缺 T3 identity 输入；GT T3 不进入正式模型、
  损失或推理。第一轮的 `zero_init_residual` 对照已成为正式初始化（Residual V2），
  不再是独立臂，其历史结果保留在旧报告里。conditioned 臂的 last-good 额外保存
  `probe_state_dict`，并在跑完后重建 model+probe+optimizer 重载做 roundtrip 校验
  （`roundtrip.status=MATCH` 表示记录指标可逐项复现）。
- `accumulation_smoke` 覆盖正式形状（physical batch4、effective48、accumulate12）的状态机：
  optimizer step 只在 accumulation 边界发生、scheduler 只在真实 update 上推进、AMP 无跳步、
  save+resume 后与不间断训练在**无噪声计数与 LR 轨迹上逐项相等**。CUDA+AMP 同一配置两次运行
  本就有约 1e-3 量级 run-to-run 噪声（carried SDPA backward 在 CUDA 上是 atomic），
  因此张量差只相对同进程内测得的 noise floor 判断（阈值 3×noise floor），不做逐位相等断言；
  硬判据是 optimizer step / scheduler epoch / LR 序列 / accumulation 边界 / scaler。
  它复用 `solver_utils` 的 `accumulation_window_size`/`should_optimizer_step` 与
  `MyCheckpointer`。
- `amp_recovery_smoke` 走真实生产对象（`LightningLite(precision=16)`、precision plugin 的
  `GradScaler`、Lite optimizer wrapper、进入 fp16 autocast 的 `_LiteModule`），从同一
  full arm last-good 继续，逐 update 记录 scale/内部 optimizer step/scheduler epoch/参数
  delta/是否继续；`--unguarded-scheduler` 复现修复前"skip 也推进 scheduler"的循环作对照。

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
