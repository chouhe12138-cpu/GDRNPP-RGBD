# EXP025：统一 T3 分类与层级 CAD attention

本页只描述当前方法与工具接口。实验状态见 [STATUS](../STATUS_CN.md)，运行事实、指标与决策见
[RECORD](../experiments/EXP-20260920-025-hierarchical-cad-attention/RECORD.md)。

## 模型契约

- ConvNeXt 输出 `[B,1024,8,8]`，经 `1×1 Conv` 适配到 512 维。
- 图像分支使用四级 Image-SA：8×8 global、16×16 global、32×32 window8+shift4、
  64×64 window8+shift4。每级 attention 结果 residual 写回 feature 后再进入下一层 transition；
  末级 tokens `[B,4096,256]` 直接作为后续 query。
- Spatial Transition 为 `1×1 Conv/GN/GELU + 3×3 Conv/GN/GELU` 与 bilinear 上采样，产生
  256@16、128@32、64@64。Image-SA 为 PreNorm、8 heads、FFN ratio2、dropout0。
- CAD 分支在图像解码阶段独立，最终按 T0/T1/T2/T3 顺序与图像 tokens cross-attention。
- 固定 hierarchy 只使用 T1=8、T2=64、T3=512；T4 不进入模型。共享 10 维几何 descriptor
  MLP，T1 全局 attention，T2/T3 在 parent broadcast 后做 sibling attention。
- LM-O hierarchy 为 `.local/dataset_cache/exp025/consistent_v3.npz`，SHA256
  `02ce090949bc40b2732417fec23984f3f748431098c5f67c853839d10ff1a373`。运行时路径由
  `GDRN_DATASET_CACHE_DIR` 决定，preflight 和 report 都校验 identity。

## 监督与解码

- 唯一分类输出是 `[B,512,64,64]` T3 logits。FP32 `log_softmax` 后，每 8 个 child 用
  `logsumexp` 依次得到 T2、T1；三层 NLL 相加，各权重为 1。
- GT 在每个 parent 的 8 个 children 内做 nested 最近邻，满足
  `T2_id=T3_id//8`、`T1_id=T3_id//64`。最多两个 SE(3) 对称分支按整实例
  route+residual loss 选取同一分支。
- Residual V2 用预测 T3 分布条件化：`logits → FP32 softmax → detach`，再对投影到 64 维的
  T3 CAD token 求期望，与 image token 拼接后预测三维残差。最后一层零初始化；训练和推理
  使用同一 forward，不读取 GT id。
- residual 经 `tanh` 限制后按 T3 anchor/radius 解码，SmoothL1 beta=.1、权重1；可见 mask
  使用 BCE、权重1。
- 推理以 `argmax(T3 logits)` 选择 leaf，不使用 marginal argmax、hard routing、beam 或
  teacher forcing。

## 正式配置入口

- `train_official_frozen.py`：原 GDRNPP LM-O backbone，冻结，在 lab0 运行。
- `train_imagenet_full.py`：ImageNet ConvNeXt 初始化，完整训练，在 lab1 运行。
- 两臂共享 `common.py`；它们同时改变初始化和训练范围，只能解释为组合策略差异。
- `BACKBONE_INIT` 只初始化 backbone；`MODEL.WEIGHTS` 只接受完整 GDRN_CAD checkpoint。
  fresh train 使用空值，resume 使用 output 目录和 `--resume`。
- eval/save-results 在缺少完整 head 或 backbone checkpoint 时 fail-closed，不使用随机参数评分。

正式 LM-O 协议为 PBR40、GT box、seed42、真实 batch48、AdamW 3e-4、40 epoch、4% linear
warmup 后 cosine 到 0.01 倍初始 LR、E5–E40 每 5 epoch 固定评价、关闭 best-checkpoint 选择。
ImageNet full arm 的 backbone 与 head 均使用 3e-4。AMP 显式开启，当前初始 scale 来自配置；
运行状态和 gate 结果不在本页重复。

`train_lm13_imagenet_full.py` 是 LM-O 之后的准备项：LM real + DeepIM renders、ImageNet
ConvNeXt 全量训练、Ranger 1e-4、backbone LR×0.1、effective batch24、160 epoch。该 profile
尚未加入服务器 launcher，不能作为当前正式入口。

## 本地工具

先激活 `pytorch22` 并设置 ImageNet ConvNeXt 权重路径。常用入口：

```bash
python -m pytest -q research/exp025/tests
python -m research.exp025.preflight
python -m research.exp025.real_smoke --output output/diagnostics/exp025_smoke_NEW
python -m research.exp025.amp_boundary_probe --output output/diagnostics/exp025_boundary_NEW
python -m research.exp025.learnability --output output/diagnostics/exp025_fixed_NEW --amp-scale 16384
python -m research.exp025.accumulation_smoke --output output/diagnostics/exp025_accum_NEW
```

- `preflight` 校验配置、hierarchy、参数覆盖和初始化契约。
- `real_smoke` 默认 CUDA、CPP、batch4、8 steps，可显式选择 backbone、renderer、固定 batch
  和 AMP scale；它检查四级 Image-SA 的梯度与更新覆盖。
- `amp_boundary_probe` 在同一 seed/batch 比较 FP32 与 AMP 的未缩放梯度量级，只用于定位数值
  边界。
- `learnability` 在固定 batch 上隔离 residual/mask 与完整 loss；它不是泛化或正式精度结果。
- `accumulation_smoke` 检查 optimizer step、scheduler、GradScaler、checkpoint/resume 与本地
  4×12 accumulation 状态机。

## 生产路径边界

- `SOLVER.AMP.INIT_SCALE` 只改变 GradScaler 初值；动态 growth/backoff 和 checkpoint 恢复
  语义不变。配置要求 AMP 开启且 scale ≥1。
- GradScaler 跳过 optimizer update 时 scheduler 也不推进；Lite optimizer resume 必须恢复
  真正执行 step 的底层 optimizer。
- 几何 buffer 不持久化，因此 checkpoint 之外必须保留并校验相同 hierarchy artifact。
- 本机 batch4、fixed-batch、smoke、profile 和 oracle 只证明工程链路或机制响应，不替代
  服务器 batch48 gate、formal 指标或跨 seed 稳定性。
- 服务器由用户按 [RUNBOOK](../RUNBOOK_CN.md) 的 bundle、只读 release 和受管容器流程运行；
  Agent 不主动 SSH。
