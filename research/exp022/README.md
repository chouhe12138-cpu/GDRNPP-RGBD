# EXP022 — 渐进式层级 CAD 对应与多尺度 PCC（第一阶段）

本阶段仅使用冻结的官方 RGB ConvNeXt 8×8 特征，训练四级 PCC、空间细化、局部残差和可见 mask。第二阶段全量 backbone 训练、ResNet-50 与部署不在当前范围。

## 固定协议

- 主层级 `reused_v1.npz`：把 EXP021 的 64×64 叶子无损重排为 8⁴；叶子 anchor、法向、半径不变。消融层级 `independent_v2.npz` 从模型表面重新采样并递归分组；单点 cell 的半径下限为 `1e-5 m`，仅做 smoke，不安排 40 epoch。
- 四级分辨率 8/16/32/64，每级 8 路；训练按当前分辨率的 GT 父路径 teacher forcing，推理保留 Top-2 完整路径 ID 和归一化累计 log score。8 路条件概率仅用于 soft CAD context。跨尺度路由状态按双线性权重合并同 ID 路径后再取 Top-2。
- 对称物体按完整 SE(3) 等价变换选择一个 instance-consistent 分支；选择分数仅为加权四级 CE 与加权 residual loss 之和，不含 mask loss。选定后该分支的四级 CE、residual 和 mask loss 一起训练。最终点为叶子 anchor 加不超过该叶子半径的 3D 残差。
- PBR40、batch 48、40 epoch、16 workers、seed 42、AdamW `3e-4`、4% linear warmup + cosine、显式 FP16 AMP。正式配置每 5 epoch 做原有 direct-pose 评估，输出仍通过 explicit RANSAC-PnP。
- V1 固定值：route/residual/mask 权重均为 `1.0`，residual Smooth L1 `beta=0.1`，`beam_k=2`，PCC token 维度 `256`，四级 fusion gate logit 初值均为 `-4.0`；AdamW `weight_decay=0.01`、`betas=(0.9,0.999)`，warmup ratio `0.04`，cosine 终点 LR factor `0.01`。warmup 结束即进入 cosine，不额外保持平坦学习率。
- 主比较使用官方模型生成的固定 support、相同 2D 点与 RANSAC 设置；EXP021 B 或 C 的正式 comparator 在其结果完整后选择。EXP022 自身 mask 的 native-support pose 单独标为 supplemental，不与 fixed-support 主结果混用。

## 本地检查

在仓库根目录、Conda `pytorch22` 中执行：

```bash
python -m research.exp022.build_hierarchy --mode reused
python -m research.exp022.build_hierarchy --mode independent
python -m pytest -q research/exp022/tests
python -m research.exp022.preflight
```

层级文件写到 ignored dataset cache；已存在时生成器拒绝覆盖。可用 `GDRN_DATASET_CACHE_DIR` 指向独立 cache，正式训练容器内要使用同一数据源。配置不将 artifact 写入 Git。
服务器容器的 cache 环境变量指向 `/home/gdrn/.cache/gdrnpp_datasets`，对应项目宿主机 `/data/labs/<lab0或lab1>/docker_data/chx/cache/gdrnpp_datasets`。用户需在启动前把本地主层级 `reused_v1.npz` 传入该 cache 的 `exp022/`，或在容器中以已有 EXP021 cache 为输入生成；不要把 NPZ 放进 release 或 Git。独立消融仅需 `independent_v2.npz`。

有本地 CUDA/CPP 时再运行固定真实 batch 数步检查；服务器正式训练前需在项目容器里用 `--renderer egl` 复核，测得 step time、峰值显存、loss 和 AMP 跳步状态后再决定是否使用初始 batch 48：

```bash
python -m research.exp022.real_smoke --config smoke_reused.py --renderer cpp --device cuda:0 --batch-size 4 --steps 2
python -m research.exp022.real_smoke --config smoke_independent.py --renderer cpp --device cuda:0 --batch-size 4 --steps 2
```

正式训练配置：`configs/gdrn/lmo_pbr/research/exp022_progressive_pcc/train_reused.py`。训练入口、bundle/release 与容器限制遵循 [RUNBOOK](../RUNBOOK_CN.md) 和 [服务器安全](../SERVER_SAFETY_CN.md)；Agent 不连接服务器，服务器运行由用户执行。`OUTPUT_DIR` 必须由唯一 run ID 覆盖，不能沿用占位目录。

训练后在完整 LM-O 上执行 `research.exp022.matched_pnp_eval`：`--pcc-checkpoint`、`--output` 为必填；可同时传入 `--exp021-config` 与 `--exp021-checkpoint` 以加入选定 B/C 的 K=4 对照。输出目录必须不存在。`--limit` 只用于接线 smoke，完整运行可加 `--bop-eval` 生成 BOP AR/ADD(-S)/reS/teS。正式结果仅写入 [RECORD](../experiments/EXP-20260916-022-progressive-pcc/RECORD.md)。
