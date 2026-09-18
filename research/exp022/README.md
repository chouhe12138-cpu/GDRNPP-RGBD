# EXP022 — 渐进式层级 CAD 对应与多尺度 PCC

## 多数据集配置（2026-09-18）

旧 LM-O 训练配置仍在 `configs/gdrn/lmo_pbr/research/exp022_progressive_pcc/`，使用原始 `reused_v1.npz` 和冻结的官方 ConvNeXt。其方法参数现在引用 `configs/gdrn/research/exp022_progressive_pcc/method.py`；有效配置仅增加 `DATASET_CONTEXT`，不更改 PCC 数学结构。

新实验从 `configs/gdrn/research/exp022_progressive_pcc/common.py` 继承公共模型，从 `configs/gdrn/research/_base_/research_runtime.py` 继承可调整的训练协议。LM13 已按 GDR-Net 的 LM 协议拆成三条**显式**协议，不再由一个配置代表三种训练定义：

| 配置 | 协议 | 训练数据 |
|---|---|---|
| `train_lm13_gdrn.py` | `lm13_gdrn` 主实验 | `lm_13_train_online` + `lm_imgn_13_train_1k_per_obj_online` |
| `train_lm13_real_only.py` | `lm13_real_only` 数据消融 | `lm_13_train_online` |
| `train_lm13_pbr.py` | `lm13_pbr` BOP/PBR 域 | `lm_pbr_13_online_train` |

主实验与 real_only 共用 `lm13_gdrn_protocol.py`（160 epoch、effective batch 24、Ranger 1e-4、
`COLOR_AUG_PROB=0`、VOC 背景 0.5、DZI 1.5/0.25/0.25、cosine 终点 `TARGET_LR_FACTOR=0`）；
主实验每 20 epoch 做一次 GT-bbox RANSAC-PnP 诊断评估（`EVAL_PERIOD=20`），测试集用
`lm_13_test_online` 以免依赖评估不需要的预生成 `xyz_crop`。协议文件放在实验目录而不是 `_base_/`，
因为 mmcv 不允许兄弟 base 之间出现重复键。`smoke_lm13_gdrn.py` / `smoke_lm13_pbr.py` 是各自的
小样本 smoke。`tless_reserved.py` 只保留配置接口，待数据和 variable-S 对称监督就绪后再检查或训练。
LM real 与 lm_imgn 的 loader 是 `core/gdrn_modeling/datasets/lm_dataset_d2.py` 与 `lm_syn_imgn.py`；
LM real 从 `lm/test/{scene}` 按 `image_set/{obj}_{train,test}.txt` 取片（`lm/train/` 是便利副本，
与官方划分并不完全一致）。

`DATASET_CONTEXT` 集中定义数据集身份、CAD ref、BOP 名称和目标文件；`resolve_dataset_context` 从注册的 train/test metadata 取得对象顺序和真实 BOP ID，并校验 hierarchy 行顺序。新增数据集需注册 split、配置 CAD ref、生成 hierarchy 并指定对应 reference 模型。LM13 当前没有 matched reference checkpoint，因此 matched evaluator 仅完成协议验证，尚未产生正式指标。

在仓库根目录激活 Conda `pytorch22` 后，LM13 本地检查入口为：

```bash
export GDRN_CONVNEXT_BASE_WEIGHTS=/path/to/convnext_base_1k_224_ema.pth
CFG=configs/gdrn/research/exp022_progressive_pcc/train_lm13_gdrn.py
python -m research.exp022.build_hierarchy --config $CFG --mode independent
python -m research.exp022.preflight --config $CFG
# 逐条核对数据：位姿单位、相机、CAD 尺寸、mask/bbox 与在线渲染的一致性
python -m research.exp022.check_lm_data --config $CFG --renderer cpp --device cuda:0 --batch-size 4
python -m research.exp022.real_smoke --config configs/gdrn/research/exp022_progressive_pcc/smoke_lm13_gdrn.py --renderer cpp --device cuda:0 --batch-size 4 --steps 3
python -m research.exp022.matched_pnp_eval --config $CFG --validate-only
# 评估 provenance；正式评估前对每条协议各写一份
python -m research.exp022.eval_manifest --config $CFG --checkpoint <ckpt> --output <eval-dir>
python -m research.exp022.eval_manifest --config configs/gdrn/research/exp022_progressive_pcc/eval_lm13_bop.py --checkpoint <ckpt> --output <eval-dir>
```

层级生成器拒绝覆盖已有文件；首次生成后，保持配置中的 hierarchy 路径与产物一致。LM13 正式训练配置仍标记 `FORMAL_READY=False`，需要在确定完整训练、评估和对照协议后再启用。

服务器侧的资源契约（`lm_imgn` 只读 mount、`GDRN_CONVNEXT_BASE_WEIGHTS`、
profile-aware runtime gate、`server_preflight`、EGL smoke 与两段 release）见
[RUNBOOK](../RUNBOOK_CN.md) 的 “EXP023 LM13 server profile”。

## LM-O 第一阶段固定协议

旧 LM-O 第一阶段使用冻结的官方 RGB ConvNeXt 8×8 特征，训练四级 PCC、stage transition、局部残差和可见 mask。新 LM13 配置允许全量 backbone 训练；ResNet-50、部署、Geometry-adaptive partition 与 fragment adjacency 仍为 **DEFERRED**。

EXP024 是单独的 LM-O 全主干训练：保持 EXP022 的 PBR40、GT-box、`reused_v1.npz`、
PCC 方法与 E5–E40 评估点，改为 ImageNet ConvNeXt-Base 初始化并解冻 backbone。
`train_lmo_full_imagenet.py` 使用物理 batch 4、梯度累积 12 次、effective batch 48，
backbone 学习率为 PCC 的 0.1 倍。`smoke_lmo_full_imagenet.py` 用于短步 EGL smoke；
lab1 EGL/AMP 真实 batch4 smoke 通过后，正式配置已开启 `FORMAL_READY=True`，
正式训练仍须从解锁版的独立 release 启动。
这同时改变 backbone 初始化来源与冻结状态，因此与 EXP022 冻结臂的性能差不能
单独归因于“解冻”。事实和 gate 见 EXP024 RECORD。

当前结构：`1024→512→256→128→64` CNN 通道阶梯；stage 间为双线性上采样加 1×1 Conv。各级以 256 维 image token 做图像 self-attention，S1/S2 为全局 attention，S3/S4 各为 8×8 window + shift=4 window attention。每级仅匹配当前 parent 下 8 个 CAD child：独立 Q/K/V 投影得到 raw route logits 与 soft CAD context，前者训练 CE/更新 Top-2 beam，后者直接经 stage 的 `context_proj` 与小 gate 以残差方式回注 CNN feature。图像 attention 使用 Pre-Norm、8 heads、无 FFN；全局及普通 window 使用 `nn.MultiheadAttention(need_weights=False)`，shifted window 复用其 Q/K/V/out 权重并用广播 mask 的 SDPA。CAD 局部匹配保留显式 8-way logits 与 packed routes。最终 fused Stage-4 feature 经同一个 image projection 进入 leaf-local residual head；visible mask 来自 64 通道特征。

## 固定协议

- 主层级 `reused_v1.npz`：把 EXP021 的 64×64 叶子无损重排为 8⁴；叶子 anchor、法向、半径不变。消融层级 `independent_v2.npz` 从模型表面重新采样并递归分组；单点 cell 的半径下限为 `1e-5 m`，仅做 smoke，不安排 40 epoch。
- 四级分辨率 8/16/32/64，每级 8 路；训练按当前分辨率的 GT 父路径 teacher forcing，推理保留 Top-2 完整路径 ID 和归一化累计 log score。8 路条件概率仅用于 soft CAD context。跨尺度路由状态按双线性权重合并同 ID 路径后再取 Top-2。
- 固定宽度 Top-2 beam 每步精确选择当前保留父路径的最多 16 个 child 候选，但一般不等于对全部 8⁴ 路径的全局穷举 Top-2；本阶段不改变 beam 协议。
- 对称物体按完整 SE(3) 等价变换选择一个 instance-consistent 分支；选择分数仅为加权四级 CE 与加权 residual loss 之和，不含 mask loss。选定后该分支的四级 CE、residual 和 mask loss 一起训练。最终点为叶子 anchor 加不超过该叶子半径的 3D 残差。
- EXP022 V1 的训练分支只比较 canonical 与一个 alternate；层级 loader 要求每个物体有 1–2 个等价变换，超过 2 个时明确报错。preflight 输出各物体 `symmetry_counts`；将来扩展到更多对称变换需另行设计分支选择。
- PBR40、batch 48、40 epoch、16 workers、seed 42、AdamW `3e-4`、4% linear warmup + cosine、显式 FP16 AMP。正式配置每 5 epoch 做原有 direct-pose 评估，输出仍通过 explicit RANSAC-PnP。
- V1 固定值：route/residual/mask 权重均为 `1.0`，residual Smooth L1 `beta=0.1`，`beam_k=2`，PCC token 维度 `256`，四级 fusion gate logit 初值均为 `-4.0`；AdamW `weight_decay=0.01`、`betas=(0.9,0.999)`，warmup ratio `0.04`，cosine 终点 LR factor `0.01`。warmup 结束即进入 cosine，不额外保持平坦学习率。
- 调试模式可记录每级 fusion update / feature 范数比、可见位置的条件 route entropy、top1 概率及剪枝前 8 路 top2 概率质量；这些量 detach，不增加 loss。正式训练默认关闭逐级诊断，保留轻量的 fusion gate 与对称分支选择统计。
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

`real_smoke` 可用 `--save-batch` 将一次真实 batch 保存到 ignored `.local/`，再用 `--load-batch` 在修改前后重放同一批数据；`--seed` 固定随机初始化，`--warmup-steps` 指定不计入稳定时间中位数的前几步。`--diagnostics on` 为 smoke 默认值，`off` 测正式训练路径。输出包括 forward/backward、峰值 allocated/reserved 显存、参数量、loss、所选诊断和 AMP 跳步检查。固定 batch 计时不包含每步 DataLoader/renderer，也不是服务器 EGL 吞吐。

正式训练配置：`configs/gdrn/lmo_pbr/research/exp022_progressive_pcc/train_reused.py`。训练入口、bundle/release 与容器限制遵循 [RUNBOOK](../RUNBOOK_CN.md) 和 [服务器安全](../SERVER_SAFETY_CN.md)；Agent 不连接服务器，服务器运行由用户执行。`OUTPUT_DIR` 必须由唯一 run ID 覆盖，不能沿用占位目录。

训练后在完整 LM-O 上执行 `research.exp022.matched_pnp_eval`：`--pcc-checkpoint`、`--output` 为必填；可同时传入 `--exp021-config` 与 `--exp021-checkpoint` 以加入选定 B/C 的 K=4 对照。输出目录必须不存在。`--limit` 只用于接线 smoke，完整运行可加 `--bop-eval` 生成 BOP AR/ADD(-S)/reS/teS。正式结果仅写入 [RECORD](../experiments/EXP-20260916-022-progressive-pcc/RECORD.md)。
