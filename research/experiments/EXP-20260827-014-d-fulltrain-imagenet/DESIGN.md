# EXP014-D 实验设计：ImageNet 全量端到端训练

## 目标

此前 EXP012/EXP013A/B/C 全部是"冻结主干+几何头、只训新姿态头"的变体，从未在
官方完整协议下检验过新头。本实验复刻官方 GDRNPP 训练协议（ImageNet 预训练
ConvNeXt-Base + 随机初始化几何/姿态头 + 全量解冻），回答：

1. 新头（`RTDecoupledGeometryPnPNet`）能否达到甚至超过官方 ckpt 的水平
   （官方 ckpt GT 框直评 0.690399，参考点）；
2. 全量训练相比冻结特征版本（EXP013A E40 0.683956）的增量有多大，间接检验
   "冻结特征是否才是瓶颈"。

## 协议

| 项 | 取值 |
|---|---|
| 主干 | timm `convnext_base`，ImageNet-1k 预训练（`convnext_base_1k_224_ema.pth`，~330MB） |
| 姿态头 | `RTDecoupledGeometryPnPNet`（继承 C），geometry_scale_r/t init 0.1 |
| 初始化 | `MODEL.WEIGHTS=""`：主干 timm 预训练，几何头/姿态头随机初始化 |
| 解冻 | BACKBONE / GEO_HEAD / PNP_NET 全部 FREEZE=False |
| 几何监督 | `GEO_HEAD.TRAIN_SUPERVISION=True`（解冻几何头的守卫要求，GDRN_double_mask.py:119-124） |
| 渲染器 | 训练 `egl`（服务器）；本地 smoke/audit48 用 `cpp`（WSL 无 GL） |
| 其余 | 40ep / flat_and_anneal / Ranger lr 8e-4 / WARMUP_ITERS=1000 / batch 48 / 每5ep评估 / GT 框 BOP19 / seed 42 |
| 决策点 | 固定 `epoch_040`（`model_best` 有已知 bug，一律以 epoch_040 为准） |

## 预注册决策规则

- 主对照：D E40 vs 官方 ckpt GT 框直评 **0.690399**；
- 次对照：vs EXP013A E40 **0.683956**（冻结特征 + 新头）；
- 参考：EXP005/B 0.691912、C2 0.693006、EXP012 0.678800；
- **SCREEN 规则**：D < 0.686 时**先查环境**（timm 权重是否真加载、egl 渲染器是否
  生效、解冻梯度是否真的在更新——preflight 已断言三桶梯度非空），不直接判结构失败；
- seed 42 首轮；正式结论前补第二 seed；
- 渲染器预案：egl smoke 失败则降级 cpp，必须在 RECORD/结果中记录所用渲染器。

## 风险与预案

1. **训练渲染器（egl）首次上服务器**：服务器先跑模式= smoke 的 egl 渲染器 +
   前反向验证，PASS 后才 formal；不过则报告，不静默绕过。
2. **服务器离线（连不上 GitHub/Gitee）**：bundle 交付；ConvNeXt ImageNet 权重
   也一并复制到 `E:\6D姿态估计\EXP-014\`，放入容器 torch hub 缓存，
   避免训练启动时 timm 在线下载失败。
3. **随机初始化头 + 解冻的早期不稳**：WARMUP_ITERS=1000（覆盖 A 的 200）。
4. **95M 参数对 LM-O PBR 过拟合风险**：与官方同配置一致，属可接受范围，以
   预注册 SCREEN 规则判定，不凭中途情绪停止。

## 回退最小化（本实验的全部文件清单）

新增（删除即回退，不含任何既有文件修改）：
- `configs/gdrn/lmo_pbr/research/exp013/d_fulltrain/{train,smoke,audit48,eval}.py`
- `research/exp014/preflight.py`
- `research/experiments/EXP-20260827-014-d-fulltrain-imagenet/{EXPERIMENT.json,DESIGN.md,LOCAL_GATE.json,RECORD.md}`

既有文件的修改仅为注册表**逐行追加**：`research/experiment_index.json`、
`research/EXPERIMENT_INDEX.md`、`research/STATUS_CN.md`、`research/RUNBOOK_CN.md`。

分支：`exp014-d-fulltrain`（`git branch -D exp014-d-fulltrain` 即可整体回退）。
