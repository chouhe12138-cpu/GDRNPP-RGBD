# EXP026 — Residual-Aligned Full Sampling Ablation

## 新阶段协议（`SERVER_RELEASE_AUTHORIZED / BATCH48_GATE_PENDING`）

- 2026-09-23 用户明确授权启动 EXP026：lab0=`uniform_full`（普通采样），
  lab1=`adaptive_l1_full`（几何自适应采样）。EXP025 的两臂 E5–E40 已全部归档；
  其最终组合策略结论仍待用户判定，本次授权不构成 EXP025 选模或结论。
- release 配置已解除 `SERVER_RELEASE_ALLOWED` 和 `FORMAL_READY` 阻断；两臂真实
  batch48/EGL gate 尚未产生服务器证据。须分别得到 PASS 后才启动相应 formal。

- experiment ID：`EXP-20260922-026-residual-aligned-sampling-ablation`。
- 问题：在两臂都修复 predicted-route residual 监督、同为 ImageNet ConvNeXt Full training 的条件下，三层 GA-HFPS λ1 相对三层 uniform_512 对 correspondence 与 LM-O 6D pose 有何影响？
- 两臂唯一变量为 hierarchy artifact：`uniform_full` 使用下表 `uniform_512`，`adaptive_l1_full` 使用下表 `adaptive_512_l1`。旧 λ2 只留作离线设计证据，不进入 formal。
- 两臂共同使用 LM-O PBR40、GT box、seed42、真实 batch48、AdamW 3e-4（backbone/head 同 LR）、weight decay .01、40 epoch、4% linear warmup + cosine、AMP 初始 scale **16384**、E5/E10/E15/E20/E25/E30/E35/E40 固定评价、RANSAC-PnP；均从同一 ImageNet backbone 初始化重新训练，不从 EXP025 权重续训，也不选 best checkpoint。16384 是下述本地数值 gate 后两臂共同修订的初值，未来仍须服务器真实 batch48 重新 gate。
- 正式输出预定保留每个固定点的 BOP AR、ADD(-S)0.1d、AR_reS、AR_teS，及 predicted-cell representability、anchor/full XYZ error、residual gain。**不预设采样优胜门槛；最终研究判断由用户结合完整证据决定。**
- 配置位于 `configs/gdrn/lmo_pbr/research/exp026_residual_aligned_sampling_ablation/`。EXP025 两臂全部固定点评价入档、EXP026 本地 gate PASS、用户另行明确放行三项均已满足；本次准备 release，服务器 gate/训练状态仍待回填。
- 新 residual target 以 detached 预测 T3 argmax 的 anchor/radius 计算；GT route NLL 和历史 symmetry branch score 不变。仅可见且预测球可达的点参与 residual loss；默认无 warmup。

## 本地验证（Observed，2026-09-22）

- 本地准备代码基于 `8fb92422179f7aaa41cae5caa9fd919dc90ade3a` 的未提交工作树；本节不是可凭该 commit 单独复现的正式 run。本次未启动服务器、未提交或 push。
- 两份原始 NPZ 的 SHA 分别为 `35287beb3dc67b5f3cd376cc445f7727857aefa888dd11d639ab65b9193235c1`、`7ac75daa756ed7ae59463614737ccc46cd746173452cd1943a4a024d0817c631`；object IDs `1/5/6/8/9/10/11/12`，mode=`geometry_adaptive`、version=1、depth=3、counts=8/64/512、variant/λ 与配置一致，`hierarchy_sanity` 两臂 PASS。EXP025 仍由旧 SHA gate 锁定。
- 两臂 resolved config 差异只在 arm、output 和 hierarchy path/SHA/variant/λ；ImageNet Full、head/backbone LR `3e-4`、batch48、seed42、调度、mask、PnP、评价点、predicted-route residual mode 相同。CPU 构造、forward/loss/backward、残差梯度、ImageNet 权重 340 tensors 精确装载、strict state_dict 往返均 PASS。
- `pytorch22` 回归测试（EXP026、EXP025、hierarchy、launcher/run contract/active config）`94 passed`；额外公共研究与 EXP025 checkpoint 诊断测试 `43 passed, 4 skipped`（部分测试与前组重叠）。`bash -n`、Python compile、`git diff --check` PASS。旧 EXP025 `official_frozen` CPU preflight 在当前代码下仍 PASS，旧 hierarchy SHA `02ce090949bc40b2732417fec23984f3f748431098c5f67c853839d10ff1a373` 未放宽。
- 同一固定 LM-O PBR object 11（ROI class 6，1225 个可见像素）和本地 RTX 4060，batch1、CPP 生成目标。`uniform` 的 32768 8-step smoke PASS；`adaptive` 的 32768 run `exp026_adaptive_smoke_20260922_a01` 在第 4 步出现 `backbone.stages_0.blocks.0.conv_dw.bias` 非有限梯度，**不进入通过证据**，失败摘要见 [failure](evidence/adaptive_smoke_a01_failure.json)。因此两臂共同采用初始 scale 16384；两臂重新跑的 8-step AMP smoke 均无 skip/非有限梯度，backbone、classifier、residual predictor 更新和 model/optimizer/scaler checkpoint 往返 PASS，见 [uniform](evidence/uniform_smoke_a02.json) 与 [adaptive](evidence/adaptive_smoke_a02.json)。16384 只经本地 batch1 gate，不能替代未来服务器 batch48 gate。
- 同一批次、相同 seed42 初始权重、共同 scale16384、无 warmup、各 100 步 fixed-batch：结果如下。对象 11 有对称分支；XYZ 误差选与锚点最近的同一等价分支评估，两臂口径一致，仅用于工程可学习性，不是 BOP 姿态结果。原始步进记录见 [uniform](evidence/uniform_fixed_a01.json) 与 [adaptive](evidence/adaptive_fixed_a01.json)。

| arm | route loss 0→100 | predicted-cell 可达率 0→100 | residual 有效像素 0→100 | anchor/full XYZ@100 (mm) | residual gain@100 (mm) |
|---|---:|---:|---:|---:|---:|
| uniform | 12.844→0.701 | 1.06%→96.08% | 13→1177 | 4.891 / 4.380 | +0.511 |
| adaptive λ1 | 12.696→0.456 | 0%→98.29% | 0→1204 | 4.789 / 4.245 | +0.545 |

- adaptive 在 step0 无 residual 有效点，但 step1 有 21、step20 有 492、step100 有 1204；持续有效监督已建立，因此没有理由加入 warmup。两臂最后三个记录点都有正 residual gain，预测残差无近饱和像素，classifier 与 residual predictor 梯度/更新存在。此单批次不能比较两种 sampling 的泛化或正式精度，也不能把 adaptive 的小幅数值差当作胜出。
- 本地 gate：编号/证据迁移、单元测试、artifact identity、配置匹配、CPU、CUDA、fixed-batch 均 PASS；当时正式服务器仍由 `SERVER_RELEASE_ALLOWED=False` 和 `FORMAL_READY=False` 双重阻断。2026-09-23 用户放行后另在本记录顶部登记 release 状态；该历史本地检查不代替真实 batch48 server gate。用户未预设采样优胜门槛。

## Preliminary offline sampling evidence（原 EXP026）

以下为 2026-09-21 离线设计阶段的原始记录与当时决策，保留 run、源码和 artifact 溯源；当时建议的 `official_frozen` 后续方案已由上方新的两臂 Full 协议取代，不作为当前执行指令。

## Status / research question

- lifecycle：`LOCAL_OFFLINE_COMPLETE`；server training：`NOT_STARTED`
- 当时记录的 experiment ID：`EXP-20260921-026-geometry-adaptive-hfps`（历史离线 run 身份；当前 EXP026 ID 见上）。
- canonical local run ID：`RUN-20260921-ga-hfps-s20260919-a01`
- source commit：`d9b4ab95d431a40284036490e7a4911357a6ae61`。本地 artifact 在该提交前、同一份实现代码上生成；提交仅固化源码与测试，未改变算法或产物。本地诊断不是正式训练。
- 问题：在固定 T1/T2/T3=8/64/512、相同表面采样和最近 child route 下，按局部几何复杂度加权 FPS 能否改善 T3 复杂区域锚点覆盖？同时隔离旧 T4→T3 radius 回传的影响。

## Protocol and artifacts

- LM-O 八个物体：1、5、6、8、9、10、11、12；mesh 单位米。构造每物体面积均匀采样 1,000,000 点，seed `20260919 + object_id`，并追加 mesh vertices 拟合节点几何。
- face 复杂度：共享边 1-ring（含自身）的面积加权单位法向均值 `m`；`g=1-||m||²`，仅有效 face 参与物体内 Q95；`g_hat=clip(g/Q95,0,1)`。退化 face 排除，Q95 近零时退化普通 FPS。
- matched variants：`uniform_512`（λ=0）、`adaptive_512_l1`（λ=1）、`adaptive_512_l2`（λ=2）；每个 parent 的 FPS score 为 `min_distance × (1+λg_hat)`，区域始终由 parent-local 欧氏最近 child anchor 定义。
- T3 radius 用三层 traversal 的真实采样点和 vertices 直接拟合，margin 1.05、floor 1e-5 m；仅向 T2、T1 传播覆盖。新 artifact 无 T4；旧 EXP025 artifact/配置/SHA 不变。
- held-out 每物体重新面积采样 200,000 点，seed `20261019 + object_id`。复杂/平滑组按每物体 held-out `g_hat` 的 Q75/Q25 定义；八物体样本数相同，pooled 为等物体采样权重。连通性使用有效 mesh face centroid 经相同 T3 traversal 后，在共享边 face graph 上计数，仅作诊断。
- 本地原始 artifact、build stats 和完整比较 JSON：`.local/dataset_cache/exp026/RUN-20260921-ga-hfps-s20260919-a01/`，均不进入 Git。生成入口为 `python -m research.cad_hierarchy.build_geometry_adaptive`；比较入口为 `python -m research.cad_hierarchy.compare_geometry_adaptive`。

| artifact | SHA256 |
|---|---|
| `uniform_512.npz` | `35287beb3dc67b5f3cd376cc445f7727857aefa888dd11d639ab65b9193235c1` |
| `adaptive_512_l1.npz` | `7ac75daa756ed7ae59463614737ccc46cd746173452cd1943a4a024d0817c631` |
| `adaptive_512_l2.npz` | `b83d6334fd008c96aa4a9fec531f5ec04b43c15310ab222beb6cda997164d3ee` |

## Observed — integrity and T4 repair

- 三个 artifact 均为三层 8/64/512，generic loader 与 `hierarchy_sanity` PASS；构造中各层 empty node / duplicate-anchor node 均为 0。三者的 T1/T2、T2/T3 父子覆盖最小比值均为 1.05，`below_one=0`。物体 12 的 mesh 有 4 个退化 face，已排除在面积采样与复杂度归一化之外。
- 新 `uniform_512` 的 T1–T3 anchor 与锁定的 EXP025 `consistent_v3.npz` **逐元素完全一致**，可将旧/新 uniform radius 差异归因于 T4 回传，而非重新选点。旧 SHA256 仍为 `02ce090949bc40b2732417fec23984f3f748431098c5f67c853839d10ff1a373`。
- 旧四层与新三层 uniform 的 T3 radius 比值：4096/4096 个节点大于 1.001；中位数 1.3251、P95 1.5254、最大 1.7338。旧半径的扩大来自 T4 child-ball 覆盖回传；新 T3 直接按自身 traversal support 拟合。
- 本地 `pytorch22` 的 `research/cad_hierarchy/tests`：31 passed；未使用 GPU，未运行服务器训练或正式姿态评估。

## Observed — held-out T3 对比

所有距离和 radius 单位为 mm；归一化 residual 为 `distance/radius`。百分比均相对**新三层** `uniform_512`，不是旧四层 artifact。

| variant | global mean distance | complex Q75 mean distance | smooth Q25 mean distance | global mean radius | normalized residual mean | traversal coverage | extra components / occupied node |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform λ0 | 5.638 | 5.605 | 5.659 | 10.924 | 0.5160 | 99.9997% | 0.118 |
| adaptive λ1 | 5.436 (-3.58%) | 3.950 (-29.51%) | 6.313 (+11.56%) | 11.188 (+2.42%) | 0.4894 | 99.9996% | 0.152 |
| adaptive λ2 | 5.606 (-0.57%) | 3.503 (-37.49%) | 7.162 (+26.57%) | 12.002 (+9.87%) | 0.4711 | 99.9994% | 0.179 |

λ1 的逐物体 gate 相关变化及连通性：

| object | global mean distance | complex Q75 mean distance | smooth Q25 mean distance | extra components/node uniform→λ1 |
|---:|---:|---:|---:|---:|
| 1 | -4.99% | -26.80% | +17.61% | 0.088→0.090 |
| 5 | +2.13% | -24.52% | +4.55% | 0.084→0.117 |
| 6 | -8.05% | -36.47% | +5.70% | 0.088→0.107 |
| 8 | -1.87% | -27.24% | +14.11% | 0.094→0.115 |
| 9 | -7.77% | -24.30% | +5.89% | 0.084→0.100 |
| 10 | -4.70% | -33.80% | +20.35% | 0.086→0.117 |
| 11 | -1.34% | -30.52% | +9.45% | 0.076→0.119 |
| 12 | -4.84% | -31.83% | +15.80% | 0.342→0.456 |

完整逐物体 mean/median/P90/P95 距离、radius、normalized residual、coverage、连通分量和复杂度对节点支持样本数/radius 的 Spearman 关系列在本地 `offline_compare.json`。物体 12 的 fragmented-node fraction 从 0.268 增至 0.330；该诊断不参与在线 target，也不等同于连续表面拓扑证明。

## Gate, interpretation, decision

- 预设工程 gate：结构/父子覆盖 PASS、构造 empty/degenerate=0、复杂 Q75 平均锚点距离降低至少 10%、全局平均恶化不超过 5%。连通性工具另以 pooled extra components/node 同时增加至少 0.1 绝对值和超过 20% 相对值标记 warning。原 evaluator 对 λ1、λ2 均输出 `engineering_pass=True`、`connectivity_warning=False`；原始输出保留于本地 JSON。
- Derived：λ1 在八个物体的复杂 Q75 均改善 24.3%–36.5%，但平滑 Q25 均恶化；λ2 更偏向复杂区域，平滑代价和 radius 增长都更大。λ1 的 pooled connectivity 从 0.118 升至 0.152，未触发原 pooled warning，但物体 12 从 0.342 升至 0.456，是单物体不利信号；不能由 pooled PASS 掩盖。
- Mechanism：每个物体在占用的 T3 node 上，以 held-out 命中数作为 cell 表面积代理，计算 node 平均复杂度与命中数/radius 的 Spearman 相关；八物体相关系数的中位数，uniform 为 `+0.129/+0.061`，λ1 为 `-0.099/-0.223`，λ2 为 `-0.349/-0.426`。这支持复杂区域获得更小 cell 的趋势，但不是跨 seed 稳定性证明。
- Review / decision（2026-09-21）：**建议 λ1 仅作为将来的 official_frozen matched 训练候选**，不直接替换 EXP025 或认定姿态精度改善。λ2 不优先；物体 12 连通性及所有物体平滑区域代价应作为训练后的预定分层诊断。EXP025 formal 完成前不启动 EXP026 服务器训练。
- 后续最小 matched 设计：同一三层构造协议下，仅比较 `uniform_512` 与 `adaptive_512_l1`；相同 official_frozen 初始化、数据、seed、batch、optimizer、40 epoch 与固定 E5–E40 评价，仅 hierarchy artifact 不同。届时建立独立配置、SHA 注册及实验专用 head mode 验证，并重新通过真实 batch48 gate；不得放宽 EXP025 的旧 fail-closed 检查或复用其 gate 代表新 artifact。

## Evidence boundary

- 本结果是 mesh 表面覆盖的单 seed 离线机制证据，不是 BOP pose、模型可训练性或跨 seed 稳定性结果。节点 radius 大小直接影响 normalized residual；不能把其下降全部解释为锚点改善。
- held-out traversal coverage 接近 1 只代表有限表面样本；不证明连续表面或深度 GT 的完整覆盖。face connectivity 是 centroid 离散诊断，不用于 target 定义。
- 新 artifact 尚不满足 EXP025 的 `consistent_v3` mode/version/SHA，故不能经当前 formal 配置加载；此 fail-closed 边界未放宽。
