# EXP022 渐进式层级 CAD 对应与多尺度 PCC

## 2026-09-20 收口

Decision：`CLOSED / SUPERSEDED_BY_EXP025 / FORMAL_NOT_RUN`。服务器 EGL、正式训练和完整
matched PnP 均未执行，缺口保留。配置、runner 和专用测试已退出 HEAD；删除前最终完整
快照为 `8c6ca86aa0668777548d4f30b8cc6f6ad9864067`，历史复现使用独立 worktree。

## 2026-09-18 多数据集抽象回归（Observed）

统一 Dataset Context 与动态对象数接入后，旧 LM-O `train_reused.py` 的有效配置相对修改前只增加 `DATASET_CONTEXT`，其余字段逐项相同；原 `reused_v1.npz` 保持不变，8 个 BOP ID 顺序仍为 `[1,5,6,8,9,10,11,12]`。CPU preflight PASS：官方 backbone 340 个张量、可训练 PCC `3,728,456` 参数及有限 loss/梯度。使用旧保存的真实 PBR batch48（类别直方图 `[4,4,13,6,9,1,3,8]`）、seed 42、本机 CUDA+CPP、FP16 AMP 运行一步 PASS，无跳步；route/residual/mask loss `2.089756/0.458948/0.725601`，峰值 allocated/reserved `5.631780/6.494880 GB`。仅一步且含预热，不作为稳定耗时或正式结果。旧 LM-O 评价链只做配置/target 协议检查，正式 matched PnP 仍未生成。LM13 新协议另见 [EXP023 RECORD](../EXP-20260918-023-lm13-progressive-pcc-fulltrain/RECORD.md)。

- `experiment_id`: `EXP-20260916-022-progressive-pcc`
- 状态：`STAGE1_PERFORMANCE_REVISED / LOCAL_CPP_SMOKE_PASS / SERVER_EGL_PENDING / FORMAL_NOT_STARTED`
- 日期：2026-09-17（2026-09-16 起）
- 主训练 `run_id`：未分配；修订源码 commit：以本次 Git 提交为准；seed：42

## 问题、变量与范围

在冻结官方 ConvNeXt、RGB/PBR40/GT-box 不变条件下，以 8⁴ CAD 路由、多尺度 PCC、局部球形有界残差和显式 RANSAC-PnP，测试是否较 EXP021 的 64→64 层级降低切向/重投影 correspondence error 并改善 matched pose。主变量是完整 EXP022 PCC 结构。`independent_v1.npz` 是层级来源消融，只做实现 smoke；本阶段不进入全量 backbone 训练。

主臂 `train_reused.py`，正式训练 40 epoch、batch 48、16 workers、AdamW `3e-4`、4% warmup 后 cosine、AMP FP16，E5/E10/E15/E20/E25/E30/E35/E40 为预定正式评估点。EXP021 comparator 选择待其结果完整后确定，不据现有 E15 指标事后指定。

2026-09-16 正式训练前协议修正：先前本地 smoke 的对称分支选择分数包含 mask loss；正式 V1 改为仅以加权 route CE 与 residual loss 选择等价 SE(3) 分支，选定后该分支的 route、residual、mask 三项 loss 均参与训练。既有本机性能与 loss 观察保留为修正前代码的工程记录，不冒充修正后的正式结果。其余 V1 固定值已在正式配置中明确：三项 loss 权重均为 `1.0`，residual beta `0.1`，beam K `2`，token 维度 `256`，四级 fusion gate logit 初值 `-4.0`；AdamW `lr=3e-4`、`weight_decay=0.01`、`betas=(0.9,0.999)`，warmup ratio `0.04` 后立即 cosine，终点 LR factor `0.01`，seed `42`。本次不调整这些值。

## 评价与 gate（预注册）

- Integrity：层级 8⁴/对象顺序/叶子映射、官方 340 个 backbone 张量加载、冻结/梯度隔离、四级概率归一化/路径合法、残差半径、可见 mask、GT 几何和 fixed support 评价链。失败则不能进入机制判断。
- Mechanism：四级 GT-parent/top1/top2 路由、错误累积、叶子误差、对称等价下的 correspondence mean/median、normal/tangent、重投影，以及同一固定支持集的 RANSAC solve/BOP/ADD；比较 EXP021 选定臂和官方 A。文档要求切向与重投影改善方向一致、PnP 能稳定消费、Beam K2 有恢复效应；未给硬性数值阈值，不能事后用单次细小差异宣称稳定机制成立。
- Advisory：每项 loss/梯度尺度、gate collapse、batch48 显存/吞吐、native-own-mask 结果；资源不可接受时不启动正式训练，先记录测量和处理。native support 不替代主 fixed-support 结论。
- Reproduction：与 EXP021 相同官方 checkpoint、数据/GT box、RANSAC seed/threshold/iterations 和固定 support。EXP021 B/C 源码、checkpoint、epoch 与指标需在正式比较时明确记录。

## Observed（本地工程检查，不是正式科学结果）

- `reused_v1.npz` 已生成至 `.local/dataset_cache/exp022/`，不进入 Git。主层级每对象四级节点数为 8/64/512/4096；主层级 4096 个原叶子索引均唯一，叶子 anchor 保持 EXP021 数值。独立采样初版 `independent_v1.npz` 存在 level4 零半径，未通过 loader；文件保留于 ignored cache、不用于实验。修复后生成 `independent_v2.npz`，单点 cell 半径下限 `1e-5 m`。
- EXP022 单元/配置测试：9 passed；完整 `pytest -q research`：197 passed。主臂与独立消融臂官方 backbone CPU preflight 均 PASS，加载 340 个张量，仅 PCC 的 3,684,168 参数可训练，三项 loss finite，反传/优化步和推理形状通过。该 CPU synthetic 不是实际数据性能。
- 本机 RTX 4060/CUDA+CPP 真实 PBR batch 4，两步 AMP smoke PASS，无跳步；loss route `2.1083→1.7396`、residual `0.4414→0.4470`、mask `0.7184→0.4296`；固定 batch 的两步 optimizer 时间 `1762.2/168.1 ms`，峰值 allocated `0.8862 GB`。首步含初始化/预热，不能将两步中位数视为稳定训练吞吐。
- 同机 batch 48 两步 AMP smoke PASS，无跳步；loss route `2.1264→1.8956`、residual `0.4757→0.4927`、mask `0.7648→0.3478`；固定 batch 的两步时间 `1600.3/669.5 ms`，峰值 allocated `3.6321 GB`。这不包含每步 DataLoader 和 renderer，也不是服务器 EGL 总耗时。
- 独立的本机 batch48 同一真实 batch 10 步 AMP 检查 PASS，无跳步；第 2–10 步固定 batch 优化时间中位数 `660.1 ms`（范围 `650.9–667.5 ms`），峰值 allocated `3.6342 GB`。loss route `2.1191→1.5583`、residual `0.4635→0.4249`、mask `0.7253→0.1029`。重复同一 batch 只作优化/性能检查，不代表全数据收敛，也不含每步加载与渲染。
- 提交 `2ad21a1` 后增加分段计时、参数与类别计数的本机诊断；batch48 固定真实 batch 10 步 AMP PASS。模型参数共 `91,248,584`，其中 frozen backbone `87,564,416`，trainable PCC `3,684,168`。第 2–10 步中位数：forward `349.85 ms`、backward `282.02 ms`、unscale+逐参数 finite 检查 `7.97 ms`、optimizer `1.45 ms`；整步中位数 `640.30 ms`，峰值 allocated `3.6353 GB`。另一 batch48 三步采样中，对称对象 `13/48`；其第 2–3 步 forward/backward 均值约 `351.63/296.47 ms`。不同 batch 的对称占比与内容未严格匹配，不能把两次耗时差归因于某一项。
- 2026-09-16 本地匹配性能对照：同一 RTX 4060、CUDA+CPP、seed 42、同一类别直方图 `[8,6,8,6,5,4,6,5]`（对称实例 `10/48`）、batch48、固定真实 batch 各 6 步 AMP。原实现第 2–6 步整步中位数 `664.30 ms`，forward/backward 中位数 `354.15/299.97 ms`，峰值 allocated `3.6429 GB`；静态 16 像素分组 packing、单次输出重排、共享 Stage-1/token 编码及按分辨率集中准备标签后，对应为 `490.38 ms`、`253.05/225.30 ms`、`3.6065 GB`。整步差值 `-173.93 ms`，按原值计算下降 `26.2%`。两次均无 AMP 跳步；第一步 route/residual/mask loss 原实现为 `2.168833/0.456654/0.776884`，新实现为 `2.168796/0.456656/0.776884`。这只是同机固定 batch 工程对照，不含 DataLoader/renderer 的逐步耗时，不是服务器 EGL 或正式训练吞吐。
- 此次工程改动后，EXP022 单元/配置测试 `12 passed`，完整 research 回归 `200 passed`；官方权重 CPU preflight PASS，加载 backbone 340 个张量，可训练参数仍为 `3,684,168`。匹配器旧分组实现与新实现的 FP32 logits/context/label/输入和 token 梯度有数值对照测试（含稀疏 route）；空 visible mask 与对称分支 Stage-1 单次执行有单元检查。本机主层级 CUDA+CPP batch4 两步 AMP smoke PASS，无跳步，第二步固定 batch 耗时 `138.52 ms`，峰值 allocated `0.9934 GB`。独立层级 CUDA+CPP batch4 两步 AMP smoke 同样 PASS，无跳步，第二步 `131.60 ms`、峰值 `0.9934 GB`。
- 正式训练前的对称分支选择修正后，EXP022 测试 `14 passed`、完整 research 回归 `202 passed`。新增检查覆盖 mask loss 不参与分支选择、选定分支 mask loss 的梯度，以及上述正式 V1 固定参数。本机 CUDA+CPP、seed 42、batch4 含 `2/4` 个对称实例的两步 AMP smoke PASS、无跳步；第二步 route/residual/mask loss 为 `1.720196/0.412373/0.375587`，峰值 allocated `0.9934 GB`。这是修正后本机固定 batch 检查，不是服务器 EGL 或正式训练结果。
- 独立消融 `independent_v2.npz` 的本机 CUDA+CPP batch4 两步 AMP smoke PASS，无跳步；第二步固定 batch 优化耗时 `195.2 ms`，峰值 allocated `0.9977 GB`。不进入正式 40 epoch。
- LM-O 单目标 fixed-support evaluator 接线 smoke COMPLETE：官方 A 和随机初始化 PCC 均生成固定支持集 correspondence、原生分辨率四级路由、PnP、own-mask supplemental 与计时字段；随机 PCC 数值不进入科学结论。最终接线输出在 ignored `output/experiments/exp022-evaluator-random-wiring-smoke-3/`；第一次索引错误的失败产物和第二次旧口径输出均保留，不参与正式结果。
- 服务器 CUDA/EGL 真 batch、稳定 batch48 profile、正式训练、完整 matched PnP/BOP：未运行或未生成。所有 E5–E40 指标：未生成。正式 checkpoint/运行目录：未生成。

## 2026-09-17 第一阶段网络结构重构（Observed / Derived）

- 在首次 formal 之前，将四级 image token 改为 S1/S2 全局 self-attention、S3/S4 的 8×8 window + shift=4 self-attention；均为 PyTorch `nn.MultiheadAttention`、8 heads、Pre-Norm、`need_weights=False`、无 FFN。各级 CAD matcher 增加独立 Q/K/V/out 投影，保持局部 8 路 raw logits、packed route、soft context fusion、GT-parent 监督和 Top-2 beam。S1 使用无分组 root fast path；S4 residual 使用融合后特征的同一 image projection。hierarchy 生成器、`reused_v1.npz`、symmetry、loss、optimizer、PnP 未改。旧 PCC head checkpoint 与新层参数不兼容；无正式 EXP022 checkpoint 需要迁移。
- 方案原文要求固定宽度 beam 的 Top-2 与全部路径穷举 Top-2 一致，但这在一般条件下不成立：上一层剪掉的较低概率 parent 若有集中的 child，仍可能产生全局最高的完整路径。按用户选择保留原 beam，测试精确性仅针对每一步已保留 parent 的 16 个候选。新增 top2 概率质量取剪枝前局部 8 路条件分布，不使用重归一化后恒为 1 的 beam 质量。
- 本地 `pytorch22`：EXP022 测试 `24 passed`，完整 research `212 passed`；官方权重 CPU preflight PASS，加载 340 个 backbone 张量，6,314,824 个 PCC 参数可训练，三项 loss finite、梯度与推理合同通过。`reused_v1` 的源叶子一一对应测试仍通过。新增测试覆盖 shifted-window wrap-around 隔离、Q/K/V 梯度、packed 与简单 gather 参考的 logits/context/labels/梯度、beam 保留候选精确性和 symmetry Stage-1 共享。
- 同一本机 RTX 4060 Laptop 8 GB、PyTorch 2.2.0、CUDA+CPP、FP16 AMP、同一保存的真实 PBR batch48（类别直方图 `[4,4,13,6,9,1,3,8]`，对称实例 4/48）、各 12 步并排除前 2 步，旧版 HEAD `15c5cbc` 与本次未提交重构版均 PASS，无 AMP 跳步。测量只覆盖重复固定 batch 的模型前后向、梯度检查和优化步；生成该 batch 的 DataLoader/renderer 不计入逐步耗时。ignored 原始日志和 batch 在 `.local/exp022-refactor-benchmark-20260917/`，不进入 Git。

| 本地固定 batch48 工程量 | 旧版 | 重构版 | 新−旧（Derived） |
|---|---:|---:|---:|
| 稳定整步中位数 | 462.083 ms | 710.912 ms | +248.829 ms（+53.85%） |
| forward 中位数 | 240.101 ms | 327.006 ms | +86.905 ms |
| backward 中位数 | 210.856 ms | 365.757 ms | +154.901 ms |
| 峰值 allocated | 3.605723 GB | 6.198546 GB | +2.592823 GB（+71.91%） |
| 峰值 reserved | 5.324669 GB | 6.977225 GB | +1.652556 GB（+31.04%） |
| 可训练 PCC 参数 | 3,684,168 | 6,314,824 | +2,630,656（+71.40%） |
| 模型总参数 | 91,248,584 | 93,879,240 | +2,630,656（+2.88%） |

- 独立层级 `independent_v2.npz` 的新结构本机 CUDA+CPP 真实 batch4、3 步 AMP smoke PASS、无跳步；排除首步后的两步中位数 `148.422 ms`，峰值 allocated/reserved `1.382921/1.522532 GB`。它仍只作实现 smoke，不进入正式 40 epoch。
- Interpretation：新增 image self-attention 与 Q/K/V 投影使本机固定 batch 资源明显上升；以上数值不外推为服务器 EGL 吞吐、正式训练资源 gate 或姿态精度。服务器真实 batch EGL、正式训练和完整 matched PnP/BOP 均未生成。

## Decision

第一阶段结构重构、本机 CPU 测试及 CUDA+CPP 工程对比完成；正式训练启动仍需服务器真实 batch CUDA/EGL smoke 和 batch48 时间/显存检查。未作机制通过或失败结论。

## 2026-09-17 性能审查后的四项修订（Observed / Derived / Decision）

- 依据 `EXP022_performance_review_and_modification.md` 的最终优先顺序，仅实施 Stage Transition、shifted-window mask、正式路径诊断开关和 matcher `out_proj` 移除。源码基准 `8b213e4fc384eece4d08609ba1842c1edf46cc34`；本节对应的最终源码 commit 以本次 Git 提交为准。运行 ID：`RUN-20260917-exp022-local-performance-review-s42`。正式训练 run ID、checkpoint 和 E5–E40 指标仍未生成。
- Transition 将三处旧 residual CNN 改为双线性上采样加 1×1 Conv，减少 2,324,224 参数。这是 formal 前的架构修订，输出与旧 block **不数值等价**。Shifted-window 使用原 MHA Q/K/V/out 权重与广播布尔 mask 的 SDPA；FP64 CPU reference 检查输出、输入梯度和 MHA 参数梯度与旧 MHA 一致，wrap-around 隔离测试通过。调试诊断现在按需开启；正式训练默认不计算各 stage 的 entropy、top1/top2 或 fusion update 比率。Matcher 移除四个 256×256 `out_proj`，减少 262,144 参数；两个连续线性层可折叠的函数空间相同，但随机初始化和优化轨迹不同，未宣称旧 checkpoint 数值等价。
- 同一台 RTX 4060 Laptop GPU、PyTorch 2.2.0、官方 340 个 backbone 张量、固定保存的真实 PBR batch48（类别直方图 `[4,4,13,6,9,1,3,8]`，对称实例 4/48）、seed 42、CUDA+CPP、FP16 AMP、12 步且排除前 2 步。每阶段重新以相同 seed 初始化 PCC；架构变化后不存在可共用的完整 PCC checkpoint。每步不计 DataLoader/renderer；各阶段原始日志及 batch 位于 ignored `.local/exp022-performance-review-20260917/` 和 `.local/exp022-refactor-benchmark-20260917/`。全部阶段训练及推理检查 PASS，AMP skipped steps 为 0。BASE 为本次固定 seed 重新测量，不能将旧记录的 710.912 ms 与本表各行当作同一次 A/B 对照。

| 阶段 | PCC 参数 | forward 中位数 ms | backward 中位数 ms | 整步中位数 ms | 峰值 allocated GB | 峰值 reserved GB |
|---|---:|---:|---:|---:|---:|---:|
| BASE（诊断开） | 6,314,824 | 435.75 | 491.13 | 949.30 | 6.198546 | 6.977225 |
| P0 Transition（诊断开） | 3,990,600 | 327.66 | 349.85 | 693.59 | 5.944746 | 6.614417 |
| P0 Shift mask（诊断开） | 3,990,600 | 301.93 | 325.77 | 642.49 | 5.807121 | 6.360662 |
| P0 Diagnostics off | 3,990,600 | 294.46 | 325.12 | 636.66 | 5.807111 | 6.360662 |
| P1 Matcher out_proj removed | 3,728,456 | 294.07 | 322.43 | 633.88 | 5.669223 | 6.712984 |

- Derived（最终−BASE）：PCC 参数 `−2,586,368`（`−40.96%`），整步中位数 `−315.42 ms`（`−33.23%`），forward `−141.68 ms`、backward `−168.70 ms`，allocated `−0.529323 GB`（`−8.54%`）。最终 reserved 虽低于 BASE，却高于上一阶段；单次 PyTorch 缓存保留峰值有波动，不将逐阶段 reserved 变化归因于单项代码。阶段间训练 loss 不直接作为科学质量比较，因为 PCC 架构和初始化函数已变。
- 独立 shifted-window 微基准：batch48、256 维、8 heads、FP16 AMP、同 seed 权重/输入，7 次前反向排除前 2 次。Stage3 32×32 旧 MHA mask→广播 SDPA：forward `10.61→8.37 ms`、backward `16.57→13.48 ms`、peak allocated `0.575→0.498 GB`；Stage4 64×64：`39.15→32.68 ms`、`63.72→53.60 ms`、`2.242→1.934 GB`。该微基准只含一个 shifted block，不等于完整训练 stage 耗时。
- `pytorch22` EXP022 测试 `27 passed`，完整 `pytest -q research` `215 passed`；CPU preflight PASS，官方 backbone 340 张量，最终 PCC 3,728,456 参数，loss finite。最终 batch48 CUDA+CPP/AMP 真实 batch smoke PASS，推理 XYZ finite、残差界通过、无 AMP 跳步。Hierarchy artifact 与生成器、geometry-adaptive partition、fragment adjacency 均未修改。
- Review 文档的“Transition 不改变语义”只适用于研究流程，不适用于逐点函数值；旧重型 block 与 1×1 Conv 不等价。删除 `out_proj` 只保证线性层可折叠的表达能力，不保证相同随机初值/训练轨迹。后续 Stage4 token 复用将改变 residual head 的输入定义；双 LayerNorm 清理也改变归一化位置。两项不应描述为数值无损优化。本轮未实施 active-parent K/V、Stage4 token 复用、推理 Q 去重、block size 调优或 P2 修改，等待下一轮独立选择和验证。

Decision：四项工程修订及本地检查完成。服务器 EGL profile、正式训练、完整 matched PnP/BOP 与机制 gate 仍待执行；本次固定 batch 的资源收益不外推至服务器吞吐或姿态精度。

## 2026-09-17 正式冻结前 symmetry 数量检查

- Observed：EXP022 V1 训练只计算 canonical 与固定的第二个 equivalent 分支；原 loader 虽读取 `symmetry_counts`，却未拒绝计数超过 2 的层级。当前本地主层级 `reused_v1.npz` 和消融层级 `independent_v2.npz` 的计数均为 `[1,1,1,1,1,2,2,1]`，最大值 2；对象顺序为 `(1,5,6,8,9,10,11,12)`。这证明当前两份本地 artifact 满足 V1 边界，不证明未来数据集也满足。
- 修改：loader 检查计数和变换数组形状、计数范围，并在最大计数 `>2` 时以明确错误拒绝；preflight JSON 输出完整计数与最大值。构造 `count=3` 的层级文件，单测确认 loader 报错。未改对称分支算法、层级生成器或 artifact。
- 验证：`pytorch22` 下 EXP022 `28 passed`、完整 research `216 passed`；主臂 `train_reused.py` 与独立消融 `smoke_independent.py` 的 CPU preflight 均 PASS，报告最大计数 2、官方 backbone 340 张量、PCC 参数 3,728,456。未运行服务器或正式训练。本次是数据适用性检查，不是多对称分支支持；T-LESS 等有更多等价姿态的数据仍需独立设计与验证。

## 2026-09-19 CAD 公共模块与全层 sanity 纠正

### Observed：修改与历史输出

基线 source 为 `86da1174b67a954fcdc6bd592ce45aa28784c83d`；修改后为该 commit
上的工作树，经用户确认后作为单次提交纳入 Git；本轮不启动训练。新增公共 CADHierarchy loader、NumPy
几何及层级诊断；builder 与 oracle CLI 调用公共实现。旧 PCC 只改 loader 适配器，
整个 ProgressivePCCHead 类与 HEAD 的 AST 精确一致；正式配置、训练 launcher、
DatasetContext、PnP/BOP 链路未改，consistent_v3 仍被旧模型拒绝。

现有 `output/diagnostics/exp022_t3_residual_oracle_full/hierarchy_sanity.json`
将 reused_v1 标为 PASS；这是原程序只判断 T3→T4 的输出，原文件保留。
本轮用各层原始 anchors/radii 复算
`parent_radius / max_j(norm(child_anchor-parent_anchor)+child_radius)`，
全部关系的 `<1` 节点数如下（按 T1→T2、T2→T3、T3→T4 排序）：

| artifact | `<1` 节点数 | 修正后静态 sanity |
|---|---|---|
| reused_v1 | 0 / 512 / 0 | FAIL；failed_relations 包含 level2_over_level3 |
| independent_v2 | 64 / 512 / 4084 | FAIL |
| consistent_v3 | 0 / 0 / 0 | PASS |

这只纠正静态检查，不覆盖历史 evaluator decision，也不改变第五轮 depth/render
两族原阈值。原 full summary 中 consistent_v3 的 depth GT 为 FAIL、rendered
surface GT 为 PASS，二者继续并列保留。表面 global_coverage 的实际定义为
“全局最近锚点对应球覆盖”，不是所有球并集；本轮保留原字段与数值，补充描述。
公共采样报告新增未命中节点数，但不把未命中判为空节点，不新增科学 gate。

### Observed：本地工程与数值回归

以下 run 标识直接对应唯一目录，均为 CPU、24 targets、无模型/训练/checkpoint，
`--hierarchy .local/dataset_cache/exp022/consistent_v3.npz --limit 24`，其余 CLI
默认参数不变（train_reused 配置、GT bbox、seed 20260730、max-correspondences=0）。
原始输出均位于 ignored `output/diagnostics/`：

- `refactor_baseline_20260919_a01`：修改前 source 86da117，退出码 0。
- `refactor_after_20260919_a01`：初次抽取后，退出码 0。
- `refactor_after_20260919_a02`：补齐通用报告后最终复核，退出码 0。

baseline 与最终 a02 的 `summary.json`、`per_object.json`、`per_target.jsonl`、
`residual_norm_histogram.csv` **逐字节一致**，因此包括逐 target support 计数、
两族 residual/correspondence/pose、聚合指标与 decision；无需使用浮点容差。
原有 hierarchy_sanity/coverage 字段值也全部精确一致，仅增加 failed_relations、
采样解释及未命中节点字段。完整 run metadata 包含时间、source 和路径差异，未要求相同。
本次没有重复全量 BOP、第五轮完整评价或正式 1M sample 建树，也没有覆盖任何旧产物。

- 公共模块测试：`29 passed`；涵盖合成结构、动态深度、对象顺序、非法数值、
  数据集身份、symmetry、路径/residual、全层 sanity、导入隔离、真实三棵树和旧 Torch 对照。
- 完整 `python -m pytest -q research`：`317 passed, 3 failed`。三个失败为
  `test_lm_real_loader_returns_bop_records`、`test_lm_imgn_loader_uses_the_render_conventions`、
  `test_benchviseblue_renders_map_to_the_benchvise_object`，均在 LM models 目录写
  `models_*.pkl` 时抛 `OSError: [Errno 30] Read-only file system`，与 EXP024
  已记录的环境问题一致。没有改外部数据权限或 loader 来绕过失败，不标全套 PASS。
- builder CLI `--help` 通过。相同 seed=17 的 100,000 个合成均匀立方体点，
  与 HEAD 原 builder 比较，四层 anchors、walk 与退化清单精确一致。
  初次检查错误地要求该小样本无退化而失败；复核发现两版本均有 8 个退化 cell，
  因此该项只验证函数等价，不标为成功完整建树，不输出 artifact。
- `--hierarchy-only` CLI 实测：reused_v1 返回 FAIL / exit 1，consistent_v3
  返回 PASS / exit 0；保留原命令行退出语义。`git diff --check` 通过。

### Decision 与证据边界

本轮轻量整理完成，API 与使用边界见 `research/cad_hierarchy/README.md`。
旧训练路径隔离保留。T3+residual 是后续候选，surface oracle 不证明网络路由与
残差可学习性；不创建 EXP025、不追加训练或 seed。既有 NPZ 与第四/第五轮原始
输出继续留在原位置，本轮不扩展为常规源码快照或哈希链。
