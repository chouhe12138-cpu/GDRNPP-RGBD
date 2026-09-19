# 当前研究状态

最后核对：2026-09-19。

## EXP024 LM-O ImageNet 全主干训练准备（2026-09-19）

用户指定在空闲 lab1 上训练 LM-O 的全量解冻 ConvNeXt + PCC，主干从 ImageNet
ConvNeXt-Base 权重初始化。新实验沿用 EXP022 的 PBR40/GT-box、`reused_v1.npz`、
PCC 方法和 E5–E40 评价点；物理 batch4 累积 12 次至 effective batch48，40 epoch、
FP16 AMP。独立实验 ID 与 launcher profile 已准备。本地 CPU preflight 验证 340 个
ImageNet 主干张量及 `91,292,872` 个可训练参数；本会话本地 CUDA 不可用。用户在
lab1 准备版 release 上完成 CPU preflight 与 EGL 真实 batch4、8 步 AMP smoke，
两项 PASS、无跳步、峰值 allocated/reserved `2.826/3.012 GB`；固定批次计时不含
逐步加载/渲染。现已按用户确认开启 `FORMAL_READY=True`，等待第二段 release 与正式
启动；目前没有 formal run 或精度结论。初始化来源和冻结状态同时变化，不能把 EXP024 与
EXP022 的差值解释为单独的解冻效应。详见 [EXP024 RECORD](experiments/EXP-20260919-024-lmo-progressive-pcc-fulltrain/RECORD.md)。

## EXP023 LM13 数据/训练/评估协议重整（2026-09-18）

按用户提供的 `GDRNPP_LM_dataset_training_evaluation_adjustment_plan.md` 做**协议层整理**，
不动 `GDRN_PCC` / `ProgressivePCCHead` / `DatasetContext` 的模型结构。原 `train_lm13.py`
名义上是 LM13、实际数据是 `lm_pbr_13_online_train`（GDRNPP 的 PBR/BOP 协议），现在拆成
`lm13_gdrn`（主实验，LM real + lm_imgn）/ `lm13_real_only`（数据消融）/ `lm13_pbr`
（BOP/PBR 域）三条显式协议，模型与在线 XYZ 路径不变。

**数据**：用户提供的 `lm_imgn.zip`（DeepIM 的 OpenGL 渲染集，1k/obj）已解压到
`/home/wsluser/Datasets/lm_imgn`（`datasets/lm_imgn` 软链接，505,191 文件 / 6.1 GB）。
新增 `lm_dataset_d2.py`（LM real，从 `lm/test` 按 `image_set/{obj}_{train,test}.txt` 取片）
与 `lm_syn_imgn.py`，逐类计数与官方划分一致：train `2,375`、test `13,425`、渲染 `13,000`。
核实到 **`lm/train/` 不等于 image_set 选片**（benchvise 少 891/892 两张），因此不采用该目录。
`xyz_crop_imgn/` 实际存在，但主实验仍走在线路，预生成的 xyz 留作交叉核对。

**修复**：`img_type` 背景替换改为显式分流（`syn/real/syn_pbr` 各自策略，未知类型报错），
两处 mapper 共用同一策略函数且不改变 RNG 流；修复 `COLOR_AUG_SYN_ONLY` 因两支都调用
`_color_aug` 而完全失效的问题；去掉 `lm_pbr.py` 残留的 `ref.lm_full`；`DatasetContext`
支持多训练 split 并要求对象顺序一致。

**验证**：CPU preflight 三个配置全 PASS（340 张量、`91,292,872` 参数、协议校验）。
`check_lm_data.py` 数值检查：相机与标准内参误差 `0.0 px`、`bbox` 与 mask 误差 `0.0 px`、
`bbox3d` 与 `models_info` 尺寸精确一致、平移为米；在线渲染与源 mask 重合 `0.954–0.989`。
`real_smoke` 三步 AMP PASS、无跳步；真实训练入口 `main_gdrn.py` 用 smoke 配置跑完 1 epoch
并保存 checkpoint（`RUN-20260918-lm13-gdrn-smoke-s42`，退出码 0）。完整回归 `243 passed`。

**P2**：只做配置与 manifest 接线（新增 `eval_lm13_bop.py`、`eval_manifest.py`，
`matched_pnp_eval` 完成时写 `eval_manifest.json` 并标为 diagnostic）。**未运行完整评估**。
两条评估链当前都只能用 GT bbox —— 官方 `bbox_faster_all.json` 全盘缺失（需从 GDR-Net
README 的 `image_sets`/`test_bboxes` 网盘包补齐），所以 Protocol A/B 是诊断口径，
不能与 GDR-Net 论文的 detector-bbox 数字直接比较。本轮**没有精度结论**，正式训练未启动。

**正式训练前收口（2026-09-18 同日）**：按
`EXP022_LM13_Pretraining_Modification_Task.md` 再作四处修改，未动 PCC 网络主体 ——
`lm13_gdrn_protocol.py` 显式 `TARGET_LR_FACTOR=0.0`（原继承 `0.01`，160 epoch 只能衰减到
`1e-6`）、`preflight.py` 把该字段纳入硬校验并修掉 smoke 配置被正式协议检查拒绝的既存缺陷、
`train_lm13_gdrn.py` 的 `TEST` 换成 `lm_13_test_online` 且 `EVAL_PERIOD=20`、
`lmo_bop_test.py` 的 `obj2label` 反向映射修正。验证：5 个配置加载、三个 preflight PASS、
`check_lm_data` 全部 0 误差、test split 在无 `xyz_crop` 机器上可加载（旧 split 断言失败作对照）、
20 步 CUDA+CPP smoke loss 单调下降且 AMP 无跳步、真实入口 1 epoch 退出码 0 并写出 checkpoint
（`iteration=3`、optimizer `step=4`、`base_lrs=[1e-4,1e-5]`、scheduler `_last_lr=4.996e-7`
与 warmup 理论值一致）、`pytest -q research` **248 passed**。

本机 EGL 实测抛 `RuntimeError: Bindless Textures not supported`，按任务书不伪造 PASS，
故 `FORMAL_READY` 保持 `False`，结论 **NO-GO**，唯一 blocker 是服务器 EGL smoke：
`BLOCKED: final CUDA/EGL smoke must be run on the training server.`

**服务器集成（2026-09-18 同日）**：按 `EXP023_LM13_Server_Integration_Modification_Task.md`
补齐服务器侧契约。`docker/l40/experiment.sh` 的 `create` 新增 `${root}/datasets/lm_imgn`
只读 mount 与 `GDRN_CONVNEXT_BASE_WEIGHTS` 注入；`check_host()` 拆成通用项，数据集与权重
改由 profile gate 按容器内读回的 `TRAIN_PROTOCOL.NAME` 选择
（`lm13` / `lm13_pbr` / `legacy_lmo`，旧 LMO 配置走 legacy，资源清单不变）。新增
`research/exp022/server_preflight.py`，在容器内报告 profile、split、hierarchy、ConvNeXt、
VOC 与各 split 记录数，并接进 LM13 的 runtime gate，失败发生在 run 目录创建之前。
本机实测 `server_preflight`：`lm_13_train_online=2375`、`lm_imgn=13000`、
`lm_13_test_online=13425`，缺 ConvNeXt env 或非 LM13 配置均 FAIL。launcher 单测由 16 增至
40 个用例（全部用模拟容器），`pytest -q research` **275 passed**。

**本地 Agent 不连接服务器**，没有在 lab0/lab1 创建容器或运行 runtime gate，因此真实 mount、
容器内 env、`server_preflight` 的服务器行为与 EGL smoke 均**未标 PASS**，需用户在服务器执行
（步骤见 [RUNBOOK](RUNBOOK_CN.md) 的 “EXP023 LM13 server profile”）。

**launcher 收口（2026-09-18 同日）**：按 `EXP023_Final_Server_Code_Closure_Task.md` 修完剩余
四个工程问题，仍未动 PCC 网络主体与 LM13 protocol 数学。`check_host()` 只查 user/Docker/GPU/
`${root}`，运行目录（`datasets`/`weights`/`cache/gdrnpp_datasets`/`home/.cache`）改由 `create`
建立，干净 profile 的首次 `create` 不再被尚未创建的目录挡住，且 `create` 仍不生成任何真实数据
内容；`lm13` profile 下是否需要 `lm_imgn` 改由容器内读回的 `DATASETS.TRAIN` 决定
（`lm13_real_only` 因此不再要求 DeepIM 渲染图，与 `server_preflight.py` 口径一致）；未知非空
`TRAIN_PROTOCOL.NAME` 改为 fail-closed（`unknown TRAIN_PROTOCOL.NAME`），只有空名才回落
`legacy_lmo`，仓库现存合法名仍只有 `lm13_gdrn`/`lm13_real_only`/`lm13_pbr`；RUNBOOK 的
hierarchy 准备改为 container-safe（复制本地已验证 artifact，或 `create` 后在容器内
`build_hierarchy` 并重跑两项 preflight），不再指导在 host release 目录直接跑项目 Python。
验证：launcher 单测 40 → 50 用例、`pytest -q research` **285 passed**；用 launcher 内嵌的同一
段 Python 直读三个配置确认 `TRAIN_PROTOCOL.NAME` 与 `DATASETS.TRAIN` 的实际取值，EXP017
legacy 配置读回空名。正式 blocker 仍只剩服务器资源部署、容器 runtime gate 与 EGL smoke。

详见 [EXP023 RECORD](experiments/EXP-20260918-023-lm13-progressive-pcc-fulltrain/RECORD.md)
和 [EXP022 README](exp022/README.md)。

## EXP022 多数据集接入 / EXP023 LM13 准备（2026-09-18）

EXP022 方法、层级 loader/builder、CPU preflight、真实 CUDA smoke 与 matched evaluator 已接入统一 Dataset Context。LM-O 旧配置只增加上下文字段，仍用原始层级与冻结官方 backbone；LM-O 本机保存真实 batch48 的 AMP smoke PASS，340 个官方 backbone 张量加载，PCC 可训练参数 `3,728,456`，无 AMP 跳步。LM13 训练/测试对象顺序为标准非连续 13 个 BOP ID，独立层级已生成；ImageNet ConvNeXt 340 个张量 CPU 验证与本机 CUDA+CPP 在线 batch1 两步 AMP smoke PASS，完整模型 `91,292,872` 参数可训练。LM13 测试集实读 2600 个实例、类别覆盖 0..12；matched evaluator 对 2600 个 BOP target 的协议验证通过。research 回归 `223 passed`，后加的对象顺序测试单独通过。尚无 LM13 reference/PCC 正式 checkpoint，也无正式训练或评价。T-LESS 当前只有配置预留，数据与 variable-S 对称监督待后续处理。详见 [EXP023 RECORD](experiments/EXP-20260918-023-lm13-progressive-pcc-fulltrain/RECORD.md) 和 [EXP022 README](exp022/README.md)。

## EXP022 第一阶段（2026-09-16 起）

EXP022 渐进式层级 CAD 对应与多尺度 PCC：冻结官方 RGB ConvNeXt，主臂把 EXP021 的
4096 个叶子重排为 8⁴；四级 8 路局部匹配使用 GT-parent 监督与 Top-2 完整路径推理，
经多尺度细化、球形有界残差和可见 mask 输出连续 XYZ，再交 explicit RANSAC-PnP。
独立表面重采样 8⁴ 只做消融 smoke。第二阶段的全量 backbone 训练暂缓。

当前状态：`STAGE1_PERFORMANCE_REVISED / LOCAL_CPP_SMOKE_PASS / SERVER_EGL_PENDING /
FORMAL_NOT_STARTED`。正式训练前已加入 S1/S2 global、S3/S4 window+shifted-window
image self-attention，以及独立 Q/K/V 的局部 8 路 CAD matcher；随后按性能审查完成
四项修订：stage transition 改为上采样+1×1 Conv、shift mask 改用广播 SDPA、正式
路径关闭逐级诊断、删除 matcher `out_proj`。保留固定 hierarchy、GT parent、Top-2
beam、受限 residual、symmetry 和三项 loss。主层级与独立消融层级仍在 ignored cache。
EXP022 测试 `28 passed`、完整 research `216 passed`；官方 340 个 backbone 张量加载的
CPU preflight PASS。模型总参数 `91.293M`，其中冻结 backbone `87.564M`、可训练 PCC
`3.728M`。
正式冻结前增加 symmetry 数量检查：当前主层级和独立层级均为
`[1,1,1,1,1,2,2,1]`，V1 loader 拒绝 `>2`，preflight 公开打印计数；两种层级
CPU preflight PASS。

本机 RTX 4060/CUDA+CPP/FP16 AMP 对同一保存的真实 batch48 各运行 12 步、排除前
2 步后，旧版→重构版固定 batch 整步中位数 `462.083→710.912 ms`，峰值 allocated
`3.606→6.199 GB`、reserved `5.325→6.977 GB`；均无跳步。计时不含每步
DataLoader/renderer，不等于服务器 EGL 吞吐。独立层级新结构 batch4 三步 smoke
PASS。修改前的 14/202 项测试与 `3.684M` 参数等观察保留在 RECORD，属于旧结构。
这组旧版→重构版耗时来自上一轮，与下述重新固定 seed 的性能审查不是同一次 A/B。
本轮以相同保存 batch 和 seed 42 对当轮 BASE→四项修订各运行 12 步、排除前 2 步：
PCC 参数 `6.315→3.728M`，整步中位数 `949.30→633.88 ms`，forward
`435.75→294.07 ms`，backward `491.13→322.43 ms`，峰值 allocated
`6.199→5.669 GB`，reserved `6.977→6.713 GB`；各阶段无 AMP 跳步。reserved
逐阶段不单调，不能将单次波动直接归因于某项改动。Stage3/4 shifted block 的独立
同权重微基准也显示广播 SDPA 降低 forward/backward 和 allocated；详见 RECORD。
对称等价分支仍只按 route CE 与 residual loss 选择，选中后 mask loss 一起训练。
LM-O 单目标随机 PCC 的 fixed-support evaluator 历史接线通过，随机数值不进入科学
结论。服务器 EGL profile、正式 E5–E40、完整 matched PnP/BOP 尚未生成。EXP021 B/C
comparator 待现有实验完成后确定；不能用当前 E15 direct-pose 指标代替 matched
correspondence 判断。协议和入口见 [EXP022 README](exp022/README.md)，原始工程观察见
[EXP022 RECORD](experiments/EXP-20260916-022-progressive-pcc/RECORD.md)。

## Active mainline（2026-09-14 起）

**EXP021 global-guided hierarchical CAD correspondence**：用固定 CAD `64×64`
层级统一粗区域、父区域内子区域与连续 XYZ 解码，受限残差限制跨叶子区域滑动；
C 臂在 8×8 特征上增加两层图像—CAD Transformer、全局粗区域偏置和零初始化残差
注入。V1 固定为 RGB 与冻结阶段，只训练新增 CAD head；不执行 backbone 联合微调。
对称监督按实例从完整 BOP SE(3) 等价路径中选择一条，三项 loss 共用该分支。

当前状态：`FORMAL_BC_E40_COMPLETE / EXIT_CODE_UNCONFIRMED / MATCHED_PNP_PENDING`。
B/C 两臂的正式训练均已跑到 `iter 255919/255920`，E5–E40 八个固定评估点的常规
direct-pose 指标与逐物体结果已全部同步。用户 2026-09-19 确认 EXP021 结束，其记录与
结果可直接提交并推送；同时指出当前代码与网络结构设计存在问题，后续不在现有设计上
直接继续，EXP022 及之后的实验等待用户安排。确定性 hierarchy 已生成到 ignored dataset cache；EXP021
20 项测试通过；B/C CPU preflight 均通过，分别有 233,347 / 2,923,587 个 trainable
参数，官方 checkpoint 只缺 `cad_head.*`，优化步后冻结张量不变。真实 LM-O 单目标
evaluator 接线 smoke 已完成，能输出 fixed support、K=1/2/4/8、对称/路由/几何与
RANSAC 计时；随机初始化 B 的数值不进入科学结论。本机 CUDA+CPP 真实 batch 标定
建议 coarse/fine/XYZ 权重 `0.125/1/16`，加权梯度相对中位数
`0.894/1.160/1.000`，已写入 B/C 共享配置；B/C CUDA+CPP one-step smoke PASS，
峰值 allocated memory 约 0.979/1.035 GB。因本机 EGL 不支持 Bindless Textures，
source `9399608` 的服务器 EGL 标定、B/C smoke 与 batch-48 audit 已通过，但 B smoke
累计耗时 `0.9508 s/iter`，因此未启动 formal。定位发现旧 CAD loss 在本机 batch 48
的 backward 超线性增至约 9.6 秒；向量化并按唯一类别复用 descriptor 后，B/C
batch-48 前向+反向分别约 1.93/2.23 秒，峰值约 1.58/3.30 GiB，数值与梯度 reference
测试一致。source `f29f9a0` 的 lab1/EGL profile 发现 2 workers 导致 DataLoader
均值约 2.8–3.6 秒；8 workers 将 B/C 总耗时均值降至约 2.21/3.00 秒，但仍有偶发
等待峰值。EXP021 formal 已显式对齐近期协议为 16 workers，执行固定为 B→lab0、
C→lab1。用户随后启动的 FP32 B/C formal 当前约为 `1.89/2.96 s/iter`，已决定在新
release 通过 EGL gate 后终止并从官方 checkpoint 重启，不混合精度续训。新实现将
训练路径改为可选全局增强后只运行一次 feature decoder，跳过零权重旧输出和 PnP，
并显式启用 FP16 AMP；几何 target/`cdist` 保持 FP32。本机 CUDA+CPP batch-1 AMP
smoke 与标定 PASS，GradScaler 保持 65536。本机建议权重为 `0.125/1/16`；source
`8a736c8` 的 lab0/L40 EGL 标定建议 `0.25/1/16`，加权梯度相对中位数为
`1.133/1.000/0.582`，因此已按预定标定规则更新共享配置，等待新 release 复核。同代码 matched
batch-48 profile 中，B AMP 相对 FP32 的模型前反向缩短约 17.0%、端到端中位数缩短
13.5%；C 分别缩短约 8.4%/3.0%，峰值显存下降约 8.6%。本机 DataLoader 有秒级长尾，
完整 research 回归为 188 passed。source `8a736c8` 的服务器 B/C EGL AMP smoke 均
PASS，无跳步且冻结参数不变；batch-48 profile 中 B FP32→AMP 端到端中位数
`1.457→1.420 s`，C 为 `1.650→1.335 s`，但 DataLoader 仍有 4.8–6.9 秒长尾，B 的
均值与模型前反向没有稳定加速。完整 matched PnP/BOP 尚未运行。

2026-09-15 本地工程优化进一步删除 CAD loss 的 CPU route 列表和 data-dependent group
loop，以 GPU 256 点 padded blocks、FP32 平方距离和 batched matmul 计算 fine label、
logits 与 residual；C 的全局交互和 loss 复用同一次 descriptor 编码。CAD-only online
batch 同时跳过未使用的旧 Region `cdist` 与 pose-head targets，renderer 仍保持原生逐
ROI EGL/CPP 接口。EXP021 20 项测试、完整 research 188 项、B/C CPU preflight 通过；
本机 CUDA+CPP AMP 固定 batch 20-step 的总 loss 均下降且无跳步。batch-48 profile 的
B/C forward+backward 均值约为 `0.426/0.523 s`，端到端中位数约
`0.882/1.008 s`，峰值约 `2.659/4.186 GB`。相对同机优化前 AMP 记录，B/C 模型
前反向分别加速约 `4.34×/3.88×`（耗时下降 `77.0%/74.2%`），端到端中位数加速约
`2.66×/2.55×`（耗时下降 `62.4%/60.7%`）；该比较受 sampled batch、权重版本和
DataLoader 长尾限制，不设严格时间 gate。本次加速作为同一 EXP021 formal 的工程
实现更新，不新建 formal 实验；实际训练 run 的 source 与进度见下段。

2026-09-19 同步的 B/C 日志记录 source `effc99b`、seed 42、FP16 AMP；B run
`RUN-20260914-125103-formal-s42-a01`，C run `RUN-20260914-125349-formal-s42-a01`。
两臂均到达 `iter 255919/255920[100.0%]` 并有
`FINAL_EVAL_REUSED periodic_epoch=40`；日志中无 `Traceback`/OOM/`RuntimeError`，
但也**没有退出码行**，进程退出状态仍未确认。E5–E40 全部八个点的 checkpoint 保存
日志、常规 direct-pose `EVAL_SUMMARY` 与 BOP score JSON 已记录于 EXP021 RECORD；
16 份 score 的 BOP AR 均与对应 epoch 日志逐点相等。E40 B/C BOP AR
`0.688999/0.693675`、ADD(-S)0.1d `0.525952/0.534256`、reS
`0.537024/0.546021`、teS `0.795386/0.806690`；八个固定点中 C 高于 B 的为 BOP
`8/8`、reS `8/8`、teS `7/8`、ADD(-S)0.1d `7/8`（两个负项都在 E5）。这些都只是
常规 direct-pose 描述，不用于重选模型。所获 formal 源码是 `effc99b`，不包含后续本地
`f01dff5` 加速；完整 fixed-support matched PnP、K sweep 和资源 gate 仍未生成，
作为证据缺口保留，不因实验收口补做。

协议、gate 和入口见 [EXP021 README](exp021/README.md)，事实记录见
[EXP021 RECORD](experiments/EXP-20260914-021-global-guided-hierarchical-cad-correspondence/RECORD.md)。

## 已结束的前序实验：EXP020（2026-09-16 决定不再补证）

**EXP020 correspondence supervision + ordinary PnP/RANSAC**：在保留 continuous
normalized XYZ 的前提下，用 GT-pose per-pixel correspondence reprojection loss
监督 Geometry/Correspondence Head，主下游用 matched classical PnP/RANSAC 判断
“producer 是否更真实、更适合普通 PnP”。唯一核心变量 `REPROJ_LW`（A=0 / B=1）；
backbone/PNP_NET 冻结、GEO_HEAD trainable、pose-level losses 显式清零以隔离
producer；无新增模型参数，官方 checkpoint strict 兼容。

当前状态：`CLOSED / E40_DIRECT_POSE_RECORDED / MATCHED_PNP_NOT_RUN /
EXIT_CODE_UNCONFIRMED`。用户决定结束后续工作；原始结果及证据缺口保留，
不补做 matched PnP，不据 direct-pose telemetry 增加机制结论。第一阶段实现
（commit `64e9098`）与 2026-09-09 审查修复（matched evaluator、A/B 跨 checkpoint
fixed support、diagnostics、USE_MTL guard、gradient-scale calibration）均已完成并
本地验证。首个服务器 release `698a8fe` 错配 CPP online training renderer，运行
时间异常；该 release 的 run 不作为有效协议。EGL/cache 修复后，source commit
`c2a7723` 的 A/B formal 日志均已到达 E40 最后 iteration 255919，并记录保存
`model_epoch_040.pth`；E5/10/15/20/25/30/35/40 全部 direct-pose evaluation 已齐，
BOP evaluation renderer 仍为 CPP。run exit code 与正式 matched PnP 评价尚未提供。
原始 score JSON 及日志抽取的 EVAL_SUMMARY 已按 run ID 随 EXP020 RECORD 保存紧凑副本。

修正后的首轮 EGL server smoke 已确认 CUDA device 0 上 EGL 1.5 context 能创建，但
PLY mesh cache 默认写入只读源码根目录 `.cache` 而失败。第二次重跑确认外层 cache
注入有效，但内部 pyassimp loader 未透传路径并再次回落到 `.cache`。训练 renderer
现于同步模型加载期间把两层相对 cache 一并定向到可写 `XDG_CACHE_HOME`，且不改变
镜像原生输入；修复后的 smoke 已通过并进入 formal，先前失败 run 不进入科学结论。

Direct-pose telemetry：E40 A/B BOP `0.460300/0.462563`、ADD(-S)0.1d
`0.159170/0.152941`、reS `0.326182/0.325952`、teS `0.495502/0.494118`。
两臂 BOP/ADD 在全部固定点中均于 E5 最大；这只是事后描述，不用于重选模型。
交叉核对发现 B E15/E20/E25 score JSON 与对应日志 BOP 不一致：保留原文件，
这三点 reS/teS 的 epoch 归属待核对，不能直接按文件名采用；详情见 RECORD。
整体随训练下降、B 未形成四项一致收益，但不能据此裁决 correspondence 或 matched
PnP 主假设。全部固定点评估、最终/最佳点逐物体 ADD 与日志口径见 EXP020 RECORD。
本次仅同步记录，未重新运行测试、训练或 evaluator。

### 审查修复后的关键事实（2026-09-09，Observed）

- `research/exp020/matched_pnp_eval.py`：EXP020 专用 matched classical
  PnP/RANSAC evaluator。reference checkpoint（默认 official）一次生成
  `S_fixed = reference_pred_visible ∩ gt_visible ∩ valid_depth`，冻结 support
  mask/flat indices/subsample/2D 点/K/RANSAC seed/threshold/iterations；A/B 只替换
  各自 predicted XYZ。不启动 EPro、不跑 alpha sweep、不以 Patch-PnP 为主结果，
  不改 EXP019 evaluator（EXP019 强校验 official SHA-256 且固定 EPro+alpha，故不能
  直接用于 EXP020 checkpoint）。
- Identity smoke（reference=A=B=official，`--limit 16`，真实 LM-O）：COMPLETE，
  A/B 完全一致（各 delta=0.0），16 targets；metadata 记录 `epro_started=False`、
  `alpha_sweep=False`。该 smoke 只验证 evaluator，不代表科学性能。
- Diagnostics：`reproj_px_loss` 命名废弃，改为 `reproj_loss`；真实
  `mean_reproj_px`（Euclidean px）与 loss 值分离；`valid_ratio` 分母为 GT
  foreground；新增 `positive_depth_ratio_on_gt_fg` /
  `behind_camera_ratio_on_gt_fg`。不再用 `loss_xyz_reproj × 64` 当 pixel error。
  GPU smoke arm B：`reproj_loss≈0.045`、`mean_reproj_px≈5.1 px`、
  `valid_ratio=1.0`、`behind_camera_ratio=0.0`。
- `REPROJ_LW>0 && USE_MTL=True` 直接 fail-fast（不新增 trainable log_var）。
- Gradient-scale calibration 实际运行（真实 online-geometry batch，GPU）：
  `g_xyz=4.9539`、`g_reproj_raw=0.3695`、`ratio_raw≈0.075`。REPROJ_LW 梯度比 XYZ
  三项总梯度小约 13×，同数量级；**formal `REPROJ_LW=1.0` 未修改**。
- 测试：EXP020 36 passed、仓库回归子集 127 passed（2026-09-09 review-fix 与
  2026-09-10 整理时复核）；CPU preflight A/B PASS。

## Historical / Deferred：2026-09-08 EPro-PnP solver-in-the-loop 主线

2026-09-08 曾确定：用可微 EPro-PnP 作为显式几何后端，把最终姿态监督反传约束
Geometry/Correspondence Head（XYZ、ROI2D、Mask、Region、Reliability），使对应关系
学习以“能被显式求解器正确、稳定地使用”为目标；EPro-PnP 不是主要创新点。该路线及
其 EXP019 机制证据（matched RANSAC 与 EPro-PnP 都稳定消费逐步改善的 XYZ，官方
Patch-PnP 响应不足；用户 review 判定机制通过）**保留为历史记录，当前未安排实验，
标为 Historical / Deferred**。当前 active mainline 是上方 EXP021，EXP020 的
matched PnP 缺口作为已结束实验的未生成证据保留；不启动 EPro。文献对照与口径见
[notes/20260908-solver-in-the-loop-review.md](notes/20260908-solver-in-the-loop-review.md)
与 [DECISIONS](DECISIONS.md)。

## 长期已建立结论（Historical，仍用于比较基线）

- EXP012 已完成 40 epoch，E40 BOP AR `0.678800`、ADD(-S) `0.494118`、
  AR_reS `0.491349`、AR_teS `0.791926`，是 EXP013 的固定比较基准。
- EXP013A 通过相对 EXP012 的正式门槛；B 的 ADD 改善但严格 BOP 门槛差
  `0.000109`；C 支持 R/t 解耦改善 rotation，但总体 ADD 门槛失败。
- EXP013E 官方头随机初始化对照完成：BOP `0.688581`、reS `0.535409`。
- EXP013F GLM-Pose-L 完成：BOP `0.684129`、reS `0.515802`，边缘 `SCREEN_FAIL`。
- EXP017 canonical formal 已完成 E40：BOP `0.681709`、ADD `0.512111`，
  reS `0.494348`、teS `0.800461`，结论 `SCREEN_FAIL`。
- EXP017-B 已完成 E40：BOP `0.683686`、ADD `0.498270`、reS `0.503114`、
  teS `0.800692`，结论 `NO_OVERALL_GAIN`。
- EXP018 formal 已完成 E40：BOP `0.686302`、ADD `0.516263`、reS `0.517647`、
  teS `0.800461`；相对 EXP013A 四项均提高，但 BOP 仅 `+0.002346`，未达到设计
  阶段建议的 `+0.003`。单 seed 小幅收益不足以支持继续投入，状态
  `COMPLETE / MARGINAL_GAIN / CLOSED`。
- EXP019 full run（1,445 targets）：原 Gate A/B 通过，历史复现 5/6 越界（原绝对
  容差 0.001）；原 evaluator decision `PROTOCOL_REPRODUCTION_FAILED_STOP` 保留为
  历史输出；2026-09-08 用户 review 判定机制通过。后续 gate 用相对阈值
  （±3%–±5%，饱和指标仍绝对）。

## 当前代码边界

- 保留上游 GDRNPP、EXP012、EXP013 A–F、暂停的 D、EXP017、已收口 EXP018、EXP019、
  EXP020（实现 + review-fix）、EXP021 V1 和 EXP022 第一阶段重构版。
- EXP020 review-fix 只改 shared 层的 loss stats/guard 与 exp020 目录；EXP019
  历史 evaluator/结果未改动。
- 本地 `.git` 历史是恢复兜底，禁止删除或重写；删除内容用普通提交表达。
- dataset、checkpoint、完整日志、缓存和 secrets 不进入 Git；`output/` 与
  `.local/` 是机器本地内容。

## 下一步

1. EXP021 训练已结束：B/C run_id、source 与 E5–E40 八个固定评估点（含逐物体
   ADD(-S)0.1d）均已记录并随记录提交；仍缺 run exit code 与服务器权重文件核验。
2. 用户已判定当前代码与网络结构设计需要修正，后续不在现有设计上直接继续；EXP022
   及之后的实验暂不修改，等待用户安排。EXP021 剩余的 fixed-support matched
   RANSAC-PnP/K sweep 是否补做由用户决定，不自动执行。
3. 不为本次 EXP021 工程加速另建 formal 实验；B/C K=1/2/4/8 fixed-support matched
   PnP、完整 BOP evaluator 与 batch-1 profile 仍未生成，作为证据缺口保留。
4. EXP020 后续补证已结束；B E15/E20/E25 score 归属、A/B exit code 与正式 matched
   PnP 未核实或未生成，保留缺口，不安排追加执行。
5. 暂不恢复 EXP014-D；不启动 EPro（Historical/Deferred）；不自动加 seed。
6. EXP022 在本机结构重构后仍须先过服务器 EGL 真 batch smoke、batch48 资源检查，
   再考虑正式训练；geometry-adaptive partition 与 fragment adjacency 延后。
