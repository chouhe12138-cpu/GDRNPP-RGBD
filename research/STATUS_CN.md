# 当前研究状态

最后核对：2026-09-16。

## EXP022 第一阶段（2026-09-16 起）

EXP022 渐进式层级 CAD 对应与多尺度 PCC：冻结官方 RGB ConvNeXt，主臂把 EXP021 的
4096 个叶子重排为 8⁴；四级 8 路局部匹配使用 GT-parent 监督与 Top-2 完整路径推理，
经多尺度细化、球形有界残差和可见 mask 输出连续 XYZ，再交 explicit RANSAC-PnP。
独立表面重采样 8⁴ 只做消融 smoke。第二阶段的全量 backbone 训练暂缓。

当前状态：`STAGE1_IMPLEMENTED / LOCAL_CPP_SMOKE_PASS / SERVER_EGL_PENDING /
FORMAL_NOT_STARTED`。主层级及独立消融层级已生成至 ignored cache；EXP022 单元/配置
9 项和完整 research 197 项通过；官方 340 个 backbone 张量加载和仅 3,684,168 个 PCC
参数可训练的 CPU 前后向通过。本机 CUDA+CPP batch4/batch48 两步 AMP 均无跳步，固定
batch 的第二步模型优化耗时约 `0.168/0.670 s`；另一次固定真实 batch48 十步诊断中，第
2–10 步耗时中位数 `0.640 s`（forward `0.350 s`、backward `0.282 s`），峰值 allocated
`3.635 GB`；模型共 `91.249M` 参数，其中 frozen backbone `87.564M`、trainable PCC
`3.684M`。上述固定 batch 计时不含每步 DataLoader/renderer，不等于服务器 EGL 吞吐。
LM-O 单目标随机 PCC 的 fixed-support
evaluator 接线通过，随机数值不进入科学结论。服务器 EGL profile、正式 E5–E40、完整
matched PnP/BOP 尚未生成。EXP021 B/C comparator 待现有实验完成后确定；不能用当前 E15 direct-pose
指标代替 matched correspondence 判断。协议和入口见 [EXP022 README](exp022/README.md)，
原始工程观察见 [EXP022 RECORD](experiments/EXP-20260916-022-progressive-pcc/RECORD.md)。

## Active mainline（2026-09-14 起）

**EXP021 global-guided hierarchical CAD correspondence**：用固定 CAD `64×64`
层级统一粗区域、父区域内子区域与连续 XYZ 解码，受限残差限制跨叶子区域滑动；
C 臂在 8×8 特征上增加两层图像—CAD Transformer、全局粗区域偏置和零初始化残差
注入。V1 固定为 RGB 与冻结阶段，只训练新增 CAD head；不执行 backbone 联合微调。
对称监督按实例从完整 BOP SE(3) 等价路径中选择一条，三项 loss 共用该分支。

当前状态：`FORMAL_BC_E15_AVAILABLE / TRAINING_EXIT_UNCONFIRMED /
MATCHED_PNP_PENDING`。确定性 hierarchy 已生成到 ignored dataset cache；EXP021
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

2026-09-16 同步的 B/C 日志均记录 source `effc99b`、seed 42、FP16 AMP；
B run `RUN-20260914-125103-formal-s42-a01`，C run
`RUN-20260914-125349-formal-s42-a01`。两臂 E5/E10/E15 的 checkpoint 保存日志、
常规 direct-pose `EVAL_SUMMARY` 与 BOP score JSON 已记录于 EXP021 RECORD；
六份 score 的 BOP AR 均与对应 epoch 日志相等。E10 B/C BOP AR
`0.685010/0.693179`、ADD(-S)0.1d `0.526644/0.560554`、reS
`0.521107/0.538639`、teS `0.793080/0.803460`。日志副本最后一条训练指标为
B iter `100499/255920`、C iter `97499/255920`，均处于 epoch 16；
退出码与后续进度未提供。所获 formal 源码是 `effc99b`，不包含后续本地
`f01dff5` 加速；完整 fixed-support matched PnP、E40 和资源 gate 均尚未生成。

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
  EXP020（实现 + review-fix）和 EXP021 V1。
- EXP020 review-fix 只改 shared 层的 loss stats/guard 与 exp020 目录；EXP019
  历史 evaluator/结果未改动。
- 本地 `.git` 历史是恢复兜底，禁止删除或重写；删除内容用普通提交表达。
- dataset、checkpoint、完整日志、缓存和 secrets 不进入 Git；`output/` 与
  `.local/` 是机器本地内容。

## 下一步

1. 不干扰远程 formal；B/C run_id 和 source 已记录，待训练结束后补充退出状态、
   E20–E40 checkpoint 文件名/epoch 和剩余预定正式评估点。
2. 训练完成后直接使用加速版本进入下一阶段，不为本次工程加速另建 formal 实验。
3. 对 B/C 固定 checkpoint 做 K=1/2/4/8 fixed-support matched RANSAC-PnP、完整
   BOP evaluator 与 batch-1 profile，按预注册 mechanism/resource gate 决策。
4. EXP020 后续补证已结束；B E15/E20/E25 score 归属、A/B exit code 与正式 matched
   PnP 未核实或未生成，保留缺口，不安排追加执行。
5. 暂不恢复 EXP014-D；不启动 EPro（Historical/Deferred）；不自动加 seed。
