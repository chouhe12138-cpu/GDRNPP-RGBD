# 当前研究状态

最后核对：2026-09-14。

## Active mainline（2026-09-14 起）

**EXP021 global-guided hierarchical CAD correspondence**：用固定 CAD `64×64`
层级统一粗区域、父区域内子区域与连续 XYZ 解码，受限残差限制跨叶子区域滑动；
C 臂在 8×8 特征上增加两层图像—CAD Transformer、全局粗区域偏置和零初始化残差
注入。V1 固定为 RGB 与冻结阶段，只训练新增 CAD head；不执行 backbone 联合微调。
对称监督按实例从完整 BOP SE(3) 等价路径中选择一条，三项 loss 共用该分支。

当前状态：`PERFORMANCE_FIX_LOCAL_PASS / SERVER_EGL_REVALIDATION_PENDING /
FORMAL_NOT_STARTED`。确定性 hierarchy 已生成到 ignored dataset cache；EXP021 13 项
测试通过；B/C CPU preflight 均通过，分别有 233,347 / 2,923,587 个 trainable
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
C→lab1；新 release 的双机 EGL profile/smoke、formal 和完整 matched PnP/BOP 尚未运行。

协议、gate 和入口见 [EXP021 README](exp021/README.md)，事实记录见
[EXP021 RECORD](experiments/EXP-20260914-021-global-guided-hierarchical-cad-correspondence/RECORD.md)。

## Predecessor pending：EXP020（2026-09-09 起）

**EXP020 correspondence supervision + ordinary PnP/RANSAC**：在保留 continuous
normalized XYZ 的前提下，用 GT-pose per-pixel correspondence reprojection loss
监督 Geometry/Correspondence Head，主下游用 matched classical PnP/RANSAC 判断
“producer 是否更真实、更适合普通 PnP”。唯一核心变量 `REPROJ_LW`（A=0 / B=1）；
backbone/PNP_NET 冻结、GEO_HEAD trainable、pose-level losses 显式清零以隔离
producer；无新增模型参数，官方 checkpoint strict 兼容。

当前状态：`E40_AVAILABLE / MATCHED_PNP_PENDING / EXIT_CODE_UNCONFIRMED`。第一阶段实现
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
matched PnP 缺口仍保留；不启动 EPro。文献对照与口径见
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

1. 在服务器项目容器内生成/核对 EXP021 hierarchy，运行真实 CUDA/EGL loss 梯度
   标定和 B/C one-step smoke；若权重建议不全为 1，先固定共享配置并重复 smoke。
2. smoke 通过后启动 EXP021 B/C 唯一 formal run；固定 checkpoint，不按中间 LM-O
   结果选择模型。
3. 对 B/C 固定 checkpoint 做 K=1/2/4/8 fixed-support matched RANSAC-PnP、完整
   BOP evaluator 与 batch-1 profile，按预注册 mechanism/resource gate 决策。
4. EXP020 E5–E40 BOP/ADD 已齐；补充 B E15/E20/E25 的明确 epoch 原始 score，核对 reS/teS
   归属；补充 A/B run exit code 并核对 E40 checkpoint 文件。
5. 使用明确的 A/B E40 checkpoint，用 `research/exp020/matched_pnp_eval.py` 做主下游评价（matched
   classical PnP/RANSAC，fixed support，A/B 只换 XYZ），必要时 `--bop-eval` 汇总
   BOP-AR/ADD(-S)/reS/teS。
6. Gate 沿用相对阈值政策（±3%–±5%），先看方向一致性：correspondence error →
   reprojection error → matched PnP pose 同方向。
7. 暂不恢复 EXP014-D；不启动 EPro（Historical/Deferred）；不自动加 seed。
