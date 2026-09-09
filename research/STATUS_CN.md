# 当前研究状态

最后核对：2026-09-09。

## Active mainline（2026-09-09 起）

**EXP020 correspondence supervision + ordinary PnP/RANSAC**：在保留 continuous
normalized XYZ 的前提下，用 GT-pose per-pixel correspondence reprojection loss
监督 Geometry/Correspondence Head，主下游用 matched classical PnP/RANSAC 判断
“producer 是否更真实、更适合普通 PnP”。唯一核心变量 `REPROJ_LW`（A=0 / B=1）；
backbone/PNP_NET 冻结、GEO_HEAD trainable、pose-level losses 显式清零以隔离
producer；无新增模型参数，官方 checkpoint strict 兼容。

当前状态：`IMPLEMENTED / LOCAL_TEST_PASS / READY_FOR_FORMAL_RUN`。第一阶段实现
（commit `64e9098`）与 2026-09-09 审查修复（matched evaluator、A/B 跨 checkpoint
fixed support、diagnostics、USE_MTL guard、gradient-scale calibration）均已完成并
本地验证；**没有 formal A/B 训练，不宣称任何性能提升**。等待用户选择服务器实验并
授权 `docker/l40/experiment.sh` 的 smoke/formal A/B 运行。

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
- 测试：EXP020 36 passed、仓库回归子集 127 passed（2026-09-09 本次整理均已
  复核）；CPU preflight A/B PASS。

## Historical / Deferred：2026-09-08 EPro-PnP solver-in-the-loop 主线

2026-09-08 曾确定：用可微 EPro-PnP 作为显式几何后端，把最终姿态监督反传约束
Geometry/Correspondence Head（XYZ、ROI2D、Mask、Region、Reliability），使对应关系
学习以“能被显式求解器正确、稳定地使用”为目标；EPro-PnP 不是主要创新点。该路线及
其 EXP019 机制证据（matched RANSAC 与 EPro-PnP 都稳定消费逐步改善的 XYZ，官方
Patch-PnP 响应不足；用户 review 判定机制通过）**保留为历史记录，当前未安排实验，
标为 Historical / Deferred**。当前 active mainline 是上方 EXP020 correspondence
supervision + ordinary PnP/RANSAC，不启动 EPro。文献对照与口径见
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

- 保留上游 GDRNPP、EXP012、EXP013 A–F、暂停的 D、EXP017、已收口 EXP018、EXP019
  和 EXP020（实现 + review-fix）。
- EXP020 review-fix 只改 shared 层的 loss stats/guard 与 exp020 目录；EXP019
  历史 evaluator/结果未改动。
- 本地 `.git` 历史是恢复兜底，禁止删除或重写；删除内容用普通提交表达。
- dataset、checkpoint、完整日志、缓存和 secrets 不进入 Git；`output/` 与
  `.local/` 是机器本地内容。

## 下一步

1. 用户选择 EXP020 formal A/B 服务器实验与配置，走 `docker/l40/experiment.sh`；
   服务器运行前本地提交并生成 bundle，服务器只 checkout 确定 commit 运行。
2. Formal 完成后用 `research/exp020/matched_pnp_eval.py` 做主下游评价（matched
   classical PnP/RANSAC，fixed support，A/B 只换 XYZ），必要时 `--bop-eval` 汇总
   BOP-AR/ADD(-S)/reS/teS。
3. Gate 沿用相对阈值政策（±3%–±5%），先看方向一致性：correspondence error →
   reprojection error → matched PnP pose 同方向。
4. 暂不恢复 EXP014-D；不启动 EPro（Historical/Deferred）；不自动加 seed。
