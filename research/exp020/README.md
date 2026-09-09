# EXP020 — Geometry-Aware Correspondence Reprojection Supervision

EXP020 第一阶段研究 question：

> 在保留当前 continuous XYZ representation 的条件下，加入由 GT pose + camera
> model 构造的 per-pixel geometric reprojection supervision，是否能让
> Geometry/Correspondence Head 输出更真实、更适合普通 PnP 的 correspondence？

EXP019 已证明 solver 不是当前主瓶颈（XYZ 变好时 matched classical PnP/RANSAC 与
EPro-PnP 都能稳定改善 pose），因此 EXP020 **不研究 solver**，只改变 correspondence
producer 的 supervision。EPro-PnP、differentiable PnP、confidence head、
Region+Local/surface code/spectral 表示、Patch-PnP 重构都不在本阶段范围内。

## 唯一核心变量

- A（control）：现有 XYZ/Mask/Region/FullMask supervision，`REPROJ_LW=0`。
- B（reproj）：完全相同 + per-pixel GT-pose correspondence reprojection loss
  （`REPROJ_LW=1.0`）。

隔离 producer：backbone freeze、PNP_NET freeze、GEO_HEAD trainable，geometry
supervision 打开。A/B 都显式关闭 pose-level losses（`PM_LW / CENTROID_LW / Z_LW /
ROT_LW / TRANS_LW / BIND_LW = 0`），因为**冻结 PNP_NET 参数不会阻断 pose loss 对
coor_feat/geometry 输入的梯度**。

## Loss 数学

网络输出 normalized object XYZ（`xyz_norm = xyz_obj / extent + 0.5`）：

`xyz_obj = (xyz_norm - 0.5) * extent`

再按 GT pose 转到 camera frame：

`X_cam = R_gt @ xyz_obj + t_gt`

用在线几何路径的 crop-resized 相机矩阵 `roi_zoom_K` 投影到网络输出分辨率，target
是 **output map 自身的 (x, y) pixel grid**（`x ∈ [0,W-1], y ∈ [0,H-1]`），不是
full-image normalized `roi_coord_2d`：

`L_reproj = SmoothL1( u_hat - u ) / max(H, W)`

valid 只取 GT XYZ 对应 mask 的前景像素且相机深度 `z > z_eps` 且投影 finite。该 target
与 `lib.pysixd.misc.calc_xyz_bp_batch` 使用的整数 pixel grid 同一坐标系，因此在线
back-projection XYZ target 在 GT 一致时满足 `L_reproj ≈ 0`（有单测覆盖）。

### Diagnostics（review-fix 后）

loss 模块返回的 stats 区分 **loss 值**与 **真实 pixel error**，禁止用
`loss_xyz_reproj * max(H,W)` 当 pixel error：

- `mean_reproj_px`：loss mask 有效像素的 Euclidean 重投影误差均值（真实 px）。
- `valid_ratio = valid_count / gt_foreground_count`：以 GT 前景为分母。
- `positive_depth_ratio_on_gt_fg` / `behind_camera_ratio_on_gt_fg`：GT 前景像素中
  预测点在相机前/非正深度（含非有限预测）的比例；`behind ≈ 1 - positive`。
- `gt_foreground_count`：GT 前景像素数。

本轮 review-fix **只改 stats，不改 loss valid mask**，避免在 formal 前引入新的科学变量。
`gdrn_loss` 经 `vis_extra` 继续记录 `vis/reproj_px`、`vis/reproj_valid_ratio`，并新增
`vis/reproj_gt_fg_count`、`vis/reproj_positive_depth_ratio`、
`vis/reproj_behind_camera_ratio`。`real_smoke.py` 记录 `reproj_loss`（真实 loss 标量）与
loss 模块产出的 `mean_reproj_px`（来自 EventStorage），不再命名/换算成 `reproj_px_loss`。

### USE_MTL guard

`loss_xyz_reproj` 没有对应 trainable `log_var_xyz_reproj`。若
`REPROJ_LW>0 && USE_MTL=True`，`forward` 与 `gdrn_loss` 直接
`NotImplementedError` fail-fast；不新增 log_var 参数。`REPROJ_LW=0` 历史路径不变。

## 代码路径

- `core/gdrn_modeling/losses/correspondence_reprojection_loss.py`：纯 Tensor loss。
- `core/gdrn_modeling/models/GDRN_double_mask.py`：`forward(..., roi_zoom_cams=None)`
  透传 `roi_zoom_K`；`gdrn_loss(..., roi_zoom_cams=None, vis_extra=None)` 在 L1 XYZ
  之后计算 `loss_xyz_reproj = L_reproj * REPROJ_LW`。`REPROJ_LW=0` 时不新增 loss key、
  不要求 crop K；`REPROJ_LW>0` 且缺 crop K 或 XYZ_LOSS_TYPE 非 L1 时 fail fast。
- `core/gdrn_modeling/engine/engine.py`：训练调用新增
  `roi_zoom_cams=batch.get("roi_zoom_K", None)`。
- 没有改 `data_loader.py`（第一版不需要新增 annotation），没有加任何模型参数，
  官方 checkpoint strict 兼容（preflight 验证 missing/unexpected 均空）。

## 配置

`configs/gdrn/lmo_pbr/research/exp020_geometry_aware_corr/`：

- `common.py`：共享 formal 协议（seed 42、40 epoch、batch 48、5-epoch ckpt/eval、
  Ranger lr 8e-4 / wd 0.01 / warmup 200、cpp online renderer、freeze/loss 布局）。
- `control.py`（A）与 `reproj.py`（B）唯一差异是 `LOSS_CFG.REPROJ_LW`
  （`0.0` vs `1.0`）。
- `smoke_control.py` / `smoke_reproj.py` / `eval.py`。

optimizer/LR 依据仓库最近的“从 official checkpoint fine-tune”协议（EXP013A/E 与
PnP-only control：Ranger 8e-4、wd 0.01、warmup 200）。真实 batch 的
gradient-magnitude 标定已完成，未发现数量级失衡，因此 `REPROJ_LW=1.0` 保持为
第一轮固定非零值；A/B 之间除 `REPROJ_LW` 外必须完全一致。

## 下游评价：matched classical PnP/RANSAC（review-fix 新增）

`research/exp020/matched_pnp_eval.py` 是 EXP020 **主下游评价**。它不能用 EXP019
runner：EXP019 `build_context()` 会强校验 official checkpoint SHA-256，且固定启动
EPro worker 与 alpha sweep。新 evaluator：

- 复用 EXP019 已验证的纯函数：`roi2d_norm_to_pixels`、`xyz_norm_to_metric`、
  `historical_gt_reprojection_errors`、`solve_ransac_pnp()`，以及 repo_adapter 的
  depth→object / prediction-valid-mask 语义；不修改任何 EXP019 代码，不启动 EPro，
  不做 alpha interpolation，不以 Patch-PnP 为主结果。
- **跨 checkpoint fixed support（主协议）**：reference checkpoint（默认 official）
  一次生成 `S_fixed = reference_pred_visible ∩ gt_visible ∩ valid_depth`，冻结
  support mask、flat indices、subsample indices、2D 点、K、RANSAC seed/threshold/
  iterations。A/B 只替换各自 predicted XYZ（`(xyz_norm-0.5)*extent` 后取固定
  indices）。A/B 不得各自重建主 support。native-support 仅允许作为显式标明的
  secondary analysis（本轮未启用）。
- 输出 per-target 行（`poses.jsonl`）：`num_fixed_support`、`num_selected`、
  A/B pose（RANSAC/EPNP，3px、100 iter、0.99、seed 20260730+序号）、A/B fixed-support
  XYZ metric error（mm，对 GT）、A/B GT-pose reprojection error（px）。汇总见
  `summary.json`（solve success rate、mean corr/reproj error、A/B arm consistency）。
- CLI：`--reference-checkpoint`、`--checkpoint-a`、`--checkpoint-b`、
  `--gdrn-config`、`--device`、`--output`、`--limit`；完整 run（limit=None）可另加
  `--bop-eval` 导出 BOP-AR/ADD(-S)/reS/teS。

对同一 checkpoint 临时作为 A 与 B 的 smoke（identity 测试）用于验证 evaluator
接线：A/B 输出应一致或数值误差级一致，不表示科学性能。

## REPROJ_LW 梯度尺度标定（review-fix 新增）

`research/exp020/calibrate_reproj_weight.py` 在同一真实 online-geometry batch、同一
模型状态（official 权重、geo head 唯一 trainable）上：

- Pass XYZ：只 backward `loss_coor_x + loss_coor_y + loss_coor_z`，记录
  `g_xyz` = geo_head_net 的 global L2 grad norm。
- Pass REPROJ：清梯度后只 backward **raw** reprojection loss
  （`loss_xyz_reproj / REPROJ_LW`），记录 `g_reproj_raw`。
- 输出 `ratio_raw = g_reproj_raw / g_xyz` 与可选分组（shared trunk / xyz output
  layer）norm。目的只是判断 `REPROJ_LW=1.0` 是否明显过强/过弱，不做 λ sweep。
  实际 `ratio_raw=0.0746`，formal `REPROJ_LW` 保持 `1.0`。

## 本地执行

```bash
source /home/wsluser/miniconda3/etc/profile.d/conda.sh
conda activate pytorch22
PYTHONPATH="$PWD" pytest -q research/exp020/tests
PYTHONPATH="$PWD" python -m research.exp020.preflight --arm A --device cpu
PYTHONPATH="$PWD" python -m research.exp020.preflight --arm B --device cpu
BOP_RENDERER_PATH="$PWD/.local/bop_renderer/build" \
  PYTHONPATH="$PWD" python -m research.exp020.real_smoke --arm both --device cuda:0
# matched evaluator identity smoke（同一 checkpoint 临时作为 reference/A/B）
PYTHONPATH="$PWD" python -m research.exp020.matched_pnp_eval \
  --reference-checkpoint pretrained_models/lmo_pbr/model_final_wo_optim.pth \
  --checkpoint-a pretrained_models/lmo_pbr/model_final_wo_optim.pth \
  --checkpoint-b pretrained_models/lmo_pbr/model_final_wo_optim.pth \
  --device cuda:0 --output output/experiments/EXP-20260909-020-geometry-aware-correspondence-loss/matched-pnp-smoke \
  --limit 16
# REPROJ_LW gradient-scale calibration
BOP_RENDERER_PATH="$PWD/.local/bop_renderer/build" \
  PYTHONPATH="$PWD" python -m research.exp020.calibrate_reproj_weight --device cuda:0
```

## 状态

`IMPLEMENTED / LOCAL_TEST_PASS / READY_FOR_FORMAL_RUN`。review-fix（2026-09-09）已完成：
EXP020 专用 matched PnP evaluator、A/B 跨 checkpoint fixed support、reprojection
diagnostics 修正、USE_MTL guard、gradient-scale calibration 均已实现并通过本地
测试/smoke；**没有 formal A/B 训练**，不宣称任何性能提升。正式 A/B 训练需要用户选择
服务器实验与配置后走 `docker/l40/experiment.sh`。主要下游评价复用本页 matched
evaluator（EXP019 已验证的 matched classical PnP/RANSAC consumer），不要仅因
`TEST.USE_PNP=True` 就默认协议一致；`eval.py` 只提供仓库标准的 BOP-AR/ADD(-S) 直接
姿态 telemetry，不作为主结论。
