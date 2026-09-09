# EXP020 — Geometry-Aware Correspondence Reprojection Supervision

## 研究问题与原协议

在保留当前 continuous normalized XYZ representation 的前提下，加入由 GT pose +
camera model 构造的 per-pixel geometric reprojection supervision，是否能让
Geometry/Correspondence Head 输出更真实、更适合普通 PnP 的 correspondence？

EXP019 结论（2026-09-08 用户 review）：当 XYZ correspondence 变好时，matched
classical PnP/RANSAC 与 EPro-PnP 都能稳定获得更好的 pose，而官方 Patch-PnP 对
相同 geometry improvement 响应不足。因此 EXP020 的瓶颈判断是 **correspondence
producer 的 supervision**，不在 solver：

- 唯一核心变量：`LOSS_CFG.REPROJ_LW`（A=`0.0`；B=`1.0`）。
- 明确不做：EPro-PnP / differentiable PnP / confidence head / Region+Local /
  surface code / spectral 表示 / 新 pose head / Patch-PnP 重构。
- 实验类型：`IMPLEMENTED / LOCAL_TEST_PASS / AWAITING_FORMAL_RUN`。本 RECORD
  只记录 2026-09-09 本地已发生的事实，**没有 formal 训练，不宣称任何性能提升**。

## Provenance：实现与运行信息

- experiment_id：`EXP-20260909-020-geometry-aware-correspondence-loss`。
- base commit：`ed6409ce73162d358c974fd967ab7ab130cf206b`
  （`EPro-GDRNPP研究` HEAD，与设计包制作时 HEAD 一致；未 reset，无前进）。
- 实现 commit：`64e9098a93d84087fb75ff569dd328dea8af1c55`（branch
  `exp020-geometry-aware-correspondence-loss`）。
- 设计包：用户提供 `GDRNPP_EXP020_correspondence_loss_kit.zip`（AGENT_PROMPT /
  DESIGN / INTEGRATION_GUIDE / CONFIG_PLAN / REVIEW_CHECKLIST / MANIFEST），本实现
  按仓库实际事实核对后落地，无与本提示词假设不一致的仓库事实。

## Changed files（Observed）

| 路径 | 改动 |
|---|---|
| `core/gdrn_modeling/losses/correspondence_reprojection_loss.py` | 新增纯 Tensor reprojection loss |
| `core/gdrn_modeling/models/GDRN_double_mask.py` | forward/gdrn_loss 透传 `roi_zoom_cams` 并计算 `loss_xyz_reproj`；REPROJ_LW=0 保持旧图/日志 |
| `core/gdrn_modeling/engine/engine.py` | 训练调用新增 `roi_zoom_cams=batch.get("roi_zoom_K", None)` |
| `configs/gdrn/lmo_pbr/research/exp020_geometry_aware_corr/{common,control,reproj,smoke_control,smoke_reproj,eval}.py` | 新增 A/B + smoke + eval 配置 |
| `research/exp020/{README,preflight,real_smoke}.py` | 实验说明、CPU preflight、真实数据 GPU smoke |
| `research/exp020/tests/` | 纯 loss 单测、gdrn_loss 接线测试、A/B config 测试 |

不修改 `data_loader.py`（第一版无需新 annotation）；不新增任何模型参数（官方
checkpoint strict 兼容，preflight 验证 missing/unexpected 均空）。

## 配置（Observed / A-B matched）

`REPROJ_LW` 为唯一实验变量。除它之外 A/B 的 checkpoint、freeze 布局、loss 布局、
optimizer/LR/schedule/batch/data/renderer/epochs/seed/evaluator 均由
`common.py` 共享并完全一致（`research/exp020/tests/test_exp020_config.py` 覆盖）。

- freeze：BACKBONE=True、PNP_NET=True、GEO_HEAD=False + TRAIN_SUPERVISION=True；
  XYZ_RENDERER=cpp（online geometry supervision）。
- 隔离 pose：`PM_LW=CENTROID_LW=Z_LW=ROT_LW=TRANS_LW=BIND_LW=0.0`
  （冻结 PNP_NET 参数不会阻断 pose loss 对 coor_feat 的梯度，因此显式关闭）。
- 保留 geometry supervision：XYZ L1 / mask / full mask / region。
- optimizer/LR 来源（Decision，依据仓库事实）：最近“从 official checkpoint
  fine-tune”协议为 Ranger `8e-4`、weight decay `0.01`、warmup `200`、batch 48、
  40 epoch、seed 42（EXP013A/E 与 PnP-only control 配置及 RECORD）；EXP020 复用。
  `REPROJ_LW=1.0` 为第一轮固定非零权重，未做 λ sweep；正式训练前可选做一次真实
  batch gradient-magnitude 标定，但 A/B 不得因此出现其它差异。

## 已运行的测试（Observed，本地 2026-09-09）

- 纯 loss 单测（GT 一致≈0、perturb 增大、梯度 finite/nonzero、empty mask、负深度、
  分辨率归一、形状校验等）：`research/exp020/tests/test_correspondence_reprojection_loss.py`
  → 13 passed。
- gdrn_loss 接线 + A/B config 测试：`test_exp020_gdrn_loss.py`（REPROJ_LW=0 保持旧
  六个 key、B 增 `loss_xyz_reproj` 且梯度有限、缺 crop K / CE XYZ / 缺 GT pose 均
  fail-fast、空 mask finite、GT 一致经 gdrn_loss≈0）与 `test_exp020_config.py`
  （A/B 仅差 REPROJ_LW、freeze/loss 布局、run contract smoke/formal/eval 全过）
  → 12 passed。
- 仓库回归子集（RUNBOOK 标准：research/tests + next_pose_head + exp013 +
  pose_structure）：82 passed。受共享代码影响的 EXP017/EXP017B/EXP018/EXP019 测试：
  45 passed。
- CPU preflight（加载 official checkpoint，full model do_loss forward/backward +
  Ranger step；official checkpoint missing/unexpected 均空；geo_head_net 是唯一
  trainable 且获得 finite gradient；frozen backbone/PNP 不变；REPROJ_LW=0 复检）：
  `python -m research.exp020.preflight --arm A|B --device cpu` → 两臂均 PASS。
- GPU 真实数据 smoke（cpp renderer 在线几何、真实 LM-PBR stage3 batch、2 步）：
  `python -m research.exp020.real_smoke --arm both --device cuda:0 --steps 2`
  → PASS。arm A 的 loss keys 无 `loss_xyz_reproj`；arm B 有。诊断注意：
  `loss_xyz_reproj` 是 SmoothL1 + reduction + /resolution + loss weight 后的
  **loss 标量，不是 pixel error**，不能用 `loss_xyz_reproj * 64` 换算成像素误差。
  2026-09-09 review-fix 后 real_smoke 改为记录 loss 模块真实的 `mean_reproj_px`
  （Euclidean px），重跑 smoke arm B ≈ `0.045` reproj_loss、`≈5.1 px`
  mean_reproj_px、`valid_ratio=1.0`、`behind_camera_ratio=0.0`，全程 finite。

合计本地证据：EXP020 新增测试 25 passed；仓库回归 82 + 45 passed。

## 下游评价计划（尚未运行）

- 主结论：训练完成后用 **EXP019 已验证的 matched classical PnP/RANSAC consumer**
  对 A/B checkpoint 做下游 pose 评价（XYZ decode、support、sampling、camera/crop、
  RANSAC threshold 与 metric 与 EXP019 协议一致），不以 `TEST.USE_PNP=True` 默认
  协议等价。
- `configs/.../exp020_geometry_aware_corr/eval.py` 仅提供仓库标准 BOP-AR/ADD(-S)
  直接姿态 telemetry（GT-box、USE_PNP=False），不是主结论。
- Gate 沿用 2026-09-08 相对阈值政策（±3%–±5%，饱和指标仍用绝对阈值），先看方向
  一致性：correspondence error → reprojection error → matched PnP pose 同方向。

## 未运行项（明确列出）

- **formal A/B 训练**：需要用户选择服务器实验与配置并授权
  （`docker/l40/experiment.sh`）；本任务未训练，不产生任何 epoch checkpoint。
- matched classical PnP/RANSAC 下游评价（依赖 formal checkpoint）。
- REPROJ_LW 标定/λ sweep（第一轮固定 1.0）。
- 多 seed / 稳定性复现（按仓库 policy 一次边缘结果不自动加 seed）。
- EXP019 历史证据未改动。

## 2026-09-09 审查修复（review-fix）

来源：用户提供的 `GDRNPP_EXP020_review_fix_kit.zip`（REVIEW_FINDINGS /
MATCHED_PNP_EVAL_SPEC / DIAGNOSTICS_FIX_SPEC / GRADIENT_CALIBRATION_SPEC /
TEST_CHECKLIST）。审查时 HEAD `395c8e37c5e1c63bd50fc07d0e730bde2c85519c`
（与实现 commit `64e9098` 后一致；当前分支即 exp020）。本段只记录实际发生的本地
事实；**没有 formal A/B 训练，不宣称性能提升**。

### 已通过的实现（review 结论，未重写）

reprojection loss 数学链、XYZ normalization、roi_zoom_K + output pixel grid
坐标系、engine wiring、A/B training isolation 均正确且未改；未引入 EPro-PnP、
未重构 Patch-PnP、未改 EXP019 历史 evaluator/结果。

### Changed files（Observed，review-fix commit `e5ad38bfcdbde2c5e531e2d3fa9a25954abb4547`）

| 路径 | 改动 |
|---|---|
| `research/exp020/matched_pnp_eval.py` | 新增 EXP020 专用 matched classical PnP/RANSAC evaluator（reference/A/B/device/output/limit；复用 EXP019 纯 helper；不启动 EPro、不跑 alpha） |
| `research/exp020/calibrate_reproj_weight.py` | 新增 REPROJ_LW gradient-scale 标定（g_xyz / g_reproj_raw / ratio_raw，真实 online-geometry batch） |
| `core/gdrn_modeling/losses/correspondence_reprojection_loss.py` | 只改 stats：`valid_ratio` 分母改为 GT foreground；新增 `gt_foreground_count`、`positive_depth_ratio_on_gt_fg`、`behind_camera_ratio_on_gt_fg`；`mean_reproj_px` 保持真实 Euclidean px；loss mask/数值未改 |
| `core/gdrn_modeling/models/GDRN_double_mask.py` | `REPROJ_LW>0 && USE_MTL=True` fail-fast（forward 与 gdrn_loss，不新增 log_var）；vis_extra 增加 `vis/reproj_gt_fg_count`、`vis/reproj_positive_depth_ratio`、`vis/reproj_behind_camera_ratio` |
| `research/exp020/real_smoke.py` | `reproj_px_loss` → `reproj_loss`；记录 loss 模块真实的 `mean_reproj_px` 与 depth 比例（EventStorage），不再用 loss×64 当 px |
| `research/exp020/tests/` | 新增 loss stats（fg 分母、positive/behind、Euclidean px）、USE_MTL guard、matched evaluator 协议测试；`test_matched_pnp_eval.py`（fixed-support 一致性、A==B 等价、非 finite 不重建 support、无 EPro/alpha、CLI 无 epro 参数） |
| `research/exp020/README.md`、EXP020 RECORD、`research/STATUS_CN.md`、`research/EXPERIMENT_INDEX.md`、`research/REPOSITORY_MAP.md` | 记录 review-fix 事实；STATUS 旧 EPro-PnP 主线标为 Historical/Deferred |

### 跨 checkpoint fixed support（主协议）

`S_fixed = reference_pred_visible ∩ gt_visible ∩ valid_depth` 由 reference
checkpoint（默认 official）生成一次；support mask、flat indices、subsample
indices、2D 点、K、RANSAC seed（20260730+target 序号）/threshold（3px）/
iterations（100）全部冻结。A/B 只替换各自 predicted XYZ（decode 后取固定 indices）。
native support 仅允许作为显式标明的 secondary analysis（本轮未启用）。
EXP019 runner 因强校验 official SHA-256 且固定 EPro + alpha sweep，不能直接评价
EXP020 checkpoint；未修改/未放宽 EXP019 代码。

### 已运行的测试与 smoke（Observed，本地 2026-09-09）

- EXP020 测试合计 **36 passed**（原 25 项 + 新增 11 项：loss stats 新增 3 项
  fg 分母/positive-behind/Euclidean px 语义、gdrn_loss 新增 2 项 USE_MTL guard、
  matched evaluator 新增 6 项协议测试）。
- 仓库回归子集（research/tests + next_pose_head + exp013 + pose_structure +
  exp017/017b/018 + exp019_epro tests）：**127 passed**。
- CPU preflight arm A / arm B：均 PASS（official checkpoint 兼容、geo_head 唯一
  trainable、REPROJ_LW=0 兼容复检）。
- GPU 真实数据 smoke（`real_smoke --arm both --steps 2`）：PASS。arm B
  `reproj_loss≈0.045`、真实 `mean_reproj_px≈5.1 px`、`valid_ratio=1.0`、
  `behind_camera_ratio_on_gt_fg=0.0`。
- **matched evaluator identity smoke**：reference=A=B=official、`--limit 16`、真实
  LM-O → 16 targets COMPLETE。A/B 完全相同（`max_abs_R/t/corr/reproj delta = 0.0`），
  solve success 1.0，mean corr err ≈ 10.44 mm、mean reproj err ≈ 1.99 px、mean
  support ≈ 691。metadata：`epro_started=False`、`alpha_sweep=False`、GT XYZ max
  reprojection ≈ 0.072 px（<0.5 sanity）。该 smoke 只验证 evaluator 接线，不代表
  科学性能。
- **非 official checkpoint 加载 smoke**：reference=official、A=B=扰动 geo 权重
  checkpoint（shared out_layer + features conv 加噪声）、`--limit 8` → 8 targets
  COMPLETE；A/B 一致 delta=0.0。证明 evaluator 不依赖 official SHA-256。
- **gradient-scale calibration 实际运行**（真实 online-geometry batch，GPU）：
  `g_xyz = 4.9539`、`g_reproj_raw = 0.3695`、`ratio_raw = 0.0746`（分组：xyz
  output layer 4.9511 vs 0.3692；shared trunk 0.1662 vs 0.0149）。REPROJ_LW 梯度比
  XYZ 三项总梯度小约 13×，属同数量级、不构成“小几十到几百倍”的强/弱失衡；
  **未修改 formal `REPROJ_LW=1.0`**。工具只做尺度标定，不做 λ sweep。

### 状态

`IMPLEMENTED / LOCAL_TEST_PASS / AWAITING_FORMAL_RUN`。review-fix 已收口，但
**没有 formal A/B 训练，不产生任何 epoch checkpoint，不宣称任何性能提升**。

