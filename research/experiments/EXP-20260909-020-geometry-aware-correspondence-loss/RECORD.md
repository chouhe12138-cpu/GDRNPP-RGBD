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
  → PASS。arm A 的 loss keys 无 `loss_xyz_reproj`；arm B 有且
  `loss_xyz_reproj`≈0.0448（≈2.9 px @ 64px，除以 res 后量级），全程 finite。

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

## 状态

`IMPLEMENTED / LOCAL_TEST_PASS / AWAITING_FORMAL_RUN`。
