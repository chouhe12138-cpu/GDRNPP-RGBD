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
PnP-only control：Ranger 8e-4、wd 0.01、warmup 200）。`REPROJ_LW=1.0` 是第一轮固定
非零值；正式训练前可选做一次真实 batch 的 gradient-magnitude 标定，但 A/B 之间除
`REPROJ_LW` 外必须完全一致。

## 本地执行

```bash
source /home/wsluser/miniconda3/etc/profile.d/conda.sh
conda activate pytorch22
PYTHONPATH="$PWD" pytest -q research/exp020/tests
PYTHONPATH="$PWD" python -m research.exp020.preflight --arm A --device cpu
PYTHONPATH="$PWD" python -m research.exp020.preflight --arm B --device cpu
BOP_RENDERER_PATH="$PWD/.local/bop_renderer/build" \
  PYTHONPATH="$PWD" python -m research.exp020.real_smoke --arm both --device cuda:0
```

## 状态

`IMPLEMENTED / LOCAL_TEST_PASS / AWAITING_FORMAL_RUN`。正式 A/B 训练需要用户选择
服务器实验与配置后走 `docker/l40/experiment.sh`。主要下游评价复用 EXP019 已验证的
matched classical PnP/RANSAC consumer（不要仅因 `TEST.USE_PNP=True` 就默认协议
一致）；`eval.py` 只提供仓库标准的 BOP-AR/ADD(-S) 直接姿态 telemetry，不作为主结论。
