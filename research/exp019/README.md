# EXP019 — EPro-PnP Geometry Utilization Diagnostic

本诊断把用户提供的 EPro-PnP-v2 参考实现接入历史 EXP004 的固定 geometry
intervention，不训练、不修改 Geometry Head，也不使用 EXP018 correction。

## 科学协议

- official checkpoint；LM-O BOP19 1,445 targets；GT bbox；FP32 producer；无训练。
- alpha 固定为 `[0, .25, .5, .75, 1]`。
- 只在历史 fixed support 内执行
  `xyz_alpha = (1-alpha)*pred_xyz + alpha*gt_xyz`。
- fixed support 精确恢复自 Git `d702030`：predicted visible mask 超过官方
  `MASK_THR_TEST`，XYZ 三轴不是背景零点，再与 GT visible、有效 depth 取交集。
- Patch-PnP、RANSAC、EPro 使用同一次 producer forward 和同一 `xyz_alpha`；
  RANSAC/EPro 使用同一个 correspondence set。
- Patch-PnP 保留官方 ROI2D、Region softmax、mask attention、extent 和 decode。
- RANSAC 固定 EPNP、3 px、100 iterations、confidence 0.99。
- EPro 使用 OpenCV EPNP init、EProPnP6DoF、AdaptiveHuberPnPCost、LM/GN，
  `w2d=uniform`；不加入 learned reliability、BPnP 或超参 sweep。

## 本地适配

第三方源码不复制进 Git。GDRNPP 与 EPro 都使用顶层 `lib` 包，因此 EPro 在持久
spawn 子进程运行。压缩包内旧 Pyro 版本拒绝 PyTorch 2.x；适配器在子进程提供仅覆盖
EPro inference 所需 distribution API 的兼容层，不修改 Conda 包或第三方源码。

当前 official config：
`configs/gdrn/lmo_pbr/research/_base_/lmo_gt_eval.py`。代码强制 official
ConvPnPNet、absolute ROI2D、Region attention、FP32，并确认 pose corrector 不存在。

## 执行顺序

先将用户提供的 zip 解压到 ignored 的 `.local/third_party`：

```bash
(
set -Eeuo pipefail
cd /home/wsluser/GDRNPP-RGBD
mkdir -p .local/third_party
unzip -q "/mnt/e/6D姿态估计/Zcode/EXP019/EPro-PnP-v2-main.zip" \
  -d .local/third_party
test -f .local/third_party/EPro-PnP-v2-main/EPro-PnP-6DoF_v2/lib/ops/pnp/epropnp.py
)
```

然后在 `pytorch22` 环境依次执行 tests、preflight、32-target balanced smoke。
smoke 每个 LM-O 物体最多选择四个 target，只验证 wiring，不生成正式结论。

完整 1,445-target raw inference 必须使用 clean committed source。完成后单独运行
`evaluate`，它导出 15 个 BOP CSV，调用当前 `eval_pose_results_more.py`，生成
`summary.json`、`reproduction_report.json` 和 `gate_report.json`。

Gate 前必须先通过历史复现：Patch/RANSAC alpha=0 与 RANSAC alpha=1 的 ADD/BOP
绝对差不超过 0.001。正式 EPro gate 为 alpha=1 ADD/BOP 均至少 0.95，ADD/BOP 的
Spearman 均至少 0.90，且相对 matched RANSAC 的恢复比例均至少 0.50。

## 证据边界

PASS 只支持显式 solver 能稳定消费逐步改善的 correspondence geometry，不支持
learned reliability、solver-oriented training、EPro-GDRN 优越性或部署收益。

