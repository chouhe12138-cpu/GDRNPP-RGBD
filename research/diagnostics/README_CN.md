# EXP013 Pose Head 结构诊断

> 该工具已经在本地 WSL、真实 LM-O PBR 数据和 EXP013 A/B E40 checkpoint 上验证。诊断指标不是正式 BOP 指标；当前 8 个样本只用于判断机制，不用于替代正式 E40 gate。

## 原始设计说明

# GDRNPP-RGBD 姿态头结构诊断包

该诊断框架已经适配并纳入仓库。它不修改训练权重和 A/B/C 配置；D4/D5 虽使用 autograd，但不执行 optimizer step，运行结束还会校验 pose-head state 哈希。

## 1. 诊断目标

框架只检查会实质改变结构决策的问题，不做普通超参数扫描。当前实现包含六组诊断：

| ID | 诊断 | 核心问题 | 默认成本 |
|---|---|---|---|
| D1 | R/t Oracle | 最终误差主要卡在 rotation 还是 translation | forward-only |
| D2 | Branch causal ablation | A 的 geometry bypass / B 的 attention 是否真正参与最终 pose | forward-only |
| D3 | Correspondence utilization | XYZ 更接近 GT 时，learned pose 是否稳定改善 | forward-only，多次 head forward |
| D4 | R/t gradient conflict | R/t 对共享表示的优化方向是否持续冲突 | 单次 forward + autograd，无 optimizer step |
| D5 | Geometry interface adaptation | 冻结 Geometry→Pose 接口是否存在明显适配失配 | 只优化临时输入 tensor，不更新模型权重 |
| D6 | Spatial sensitivity | rotation 是否真正使用保留下来的空间排列 | forward-only，固定 cell permutation |

这些诊断不会重新训练模型。D4/D5 会使用 autograd，但不会调用 optimizer，不会修改 checkpoint 参数。

## 2. 设计依据与当前仓库接口

当前 GDRN forward 为：Backbone → Geometry Head → XYZ/Region/Mask → `XYZ + ROI2D` → PnP/Pose Head → raw R/centroid-z → ego R,t。

EXP013-A 在 EXP012 主路径旁增加 Region-free 的 `XYZ+ROI2D` geometry residual；EXP013-B 在 A 的 geometry path 上增加 16×16、3×3 local geometry attention。当前 C 继承 B 并进行 R/t late decoupling。本框架通过运行时捕获 `pnp_net` 的真实输入，不要求修改 `GDRN.forward()`。

## 3. 运行环境

在仓库根目录激活固定环境：

```bash
source /home/wsluser/miniconda3/etc/profile.d/conda.sh
conda activate pytorch22
```

本地 WSL 使用 `--xyz-renderer cpp`。checkpoint、完整输出和数据不进入 Git。

## 4. 推荐第一次运行

先用少量样本检查代码兼容性：

```bash
python tools/run_pose_structure_diagnostics.py \
  --config-file configs/gdrn/lmo_pbr/research/exp013/a_xyz_residual/train.py \
  --checkpoint /ABS/PATH/TO/model_epoch_040.pth \
  --output-dir output/diagnostics/exp013a_e40_smoke \
  --diagnostics d1,d2,d3 \
  --max-batches 1 \
  --batch-size 2 \
  --num-workers 0 \
  --device cuda:0 \
  --xyz-renderer cpp
```

通过后再运行完整低成本诊断：

```bash
python tools/run_pose_structure_diagnostics.py \
  --config-file configs/gdrn/lmo_pbr/research/exp013/a_xyz_residual/train.py \
  --checkpoint /ABS/PATH/TO/model_epoch_040.pth \
  --output-dir output/diagnostics/exp013a_e40 \
  --diagnostics d1,d2,d3,d4,d5,d6 \
  --max-batches 4 \
  --batch-size 2 \
  --num-workers 0 \
  --device cuda:0 \
  --xyz-renderer cpp \
  --no-explicit-pnp \
  --d5-steps 3
```

B 只需替换 config/checkpoint。未来 clean C 或同风格 pose head 也使用同一入口。

## 5. 输出

每次运行生成：

```text
<output-dir>/
├── results.json
└── SUMMARY.md
```

`results.json` 是后续脚本/Codex分析的机器可读结果；`SUMMARY.md` 是同一内容的直接查看版本。

## 6. 六个诊断的判断逻辑

### D1：R/t Oracle

比较 `PredR+Predt`、`GTR+Predt`、`PredR+GTt`、`GTR+GTt`。默认记录 rotation degree error、translation cm error 和 per-class 统计。

注意：本框架的快速诊断指标**不是官方 BOP AR / ADD(-S)**。如果需要最终正式 gate，应在结构假设通过后再接入现有 BOP evaluator。D1 的作用是先定位误差归属，不是替代正式评估。

### D2：A/B 分支因果消融

自动检测当前 head 支持的 intervention：

- `normal`
- `region_zero`
- A/B：`main_only`（geometry scale=0）
- A/B：`geometry_only`
- B/C：`attention_zero`

额外记录 `main_latent_rms`、`geometry_latent_rms`、两者 cosine、`geometry_scale`、`attention_scale`、attention entropy 等。

关键判断：如果 `normal ≈ main_only`，新增 geometry residual 对最终 pose 贡献很弱；如果 `normal ≈ attention_zero`，B 的 attention 不是决定性模块；如果 Region×0 大幅下降且 geometry-only 很弱，则 Region-free bypass 并没有形成独立有效的 pose 路径。

### D3：XYZ 利用能力

默认 alpha：`0, 0.25, 0.5, 0.75, 1`。

- `fixed_region`：只替换 XYZ，Region 固定为原预测。
- `synced_region`：根据混合后的 metric XYZ + FPS points 重新计算 Region；若 FPS 信息不可用，则框架会报告 skip。
- `explicit_pnp`：当原始样本保留图像宽高时，额外尝试 OpenCV PnP 作为几何参考；缺少宽高时自动 skip，不影响 learned-head 诊断。

若显式 PnP 随 XYZ 改善，而 learned head 不改善或反向恶化，优先判定为 correspondence→pose 利用问题，而不是 Geometry Head 缺少 pose 信息。

### D4：R/t 梯度冲突

从训练 loss 中自动寻找 `loss_PM_R` 和 translation 相关 loss，对 late shared latent（若存在）以及 pose head 共享参数计算两组 gradient cosine。

持续出现负 cosine 支持“共享表示存在优化冲突”，可作为 clean R/t decoupling 的机制证据，但不能单独证明解耦一定提升最终指标。

### D5：冻结 Geometry 接口适配

不改权重。将 pose head 输入的 XYZ（可选 Region logits）作为临时可优化 tensor，依据 rotation pose objective 做 1~5 个小步更新，并和同 RMS 随机扰动比较。

如果非常小的 task-directed 输入变化就能显著降低 rotation error，而同强度 random perturbation 没效果，说明 frozen Geometry→Pose 接口存在明显的 task-adaptation 空间，才值得进一步考虑 Geometry+Pose joint adaptation。

### D6：空间排列敏感性

保持 descriptor 中所有数值和每个 channel 的统计不变，只固定打乱 spatial cell order：

- `main_grid_shuffle`
- A/B/C：`geometry_grid_shuffle`
- `both_grid_shuffle`

如果 rotation 几乎不变，说明 decoder 对空间顺序利用很弱；如果 rotation 明显受损而 translation 稳定，说明空间排列确实主要服务 rotation。

## 7. 当前实现刻意没有做的事情

1. **不把快速 train/PBR subset 指标冒充 BOP formal 指标。**
2. **不修改 GDRN forward。** 诊断通过 runtime capture/context intervention 完成，降低对主训练路径的侵入。
3. **不自动删除 online renderer。** 这是另一项训练效率修复。D1/D3/D4 需要 GT XYZ/Region/pose 时仍沿用当前 GT-enabled train path，保证诊断先能复现仓库当前数据协议。
4. **不把现有 C 当作 clean R/t control。** 本包可以诊断现有 C，但如果 A/B gate fail，建议后续重建从 EXP012 出发的 clean C。

## 8. 本地 Codex 必须优先验证的兼容点

请让 Codex 按 `CODEX_HANDOFF.md` 顺序检查。最重要的是：

- checkpoint loading 是否匹配当前 managed-run 目录；
- `get_renderer()` 在服务器现有 cpp/EGL 配置下是否正常；
- 当前 PyTorch 版本是否支持本包使用的 hook 行为；
- `do_loss=True` 时 D4 能否拿到 `loss_PM_R/loss_centroid/loss_z`；
- A/B 的 `_encode_main/_encode_geometry/geometry_scale/attention_scale` 名称是否与工作分枝一致；
- D3 的 `roi_fps_points` 是否随当前 online batch 保留；
- 少量样本 smoke 后再扩到 32~128 个实例。
