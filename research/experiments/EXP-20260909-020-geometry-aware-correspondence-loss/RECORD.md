# EXP020 — Geometry-Aware Correspondence Reprojection Supervision

## 研究问题与隔离协议

在保留 continuous normalized XYZ 的前提下，加入由 GT pose、object extent 和
crop-resized camera matrix 构造的 per-pixel correspondence reprojection loss，是否
能让 Geometry/Correspondence Head 输出更真实、更适合普通 PnP 的 correspondence？

EXP019 表明，当 XYZ correspondence 受控改善时，matched classical PnP/RANSAC 与
EPro-PnP 都能稳定改善 pose，而官方 Patch-PnP 响应不足。因此 EXP020 只研究
correspondence producer 的 supervision，不引入 EPro-PnP、differentiable PnP、
confidence head、新 pose head 或 Patch-PnP 重构。

- experiment_id：`EXP-20260909-020-geometry-aware-correspondence-loss`。
- base commit：`ed6409ce73162d358c974fd967ab7ab130cf206b`；实现 commit：
  `64e9098a93d84087fb75ff569dd328dea8af1c55`；review-fix commit：
  `e5ad38bfcdbde2c5e531e2d3fa9a25954abb4547`。
- 唯一核心变量：`LOSS_CFG.REPROJ_LW`，A=`0.0`，B=`1.0`。
- 两臂共用 official checkpoint、seed 42、40 epoch、batch 48、Ranger lr `8e-4`、
  wd `0.01`、warmup 200、数据与 evaluator；训练期在线 GT geometry 使用 EGL，
  BOP evaluation 独立保持 CPP。
- backbone/PNP_NET frozen，GEO_HEAD trainable；保留 XYZ/mask/full-mask/region loss；
  `PM/CENTROID/Z/ROT/TRANS/BIND_LW` 全部为 0，隔离 producer。
- 无新增模型参数，official checkpoint strict compatible。

## 实现（Observed）

- `core/gdrn_modeling/losses/correspondence_reprojection_loss.py`：把
  `xyz_norm` 解码为 object XYZ，经 GT pose 与 `roi_zoom_K` 投影，监督其回到对应的
  output-map pixel grid；前景且预测 camera depth 为正、投影 finite 的像素参与 loss。
- `core/gdrn_modeling/models/GDRN_double_mask.py` 与 `engine/engine.py`：透传
  `roi_zoom_K` 并接入 `loss_xyz_reproj`。`REPROJ_LW=0` 保持历史 loss graph/key；
  缺必要输入、XYZ 非 L1 或 `REPROJ_LW>0 && USE_MTL=True` 时 fail-fast。
- loss stats 将训练标量与真实 Euclidean pixel error 分开：`mean_reproj_px`、以 GT
  foreground 为分母的 `valid_ratio`，以及 positive/behind-camera ratio。
- `configs/gdrn/lmo_pbr/research/exp020_geometry_aware_corr/`：共享 formal 协议及
  A/B、smoke、eval 配置。
- `research/exp020/matched_pnp_eval.py`：EXP020 主下游 evaluator；
  `research/exp020/calibrate_reproj_weight.py`：单真实 batch 梯度尺度标定。

## 主评价协议

reference checkpoint（默认 official）只生成一次
`S_fixed = reference_pred_visible ∩ gt_visible ∩ valid_depth`。support mask、flat
indices、subsample、2D 点、K、RANSAC seed、3 px threshold 与 100 iterations 全部
冻结；A/B 只替换各自预测 XYZ。native support 只能作为明确标注的 secondary
analysis。本实验不复用 EXP019 runner，因为后者强校验 official checkpoint 且固定
启动 EPro/alpha sweep。

正式结果首先检查方向链是否一致：correspondence error → GT-pose reprojection error
→ matched PnP pose。完整 checkpoint 可追加 `--bop-eval` 汇总 BOP AR、ADD(-S)、
AR_reS、AR_teS；仓库标准 direct-pose `eval.py` 只作 telemetry。

## Formal 前证据（Observed）

- EXP020 tests：2026-09-09 review-fix 后与本次整理时均为 **36 passed**。覆盖 loss
  数学/梯度/空 mask/深度与像素统计、A/B config、模型接线与 USE_MTL guard、fixed
  support、A==B、non-finite prediction、CLI 不含 EPro/alpha。
- 受 shared code 影响的仓库回归子集：review-fix 后与本次整理时均为
  **127 passed**；CPU preflight A/B 均 PASS，official checkpoint
  missing/unexpected 为空，只有 GEO_HEAD trainable。
- GPU real smoke A/B PASS。B：`reproj_loss≈0.045`、真实
  `mean_reproj_px≈5.1 px`、`valid_ratio=1.0`、`behind_camera_ratio=0.0`。
- matched evaluator identity smoke（official=reference=A=B，16 targets）COMPLETE：
  solve success 1.0，mean correspondence error `10.4353 mm`，mean reprojection error
  `1.9940 px`，mean fixed support `690.75`，A/B 全部 delta 为 0；GT XYZ 最大重投影
  error `0.0719 px`。`epro_started=False`、`alpha_sweep=False`。
- 非 official checkpoint loading smoke（reference=official、A=B=perturbed，8
  targets）COMPLETE，A/B 全部 delta 为 0，证明 evaluator 不依赖 official SHA。
- gradient-scale calibration 已运行：`g_xyz=4.9539`、`g_reproj_raw=0.3695`、
  `ratio_raw=0.0746`。REPROJ 梯度约为 XYZ 三项总梯度的 1/13，未发现数量级失衡，
  formal `REPROJ_LW=1.0` 保持不变；未做 λ sweep。

这些 smoke 和 calibration 只验证工程链路与尺度，不是性能结果。

## 2026-09-09 renderer protocol correction

首个服务器 release `698a8fe` 将训练期 `XYZ_RENDERER` 错配为 CPP。A/B smoke
`RUN-20260909-084019-smoke-s42-a01` / `RUN-20260909-084107-smoke-s42-a01` 虽完成
1 epoch 且 loss finite，但每批在线 GT geometry 都走 CPU-oriented CPP renderer，
造成明显不必要的运行时间；这些 run 只证明旧配置能执行，不作为修正后 formal 的
smoke gate。修正将 A/B 共用 `XYZ_RENDERER` 改为 EGL，不改变唯一变量
`REPROJ_LW`；`VAL.RENDERER_TYPE=cpp` 保持不变，以保留 BOP 指标可比性。旧 release
若已启动 formal，应视为无效协议 run，不进入科学结论。本地 effective-config 与
36 项 EXP020 测试通过；本地无可用 CUDA runtime，EGL renderer 的真实执行和速度
必须由新 release 的服务器 smoke 验证。

同次处置发现 launcher 的旧 idle gate 用 `pgrep -f main_gdrn.py`，会漏掉
`setproctitle` 后名为 `control.TIMESTAMP` / `reproj.TIMESTAMP` 的存活训练进程。该
基础设施缺陷改为用 `docker top` 拒绝容器内除 PID 1 `sleep infinity` 外的任何进程；
旧 run 输出仍保留，修复不改变模型或 A/B 科学变量。

修正后的首轮 EGL smoke 中，B run
`RUN-20260909-113749-smoke-s42-a01` 在 EGL context 创建后、加载首个 PLY 时失败：
`load_mesh_sixd()` 尝试在只读源码挂载根目录创建相对路径 `.cache`，触发
`OSError: [Errno 30] Read-only file system: '.cache'`。这是运行缓存路径缺陷，不是
EGL/CUDA 初始化失败；日志已显示 CUDA device 0 上的 EGL 1.5 context 创建成功。
`503376a` 的 A run `RUN-20260909-120624-smoke-s42-a01` 进一步确认外层
`load_mesh_sixd` 已使用可写 cache，但该函数内部
调用 `load_mesh_pyassimp(use_cache=False)` 时没有透传 `cache_dir`，嵌套 loader 再次
回落到只读 `.cache`。现由训练 renderer 在同步加载全部模型期间临时将 cwd 切换到
`${XDG_CACHE_HOME}/gdrnpp_egl_meshes`，使两层 loader 的相对 `.cache` 都落入可写挂载，
并在成功或异常后恢复原 cwd。本地未设 XDG 时保持原行为。实现不修改镜像所绑定的
`lib/egl_renderer` 原生输入，因此继续复用稳定镜像。这些失败 run 不进入科学结论；
修复后的 A/B smoke 随后通过，并进入下述 formal。

## Formal partial：E5/E10 与训练日志快照（2026-09-10）

修正后的 EGL A/B formal 使用相同 source commit
`c2a7723d5cf468de97ef8aa6f860c4f474203f30`、seed 42 和镜像
`gdrnpp-research:torch220-cu121-sm89-c0be1ade7ea9`；镜像 ID 为
`sha256:124d9ba7c7754b4445bfb7a1a177e15073f8139a376ec4e0701ec336be6944da`，
build revision 为 `c0be1ade7ea94d968256d81051caeb7b6a0db9ba`：

- A：`RUN-20260909-124125-formal-s42-a01`，`control.py`。
- B：`RUN-20260909-124137-formal-s42-a01`，`reproj.py`。
- 当前外置证据：`E:\6D姿态估计\EXP020\实验A` 与 `实验B`。每臂包含
  `console.log`、`scores_bop19_5epoch.json` 和 `scores_bop19_10epoch.json`。
- A/B 日志均记录生成了 `model_epoch_005.pth`、`model_epoch_010.pth`；对应 E5/E10
  均已有 direct-pose evaluation。
- 日志快照覆盖 A 至 iteration 93,499（epoch 15），B 至 iteration 88,999（epoch 14）；
  两份日志均未包含 run exit code，外置目录中尚无 E15–E40 score JSON。

### Direct-pose 聚合指标（Observed）

| Epoch | 臂 | BOP AR | ADD(-S)0.1d | AR_reS | AR_teS |
|---:|---|---:|---:|---:|---:|
| 5 | A | 0.563426 | 0.255363 | 0.433679 | 0.634141 |
| 5 | B | 0.567089 | 0.278893 | 0.449366 | 0.638754 |
| 10 | A | 0.518987 | 0.253979 | 0.419839 | 0.571857 |
| 10 | B | 0.521398 | 0.253979 | 0.412918 | 0.573010 |

输入值来自各 checkpoint 的 `EVAL_SUMMARY` 和对应 BOP score JSON。

### A/B 差值（Derived）

差值均按 `B − A` 计算，相对变化按 `(B-A)/A` 计算。

| Epoch | 指标 | B − A | 相对变化 |
|---:|---|---:|---:|
| 5 | BOP AR | +0.003663 | +0.65% |
| 5 | ADD(-S)0.1d | +0.023529 | +9.21% |
| 5 | AR_reS | +0.015686 | +3.62% |
| 5 | AR_teS | +0.004614 | +0.73% |
| 10 | BOP AR | +0.002411 | +0.46% |
| 10 | ADD(-S)0.1d | 0.000000 | 0.00% |
| 10 | AR_reS | -0.006920 | -1.65% |
| 10 | AR_teS | +0.001153 | +0.20% |

### ADD(-S)0.1d 逐物体结果（Observed / Derived）

| 物体 | A E5 | B E5 | B−A E5 | A E10 | B E10 | B−A E10 |
|---|---:|---:|---:|---:|---:|---:|
| ape | 0.125714 | 0.068571 | -0.057143 | 0.000000 | 0.000000 | 0.000000 |
| can | 0.301508 | 0.346734 | +0.045226 | 0.276382 | 0.266332 | -0.010050 |
| cat | 0.210526 | 0.216374 | +0.005848 | 0.175439 | 0.169591 | -0.005848 |
| driller | 0.345000 | 0.355000 | +0.010000 | 0.320000 | 0.300000 | -0.020000 |
| duck | 0.033333 | 0.027778 | -0.005556 | 0.016667 | 0.016667 | 0.000000 |
| eggbox | 0.405556 | 0.477778 | +0.072222 | 0.477778 | 0.505556 | +0.027778 |
| glue | 0.314286 | 0.342857 | +0.028571 | 0.278571 | 0.271429 | -0.007143 |
| holepuncher | 0.295000 | 0.375000 | +0.080000 | 0.450000 | 0.465000 | +0.015000 |

E5 的逐物体差值计数为 6 个正值、2 个负值；E10 为 2 个正值、4 个负值、2 个零值。

### 训练日志快照（Observed）

`my_writer` 的括号外数值是最近 20 iteration 的 median，括号内数值是从 iteration 0
起累计的 global average。下表记录两份日志最后一行的括号内数值：

| 臂 | 最后 iteration | epoch | total | region | coor_x | coor_y | coor_z | xyz_reproj | mask | mask_full | max_mem |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 93,499 | 15 | 16.49 | 16.37 | 0.02731 | 0.02554 | 0.01966 | — | 0.02257 | 0.02274 | 8286M |
| B | 88,999 | 14 | 16.49 | 16.37 | 0.02719 | 0.02544 | 0.01968 | 0.009151 | 0.02251 | 0.02266 | 8295M |

按表中显示值计算，两臂 `loss_region / total_loss = 16.37 / 16.49 = 99.27%`。
两份日志均未检出 `Traceback`、CUDA OOM 或 `ERROR`。

### Logged region loss 口径（Code-derived）

当前实现对 `out_region * gt_mask` 计算 65-way cross entropy，先对图像求和，再除以
前景像素数。背景位置的输入 logits 为 0，因此其标量交叉熵为 `log(65)`；乘法链式
导数中对应 `gt_mask=0`，这些位置对原始 `out_region` 的导数为 0。上述背景标量包含在
日志的 `loss_region` 与 `total_loss` 中。

## 当前状态与待运行项

状态：`FORMAL_RUNNING / E10_AVAILABLE / PARTIAL_EVIDENCE`。

当前已有 E5/E10 direct-pose telemetry；日志快照分别覆盖 A 的 epoch 15 与 B 的
epoch 14。E15–E40 结果、run exit code 和 matched PnP/RANSAC 评价均未提供。
