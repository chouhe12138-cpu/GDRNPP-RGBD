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
`lib/egl_renderer` 原生输入，因此继续复用稳定镜像。这些失败 run 不进入科学结论，
修复后需重新执行 A/B smoke。

## 当前状态与待运行项

状态：`IMPLEMENTED / LOCAL_TEST_PASS / READY_FOR_EGL_SMOKE`。

尚未运行有效 formal A/B 训练、matched PnP/RANSAC 正式评价、BOP 聚合或多 seed，
因此不宣称性能提升。下一步以修正后的确定 commit 生成 bundle，重跑 EGL A/B smoke；
通过后才启动唯一 formal A/B，训练完成后按 fixed-support 主协议评价。
