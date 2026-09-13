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

- EXP020 tests：2026-09-09 review-fix 后与 2026-09-10 整理时均为 **36 passed**。覆盖 loss
  数学/梯度/空 mask/深度与像素统计、A/B config、模型接线与 USE_MTL guard、fixed
  support、A==B、non-finite prediction、CLI 不含 EPro/alpha。
- 受 shared code 影响的仓库回归子集：review-fix 后与 2026-09-10 整理时均为
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

## 历史快照：E5/E10 与训练日志（2026-09-10）

本节保留当时的证据范围；当前完整评估点见下方 2026-09-13 更新。

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

## 2026-09-13 更新：E5–E40 direct-pose telemetry 已齐

### 运行与来源（Observed）

状态：`E40_AVAILABLE / MATCHED_PNP_PENDING / EXIT_CODE_UNCONFIRMED`。

沿用上文相同 A/B run ID、source commit `c2a7723d5cf468de97ef8aa6f860c4f474203f30`、
seed 42、配置与镜像。此次只同步外置结果，不重新运行训练、测试或 evaluator。

- A/B 均已有 E5/10/15/20/25/30/35/40 全部预定 direct-pose 评估点。
- 两臂日志均到达 epoch 40、iteration `255919/255920[100.0%]`。
- A 在 `20260911_225832`、B 在 `20260912_031606` 记录保存
  `checkpoints/model_epoch_040.pth`；E40 EVAL_SUMMARY 分别记录于
  `20260911_230718` 和 `20260912_032456`（均为日志原始时间戳，未另行换算时区）。
  checkpoint 保存由日志证实，本次未检查服务器上的权重文件。
- 本次读取的两份 console.log 未检出 Traceback、out of memory 或 ERROR；
  未提供明确 run exit code，不能把日志到达 E40 等同于退出状态已核验。
- 正式 matched PnP/RANSAC 结果尚未提供；本地现存 matched evaluator 产物仅为
  identity/perturbed smoke，不能替代 E40 主评价。

原始来源仍为 `E:\6D姿态估计\EXP020\实验A` 与 `实验B`。每臂的 8 份
`scores_bop19_<epoch>epoch.json` 已原样复制至下方对应 run 目录；
`eval_summary.jsonl` 原样抽取 console.log 各行 `EVAL_SUMMARY ` 后的 JSON，保持字段、
精度与顺序（仅去掉日志前缀），包含全部评估点的 ADD(-S)0.1d 与逐物体 recall。

- A：[evidence/RUN-20260909-124125-formal-s42-a01](evidence/RUN-20260909-124125-formal-s42-a01/)。
- B：[evidence/RUN-20260909-124137-formal-s42-a01](evidence/RUN-20260909-124137-formal-s42-a01/)。

完整日志、checkpoint 和大型评估产物继续外置。ADD(-S)0.1d 取自
`EVAL_SUMMARY.add_s_0.1d`，不以 `bop19_average_recall_ad` 替代。

### 全部预定评估点（Observed）

| Epoch | 臂 | BOP AR | ADD(-S)0.1d | AR_reS | AR_teS |
|---:|---|---:|---:|---:|---:|
| 5 | A | 0.563426 | 0.255363 | 0.433679 | 0.634141 |
| 5 | B | 0.567089 | 0.278893 | 0.449366 | 0.638754 |
| 10 | A | 0.518987 | 0.253979 | 0.419839 | 0.571857 |
| 10 | B | 0.521398 | 0.253979 | 0.412918 | 0.573010 |
| 15 | A | 0.497206 | 0.238062 | 0.370934 | 0.546251 |
| 15 | B | 0.495430 | 0.222837 | 待核对* | 待核对* |
| 20 | A | 0.481451 | 0.215225 | 0.348558 | 0.528028 |
| 20 | B | 0.475649 | 0.193080 | 待核对* | 待核对* |
| 25 | A | 0.481423 | 0.215225 | 0.355248 | 0.522722 |
| 25 | B | 0.486946 | 0.215225 | 待核对* | 待核对* |
| 30 | A | 0.482198 | 0.217301 | 0.322953 | 0.529181 |
| 30 | B | 0.478404 | 0.204152 | 0.324798 | 0.523645 |
| 35 | A | 0.464734 | 0.164014 | 0.317878 | 0.506574 |
| 35 | B | 0.465008 | 0.154325 | 0.326413 | 0.502653 |
| 40 | A | 0.460300 | 0.159170 | 0.326182 | 0.495502 |
| 40 | B | 0.462563 | 0.152941 | 0.325952 | 0.494118 |

### Score 文件与日志的 epoch 冲突（Observed）

表中星号：B 的以下三份 score JSON 与对应 epoch 的 EVAL_SUMMARY 不一致；其余 13 点 BOP AR
一致。上述表格 BOP/ADD 按带明确 epoch 的 EVAL_SUMMARY 记录，冲突点的 reS/teS
未获得可信 epoch 归属，标为待核对，不使用其他 epoch 的近似指标替代。

| B 文件名 | 文件 BOP AR | 对应 epoch 日志 BOP AR | 数值吻合的日志 epoch（仅观察） |
|---|---:|---:|---:|
| scores_bop19_15epoch.json | 0.4756493656286043 | 0.4954302191464821 | 20 |
| scores_bop19_20epoch.json | 0.4869457900807382 | 0.4756493656286043 | 25 |
| scores_bop19_25epoch.json | 0.47840369088811996 | 0.4869457900807382 | 30 |

疑似外置文件命名/收集错位，但仅凭一个聚合值不能证明真实 checkpoint 归属。
原文件名与字节均保留，暂不重新映射分项；需从明确的服务器 epoch 目录补充原始
score 文件。本次差异不涉及 E5/E10 或 E40；不据此推断训练 run 整体无效。

### E40 差值与趋势（Derived）

差值为 `B − A`，相对变化为 `(B − A) / A`；由上述原始 JSON 未舍入值计算。

| 指标 | E40 B − A | 相对变化 |
|---|---:|---:|
| BOP AR | +0.002263 | +0.49% |
| ADD(-S)0.1d | -0.006228 | -3.91% |
| AR_reS | -0.000231 | -0.07% |
| AR_teS | -0.001384 | -0.28% |

在全部 8 个固定点中，A/B 的 BOP/ADD 最大值均位于 E5；A 的 reS/teS 及 B 已核对点的
reS/teS 最大值也位于 E5，B 冲突点分项不参与最佳点判定。上文已保留 E5
逐物体 ADD，以下补充最终 E40。这里的“最大值”仅为事后描述，不以 LM-O
中间结果重新选择正式 checkpoint。其他 epoch 的逐物体 ADD 全部保存在
`eval_summary.jsonl`；逐物体 BOP/reS/teS 未在本次外置来源中提供，未留存为紧凑副本，
不使用逐物体 ADD 代替。

### 最终 E40 ADD(-S)0.1d 逐物体结果（Observed / Derived）

| 物体 | A E40 | B E40 | B − A |
|---|---:|---:|---:|
| ape | 0.017143 | 0.045714 | +0.028571 |
| can | 0.145729 | 0.130653 | -0.015075 |
| cat | 0.134503 | 0.134503 | +0.000000 |
| driller | 0.150000 | 0.115000 | -0.035000 |
| duck | 0.038889 | 0.061111 | +0.022222 |
| eggbox | 0.338889 | 0.333333 | -0.005556 |
| glue | 0.164286 | 0.157143 | -0.007143 |
| holepuncher | 0.270000 | 0.240000 | -0.030000 |

### E40 训练日志（Observed）

延续上文 my_writer 口径，下表为最后训练日志行括号内的 global average，
不是最后 batch loss，也不是全量像素误差。

| 臂 | iteration | total | region | coor_x | coor_y | coor_z | xyz_reproj | mask | mask_full | max_mem |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 255919 | 16.48 | 16.37 | 0.02668 | 0.02493 | 0.01906 | — | 0.02218 | 0.02237 | 8286M |
| B | 255919 | 16.49 | 16.37 | 0.02649 | 0.02474 | 0.01904 | 0.00891 | 0.02211 | 0.02229 | 8295M |

region loss 的背景常数及零梯度口径见上文；不能以 loss 标量占比推断梯度占比，
不能用 `loss_xyz_reproj × 64` 代替真实重投影像素误差。

### 证据边界与后续决策（Interpretation / Decision）

Observed：A/B direct-pose 指标整体随训练下降，B 相对 A 未形成四项一致收益；
E40 BOP 略高，但 ADD、reS、teS 略低。

Interpretation：该 telemetry 反映更新后的 geometry 与冻结 Patch-PnP 的组合表现，
不足以判定 correspondence 是否更真实或更适合 classical PnP；分布适配是待验证解释，
不能作为已确认原因。单 seed 结果不证明跨 run 稳定性。

Decision：不依据 direct-pose telemetry 宣布 EXP020 机制通过或失败，不改原主评价
协议。后续补充退出状态与 E40 明确 checkpoint，使用专用 evaluator 完成 fixed-support
matched PnP/RANSAC，报告 correspondence error → reprojection error → pose 的响应关系；
沿用相对阈值政策（±3%–±5%，饱和指标仍绝对），不自动增加训练或 seed。
