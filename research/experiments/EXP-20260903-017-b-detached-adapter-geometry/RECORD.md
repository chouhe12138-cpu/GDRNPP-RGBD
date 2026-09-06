# EXP017-B — Detached Adapter Geometry

## 状态与协议

- 状态：`COMPLETE / NO_OVERALL_GAIN`；2026-09-06 核对正式结果。
- experiment ID：`EXP-20260903-017-b-detached-adapter-geometry`
- 母实验：EXP017。
- 唯一变量：adapter 输入由 `geometry_grid` 改为 `geometry_grid.detach()`。
- EXP013A 自身 geometry latent、translation、loss、optimizer、数据、40 epoch、pooling、
  position 全部不变；不从 EXP017 checkpoint warm-start。
- training renderer：disabled；BOP evaluation renderer：cpp。

## 假设

EXP017 的 rotation residual loss 会经 adapter 输入反传到 EXP013A 的共享 geometry
encoder。只切断这条新增梯度边，可以保留 adapter 的 rotation 表达力，同时减少 rotation
优化对后续共享 translation 表示的额外扰动。

## Gate

正式运行前已明确 E40 是唯一决策点。原记录未单独列出 B 的数值 gate；
以下对母实验 EXP017 五项门槛的核对作为参照，不追认成 B 独立预注册门槛。

## 本地门禁

- EXP017-B + EXP017 + pose_structure tests：`26 passed`。
- CPU preflight：`PASS`；参数 `13,000`，normal/detached forward value-exact，translation
  bitwise equal，adapter rotation 梯度保留，translation→adapter 梯度为零，
  adapter→共享 geometry encoder 梯度已切断。

## 服务器 smoke

- run：`RUN-20260903-083348-smoke-s42-a01`。
- source：`f02dadebfcb89d2cbe297d7409c10abfb5fe7013`；machine：lab1。
- config：`configs/gdrn/lmo_pbr/research/exp017/b_detached_adapter_geometry/smoke.py`。
- 用户确认训练正常完成、`exit_code=0`；该 run 只验证链路，不进入性能结论。

## Formal

- run：`RUN-20260903-085807-formal-s42-a01`；machine lab1；seed 42。
- source：`f02dadebfcb89d2cbe297d7409c10abfb5fe7013`，metadata 确认 source tree clean。
- config：`configs/gdrn/lmo_pbr/research/exp017/b_detached_adapter_geometry/train.py`。
- 与母实验 matched：LM-PBR train、LM-O BOP19 GT-box、冻结 backbone/geometry head、
  batch 48、Ranger lr 8e-4、weight decay 0.01、warmup 200、40 epoch。
- checkpoint：`model_epoch_040.pth`，epoch 40 / iteration 255919；日志确认保存、
  训练 100% 完成和 `FINAL_EVAL_REUSED periodic_epoch=40`。
- 外置证据：`E:\6D姿态估计\EXP017-B\` 的 run_metadata.json、console.log 与八个
  scores_bop19_*epoch.json。本次该目录未提供 checkpoint 本体，保存事实来自日志。
- ADD 使用日志 `EVAL_SUMMARY.add_s_0.1d` 的 1,445-target micro 口径；
  BOP/reS/teS 来自对应 JSON，BOP 已与日志交叉核对；不以 AR_ad 替代 ADD。

## 正式轨迹

| Epoch | BOP AR | ADD(-S) target-micro | AR_reS | AR_teS |
|---:|---:|---:|---:|---:|
| 5 | 0.613550 | 0.439446 | 0.313956 | 0.758016 |
| 10 | 0.655596 | 0.510727 | 0.382699 | 0.792849 |
| 15 | 0.656549 | 0.496886 | 0.434602 | 0.777393 |
| 20 | 0.653654 | 0.471972 | 0.444060 | 0.775779 |
| 25 | 0.670501 | 0.492734 | 0.459285 | 0.791465 |
| 30 | 0.674023 | 0.497578 | 0.470127 | 0.795156 |
| 35 | 0.673700 | 0.495502 | 0.472203 | 0.798385 |
| 40 | 0.683686 | 0.498270 | 0.503114 | 0.800692 |

E40 为 BOP/reS/teS 最高点，E10 为 ADD 最高点；不按中间点选模。
E30 使用上述原始日志/JSON，取代此前对话中的约数。

| Epoch | ape | can | cat | driller | duck | eggbox | glue | holepuncher |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.434286 | 0.748744 | 0.356725 | 0.785000 | 0.272222 | 0.555556 | 0.757143 | 0.200000 |
| 40 | 0.457143 | 0.778894 | 0.450292 | 0.800000 | 0.061111 | 0.394444 | 0.742857 | 0.310000 |

## E40 对比与结论

| 指标 | EXP013A | EXP017 | EXP017-B | B−EXP017 |
|---|---:|---:|---:|---:|
| BOP AR | 0.683956 | 0.681709 | 0.683686 | +0.001977 |
| ADD(-S) target-micro | 0.510727 | 0.512111 | 0.498270 | -0.013841 |
| AR_reS | 0.498039 | 0.494348 | 0.503114 | +0.008766 |
| AR_teS | 0.797693 | 0.800461 | 0.800692 | +0.000231 |

参照母实验的固定 A 基准门槛：

| 指标 | B E40 | 母实验门槛 | 参照判定 |
|---|---:|---:|---|
| AR_reS | 0.503114 | ≥0.513039 | 未达到 |
| BOP AR | 0.683686 | ≥0.686956 | 未达到 |
| ADD(-S) target-micro | 0.498270 | ≥0.505727 | 未达到 |
| AR_teS | 0.800692 | ≥0.794693 | 达到 |
| 相对 A 逐物体 ADD 非负 | 3/8 | ≥5/8 | 未达到 |

相对 A 仅 driller、glue（持平）、holepuncher 非负。参照五项门槛仅达到一项。
detach 相对 EXP017 有小幅 BOP/reS 改善，但 teS 几乎不变且 ADD 下降 1.3841 pp；
结论 NO_OVERALL_GAIN，不支持切断新增梯度边就能改善整体姿态或挽救 ADD。
该单 seed matched 实验不能证明梯度冲突、Region 依赖或某种结构存在普遍缺陷。
实验已完成收口，不自动增加 seed、追加训练或用 E10 替代最终 E40。
