# EXP027 本地交付报告

实验协议与原始紧凑证据见 [RECORD](../experiments/EXP-20260924-027-multiscale-cad-interaction/RECORD.md)。
状态：A/B 实现与本地 batch4 gate 完成；服务器 release、batch48/EGL gate、formal 均未启动。

## 结构与代码

- `multiscale_cad_head.py` 新增 A/B 结构。A：四级图像 SA→CAD CA→前三级 CAD delta
  写回→transition+lateral ADD；B：同一 FPN 的图像特征被固定身份 CAD query 逐级读取，
  image-conditioned parent 传播至 T3，query-pixel similarity 输出 512 类。
- `GDRN_CAD.py` 以显式 architecture 和严格 `out_indices` 选择 head，旧实验仍是单输出路径；
  checkpoint sentinel 按架构验证。`ImageSAStage` 复用投影进行 delta write-back。
- EXP027 A/B 配置直接继承同一新 common，不继承 EXP025/026 experiment config；
  `preflight.py`、`real_smoke.py`、`parameter_report.py` 提供身份、训练和资源检查；
  `experiment.sh`/`run_contract.py` 增加独立 server profile，formal 前校验同 source/config/
  SHA/scale 的唯一 batch48 gate PASS。
- EXP026 仅给诊断 `measure()` 增加四个输出字段（逐级 loss 和 mask loss）；训练配置、head、
  hierarchy SHA、正式协议均未改。共享 runtime 只让 B 的 query projection 可进入失败遥测。

## Effective config 差异

EXP027 A/B resolved config 仅在 `EXP027_ARM`、`OUTPUT_DIR`、
`MODEL.POSE_NET.CAD_ATTENTION_HEAD.ARCHITECTURE` 三项不同。相对 EXP026 adaptive 基线，
仅改变实验/arm/output 身份、`out_indices=(0,1,2,3)`、architecture、
`TRAIN_PROTOCOL.NAME=exp027_lmo`、AMP 初值 `16384→4096` 和当前关闭的 release/formal flags；
数据、hierarchy、优化器、40 epoch 调度、seed、batch、评价协议和 predicted-route residual 一致。

## 参数、性能与测试

| variant | total=trainable | backbone | head | lateral | cross/query | classifier | residual+mask | Δ vs EXP026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EXP026 adaptive | 98,352,068 | 87,564,416 | 10,787,652 | 0 | 2,110,464 | 131,584 | 99,652 | 0 |
| A | 98,524,551 | 87,564,416 | 10,960,135 | 172,483 | 2,110,464 | 131,584 | 99,652 | +172,483 / +0.175% |
| B | 98,721,927 | 87,564,416 | 11,157,511 | 172,483 | 2,439,424 | 0 | 99,652 | +369,859 / +0.376% |

`head` 包含表内四个子组及共同的 input adapter、image SA、transition、geometry encoder；
`parameter_report.json` 将后者单列为 `other_head`，参数不重复计数。

| variant | batch / AMP | forward / backward / optimizer / full median (ms) | peak allocated / reserved (GB) |
|---|---|---:|---:|
| EXP026 adaptive | 4 / 4096 | 53.85 / 252.77 / 40.89 / 347.03 | 3.179 / 3.295 |
| A | 4 / 4096 | 49.74 / 243.51 / 41.19 / 334.76 | 2.914 / 2.993 |
| B | 4 / 4096 | 49.20 / 245.88 / 42.15 / 336.41 | 2.857 / 2.917 |

同一保存真实 LM-O PBR/CPP batch4、seed42 和 100 步 fixed-batch 的末点：EXP026/A/B
route loss 分别为 0.638/0.749/2.037，predicted-cell representability 为
96.51%/95.58%/70.39%。B step80→100 从 1.652 回升到 2.037，是需要在服务器 gate
与后续评价中关注的不利趋势；固定批次不能证明正式 pose 提升。

- 本地 `pytorch22`：ConvNeXt 四尺度真实 forward 实测形状与设计一致；CPU preflight 两臂
  PASS，ImageNet backbone 340 tensors 精确装载；相关 pytest **70 passed**；
  `bash -n`、compile、`git diff --check` PASS。
- 最终代码两臂 batch4 8-step CUDA AMP smoke 与 100-step fixed-batch 均 PASS，0 skipped/non-finite，
  主要模块梯度及更新存在；8-step checkpoint 严格往返 PASS。初始版本更高 scale 和
  FP32 融合前的失败保留在 RECORD。当前 4096 只是本地候选，服务器 batch48 必须重新 gate。

## 服务器准备

当前 `SERVER_RELEASE_ALLOWED=False`、`FORMAL_READY=False`，EXP026 formal 期间不修改服务器
release/容器。具体 release、gate、formal 阶段和可整体复制的块见
[SERVER_PREP_CN.md](SERVER_PREP_CN.md)；只在用户后续授权后使用。

## 2026-09-24 修改包收口

- B 新增三个可训练 parent-query gate（初值 0.01），参数总量 `98,721,927→98,721,930`；
  A 仍为 `98,524,551`，EXP026 adaptive 仍为 `98,352,068`。
- A/B 的四尺度 channel/resolution/pyramid 规格现在由 config 显式声明；core wrapper
  通过 head capability 路由 backbone 特征，仍严格检查 architecture/out_indices。
  EXP027-A vs EXP026 adaptive 的 resolved-config 白名单 diff 无意外差异。
- 同批次 100 步：B route loss `2.037→1.607`，representability `70.39%→77.48%`，
  step80→100 从旧版 `1.652→2.037` 改为 `1.713→1.607`；A route loss
  `0.749→0.811`，representability `95.58%→95.29%`。新 B 整步中位时间
  `361.68 ms`，A 为 `328.82 ms`；均为本地 batch4 工程计时。
- A/B CPU preflight、batch4 8-step AMP smoke 和 100-step fixed-batch PASS，0 skipped，
  strict checkpoint roundtrip PASS。新原始紧凑报告和判断边界见 EXP027 RECORD。
- B 还替换 dense classifier，且 final residual/mask image token 路径与 A 不同；
  因此两臂不构成单独的 attention-direction 消融。正式协议仍为完整 40 epoch，
  E15 只观察；建议两臂分别进入服务器 batch48/EGL gate，当前不开放 release/formal。
