# EXP005 — Patch-PnP-Only Adaptation Control

## 协议与运行

- 状态：`COMPLETE / EPOCH_040_MATCHED_CONTROL`
- 问题：只适应官方 Patch-PnP，是否足以解释后续新姿态头的改善？
- 训练：LM-PBR 50 scenes；LM-O BOP19 GT-box evaluation；40 epoch；seed `42`
- 冻结 backbone/geometry head，只训练官方 Patch-PnP；不按中间 LM-O 结果选模
- formal run：`RUN-20260811-063606-formal-s42-a01`
- source commit：`652d7fd9d38f8ea5cea0c5a98cc9477b66623180`
- checkpoint：`model_epoch_040.pth`，epoch 40 / iteration 255919
- 历史原始配置在 source commit 中；当前树提供
  `configs/gdrn/lmo_pbr/research/controls/pnp_only/` 作为长期等价 matched control

当前 control 只保留 EXP005 的科学控制条件，并按现有 `_base_` 配置体系重写；若要
精确复现 2026-08-11 的正式结果，应 checkout 上述 source commit，而不是使用当前
core 重新解释历史结果。

2026-09-02 的 `RUN-20260902-034930-smoke-s42-a01` 因 launcher 构造训练命令时
遗漏 `--opts`，被 `main_gdrn.py` 以 `unrecognized arguments: OUTPUT_DIR ...`
拒绝。该 run 未进入模型构建或训练，没有 checkpoint 和指标，只作为 launcher
基础设施失败保留，不进入 EXP005 科学结果。后续修复同时覆盖 train/eval override。

2026-09-02 的 `RUN-20260902-042118-smoke-s42-a01` 已通过 runtime/native gate，
但 launcher 将 mmcv `DictAction` 的 override 写成 `KEY VALUE`，解析阶段报
`ValueError: not enough values to unpack (expected 2, got 1)`。该 run 未进入模型
初始化或训练，没有 checkpoint 和指标，只作为 launcher 基础设施失败保留，不进入
EXP005 科学结果；train/eval override 随后统一修正为 `KEY=VALUE`。

2026-09-02 的 `RUN-20260902-043259-smoke-s42-a01` 已通过 runtime/native gate、
CLI parsing、模型构建和 dataset parsing，但 launcher 未设置
`GDRN_DATASET_CACHE_DIR`，保存 dataset cache 时因只读 repo 下的
`/workspace/gdrnpp/.cache` 报 `OSError: [Errno 30] Read-only file system`。该 run
没有完成 smoke，也没有产生科学指标，只作为基础设施失败保留，不进入 EXP005
科学结果。

首次 managed smoke `RUN-20260811-052852-smoke-s42-a01` 因 dataset cache 指向
只读 release 而失败，没有 checkpoint 或指标，不进入科学结果。修正 writable
cache 与异常返回后，smoke/audit 通过且另建目录，没有覆盖失败 run。

## 固定 E40 结果

| 指标 | 结果 |
|---|---:|
| BOP AR | 0.6919123414 |
| ADD(-S)@0.1d macro-object | 0.5065743945 |

逐物体 ADD(-S)@0.1d：

| obj | 1 | 5 | 6 | 8 | 9 | 10 | 11 | 12 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| recall | 0.525714 | 0.814070 | 0.461988 | 0.815000 | 0.077778 | 0.411111 | 0.750000 | 0.215000 |

E15 的中间 BOP AR 虽更高，但协议固定使用 E40，不据 LM-O test 选 checkpoint。
该结果接近官方基线，说明单独重训 Patch-PnP 没有形成大幅改善；EXP005 只作为
matched-training control，不作为方法贡献。外置 checkpoint 位于
`E:\6D姿态估计\EXP-005\model_epoch_040.pth`。
