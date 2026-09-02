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

2026-09-02 的 bundle/fresh-readonly-release 迁移验证先后暴露并修复了 launcher
override、`DictAction` 格式、dataset cache 和背景路径 cache 契约。对应的
`RUN-20260902-034930-smoke-s42-a01`、`RUN-20260902-042118-smoke-s42-a01`、
`RUN-20260902-043259-smoke-s42-a01` 与 `RUN-20260902-044659-smoke-s42-a01` 均为
迁移基础设施验证；早期 `RUN-20260811-052852-smoke-s42-a01` 也属于已退出的
managed 流程验证。这些 run 不是科学实验失败，也不进入 EXP005 科学结果。

最终 smoke `RUN-20260902-050035-smoke-s42-a01` 成功完成 2048/2048 iterations：
forward/backward 正常、loss finite，并保存 `model_epoch_001.pth`。它验证了当前
lightweight launcher 的只读 mount、CLI、runtime/CUDA/native gate、dataset/XDG
cache 与训练闭环；该 smoke 仍不作为正式科学指标。

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
