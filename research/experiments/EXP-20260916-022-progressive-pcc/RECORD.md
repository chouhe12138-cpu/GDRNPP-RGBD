# EXP022 — 渐进式层级 CAD 对应与多尺度 PCC

## Status

- lifecycle：`SUPERSEDED`
- execution：`FORMAL_NOT_STARTED`
- superseded by：EXP025
- final historical snapshot：`8c6ca86aa0668777548d4f30b8cc6f6ad9864067`
- server EGL、formal、E5–E40 与完整 matched PnP：均未生成。

## Research question

在冻结 official ConvNeXt、LM-O PBR40/GT-box 不变时，以 8⁴ CAD route、多尺度 PCC、局部
球形 bounded residual 和显式 RANSAC-PnP，能否较 EXP021 的 64→64 hierarchy 降低切向与
重投影 correspondence error 并改善 matched pose？

主变量是完整 Progressive PCC 结构；`independent_v2.npz` 只作 hierarchy 来源 smoke，不进入
formal。该实验只设计冻结 backbone 阶段，不包含全量 backbone 训练。

## Final intended protocol

- 配置：历史 `train_reused.py`；seed42；LM-O PBR40；GT box。
- 40 epoch、batch48、16 workers、AdamW 3e-4、wd .01、4% warmup 后 cosine、AMP FP16。
- E5/E10/E15/E20/E25/E30/E35/E40 固定评价。
- route/residual/mask 权重均 1.0；residual beta .1；beam K2；token dim256。
- 对称分支仅由 route CE + residual 选择；选定分支的 route/residual/mask 均参与训练。
- 主评价为 fixed-support correspondence、reprojection 与 RANSAC pose；native support 只作
  secondary analysis。

正式 run ID、source commit、checkpoint 与指标从未生成。配置、runner 和专用测试已退出 HEAD；
精确恢复应从上方 snapshot 建独立 worktree。

## Implemented structure

- hierarchy 主臂每对象四级节点 8/64/512/4096；`reused_v1` 保持 EXP021 leaf anchors。
- image stages：S1/S2 global self-attention，S3/S4 window8 + shift4 self-attention；每级 matcher
  使用独立 Q/K/V，保留局部 8-way logits、packed route、soft context fusion、GT-parent 监督
  和 Top-2 beam。
- Top-2 仅保证每步已保留 parents 的 16 个 children 内精确，不宣称等价于全路径 exhaustive
  Top-2。
- transition 最终改为 bilinear upsample + 1×1 Conv；shifted window 使用 masked SDPA；正式
  路径关闭诊断统计；matcher 移除可折叠的 `out_proj`。
- symmetry loader 对 branch count >2 fail-closed；LM-O artifacts 的 counts 为
  `[1,1,1,1,1,2,2,1]`。

## Canonical local engineering evidence

- CPU preflight 加载 official backbone 340 张量，最终 Progressive PCC 可训练参数
  `3,728,456`，loss/gradient finite。
- 本机 CUDA+CPP batch4/batch48 AMP smoke 通过，无 skipped step；这些固定 batch 结果不含
  每步 DataLoader/renderer，不能作为服务器吞吐或正式精度。
- 性能审查的 matched fixed-batch48 对照：最终实现相对 review BASE 的 PCC 参数减少
  2,586,368（−40.96%），稳定整步中位数 `949.30→633.88 ms`（−33.23%），allocated
  `6.199→5.669 GB`（−8.54%）。架构和随机初始化已变化，不是逐点数值等价对照。
- 多数据集抽象后，用旧保存的真实 PBR batch48、seed42 跑一步 AMP PASS；对象顺序仍为
  `[1,5,6,8,9,10,11,12]`，未改变 `reused_v1.npz`。
- 单目标 evaluator wiring 完成 official A 与随机 PCC 的 route/PnP 输出；随机模型数值只证明
  接线，不进入科学结论。

## Hierarchy sanity correction

公共 CADHierarchy 模块按所有 parent-child 层复算
`parent_radius / max(norm(child-parent)+child_radius)`：

| artifact | `<1` nodes by relation | corrected static sanity |
|---|---|---|
| `reused_v1` | 0 / 512 / 0 | FAIL |
| `independent_v2` | 64 / 512 / 4084 | FAIL |
| `consistent_v3` | 0 / 0 / 0 | PASS |

旧诊断曾只检查 T3→T4 而把 `reused_v1` 标为 PASS；该历史输出保留，但不能作为全层完整性证据。
对 `consistent_v3` 的重构前后 24-target CPU diagnostic，summary/per-object/per-target/histogram
逐字节一致；新增的只是全层 failure relations 和采样说明。

surface oracle 中 consistent_v3 的 depth GT FAIL 与 rendered-surface GT PASS 继续并列；
oracle 不证明网络 route/residual 可学习性。

## Decision

本地实现、性能修订和静态 hierarchy 审计完成，但服务器真实 batch EGL、formal、matched PnP
与机制 gate 均未运行。EXP025 采用统一 T3 分类和新的 hierarchy/head contract 取代本方案，
因此 EXP022 关闭，不再补 formal，也不把本地 smoke 或 oracle 提升为科学结果。

## Evidence / artifacts

- tracked evidence：无；紧凑事实保存在本 RECORD。
- hierarchy、固定 batch、benchmark、oracle 与完整 diagnostic 输出位于 ignored `.local/` 和
  `output/diagnostics/`。
- 公共 hierarchy API 见 `research/cad_hierarchy/README.md`。

## Known gaps / boundaries

- 没有服务器 EGL、formal run、checkpoint、E5–E40 指标或完整 matched PnP/BOP。
- fixed-batch 性能仅是工程证据，不支持正式吞吐或姿态精度。
- `reused_v1`/`independent_v2` 的全层 sanity 失败不追溯改写旧输出；EXP025 使用独立契约。
