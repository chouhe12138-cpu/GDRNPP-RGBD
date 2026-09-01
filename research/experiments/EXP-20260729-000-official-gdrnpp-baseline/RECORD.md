# EXP000 — 官方 GDRNPP 基线

- 状态：`COMPLETE`
- 模型：GDRNPP ConvNeXt-Base，LM-O class-aware 官方 checkpoint
- 协议：LM-O BOP19、GT bbox、RGB-only、官方 Patch-PnP

该基线已在后续 full diagnostic 中按 1,445 个 targets 复现。它作为历史父基线，
不代表本项目重新训练的模型。正式比较必须注明 ADD(-S) 使用 macro-object 还是
target-micro 口径。
