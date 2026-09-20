# LM-O 当前研究配置

当前配置分为四类：`_base_` 定义 LM-O GT-box 评估和 40-epoch 公共协议，
`controls` 保存长期 matched control，`templates/pose_head` 提供新实验模板，
`exp025_hierarchical_cad_attention` 是唯一活动实验配置。

`controls/pnp_only` 是 EXP005 的长期等价控制入口：保留官方 pretrained ConvPnPNet，
冻结 backbone/geometry，仅适应 PnP head。它用于未来方法的统一 matched comparison；
EXP005 的历史结果若需精确复现，应 checkout 对应 RECORD 中记录的 source commit。

EXP025 用 `train_official_frozen.py` 与 `train_imagenet_full.py` 明确区分两条正式臂，
`smoke.py` 只用于本地接线。已退出当前研究链的历史配置不长期保留在 HEAD；启动器
通过配置覆盖设置唯一 OUTPUT_DIR，不复制训练循环。

`train_lm13_imagenet_full.py` 是 LM-O 完成后的预备配置：ImageNet ConvNeXt 全量训练，
沿用历史 LM real+DeepIM、160 epoch flat-and-anneal 协议。它当前不在服务器 launcher
allowlist 中，不能启动 formal。
