# LM-O 当前研究配置

当前配置分为四类：`_base_` 定义 LM-O GT-box 评估和 40-epoch 公共协议，
`controls` 保存长期 matched control，`templates/pose_head` 提供新实验模板，
EXP012 与 EXP013 A–F/D 是当前研究链。

`controls/pnp_only` 是 EXP005 的长期等价控制入口：保留官方 pretrained ConvPnPNet，
冻结 backbone/geometry，仅适应 PnP head。它用于未来方法的统一 matched comparison；
EXP005 的历史结果若需精确复现，应 checkout 对应 RECORD 中记录的 source commit。

当前实验的 `train.py`、`smoke.py`、`audit48.py`、`eval.py` 分别用于正式训练、真实
数据小跑、batch-48 工程验收和显式 checkpoint 评估。已退出当前研究链的历史配置
不长期保留在 HEAD，启动器通过配置覆盖设置唯一 OUTPUT_DIR，不复制训练循环。
