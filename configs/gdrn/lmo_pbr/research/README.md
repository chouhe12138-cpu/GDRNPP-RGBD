# LM-O 当前研究配置

保留的配置分为三层：`_base_` 定义 LM-O 评估和 40-epoch 训练协议，
`templates/pose_head` 提供新实验模板，EXP012 与 EXP013 A–F/D 是当前论文链。

每个实验的 `train.py`、`smoke.py`、`eval.py` 分别用于正式训练、本地/服务器小跑和
明确 checkpoint 评估。启动器通过配置覆盖设置唯一 OUTPUT_DIR，不复制训练循环。
