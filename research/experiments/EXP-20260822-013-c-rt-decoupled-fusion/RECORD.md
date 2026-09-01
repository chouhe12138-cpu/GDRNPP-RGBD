# EXP013C — R/t Decoupled Fusion

## 协议修订与运行

- 状态：`COMPLETE / SCREEN_FAIL / ROTATION_SUPPORTED`
- 原始 B-based C 未运行；B 严格 gate 失败后，revision 2 在正式运行前改为继承 A
- revision 2 不含 B attention；R/t 使用独立 aggregation、latent、输出和 scale
- formal run：`RUN-20260826-124748-formal-s42-a01`；lab0；seed 42
- source commit：`d702030c65347175aead005d046edc4f5e8bdd83`
- 配置：`configs/gdrn/lmo_pbr/research/exp013/c_rt_decoupled/train.py`
- checkpoint：`model_epoch_040.pth`，epoch 40 / iteration 255919

## E40 正式结果

| 指标 | A E40 | C E40 | C−A | 条件 | 判决 |
|---|---:|---:|---:|---:|---|
| BOP AR | 0.683956 | 0.684646 | +0.000690 | — | 描述性提高 |
| ADD(-S) target-micro | 0.510727 | 0.496886 | -0.013841 | 不得明显下降 | FAIL |
| AR_reS | 0.498039 | 0.525029 | +0.026990 | rotation improve | PASS |
| AR_teS | 0.797693 | 0.794233 | -0.003460 | drop within bound | PASS |

C 的 ADD macro-object 为 `0.498841`。rotation 与 translation-drop 条件通过，
但 ADD 明显下降，最终为 `SCREEN_FAIL / ROTATION_SUPPORTED`：支持 R/t 专用表示
改善 rotation 读出，不支持整体更优。外置证据位于
`E:\6D姿态估计\EXP-013\实验C\`。
