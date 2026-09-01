# EXP013B — Geometry Attention Residual

## 协议与运行

- 状态：`COMPLETE / FAIL_STRICT_BOP_GATE / ATTENTION_EFFECTIVE`
- 在 A 的 geometry residual 上增加局部 attention
- formal run：`RUN-20260822-112724-formal-s42-a01`；lab1；seed 42；40 epoch
- source commit：`e316b9f33abf51cc8d80550e9dfd52ebb0420088`
- 配置：`configs/gdrn/lmo_pbr/research/exp013/b_geometry_attention/train.py`
- checkpoint：`model_epoch_040.pth`，epoch 40 / iteration 255919

## E40 正式 gate

| 指标 | EXP012 | B E40 | 增量 | 门槛 | 判决 |
|---|---:|---:|---:|---:|---|
| BOP AR | 0.678800 | 0.683691 | +0.004891 | ≥ +0.005 | FAIL |
| ADD(-S) | 0.494118 | 0.514187 | +0.020069 | ≥ +0.010 | PASS |
| 逐物体非负 | — | 5/8 | — | ≥ 5/8 | PASS |

BOP 距门槛 `0.000109`，不能四舍五入放宽。相对 A，BOP 差 `-0.000265`，落在
预注册 `±0.001` 区间，ADD 高 `+0.003460`，因此 attention 按 B−A 规则有效，
但 B 整体 gate 仍失败。AR_reS `0.498039`，AR_teS `0.801153`；逐物体非负为
ape、driller、duck、eggbox、holepuncher。

## E40 结构诊断

normal `4.62°/1.52 cm`，关闭 attention 后 `8.10°/2.08 cm`；attention scale 从
`0.1` 学到约 `0.177`，geometry scale 约 `0.099`。Region 清零后
`11.52°/12.56 cm`；打乱 geometry 网格使 rotation 平均变化 `59.70°`。
共享参数 R/t 梯度余弦均值 `+0.143`，但 translation 梯度范数约为 rotation 的
39 倍。诊断证明 attention 被使用，不把它改写为整体方法 PASS。
