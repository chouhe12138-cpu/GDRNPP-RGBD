# EXP013E — Official Head Random Initialization

## 协议与运行

- 状态：`COMPLETE / DIAGNOSTIC_PARTIAL_SUPPORT_M2_M3`
- 与 A 匹配：冻结 backbone/geometry，只训练 pose head；40 epoch；batch 48
- 官方 `ConvPnPNet` 结构随机初始化；Ranger lr `8e-4`；seed `42`；renderer 关闭
- formal run：`RUN-20260829-080742-formal-s42-a01`；lab0
- source commit：`d231bf634b94807e01948cc7126f3f0e6d54582e`
- 配置：`configs/gdrn/lmo_pbr/research/exp013/e_official_head_random/train.py`
- checkpoint：`model_epoch_040.pth`，epoch 40 / iteration 255919

v1 smoke `RUN-20260829-063652-smoke-s42-a01` 依赖临时派生的去 PnP checkpoint，
容器间文件状态不可靠而失败，没有指标。v2 改为 wrapper 键名隔离并直接加载原始
官方 checkpoint；共享张量继承、官方 PnP 键过滤和随机初始化不被覆盖均通过验证。

## 正式轨迹

| Epoch | BOP AR | ADD(-S) target-micro | AR_reS | AR_teS |
|---:|---:|---:|---:|---:|
| 5 | 0.655557 | 0.440830 | 0.467128 | 0.768166 |
| 10 | 0.674012 | 0.514879 | 0.502422 | 0.797232 |
| 15 | 0.672540 | 0.482353 | 0.488120 | 0.789850 |
| 20 | 0.667696 | 0.460208 | 0.488120 | 0.783391 |
| 25 | 0.666990 | 0.490657 | 0.495040 | 0.787313 |
| 30 | 0.672561 | 0.480277 | 0.522491 | 0.789158 |
| 35 | 0.679548 | 0.474740 | 0.523183 | 0.786621 |
| 40 | **0.688581** | **0.510727** | **0.535409** | **0.801153** |

E40 ADD(-S) macro-object `0.512940`。预注册解释带针对 reS：`0.52–0.54` 为部分
支撑，≥`0.54` 为强支撑；E40 `0.535409` 落入部分支撑带，距强支撑线
`0.004591`。E 是诊断对照而非 PASS/FAIL screening，因此不把它改写成方法通过。
结果说明官方头结构本身可读出较强 rotation，同时预训练继承仍有作用。外置证据
位于 `E:\6D姿态估计\EXP-013\实验E\`。
