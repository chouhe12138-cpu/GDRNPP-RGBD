# EXP009 — Correspondence-Aware Moment Pose Head

## 协议与运行

- 状态：`COMPLETE / CPM_SCREEN_FAIL`
- 问题：Region-conditioned 2D–3D low-order moments 能否改善 correspondence 消费？
- 冻结 backbone/geometry head，只训练 CPM；LM-PBR → LM-O GT-box；40 epoch
- seed `42`；Ranger lr `8e-5`；只使用固定 E40 作正式判决
- formal run：`RUN-20260811-063626-formal-s42-a01`
- source commit：`652d7fd9d38f8ea5cea0c5a98cc9477b66623180`
- checkpoint：`model_epoch_040.pth`，epoch 40 / iteration 255919
- 历史配置已退出当前树，可在 source commit 中恢复

第一次 managed smoke 因只读 dataset cache 失败，没有 checkpoint 或指标，不是科学
失败。正式训练经历一次 CUDA launch failure 后由同一 run 恢复并完成 E40；失败与
恢复均不改变固定 checkpoint 的判决。

## 正式结果

| 指标 | E35 | E40 |
|---|---:|---:|
| BOP AR | 0.599423 | 0.598392 |
| ADD(-S) target-micro | 0.386159 | 0.380623 |
| ADD(-S) macro-object | — | 0.376867 |
| AR_reS | — | 0.441292 |
| AR_teS | — | 0.692042 |
| VSD AR | — | 0.406249 |
| MSSD AR | — | 0.545606 |
| MSPD AR | — | 0.843322 |

E40 逐物体 ADD(-S)：

| obj | 1 | 5 | 6 | 8 | 9 | 10 | 11 | 12 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| recall | 0.182857 | 0.603015 | 0.274854 | 0.670000 | 0.150000 | 0.277778 | 0.521429 | 0.335000 |

相对 official BOP `0.690415` 与 EXP005 `0.691912`，E40 分别下降
`9.2023/9.3520 pp`。三项预注册 gate 为 BOP 增量 ≥`+0.005`、ADD macro 增量
≥`+0.010`、非负物体 ≥`5/8`；当前逐物体只有 `2/8` 非负，三项均失败。

历史记录曾把 `0.380623` 误标为 macro-object；代码核对表明它是 target-micro，
由逐物体值等权派生的 macro 为 `0.376867`。该口径修正不改变判决。

## E30 机制预览

E30 的 1,445-target × 19-condition 诊断只作机制解释，不替代 E40 gate：

| 条件 | BOP AR |
|---|---:|
| baseline | 0.59946 |
| GT-XYZ α=.25/.50/.75/1 | 0.56205 / 0.46897 / 0.37097 / 0.30820 |
| XYZ permutation | 0.14534 |
| ROI permutation | 0.37114 |
| Region disruption / mean Region | 0.18668 / 0.18765 |
| coverage-only | 0.00000 |
| CXU-null | 0.15855 |

CPM 确实依赖 XYZ、ROI、Region 和 cross-covariance，但改善 XYZ 反而持续恶化
pose。结论 `CPM_SCREEN_FAIL` 只否定当前 low-order moment 压缩与训练协议，不否定
dense correspondence，也不能由本实验单独指定学习率、尺度或 mismatch 为唯一根因。
