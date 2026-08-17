# EXP-20260731-005 — Patch-PnP-Only Adaptation Control

Status: `FORMAL COMPLETE — FIXED EPOCH 40 MATCHED CONTROL`

## Managed formal protocol update

The historical local pilot and failed attempts retain their originally recorded
seeds.  New managed smoke, audit, formal, evaluation, and diagnostic runs use
the fixed seed `42`.  The frozen legacy Stage 3C0 config remains unchanged;
`configs/gdrn/lmo_pbr/research/exp005_pnp_control/` provides additive wrappers
for the managed run identity and output layout.

## Purpose

Test whether the existing official direct pose head can be adapted before
introducing a new quality/coverage mechanism.

Frozen protocol:

```text
research/stages/STAGE_03C0_PNP_ADAPTATION_CONTROL.md
```

## Data verification

The deep preflight passed:

| Asset | Verified count |
|---|---:|
| PBR scenes | 50 |
| PBR RGB images | 50,000 |
| PBR depth images | 50,000 |
| PBR annotations | 749,600 |
| LM-O-eight-object annotations | 399,950 |
| Full + visible mask paths | 1,499,200 |
| VOC JPEG images | 17,125 |
| Positive dining-table backgrounds | 538 |

The official checkpoint SHA-256 was
`bafa869d4e6c00410517ecb1add59f234ed1642e47fabcf3aa6e0e8a1b498a8c`.

The local split produced 8,192 effective instances from 1,389 images, exactly
1,024 instances for each of the eight LM-O objects.

## Local pilot

```text
GPU:                RTX 4060 Laptop, 8 GB
micro-batches:      2,048
optimizer updates:  171
peak GPU memory:    4,820 MiB
physical/effective: 4 / 48
checkpoint:         model_final.pth
```

All losses remained finite.  Recorded first/last-quarter medians:

| Loss | First | Last | Relative change |
|---|---:|---:|---:|
| Total | 0.038747 | 0.036650 | -5.41% |
| Rotation PM | 0.003709 | 0.003757 | +1.29% |
| Centroid | 0.008664 | 0.008476 | -2.17% |
| Z | 0.023609 | 0.022837 | -3.27% |

The checkpoint strictly reloaded with no missing or unexpected tensors.
Against the official checkpoint, 17/17 Patch-PnP trainable tensors changed,
while 0 frozen tensors changed.

## 2026-08-11 managed server smoke（无效基础设施运行）

第一次新架构服务器 smoke 使用以下冻结身份：

```text
run_id:                   RUN-20260811-052852-smoke-s42-a01
source_git_commit:        b39f68092de2609b7ee1726811c9ee965e606328
environment_image_id:     sha256:f3055cb660032bbb4c1b7cfd9b1840a6c98359d0562a3a4f0601f7238f7291ee
seed:                     42
```

release snapshot、native artifact、环境、数据和 EXP005 isolation gate 均通过，
但训练在创建 LM-PBR dataset cache 时尝试写只读 source release 下的
`/workspace/gdrnpp/.cache`，触发 `OSError: [Errno 30] Read-only file system`。
当时训练入口的 Loguru decorator 又吞掉了内部异常并返回 0，导致 managed
postprocess 继续查找不存在的 `model_epoch_001.pth`，最终表面状态被记为
`POSTPROCESS_ERROR`。

该 run 没有 checkpoint、评估指标或科学结论，必须原样保留为无效基础设施
证据，不计入 EXP005 结果。修复仅允许提供外置可写 cache/home，并让训练异常
正确返回非零；不得修改模型、loss 或正式实验定义。

## 2026-08-11 有效 managed 验收与正式启动

最终服务器 release 和环境身份为：

```text
source_git_commit:        652d7fd9d38f8ea5cea0c5a98cc9477b66623180
environment_image_id:     sha256:f3055cb660032bbb4c1b7cfd9b1840a6c98359d0562a3a4f0601f7238f7291ee
environment_build_source: 35313ae3d4139a559a97c01b2d3ee007dc16604c
seed:                     42
machine/container/GPU:    lab0 / lab0_chx / physical GPU 0
```

source snapshot、native、环境、数据、registry 与 EXP005 isolation gate 全部
PASS。新受管运行没有覆盖第一次失败目录：

```text
smoke: RUN-20260811-061212-smoke-s42-a01 — COMPLETE, exit 0
audit: RUN-20260811-062719-audit-s42-a01 — COMPLETE, exit 0
```

smoke 完成 1 epoch / 2047 iterations，生成并记录
`checkpoints/model_epoch_001.pth`；audit 使用 batch-48 对应配置并完成 169
iterations。两者只证明当前 release 下训练、checkpoint 和受管产物链可执行，
不构成正式性能结果。

## 2026-08-15 固定 Epoch 40 结果

正式 run 为 `RUN-20260811-063606-formal-s42-a01`。原始 console 记录训练自然
到达 Epoch 40 / iteration `255919/255920`，写出 `model_epoch_040.pth`，并复用
同一周期的最终评估。固定结果为：

| 指标 | 结果 |
|---|---:|
| BOP AR | 0.6919123414 |
| ADD(-S)@0.1d macro-object | 0.5065743945 |

逐物体 ADD(-S)@0.1d 为：obj 1 `0.525714`、5 `0.814070`、6 `0.461988`、
8 `0.815000`、9 `0.077778`、10 `0.411111`、11 `0.750000`、12 `0.215000`。
Epoch 15 的中间 BOP AR 虽更高，但不用于选模；正式比较只使用固定 Epoch 40。

外部 Windows 副本 `E:\\6D姿态估计\\EXP-005\\model_epoch_040.pth` 已完成
下载，大小 `519078612` bytes，SHA-256 为：

```text
39c0128526f68cf9c4f7a1780ff095e71be8cdada186136eb8699d99d68d009e
```

本地 `torch.load` 读取成功：Epoch 40、iteration `255919`，包含 optimizer、
scheduler 和 392 个模型张量。当前没有服务器端原文件 SHA-256，因此只记录
“本地副本完整可读”，不记录“已与服务器哈希一致”。

## Conclusion

```text
LOCAL_PIPELINE_AND_ISOLATION = PASS
```

The matched 40-epoch control completed. Its fixed Epoch 40 result is close to
the public official baseline, so retraining Patch-PnP alone provides no large
explanation for a future CPM gain. B is retained as a matched-training control,
not presented as an external competing method or a method contribution.

Artifacts:

```text
output/EXP-20260731-005/pnp_only_local/metrics.json
output/EXP-20260731-005/pnp_only_local/local_pilot_summary.json
output/EXP-20260731-005/pnp_only_local/model_final.pth
```
