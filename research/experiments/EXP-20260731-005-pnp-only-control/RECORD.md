# EXP-20260731-005 — Patch-PnP-Only Adaptation Control

Status: `LOCAL PILOT PASS — MANAGED FORMAL AUTHORIZED, SERVER SMOKE RETRY REQUIRED`

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

## Conclusion

```text
LOCAL_PIPELINE_AND_ISOLATION = PASS
```

This proves that the data path, online XYZ renderer, 8 GB training setup,
gradient accumulation, checkpointing, and PnP-only isolation work. It does
not prove pose improvement. The final C1 protocol freezes Patch-PnP, so the
formal PnP-only run is deferred unless C2 later unfreezes Patch-PnP; in that
case B uses all 50 PBR scenes and the same 40-epoch LM-O evaluation budget as
C2.

Artifacts:

```text
output/EXP-20260731-005/pnp_only_local/metrics.json
output/EXP-20260731-005/pnp_only_local/local_pilot_summary.json
output/EXP-20260731-005/pnp_only_local/model_final.pth
```
