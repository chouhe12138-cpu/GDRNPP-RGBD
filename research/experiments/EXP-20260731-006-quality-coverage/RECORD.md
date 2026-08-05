# EXP-20260731-006 — Quality/Coverage Residual Attention

Status: `FORMAL COMPLETE — C1_SCREEN_FAIL`

The experiment implements one identity-initialized, lightweight,
region-balanced quality/coverage mechanism feeding the official direct
Patch-PnP `R,t` head.

Frozen protocol:

```text
research/stages/STAGE_03C1_QUALITY_COVERAGE_ATTENTION.md
```

The official components are frozen in C1.  Therefore the primary comparison
is the official checkpoint A versus C1.  Patch-PnP-only B becomes mandatory
only if C2 unfreezes Patch-PnP.

## Final 40-epoch result

The L40 formal run uses all 50 LM-PBR scenes, the official LM-O checkpoint,
one pre-registered seed (`20260731`), and LM-O GT-box evaluation every five
epochs.

| Checkpoint | BOP AR (%) | Δ BOP AR (pp) | ADD(-S)@0.1d (%) | Δ ADD(-S) (pp) | Nonnegative objects |
|---|---:|---:|---:|---:|---:|
| Official checkpoint | 69.0415 | — | 50.86 | — | — |
| Epoch 5 | 69.0242 | -0.0173 | 51.00 | +0.14 | 7/8 |
| Epoch 10 | 68.9624 | -0.0791 | 50.36 | -0.50 | 5/8 |
| Epoch 15 | 68.9836 | -0.0579 | 50.79 | -0.07 | 6/8 |
| Epoch 20 | 68.9933 | -0.0482 | 50.43 | -0.43 | 4/8 |
| Epoch 25 | 68.9813 | -0.0602 | 50.94 | +0.08 | 5/8 |
| Epoch 30 | 68.9670 | -0.0745 | 50.37 | -0.49 | 4/8 |
| Epoch 35 | 68.9751 | -0.0664 | 50.59 | -0.27 | 4/8 |
| Epoch 40 | 68.9742 | -0.0674 | 50.57 | -0.29 | 4/8 |

Epoch 5 remains the best observed checkpoint.  Its per-object ADD(-S)
differences from the official checkpoint are: ape `+1.15`, can `+0.51`, cat
`-0.59`, and `0.00` for driller, duck, eggbox, glue, and holepuncher.  At
epoch 10 they are: ape `-0.57`, can `+0.51`, cat `-1.76`, duck `-2.22`, and
`0.00` for the other four objects.  At epoch 15 they are: ape `+1.15`, can
`+0.51`, cat `-1.17`, eggbox `-1.11`, and `0.00` for driller, duck, glue,
and holepuncher.  At epoch 20 they are: ape `-1.14`, can `+0.51`, cat
`-1.76`, driller `-0.50`, duck `-0.56`, and `0.00` for eggbox, glue, and
holepuncher.  At epoch 25 they are: ape `+1.72`, can `+0.51`, cat `-0.59`,
driller `+0.50`, duck `+0.55`, eggbox `-1.11`, glue `0.00`, and
holepuncher `-1.00`.  At epoch 30 they are: ape `-0.57`, can `0.00`,
cat `-1.76`, driller `-0.50`, duck `-2.22`, eggbox `+0.56`, glue `0.00`,
and holepuncher `+0.50`.  At epoch 35 they are: ape `-0.57`, can `0.00`,
cat `-0.59`, driller `-0.50`, duck `0.00`, eggbox `-0.55`, glue `0.00`,
and holepuncher `0.00`.

Epoch-20 per-object ADD(-S)@0.1d:

| Object | Official (%) | Epoch 20 (%) | Δ (pp) |
|---|---:|---:|---:|
| ape | 49.14 | 48.00 | -1.14 |
| can | 80.90 | 81.41 | +0.51 |
| cat | 47.37 | 45.61 | -1.76 |
| driller | 82.50 | 82.00 | -0.50 |
| duck | 8.89 | 8.33 | -0.56 |
| eggbox | 41.11 | 41.11 | 0.00 |
| glue | 75.00 | 75.00 | 0.00 |
| holepuncher | 22.00 | 22.00 | 0.00 |

Epoch-25 per-object ADD(-S)@0.1d:

| Object | Official (%) | Epoch 25 (%) | Δ (pp) |
|---|---:|---:|---:|
| ape | 49.14 | 50.86 | +1.72 |
| can | 80.90 | 81.41 | +0.51 |
| cat | 47.37 | 46.78 | -0.59 |
| driller | 82.50 | 83.00 | +0.50 |
| duck | 8.89 | 9.44 | +0.55 |
| eggbox | 41.11 | 40.00 | -1.11 |
| glue | 75.00 | 75.00 | 0.00 |
| holepuncher | 22.00 | 21.00 | -1.00 |

Epoch-30 per-object ADD(-S)@0.1d:

| Object | Official (%) | Epoch 30 (%) | Δ (pp) |
|---|---:|---:|---:|
| ape | 49.14 | 48.57 | -0.57 |
| can | 80.90 | 80.90 | 0.00 |
| cat | 47.37 | 45.61 | -1.76 |
| driller | 82.50 | 82.00 | -0.50 |
| duck | 8.89 | 6.67 | -2.22 |
| eggbox | 41.11 | 41.67 | +0.56 |
| glue | 75.00 | 75.00 | 0.00 |
| holepuncher | 22.00 | 22.50 | +0.50 |

Epoch-35 per-object ADD(-S)@0.1d:

| Object | Official (%) | Epoch 35 (%) | Δ (pp) |
|---|---:|---:|---:|
| ape | 49.14 | 48.57 | -0.57 |
| can | 80.90 | 80.90 | 0.00 |
| cat | 47.37 | 46.78 | -0.59 |
| driller | 82.50 | 82.00 | -0.50 |
| duck | 8.89 | 8.89 | 0.00 |
| eggbox | 41.11 | 40.56 | -0.55 |
| glue | 75.00 | 75.00 | 0.00 |
| holepuncher | 22.00 | 22.00 | 0.00 |

Epoch-40 per-object ADD(-S)@0.1d:

| Object | Official (%) | Epoch 40 (%) | Δ (pp) |
|---|---:|---:|---:|
| ape | 49.14 | 49.14 | 0.00 |
| can | 80.90 | 80.90 | 0.00 |
| cat | 47.37 | 46.78 | -0.59 |
| driller | 82.50 | 82.00 | -0.50 |
| duck | 8.89 | 8.89 | 0.00 |
| eggbox | 41.11 | 40.56 | -0.55 |
| glue | 75.00 | 74.29 | -0.71 |
| holepuncher | 22.00 | 22.00 | 0.00 |

The epoch-40 periodic evaluation and the immediately repeated final
in-memory evaluation are identical.

## Screening assessment

| Frozen C1 gate | Required | Current best (epoch 5) | Result |
|---|---:|---:|---|
| Δ BOP AR | at least +0.50 pp | -0.0173 pp | Fail |
| Δ ADD(-S)@0.1d | at least +1.00 pp | +0.14 pp | Fail |
| Nonnegative objects | at least 5/8 | 7/8 | Pass |

The best observed checkpoint fails the BOP and ADD gates, while fixed epoch
40 fails all three gates.  The completed experiment is therefore
`C1_SCREEN_FAIL`.

## Training health and interpretation

- Training is numerically stable: no NaN, Inf, or OOM occurred.  The only
  traceback is the post-run summarizer path mismatch after both final
  evaluations completed.
- Peak GPU memory is 5,285 MiB and throughput is about 1.22 seconds/step.
- Sampled mean total loss fell from `0.055865` at epoch 1 to `0.052811` at
  epoch 5, then plateaued at `0.053341` at epoch 10 and `0.053578` at epoch
  15; the sampled means are `0.053992` at epoch 20 and `0.053158` at epoch
  25, `0.053304` at epoch 30, `0.053021` at epoch 35, and `0.054606` at
  epoch 40 across sampled progress records.  The writer's cumulative
  epoch-40 value is `0.05487`.

The module is learning on synthetic PBR data, but the improvement has not
transferred to real LM-O pose accuracy.  The leading explanations are the
PBR-to-real occlusion/appearance gap and the frozen Patch-PnP head being
unable to adapt to the reweighted region features.  These are hypotheses for
the conditional matched controls, not conclusions proven by this run alone.

## Evaluation note

ADD(-S) is computed and printed, and periodic checkpoints are saved.  However,
the automatic lookup warns that it found no ADD result because evaluator
directories use `error:ad_ntop:1` while the lookup expects
`error=ad_ntop=*`.  This affects automatic secondary-metric lookup,
summarization, and plotting.  It does not change the current epoch-5 choice,
which remains better than epochs 10, 15, 20, and 25 on the primary metric.
Epochs 30 and 35 also remain below epoch 5 and the official checkpoint.

The complete log contains excessive repeated dependency warnings: 1,365
`UserWarning` occurrences and 5,452 `NumbaDeprecationWarning` occurrences.
The completed run is preserved unchanged.  The compatible ADD lookup,
compact logging, and warning reduction apply only to subsequent experiments.

## Evidence

```text
source log: E:\6D姿态估计\26-08-02\26-08-02-14.log
SHA-256:   fc4551e073175ed40e374faf154d32c5ecb2225646b3029c221f7f5acf2b7205
size:      3291926 bytes
snapshot:  2026-08-02 14:17:26 +0800
```

Epoch-15 evidence:

```text
source log: E:\6D姿态估计\26-08-02\26-08-03-15epoch.log
SHA-256:   b4c2a74d638b63276ebabec0e53f1edbbb99507f44eea492e80061dd8e1d6420
size:      4209158 bytes
snapshot:  2026-08-03 00:11:45 +0800
```

Epoch-20 evidence:

```text
source log: E:\6D姿态估计\26-08-02\26-08-03-20epoch.log
SHA-256:   fe5643ece26f69767488662e2a78611e0ca888e41e058d2d167478dcdf432f2e
size:      5139428 bytes
snapshot:  2026-08-03 11:45:20 +0800
```

Epoch-25 evidence:

```text
source log: E:\6D姿态估计\26-08-02\26-08-03-25epoch.log
SHA-256:   b92e56587dd7b7e173d2041240dc399fbb1b1fd3a7cda2464edb7a2c54022410
size:      6073473 bytes
snapshot:  2026-08-03 23:48:25 +0800
```

Epoch-30 evidence:

```text
source log: E:\6D姿态估计\26-08-02\26-08-04-30epoch.log
SHA-256:   b7b6a3ecbdd07f58743baaa7278f51316aebc8871b7518fddde2ff9fa35712be
size:      7023674 bytes
snapshot:  2026-08-04 14:00:39 +0800
```

Epoch-35 evidence:

```text
source log: E:\6D姿态估计\26-08-02\26-08-04-35epoch.log
SHA-256:   b07f5d9dfe56da86a33001bcf1107fc0d4a8cccdbb907be55818d0bcd5d8292d
size:      7915641 bytes
snapshot:  2026-08-04 20:41:38 +0800
```

Final epoch-40 evidence:

```text
source log: E:\6D姿态估计\26-08-02\26-08-05-40epoch.log
SHA-256:   a7333b54f64aa9effd2f14677f047560727d489a9e5b0987a6bc70bdf7a5009a
size:      9678712 bytes
snapshot:  2026-08-05 12:21:07 +0800
checkpoint: /workspace/gdrnpp/output/EXP-20260731-006/quality_coverage_full/model_0255919.pth
checkpoint SHA-256: d3ab7167f2fc5f6aab8d7e8444c5b816036bd64e38f647a26e994c8e91939aa6
local copy: E:\6D姿态估计\26-08-02\model_0255919.pth
local size: 411145010 bytes
verification: local copy SHA-256 matches the server checkpoint
```
