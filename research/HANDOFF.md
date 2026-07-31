# GDRNPP-RGBD Research Handoff

Last updated: 2026-07-31

## Start Here

Repository:

```text
/home/wsluser/GDRNPP-RGBD
```

Read this file, then inspect `git status` before doing any work.  The worktree
contains intentional uncommitted research files and pre-existing changes; do
not reset, clean, overwrite, or commit them without confirming scope.

Current research state:

```text
Stage 1:  COMPLETE — FAIL
Stage 2:  COMPLETE — PASS (XYZ GEOMETRY)
Stage 3A: COMPLETE — CALIBRATION_MISMATCH (NOT FORMAL VALIDATION)
Stage 3B: COMPLETE — PATCH_PNP_UNDERUTILIZATION
Stage 3C-0: LOCAL PILOT PASS — RETAINED AS CONDITIONAL B CONTROL
Stage 3C-1: IMPLEMENTED — LOCAL C1 PILOT PENDING
```

A single lightweight quality/coverage residual-attention architecture is now
implemented. No C1 training or pose-accuracy result exists yet.

## Stage 1 — Pose Aggregation Diagnostic

Protocol: official ConvNeXt-Base GDRNPP, LM-O BOP19, 1,445 targets, GT boxes,
RGB only, one shared network forward.

Key result:

| Method | BOP AR (%) | ADD(-S) (%) |
|---|---:|---:|
| Patch-PnP | 69.021 | 50.242 |
| RANSAC-EPnP | 69.594 | 53.080 |
| Current reliability Top-50% | 68.719 | 51.696 |

RANSAC gives a small, object-dependent gain but improves/ties only 5/8 objects.
The mask × region reliability proxy is rejected.

Evidence:

```text
research/experiments/EXP-20260730-001-gdrnpp-pose-aggregation-diagnostic/
output/EXP-20260730-001/full/
```

## Stage 2 — Causal Oracle Diagnostic

The same LM-O protocol was frozen and decomposed into mask, XYZ, reliability,
aggregation, and axis oracles.  Test depth/masks/poses were oracle-only and
must never be described as deployable RGB input.

Formal decision:

```text
PASS — predicted dense XYZ geometry is the primary causal bottleneck.
```

Key result:

| Method | BOP AR (%) | ADD(-S) (%) |
|---|---:|---:|
| Patch-PnP | 69.021 | 50.242 |
| Pred XYZ + Pred visible | 69.594 | 53.080 |
| Pred XYZ + GT visible | 69.142 | 54.256 |
| GT XYZ + GT visible | 100.000 | 100.000 |
| True-XYZ-error Top-50% | 68.820 | 52.042 |
| Oracle best R/t axes | 72.614 | 61.107 |

GT XYZ closes 97.49% of the complete oracle gap and is non-negative on 8/8
objects.  GT mask is neutral.  Even retaining the 50% lowest true-XYZ-error
points hurts LM-O, showing that pointwise accuracy alone is insufficient;
coverage and conditioning matter.  Axis-wise Patch-PnP/RANSAC selection is a
strong secondary signal (+3.020 BOP AR, +8.028 ADD(-S)) but fails the frozen
30% oracle-gap-closure primary gate.

Evidence:

```text
research/experiments/EXP-20260731-002-gdrnpp-causal-oracle/
research/stages/STAGE_02_GDRNPP_CAUSAL_ORACLE.md
output/EXP-20260731-002/full/oracle_summary.json
```

## Stage 3A — PBR Validation Infrastructure

Dataset facts:

- Dataset root: `/home/wsluser/Datasets/BOP_DATASETS`.
- VOC root: `/home/wsluser/Datasets/VOC`.
- All 50 PBR scenes / 50,000 images are extracted and deep-verified.
- PBR contains 749,600 annotations and 1,499,200 full/visible mask paths.
- The eight LM-O objects contribute 399,950 PBR annotations.
- VOC2012 contains 17,125 JPEGs and 538 verified table backgrounds.
- Repository dataset paths are symbolic links; no dataset was copied.

Historical Stage 3A calibration split (not used by the final C1 training
protocol):

```text
Train scenes:       00–11 and 15–49 = 47,000 images
Validation reserve: 12–14           =  3,000 images
Tracked diagnostic subset:          =  1,500 images
Local calibration subset:           =    300 images
```

The official checkpoint may already have trained on validation scenes.  The
current PBR result is calibration only, not held-out validation and not model
selection evidence.

Calibration result, 300 images / 2,087 valid instances:

| Method | ADD(-S) (%) |
|---|---:|
| Patch-PnP | 81.696 |
| Pred XYZ + Pred visible | 80.307 |
| Pred XYZ + GT visible | 80.498 |
| GT XYZ + GT visible | 99.952 |
| True-XYZ-error Top-50% | 84.236 |
| Oracle best R/t axes | 85.721 |

Frozen decision:

```text
CALIBRATION_MISMATCH
```

- GT XYZ improves by 19.454 points, 0.546 below the frozen 20-point threshold.
- GT mask is neutral at +0.192 points.
- Axis Oracle improves by 4.025 points, 0.975 below the 5-point threshold.
- True-error Top-50% improves PBR by +3.833 but hurt LM-O by -1.799.
- PBR median normalized XYZ error is 0.929% diameter versus 5.010% on LM-O.
- PBR boundary XYZ error is 2.977 mm versus 1.382 mm in the interior.

The cross-domain hypothesis is therefore:

> Correspondence aggregation must jointly balance coordinate precision and
> geometric coverage.  Scalar reliability alone does not transfer from clean
> synthetic data to real occlusion.

Evidence:

```text
research/experiments/EXP-20260731-003-pbr-validation-calibration/
research/stages/STAGE_03A_PBR_VALIDATION_CALIBRATION.md
research/splits/lmo_pbr_stage3_scene_split.json
output/EXP-20260731-003/calibration300/calibration_summary.json
```

## Verification

Current test command:

```bash
cd /home/wsluser/GDRNPP-RGBD
PYTHONPATH=/home/wsluser/GDRNPP-RGBD \
PYTHONPYCACHEPREFIX=/tmp/gdrnpp-pycache \
conda run -n pytorch22 python -m pytest -q \
  -o cache_dir=/tmp/gdrnpp-pytest-cache \
  research/quality_coverage/tests \
  research/pnp_control/tests \
  research/pose_head_utilization/tests \
  research/pbr_validation/tests \
  research/oracle_diagnostic/tests \
  research/pose_aggregation/tests
```

Expected result after Stage 3C-1 implementation: `43 passed`.

Official checkpoint SHA-256:

```text
bafa869d4e6c00410517ecb1add59f234ed1642e47fabcf3aa6e0e8a1b498a8c
```

## Stage 3B — Frozen Pose-Head Utilization

The full 1,445-target diagnostic and all 10 BOP19 evaluations completed.

| Path | ADD(-S), alpha 0 | ADD(-S), alpha 1 | BOP AR, alpha 0 | BOP AR, alpha 1 |
|---|---:|---:|---:|---:|
| Direct Patch-PnP | 50.242% | 49.550% | 69.021% | 68.324% |
| Diagnostic RANSAC | 53.841% | 99.377% | 69.255% | 99.377% |

Decision:

```text
PATCH_PNP_UNDERUTILIZATION
```

RANSAC improves monotonically and on 8/8 objects; Patch-PnP is not monotonic
and is non-negative on only 5/8. The official direct head therefore does not
convert improved XYZ into better `R,t`. RANSAC remains diagnostic only.

Evidence:

```text
research/experiments/EXP-20260731-004-gdrnpp-pose-head-utilization/
research/stages/STAGE_03B_POSE_HEAD_UTILIZATION_DIAGNOSTIC.md
output/EXP-20260731-004/full/utilization_summary.json
```

## Stage 3C-0 — Patch-PnP Adaptation Control

The required control now exists and its local pilot passed:

```text
data:               8,192 instances, 1,024 per object
physical/effective: batch 4 / 48
execution:          2,048 micro-batches, 171 optimizer updates
peak memory:        4,820 MiB
total-loss trend:   -5.41% first-to-last-quarter median
checkpoint reload:  PASS
changed tensors:    17 pnp_net, 0 frozen
```

Evidence:

```text
research/stages/STAGE_03C0_PNP_ADAPTATION_CONTROL.md
research/experiments/EXP-20260731-005-pnp-only-control/
research/pnp_control/
output/EXP-20260731-005/pnp_only_local/
```

The earlier five-epoch/47-scene control protocol was superseded by the
user-selected full protocol on 2026-07-31. Stage 3C-0 is now the conditional B
control and is run only if C2 unfreezes Patch-PnP.

## Stage 3C-1 — Quality/Coverage Attention

The C1 implementation keeps official XYZ values, input dimensions, and direct
`R,t`. It adds identity-initialized residual reweighting of the existing 64
region-attention maps. The backbone, geometry head, and Patch-PnP are frozen;
only the new module is trained.

Formal protocol: all 50 PBR scenes, 40 epochs, batch 48, LM-O GT-box
evaluation after every five completed epochs, direct `R,t`, and best-one plus
latest-two checkpoint retention.

Next execution:

```bash
conda activate pytorch22
research/quality_coverage/run_local.sh
```

Only after that local architecture gate passes should
`research/quality_coverage/run_l40.sh` be run inside the L40 Docker container.
Do not describe implementation/preflight results as pose-accuracy evidence.
