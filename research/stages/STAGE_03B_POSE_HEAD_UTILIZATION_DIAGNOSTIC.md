# Stage 3B — Frozen Patch-PnP XYZ-Utilization Diagnostic

Status: `COMPLETE — PATCH_PNP_UNDERUTILIZATION`

## Goal

Determine whether the official frozen GDRNPP Patch-PnP head converts a
controlled improvement in dense XYZ into a corresponding improvement in
rotation and translation.

This stage tests the missing causal link between Stage 2's XYZ oracle and the
official direct pose head. It contains no training and does not authorize a
new architecture.

## Frozen Inputs

```text
Checkpoint: official ConvNeXt-Base LM-O GDRNPP
Dataset:    LM-O BOP19, 1,445 targets
BBox:       GT
Seed:       20260730
PnP ref:    RANSAC-EPnP, 3 px, 100 iterations
Alpha:      0.00, 0.25, 0.50, 0.75, 1.00
```

The checkpoint SHA-256 remains:

```text
bafa869d4e6c00410517ecb1add59f234ed1642e47fabcf3aa6e0e8a1b498a8c
```

## Intervention

On the fixed support `predicted visible ∩ GT visible ∩ valid depth`, construct:

\[
X_\alpha=(1-\alpha)X_\text{pred}+\alpha X_\text{GT}.
\]

Pixels outside the correction support retain the original prediction. Visible
mask, region logits, 2D coordinates, Patch-PnP weights, bbox, and the
correction support remain fixed.

For every alpha:

1. rerun only the frozen official Patch-PnP head to obtain direct `R,t`;
2. solve RANSAC-EPnP on the same fixed correctable support as a diagnostic
   reference.

GT depth, masks, poses, and interpolated XYZ are oracle-only. None is a
deployable model input.

## Validation

- Pure helper tests must verify support preservation, exact interpolation, and
  decision logic.
- The 8-object smoke run must complete all 10 methods.
- Alpha zero Patch-PnP uses the original official forward pose; a separate
  re-entry audit records AMP numerical differences.
- Full alpha-zero ADD(-S) must exactly reproduce Stage 2 `patch_pnp` and
  `pred_inter_gt_vis_ransac`.
- Full execution must contain exactly 1,445 targets and all official BOP19
  evaluations.

## Frozen Decision

Let the alpha-zero to alpha-one ADD(-S) gains be `ΔPatch` and `ΔRANSAC`.

`PATCH_PNP_USES_IMPROVED_XYZ` requires:

- `ΔPatch >= 5` percentage points;
- Patch-PnP BOP AR gain at least 1 point;
- non-negative Patch-PnP ADD(-S) change on at least 6/8 objects;
- the Patch-PnP ADD(-S) curve is monotonic non-decreasing;
- `ΔPatch / ΔRANSAC >= 0.50`.

`PATCH_PNP_UNDERUTILIZATION` requires:

- the RANSAC reference gains at least 5 ADD(-S) points and 1 BOP AR point;
- RANSAC is non-negative on at least 6/8 objects and monotonic;
- `ΔPatch < max(1 point, 0.30 × ΔRANSAC)`;
- `ΔRANSAC - ΔPatch >= 5` points.

All other outcomes are `MIXED_OR_INCONCLUSIVE`.

Next action is mechanically determined:

- `PATCH_PNP_USES_IMPROVED_XYZ` → improve XYZ geometry first;
- `PATCH_PNP_UNDERUTILIZATION` → train a direct, region-balanced
  quality/coverage attention for Patch-PnP;
- `MIXED_OR_INCONCLUSIVE` → analyze rotation/translation curves before any
  training.

Stage 3B stops after recording the causal result.

## Result

The full run completed all 1,445 LM-O BOP19 targets and all 10 official BOP19
evaluations. Alpha-zero ADD(-S) exactly reproduced both Stage 2 baselines.

| Alpha | Patch-PnP ADD(-S) (%) | Patch-PnP BOP AR (%) | RANSAC ADD(-S) (%) | RANSAC BOP AR (%) |
|---:|---:|---:|---:|---:|
| 0.00 | 50.242 | 69.021 | 53.841 | 69.255 |
| 0.25 | 49.827 | 68.971 | 61.592 | 71.769 |
| 0.50 | 49.550 | 68.950 | 73.910 | 78.356 |
| 0.75 | 49.204 | 68.958 | 85.329 | 85.392 |
| 1.00 | 49.550 | 68.324 | 99.377 | 99.377 |

The RANSAC reference is monotonic, gains 45.536 ADD(-S) points and 30.122
BOP AR points, and is non-negative on 8/8 objects. Patch-PnP loses 0.692
ADD(-S) points and 0.697 BOP AR points, is non-negative on only 5/8 objects,
and is not monotonic. Its measured conversion ratio is -1.52%.

The frozen decision is therefore:

```text
PATCH_PNP_UNDERUTILIZATION
```

This does not make RANSAC the proposed deployment method. It establishes that
the corrected XYZ contains pose-useful information while the current direct
pose head does not convert that information into better `R,t`. The next
mechanism to test is a lightweight direct quality/coverage attention inside
Patch-PnP.
