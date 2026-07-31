# EXP-20260731-002 — GDRNPP Causal-Oracle Diagnostic

## Status

`COMPLETE — PASS (XYZ GEOMETRY)`

## Purpose

Determine whether GDRNPP's main recoverable error source is visible support,
dense XYZ, pixel reliability, double-mask usage, pose aggregation, or separate
rotation/translation aggregation.

## Frozen Evidence Boundary

- LM-O test GT and depth are oracle-only diagnostic information.
- No training, tuning, architecture change, or paper performance claim is
  permitted from the oracle numbers.
- Full protocol: `research/stages/STAGE_02_GDRNPP_CAUSAL_ORACLE.md`.

## Validation

- CPU unit tests: 13/13 PASS, including all Stage 1 tests.
- GPU smoke: 16 targets, two per LM-O object, 192/192 method rows.
- All 12 methods returned successful poses on the smoke set.
- GT-XYZ maximum reprojection error: 0.0719 px (required < 0.5 px).
- Full run: exactly 1,445/1,445 targets and 17,340 method rows.
- Official BOP19 evaluation: 12/12 methods complete.
- Stage 1 `patch_pnp` and `pred_vis_ransac`: exact ADD(-S) and BOP AR
  reproduction.
- Full-run GT-XYZ maximum reprojection error: 0.4219 px.

## Formal Result

The pre-registered gate identifies predicted XYZ geometry as the primary
bottleneck.  Replacing predicted XYZ with GT XYZ while holding GT-visible
support fixed changes BOP AR from 69.142% to 100.000% and ADD(-S) recall from
54.256% to 100.000%.  The gains are +30.858 and +45.744 percentage points;
the paired ADD(-S) 95% interval is [+43.577, +47.933], all 8/8 objects are
non-negative, and 97.49% of the complete oracle gap is closed.

The support-matched comparison confirms the same result: GT XYZ raises BOP AR
from 69.255% to 99.377% and ADD(-S) from 53.841% to 99.377%, with 8/8 objects
non-negative and 97.05% gap closure.

## Rejected Primary Explanations

- Replacing predicted visible support with GT visible support adds only 1.176
  ADD(-S) points, reduces BOP AR by 0.452 points, and has a confidence interval
  crossing zero.
- Predicted full-mask support reduces BOP AR by 0.348 points and ADD(-S) by
  0.554 points relative to predicted visible support.
- Even true XYZ-error ranking followed by retaining the best 50% reduces BOP
  AR by 0.435 points and ADD(-S) by 1.799 points relative to the unfiltered
  shared support.  Pointwise error ranking alone is therefore insufficient;
  spatial coverage and joint correspondence geometry matter.
- The current mask × region score has mean Spearman correlation -0.062 with
  true XYZ error and is not a calibrated reliability measure.

## Secondary Signal

Oracle selection between Patch-PnP and explicit RANSAC raises BOP AR by 2.314
points and ADD(-S) by 7.751 points.  Selecting rotation and translation axes
separately raises BOP AR by 3.020 points and ADD(-S) by 8.028 points, with a
paired 95% interval of [+6.612, +9.497] and 8/8 objects non-negative.

This is a real complementary signal, but it closes only 17.11% of the full
XYZ oracle gap and therefore fails the frozen 30% primary-factor requirement.
It is recorded as a candidate for a separately authorized validation-stage
experiment, not as the Stage 2 primary conclusion.

## Dense Diagnostics

- Median instance-level visible-mask IoU: 0.803.
- Median instance-level full-mask IoU: 0.863.
- Median per-instance XYZ median error: 7.70 mm, or 5.01% of object diameter.
- Across instances, the 90th percentile of normalized median XYZ error is
  27.15% of object diameter.
- Normalized median XYZ error has Spearman correlation -0.346 with RANSAC
  ADD(-S) success; mask correlations are descriptive difficulty indicators,
  not causal evidence, because mask oracle replacement did not improve pose.

## Runtime Artifacts

```text
output/EXP-20260731-002/smoke/
output/EXP-20260731-002/full/
```

Stage 2 stops here.  It does not authorize training or an architecture change.
