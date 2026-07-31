# Stage 3A — PBR Validation Infrastructure and Calibration

Status: `COMPLETE — CALIBRATION_MISMATCH (NOT FORMAL VALIDATION)`

## Purpose

Create a scene-disjoint synthetic validation protocol for future retraining,
and check whether the Stage 2 XYZ/mask/aggregation pattern is visible on PBR.

The official checkpoint may have been trained on all LM-PBR scenes.  Therefore
the current run is explicitly a leakage-marked calibration, not formal model
selection evidence.

## Frozen Split

```text
Archive:             LM train_pbr, 50 scenes / 50,000 images
Future train scenes: 00–11, 15–49 (47,000 images)
Validation reserve:  12–14 (3,000 images)
Tracked diagnostic:  500 images per validation scene (1,500 total)
Local calibration:   first 100 tracked images per scene (300 total)
Seed:                20260731 for split, 20260730 for solvers
```

The split is tracked in
`research/splits/lmo_pbr_stage3_scene_split.json`.  Images remain on the
existing drive and are accessed through a symbolic link; no dataset is copied.

## Frozen Calibration Methods

- Official Patch-PnP.
- Predicted XYZ with predicted visible/full/GT-visible support.
- GT XYZ with GT-visible support.
- Current top-50% reliability filtering.
- True-XYZ-error top-50% filtering.
- Oracle best Patch-PnP/RANSAC pose.
- Oracle best Patch-PnP/RANSAC rotation and translation axes.

Dense diagnostics include per-axis XYZ error, boundary/interior XYZ error,
mask IoU, reliability/error correlation, and 2D/3D point-set eigenvalue ratios.

## Calibration Decision

`CALIBRATION_MATCH` requires:

- GT XYZ improves ADD(-S) by at least 20 points, with a positive clustered
  bootstrap interval and non-negative effect on 8/8 objects.
- GT-visible support improves ADD(-S) by less than 5 points.

Scalar reliability and axis complementarity are recorded as separate patterns.
Axis complementarity requires at least +5 ADD(-S) points, a positive interval,
and non-negative effect on at least 6/8 objects.

Regardless of the result, Stage 3A cannot authorize a paper claim or network
selection because the official checkpoint may have seen the validation scenes.
Formal Stage 3B requires baseline and variants retrained on the 47 training
scenes with scenes 12–14 excluded.
