# Stage 2 — GDRNPP Causal-Oracle Diagnostic

Status: `COMPLETE — PASS (XYZ GEOMETRY)`

## Goal

Identify a GDRNPP-specific bottleneck without transferring the RDPN6D
conclusion.  The frozen diagnostic separates visible support, dense XYZ,
pixel reliability, double-mask usage, pose aggregation, and axis coupling.

This is a diagnostic-only experiment.  Test depth, masks, and poses construct
oracles and are never presented as inputs to a deployable RGB method.

## Frozen Inputs

```text
Checkpoint: official ConvNeXt-Base LM-O GDRNPP
Dataset:    LM-O BOP19, 1,445 targets
BBox:       GT
Seed:       20260730
PnP:        RANSAC-EPnP, 3 px, 100 iterations
Top-k:      50%, disabled below 32 correspondences
```

All methods share one frozen network forward.  No model parameter, solver
parameter, threshold, or method is selected from the test results.

## Frozen Methods

| Method | Coordinates | Support / selection | Solver |
|---|---|---|---|
| `patch_pnp` | predicted | learned head | Patch-PnP |
| `pred_vis_ransac` | predicted | predicted visible mask | RANSAC |
| `pred_full_ransac` | predicted | predicted full mask | RANSAC |
| `pred_inter_gt_vis_ransac` | predicted | predicted ∩ GT visible | RANSAC |
| `pred_gt_vis_ransac` | predicted | GT visible | RANSAC |
| `gt_inter_vis_ransac` | GT | predicted ∩ GT visible | RANSAC |
| `gt_gt_vis_ransac` | GT | GT visible | RANSAC |
| `current_top50_ransac` | predicted | current score, top 50% | RANSAC |
| `current_shared_top50_ransac` | predicted | current score on shared support | RANSAC |
| `oracle_xyz_top50_ransac` | predicted | true XYZ error, best 50% | RANSAC |
| `oracle_best_pose` | predicted | per-instance best Patch/RANSAC | Oracle selector |
| `oracle_best_axis` | predicted | per-axis best Patch/RANSAC | Oracle selector |

GT XYZ is reconstructed from observed depth and the GT pose:

```text
X_cam = depth * K^-1 [u, v, 1]^T
X_obj = R_gt^T (X_cam - t_gt)
```

## Decision Gate

A factor passes only if all conditions hold relative to its matched baseline:

- BOP AR gain at least 1 percentage point.
- ADD(-S)-0.1d recall gain at least 5 percentage points.
- Image-clustered 10,000-iteration paired-bootstrap 95% lower bound above 0.
- Non-negative ADD(-S) effect on at least 6/8 objects.
- At least 30% of the complete GT-XYZ + GT-visible oracle gap is closed.

The passing comparison with the largest gap closure becomes the primary
finding.  If none passes, the formal result is `FAIL`; no architecture change
is authorized.

## Validation and Exit

- Pure CPU solver/oracle tests must pass.
- The 16-instance smoke run must contain 12 × 16 pose rows and all 8 objects.
- GT-XYZ maximum reprojection error must be below 0.5 px.
- The full run must contain exactly 1,445 targets and reproduce Stage 1
  `patch_pnp` and `pred_vis_ransac` results.
- Save per-instance, per-object, visibility-bin, dense-quality, BOP, protocol,
  and conclusion artifacts.
- Stop after recording the causal conclusion; do not train or change GDRNPP.
