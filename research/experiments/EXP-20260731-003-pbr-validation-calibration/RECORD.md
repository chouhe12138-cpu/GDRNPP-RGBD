# EXP-20260731-003 — PBR Validation Calibration

## Status

`COMPLETE — CALIBRATION_MISMATCH (NOT FORMAL VALIDATION)`

## Data Facts

- The 21 GB archive contains 50 scenes and 50,000 RGB images.
- Only scenes 00–14 (15,000 images) are currently extracted.
- Validation scenes 12–14 are available; the remaining 35 training scenes are
  not extracted in this stage.
- Scene-disjoint future split: 47,000 train / 3,000 validation images.
- Fixed diagnostic manifest: 1,500 validation images.
- Local calibration subset: 300 images.
- The dataset is reused through `datasets/BOP_DATASETS/lmo/train_pbr`; no
  image, depth, or mask file is copied.

## Validation So Far

- Unit tests: 16/16 PASS.
- Manifest inspection: 1,500 images, 10,463 valid LM-O instances.
- Fixed path sample: 30 images, zero missing RGB/depth/mask files.
- GPU smoke: 16 instances, two per object, all 10 methods successful.

## Calibration Run

- Images: 300, 100 from each validation scene.
- Valid LM-O instances: 2,087.
- Pose methods: 10, producing 20,870 per-instance rows.
- Network inference time: recorded in the runtime summary.
- Clustered paired bootstrap: 2,000 iterations over scene/image IDs.

| Method | ADD(-S)-0.1d (%) |
|---|---:|
| Official Patch-PnP | 81.696 |
| Pred XYZ + Pred visible | 80.307 |
| Pred XYZ + GT visible | 80.498 |
| GT XYZ + GT visible | 99.952 |
| True-XYZ-error Top-50% | 84.236 |
| Oracle best Patch/RANSAC | 85.673 |
| Oracle best R/t axes | 85.721 |

## Frozen Decision

The result is `CALIBRATION_MISMATCH`:

- GT XYZ improves ADD(-S) by 19.454 points, with interval
  [+17.437, +21.422] and 8/8 objects non-negative.  This is the largest causal
  factor but falls 0.546 points below the frozen 20-point pattern threshold.
- GT visible support improves ADD(-S) by only 0.192 points, with an interval
  crossing zero.  Mask remains non-primary.
- True-error Top-50% improves ADD(-S) by 3.833 points on PBR, unlike its
  -1.799-point LM-O effect.
- Axis-wise Oracle selection improves ADD(-S) by 4.025 points with 8/8 objects
  non-negative, but falls 0.975 points below the frozen 5-point threshold.

Thresholds are not changed after observing the results.

## Dense and Domain Diagnostics

- Median visible-mask IoU: 0.946 on PBR versus 0.803 on LM-O.
- Median normalized XYZ error: 0.929% of object diameter on PBR versus 5.010%
  on LM-O.
- Median boundary XYZ error: 2.977 mm.
- Median interior XYZ error: 1.382 mm.
- Mean reliability/error Spearman correlation: -0.177 on PBR versus -0.062 on
  LM-O; reliability is better calibrated in the synthetic/training domain.
- True-error Top-50% changes median 2D eigenvalue ratio from 0.401 to 0.357 and
  median 3D ratio from 0.090 to 0.057.  Selecting accurate points reduces
  spatial diversity even when it helps on clean PBR.

The cross-domain sign reversal supports a precision/coverage trade-off:
pointwise reliability is useful only when coordinate noise is low enough that
the loss of spatial conditioning does not dominate.

## Evidence Boundary

The official checkpoint may already include scenes 12–14 in its original
training.  Results are pipeline calibration only.  They cannot select a model
or be reported as held-out validation performance.

Formal Stage 3B remains blocked on extracting the remaining training scenes
and retraining both baseline and variants with validation scenes excluded.
