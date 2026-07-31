# Stage 3A Conclusion — PBR Validation Calibration

## Decision

`CALIBRATION_MISMATCH — NOT FORMAL VALIDATION`

The official checkpoint may have trained on scenes 12–14, so these results are
calibration evidence only.  The frozen mismatch decision is retained even
though GT XYZ misses its threshold by only 0.546 ADD(-S) points.

## What Replicated

- Mask support is not the primary bottleneck: GT mask changes ADD(-S) by only
  +0.192 points and its confidence interval crosses zero.
- XYZ remains the largest causal factor: GT XYZ adds +19.454 points with 8/8
  objects non-negative.
- Patch-PnP and RANSAC remain complementary, but the axis oracle adds +4.025
  points rather than the required +5.

## What Did Not Replicate

True-XYZ-error Top-50% filtering improves PBR ADD(-S) by +3.833 points but
reduced LM-O ADD(-S) by -1.799 points.  Scalar reliability therefore does not
transfer consistently from synthetic to real occlusion.

PBR is substantially easier:

| Diagnostic | PBR calibration | LM-O test |
|---|---:|---:|
| Median visible-mask IoU | 0.946 | 0.803 |
| Median normalized XYZ error | 0.929% diameter | 5.010% diameter |
| Mean reliability/error Spearman | -0.177 | -0.062 |

On PBR, boundary XYZ error is about 2.15× interior error.  Selecting the
lowest-error half of points also reduces both 2D and 3D spatial eigenvalue
ratios.  The resulting research hypothesis is not “learn a better confidence
score,” but:

> Pose correspondence selection must jointly balance coordinate precision and
> geometric coverage.  Synthetic data favors point accuracy, while real
> occlusion makes coverage and conditioning more important.

## Required Next Step

Before any method comparison:

1. Extract the remaining PBR training scenes.
2. Keep scenes 12–14 completely excluded from training.
3. Retrain the official baseline and every proposed variant under the same
   47-scene schedule.
4. Select methods only on the held-out PBR validation set.
5. Freeze the chosen method before one final LM-O test evaluation.

Stage 3A stops here and does not authorize extraction, retraining, or network
modification.
