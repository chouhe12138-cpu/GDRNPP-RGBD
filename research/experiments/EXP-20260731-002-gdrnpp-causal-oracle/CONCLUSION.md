# Stage 2 Conclusion — GDRNPP Causal Oracle

## Decision

`PASS — PREDICTED XYZ GEOMETRY IS THE PRIMARY BOTTLENECK`

This is a causal diagnostic result, not a deployable-method result.  GT depth,
masks, and poses were used only to construct frozen oracles on LM-O test.

## Key Results

| Method | BOP AR (%) | ADD(-S) (%) | Interpretation |
|---|---:|---:|---|
| Official Patch-PnP | 69.021 | 50.242 | Official baseline |
| Pred XYZ + Pred visible | 69.594 | 53.080 | Explicit geometry baseline |
| Pred XYZ + GT visible | 69.142 | 54.256 | Mask replacement is not causal |
| GT XYZ + GT visible | 100.000 | 100.000 | Complete coordinate oracle |
| True-error best 50% | 68.820 | 52.042 | Pointwise reliability is insufficient |
| Oracle best Patch/RANSAC | 71.908 | 60.830 | Aggregators are complementary |
| Oracle best R and t axes | 72.614 | 61.107 | Strong secondary axis signal |

The primary XYZ comparison passes every frozen condition:

- +30.858 BOP AR points.
- +45.744 ADD(-S) points.
- 95% paired-bootstrap interval: [+43.577, +47.933] points.
- Non-negative on 8/8 objects.
- 97.49% of the full oracle gap closed.

Mask support, full-mask support, and pixel reliability do not pass.  Notably,
selecting the half of correspondences with the lowest true XYZ error still
hurts pose, which shows that correspondence spatial distribution and joint
conditioning cannot be replaced by a scalar per-pixel accuracy score.

Axis-wise aggregation is the most actionable secondary observation, but its
17.11% oracle-gap closure is below the frozen 30% gate.  It must be validated
on a held-out PBR split before it can motivate a learned module.

## Research Interpretation

The defensible GDRNPP-specific problem statement is:

> GDRNPP predicts sufficiently accurate visible/full supports, but pose
> accuracy is limited by the geometric quality of its dense object-coordinate
> field.  Moreover, low pointwise XYZ error alone does not identify a good PnP
> subset because pose recovery also depends on correspondence coverage and
> conditioning.  Patch-PnP and explicit geometry contain complementary,
> axis-dependent errors, but that secondary opportunity is substantially
> smaller than the dense-coordinate oracle gap.

No network modification is authorized by this diagnostic alone.
