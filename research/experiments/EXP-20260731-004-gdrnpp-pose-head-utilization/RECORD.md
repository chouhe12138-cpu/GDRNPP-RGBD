# EXP-20260731-004 — GDRNPP Pose-Head XYZ Utilization

Status: `COMPLETE — PATCH_PNP_UNDERUTILIZATION`

## Purpose

Test whether progressively corrected dense XYZ is converted into pose
improvements by the frozen official Patch-PnP head.

Protocol:

```text
research/stages/STAGE_03B_POSE_HEAD_UTILIZATION_DIAGNOSTIC.md
```

## Smoke

The 8-object / 8-instance GPU smoke completed all 80 method rows.

- Frozen support and five alpha levels were present.
- Alpha-zero official pose was reused exactly.
- Separate Patch-PnP re-entry audit differences were bounded by
  `6.013e-4` for rotation-matrix elements and `7.184e-4 m` for translation
  under AMP.
- The smoke is not a formal decision.

Artifacts:

```text
output/EXP-20260731-004/smoke/
```

## Full Result

The full GPU run completed 1,445 targets, 14,450 per-method instance rows, and
all 10 official BOP19 evaluations.

| Alpha | Patch ADD(-S) | Patch BOP AR | RANSAC ADD(-S) | RANSAC BOP AR |
|---:|---:|---:|---:|---:|
| 0.00 | 50.242% | 69.021% | 53.841% | 69.255% |
| 0.25 | 49.827% | 68.971% | 61.592% | 71.769% |
| 0.50 | 49.550% | 68.950% | 73.910% | 78.356% |
| 0.75 | 49.204% | 68.958% | 85.329% | 85.392% |
| 1.00 | 49.550% | 68.324% | 99.377% | 99.377% |

Alpha-zero ADD(-S) reproduced the Stage 2 Patch-PnP and RANSAC baselines
exactly. The BOP result CSV hashes were verified before finalization.

Object-level alpha-zero to alpha-one ADD(-S) changes:

| Object | Patch-PnP | RANSAC |
|---|---:|---:|
| ape | -1.143 | +48.571 |
| can | -4.020 | +20.101 |
| cat | -4.678 | +52.632 |
| driller | +0.000 | +14.000 |
| duck | +0.000 | +71.667 |
| eggbox | +1.667 | +61.111 |
| glue | +1.429 | +22.143 |
| holepuncher | +1.500 | +72.500 |

## Conclusion

```text
PATCH_PNP_UNDERUTILIZATION
next_action = TRAIN_DIRECT_QUALITY_COVERAGE_ATTENTION
```

The XYZ intervention is effective: under the fixed correspondence support,
RANSAC improves monotonically by 45.536 ADD(-S) points and 30.122 BOP AR
points, with non-negative changes on 8/8 objects. The frozen official
Patch-PnP head instead loses 0.692 ADD(-S) and 0.697 BOP AR points and is
non-negative on only 5/8 objects.

Thus, the dense XYZ already carries enough geometric information to support a
near-perfect pose on this oracle path, but the current direct regression head
does not use progressively better XYZ effectively. RANSAC remains diagnostic
only; the next deployable experiment must preserve direct `R,t` output and
modify only lightweight correspondence aggregation.

Artifacts:

```text
output/EXP-20260731-004/full/utilization_summary.json
output/EXP-20260731-004/full/per_instance.csv
output/EXP-20260731-004/full/per_object.csv
output/EXP-20260731-004/full/bop_eval/
```
