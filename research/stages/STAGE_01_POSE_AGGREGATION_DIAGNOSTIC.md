# Stage 1 — GDRNPP Pose-Aggregation Diagnostic

Status: `COMPLETE — FAIL`

## Goal

Determine whether the official GDRNPP Patch-PnP head leaves recoverable pose
information in its dense XYZ, visible-mask, and region outputs, especially for
occluded LM-O instances.

## Inputs

```text
Config:
  configs/gdrn/lmo_pbr/
  convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo.py
Checkpoint:
  pretrained_models/lmo_pbr/model_final_wo_optim.pth
Dataset:
  datasets/BOP_DATASETS/lmo
Protocol:
  LM-O BOP19, 1,445 targets, GT bbox
Seed:
  20260730
```

The checkpoint SHA-256 must be:

```text
bafa869d4e6c00410517ecb1add59f234ed1642e47fabcf3aa6e0e8a1b498a8c
```

## Pre-registered Methods

1. `patch_pnp`: official network pose.
2. `epnp_all`: EPnP over every valid dense correspondence.
3. `ransac_epnp`: EPnP-RANSAC, 3 px threshold, 100 iterations.
4. `reliable_ransac`: retain the top 50% by visible-mask probability
   multiplied by maximum foreground-region probability, then use method 3.
5. `geom_R_net_t`: method 4 rotation with network translation.
6. `net_R_geom_t`: network rotation with reliability-weighted linear
   translation.

Points are filtered only by the official mask threshold and the official
background-coordinate rule.  Reliability filtering is disabled when fewer
than 32 valid points exist.  Fewer than four points is an explicit PnP failure;
failed estimates remain in the record.

## Execution

```bash
conda run -n pytorch22 pytest -q research/pose_aggregation/tests

conda run -n pytorch22 bash research/pose_aggregation/run_local.sh \
  --config-file configs/gdrn/lmo_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo.py \
  --weights pretrained_models/lmo_pbr/model_final_wo_optim.pth \
  --dataset lmo_bop_test \
  --bbox-source gt \
  --device cuda:0 \
  --smoke-per-object 2 \
  --output-dir output/EXP-20260730-001/smoke

conda run -n pytorch22 bash research/pose_aggregation/run_local.sh \
  --config-file configs/gdrn/lmo_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo.py \
  --weights pretrained_models/lmo_pbr/model_final_wo_optim.pth \
  --dataset lmo_bop_test \
  --bbox-source gt \
  --device cuda:0 \
  --bop-eval \
  --output-dir output/EXP-20260730-001/full
```

## Acceptance and Decision

- CPU tests, imports, asset checks, and the 16-instance smoke test must pass.
- A full run must account for exactly 1,445 targets.
- Methods must consume the same prediction tensors and GT boxes.
- BOP AR, ADD(-S), rotation/translation error, per-object results, visibility
  bins, failure rate, correspondence count, inlier rate, and runtime are saved.

Decision rules:

- `PASS`: BOP AR and ADD(-S) each improve by at least 0.5 percentage points,
  with non-negative ADD(-S) recall on at least 6/8 objects.
- `AXIS_PASS`: rotation or translation error improves by at least 10%, the
  other worsens by no more than 5%, and BOP AR drops by no more than 0.2
  percentage points.
- `PARTIAL`: overall metrics are neutral but BOP AR improves by at least one
  point for visibility below 0.5.
- `FAIL`: no stable effect, or an apparent gain depends on dropped/failed
  instances.

Stage 1 stops after recording the decision.
