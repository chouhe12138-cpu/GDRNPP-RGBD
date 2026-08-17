# Patch-PnP information-flow diagnostic

This package implements frozen, aggregate-only Patch-PnP diagnostics. It does
not train, change model parameters, or replace preregistered training
conclusions.

## Scope

- Re-analyze the existing complete 1,445-target Stage 3B response curves.
- Validate on 8-target smoke and 80-target audit sets before a complete
  1,445-target run.
- Observe the Patch-PnP input, three convolution stages, flattened feature,
  two fully connected activations, raw rotation/translation heads, and pose.
- Apply fourteen fixed-support interventions, including axis corrections,
  spatial permutations, boundary/interior controls, and high-error/random
  matched controls.
- Persist aggregate statistics only. Full intermediate activations and
  instance-level layer-response tables are never written.

The experiment record and current execution gate are in
`research/POSE_HEAD_DIAGNOSTIC_HANDOFF_CN.md`.

## Commands

```bash
conda activate pytorch22

python -m research.pose_head_diagnostic.analyze_existing

python -m research.pose_head_diagnostic.run_statistical_diagnostic \
  --mode smoke --model-role official \
  --config-file configs/gdrn/lmo_pbr/convnext_stage3c1_official_gt_lmo.py \
  --weights pretrained_models/lmo_pbr/model_final_wo_optim.pth \
  --device cpu \
  --output-dir output/EXP-20260804-007-pose-head-information-flow/official/smoke
```

The official smoke, CUDA audit80, and complete 1,445-target full run have
passed. The formal protocol uses deterministic FP32 on CPU and CUDA; AMP is
rejected because repeated Patch-PnP calls differ at FP16 quantization scale.
C1, B, and C2 diagnostics still require their matching fixed Epoch-40
checkpoints and separate output directories.

## EXP011 CPM XYZ/Region consistency diagnostic

The EXP011 route is a frozen-checkpoint diagnostic for the verified EXP009
Epoch-40 CPM head.  It reuses the training-time `xyz_to_region` definition and
the LM-O `fps64_and_center[:-1]` points.  GT Region is converted to a 64-channel
foreground one-hot posterior on the same frozen support used by the existing
GT-XYZ intervention; predicted Region is retained outside that support.

Use the four endpoint conditions for smoke and audit, then the two Region
sources by five XYZ-alpha values for the complete run:

```bash
conda activate pytorch22

python -m research.pose_head_diagnostic.run_statistical_diagnostic \
  --mode smoke --model-role cpm \
  --condition-set cpm_xyz_region_2x2 \
  --config-file configs/gdrn/lmo_pbr/research/exp009_cpm_head/eval.py \
  --weights /mnt/e/6D姿态估计/EXP-009/model_epoch_040.pth \
  --device cpu --seed 20260817 \
  --output-dir output/EXP-20260817-011-cpm-xyz-region-consistency-diagnostic/smoke-2x2

python -m research.pose_head_diagnostic.run_statistical_diagnostic \
  --mode audit80 --model-role cpm \
  --condition-set cpm_xyz_region_2x2 \
  --config-file configs/gdrn/lmo_pbr/research/exp009_cpm_head/eval.py \
  --weights /mnt/e/6D姿态估计/EXP-009/model_epoch_040.pth \
  --device cuda --seed 20260817 \
  --output-dir output/EXP-20260817-011-cpm-xyz-region-consistency-diagnostic/audit80-2x2

python -m research.pose_head_diagnostic.run_statistical_diagnostic \
  --mode full --model-role cpm \
  --condition-set cpm_xyz_region_alpha_sweep --bop-eval \
  --config-file configs/gdrn/lmo_pbr/research/exp009_cpm_head/eval.py \
  --weights /mnt/e/6D姿态估计/EXP-009/model_epoch_040.pth \
  --device cuda --seed 20260817 \
  --output-dir output/EXP-20260817-011-cpm-xyz-region-consistency-diagnostic/full-2x5
```

The new route rejects any checkpoint whose SHA-256 differs from the verified
EXP009 Epoch-40 hash.  It creates no optimizer, runs under `eval()` and
`no_grad()`, verifies the model state before/after, and refuses non-empty output
directories.  It writes aggregate ADD(-S) micro/macro/per-object summaries,
BOP19 scores and components, Region agreement QC, and the endpoint factorial
interaction/decision.  It does not modify the checkpoint or persist
instance-level activations.

FP32 baseline re-entry uses the calibrated cross-device absolute tolerances
`5e-5` for raw head outputs, `3e-4` for rotation matrices, and `5e-5 m` for
translation.  Re-entry is an advisory numerical diagnostic and does not block
a run; exact immutability is enforced separately through model-state and
checkpoint-file hashes.

## Aggregate outputs

The statistical runner writes only overall, layer, pose-component, object
class, visibility, symmetry, and support-quartile summaries. Per-target BOP
pose CSVs are retained only under ignored `output/` because the official BOP
evaluator requires them.

The older `run_information_flow` entry remains a preliminary compatibility
smoke tool. It no longer writes its former instance-level layer-response CSV
and must not be used for the formal full diagnostic.
