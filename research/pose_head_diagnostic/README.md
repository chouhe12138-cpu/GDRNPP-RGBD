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

## Aggregate outputs

The statistical runner writes only overall, layer, pose-component, object
class, visibility, symmetry, and support-quartile summaries. Per-target BOP
pose CSVs are retained only under ignored `output/` because the official BOP
evaluator requires them.

The older `run_information_flow` entry remains a preliminary compatibility
smoke tool. It no longer writes its former instance-level layer-response CSV
and must not be used for the formal full diagnostic.
