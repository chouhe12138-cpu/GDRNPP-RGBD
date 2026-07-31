# Stage 3C-1 — Identity-Initialized Quality/Coverage Attention

Status: `IMPLEMENTED — LOCAL C1 PILOT PENDING`

## Problem and hypothesis

Stage 3B showed that corrected XYZ correspondences are highly pose-useful to a
geometric solver but are underused by the official direct Patch-PnP head.
Pointwise reliability alone also failed to transfer from synthetic PBR to
real LM-O occlusion.  The intervention therefore jointly represents:

- local correspondence quality; and
- global coverage across the 64 predicted object regions.

The module preserves GDRNPP's direct `R,t` output and does not deploy RANSAC.

## Architecture

The official coordinate values and Patch-PnP input dimensions are unchanged.
The new module only reweights the 64 region-attention maps already consumed by
ConvPnP:

```text
region_new = region * (1 + 0.25 * (quality_delta + coverage_delta))
```

- `quality_delta` is a spatial scalar inferred from XYZ/2D coordinates,
  visible-mask probability, and region probabilities.
- `coverage_delta` is a 64-value residual inferred from each region's global
  occupancy and visible occupancy.
- both deltas pass through `tanh`;
- the last layer of both branches is initialized to exactly zero.

At initialization `region_new == region` bit-for-bit, so loading the official
checkpoint does not change its behavior before training.  The module adds
only a small convolutional/MLP branch and leaves the backbone, geometry head,
and direct pose head intact.

## C1 formal protocol

```text
initialization:       official LM-O class-aware checkpoint
trainable parameters: quality_coverage_net only
frozen parameters:    backbone + geometry head + Patch-PnP
training data:        all 50 LM-PBR scenes
test data:            LM-O BOP19
bounding boxes:       GT
pose output:          direct Patch-PnP R,t
epochs:               40
batch:                48
optimizer:            Ranger, 8e-4, weight decay 0.01
evaluation:           after epochs 5,10,...,40
checkpoint retention: best BOP AR one + latest two
seed:                 20260731
```

BOP AR is the primary checkpoint metric.  When values differ by at most
0.001, ADD(-S) at 0.1 diameter is the secondary metric.

## Hardware gate

Each architectural revision first runs the one-epoch, 8,192-instance local
config on the RTX 4060 using batch 4 and 12-step accumulation.  Formal
40-epoch training runs only on the Docker-managed L40.

## Conditional control

If C1 has finite/convergent training loss but does not improve LM-O, Stage C2
may unfreeze Patch-PnP.  Only then run:

- B: Patch-PnP fine-tuning without the new module;
- C2: the same Patch-PnP fine-tuning with the new module.

Both use all 50 PBR scenes, the same 40-epoch budget, and Patch-PnP learning
rate `8e-5`.  C2 uses `8e-4` for the new module.

## Commands

```bash
conda activate pytorch22

# Official epoch-0 GT-box reference
research/quality_coverage/run_baseline.sh

# RTX 4060 one-epoch architecture gate
research/quality_coverage/run_local.sh

# Docker/L40 formal C1
research/quality_coverage/run_l40.sh

# Figures after evaluation
python -m research.quality_coverage.plot_curves \
  output/EXP-20260731-006/quality_coverage_full \
  --baseline-output output/EXP-20260731-006/official_gt
```
