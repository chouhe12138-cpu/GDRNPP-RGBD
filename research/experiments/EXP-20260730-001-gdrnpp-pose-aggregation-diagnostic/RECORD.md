# EXP-20260730-001 — GDRNPP Pose-Aggregation Diagnostic

## Status

Complete.  Formal decision: `FAIL` under the pre-registered gate, with a
positive but object-dependent signal from unfiltered RANSAC-EPnP.

## Frozen Protocol

- Official GDRNPP ConvNeXt-Base LM-O checkpoint
- LM-O BOP19 test targets
- GT bounding boxes
- Seed `20260730`
- Six methods defined in `research/stages/STAGE_01_POSE_AGGREGATION_DIAGNOSTIC.md`
- No training and no depth input

## Asset Provenance

- Dataset source: ignored link `datasets/BOP_DATASETS`
- Checkpoint source: ignored link
  `pretrained_models/lmo_pbr/model_final_wo_optim.pth`
- Checkpoint SHA-256:
  `bafa869d4e6c00410517ecb1add59f234ed1642e47fabcf3aa6e0e8a1b498a8c`
- Renderer and native extensions: compatible ignored links from the historical
  local GDRNPP environment; imports are tested before inference.

## Evidence

### Validation

- Checkpoint hash: PASS.
- Reused native extensions and BOP renderer imports: PASS.
- CPU solver/metric tests: 9/9 PASS.
- CPU one-target end-to-end inference: PASS.
- GPU smoke: 16 targets, two per object, all six methods produced poses.
- Full GPU run: exactly 1,445/1,445 BOP19 targets.
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB.
- Shared network-forward time over the full run: 14.690 s.
- BOP toolkit: C++ renderer, VSD/MSSD/MSPD, one worker.

The upstream loader exposes 1,517 GT instances from the 200 target images.
Formal inference filters these with `test_targets_bop19.json` and its
`inst_count` values to the required 1,445 targets.

### Overall Results

Percentages are absolute percentages.  Rotation is symmetry-aware.  Failed
estimates count as zero in ADD(-S) recall.

| Method | BOP AR | VSD AR | MSSD AR | MSPD AR | ADD(-S)-0.1d | Mean R (deg) | Median R (deg) | Median t (mm) | Failure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Patch-PnP | 69.021 | 51.671 | 66.713 | 88.678 | 50.242 | 7.76 | 4.64 | 17.06 | 0.000% |
| EPnP all | 68.671 | 52.399 | 67.052 | 86.561 | 52.872 | 9.54 | 4.19 | 15.34 | 0.069% |
| RANSAC-EPnP | **69.594** | **53.080** | **67.799** | 87.903 | **53.080** | 8.86 | **4.21** | **15.34** | 0.069% |
| Reliable RANSAC | 68.719 | 52.095 | 66.713 | 87.349 | 51.696 | 8.99 | 4.27 | 16.16 | 0.069% |
| Geometric R + network t | 68.447 | 51.340 | 66.055 | 87.945 | 49.827 | 8.99 | 4.27 | 17.05 | 0.069% |
| Network R + geometric t | 68.955 | 52.338 | 66.900 | 87.626 | 51.834 | **7.51** | 4.63 | 16.06 | 0.346% |

RANSAC-EPnP improves BOP AR by 0.573 points and ADD(-S) recall by
2.838 points over Patch-PnP.  It nevertheless improves or ties ADD(-S) on only
5/8 objects, below the frozen 6/8 PASS requirement.  Its BOP gain comes from
VSD and MSSD while MSPD decreases by 0.775 points.

The fixed reliability proxy is not validated: top-50% filtering reduces BOP AR
from 69.594 to 68.719.  Mask probability multiplied by maximum foreground
region probability therefore cannot be treated as a calibrated correspondence
reliability signal.

Four unfiltered and five filtered RANSAC estimates have extreme translation
errors above 10 m.  They are concentrated in eggbox/glue instances with
visibility between 0.003 and 0.071.  They remain in the raw record and explain
the non-robust translation means; medians are reported separately.

### Artifacts

Ignored runtime artifacts:

```text
output/EXP-20260730-001/full/per_instance.csv
output/EXP-20260730-001/full/per_object.csv
output/EXP-20260730-001/full/visibility_bins.csv
output/EXP-20260730-001/full/protocol.json
output/EXP-20260730-001/full/summary.json
output/EXP-20260730-001/full/bop_results/
output/EXP-20260730-001/full/bop_eval/
```

The per-instance table contains 8,670 rows: 1,445 targets × 6 methods.
