# GDRNPP-RGBD Research Plan

## Objective

Develop a graduation-priority RGB-D 6D object pose estimation method for known
CAD models and seen objects. The main question is whether Camera XYZ can
provide sparse, one-way geometric residuals to GDRNPP under occlusion without
the cost and redundancy of full-flow bidirectional fusion.

The target is an English manuscript ready for submission within six months.
The Chinese degree thesis will reuse the method and experiment evidence while
including the earlier RDPN6D/D1/D2 work as a separate preliminary chapter.

## Frozen Research Boundaries

- Parent method: GDRNPP with ConvNeXt-Base.
- Main dataset: LM-O.
- Transfer dataset: YCB-Video.
- Modality: RGB-D, represented initially as Camera XYZ plus a validity mask.
- Main mechanism family: sparse, one-way geometric residual fusion.
- The GDRNPP double-mask output, dense XYZ/region losses, and Patch-PnP remain
  unchanged in the first method version.
- Official depth refinement is disabled in controlled fusion experiments.
- Attention, normals, handcrafted reliability, pruning, DySample, and new
  training losses are outside the first method version.
- Local experiments must show repeatable positive evidence before L40 runs.
- Only one fallback redesign is allowed: depth-guided residual correction of
  the dense object-coordinate prediction.
- Jetson deployment is out of scope. UR3e + D455 is optional and cannot block
  the public-dataset paper.
- No SOTA or first-of-its-kind claim is allowed without a dedicated literature
  and protocol audit.

## Stage Gates

| Stage | Purpose | Entry condition | Exit condition |
|---|---|---|---|
| 0 | Import source and freeze the research plan | Verified source archive | Clean Git baseline and Stage 0 PASS report |
| 1 | Rebuild and audit the local environment and official baseline | Stage 0 PASS | Imports, native extensions, renderer, and baseline inference PASS |
| 2 | Add the depth data path and Camera XYZ generation | Stage 1 PASS | Alignment, units, validity, finite-value, and missing-depth tests PASS |
| 3 | Run the local controlled fusion pilot | Stage 2 PASS | Three-seed positive gate or one authorized fallback |
| 4 | Run formal LM-O experiments on L40 | Stage 3 PASS | Frozen protocol, three-seed main result, ablations, and efficiency evidence |
| 5 | Validate the frozen method on YCB-Video | Stage 4 PASS | Cross-dataset result and failure-boundary analysis |
| 6 | Complete statistics, English paper, and thesis mapping | Stages 4–5 evidence frozen | Submission-ready manuscript and archived evidence |

Only the active stage may be executed. Finishing one stage does not authorize
the next stage automatically.

## Current Status

```text
Stage 0: PASS
Stage 1–6: NOT AUTHORIZED
```

The implementation details and acceptance checklist for the active stage are in
`research/stages/STAGE_00_REPO_BOOTSTRAP.md`.
