# GDRNPP Pose-Aggregation Research Plan

Current handoff and exact experiment state: `research/HANDOFF.md`.

## Objective

Prioritize a low-risk, evidence-first graduation project.  The active question
is whether GDRNPP's dense XYZ, visible-mask, and region predictions contain
pose-useful information that the learned Patch-PnP head does not fully
aggregate under occlusion.

The earlier Camera-XYZ RGB-D residual-fusion proposal remains an archived
candidate.  It is not the active method and will only be reconsidered after the
pose-aggregation diagnostic has produced a PASS/PARTIAL/FAIL result.

## Frozen Stage 1 Protocol

- Parent model: official GDRNPP with ConvNeXt-Base.
- Dataset: LM-O BOP19 targets, known CAD models and seen objects.
- Input for the diagnostic: RGB only.
- Bounding boxes: GT boxes only, to isolate pose aggregation.
- Checkpoint: official LM-O class-aware model.
- No training, detector experiment, depth input, new backbone, or learned head.
- One network forward supplies all six pose-solving variants.
- Solver parameters are pre-registered in
  `research/stages/STAGE_01_POSE_AGGREGATION_DIAGNOSTIC.md`.
- Stage 1 ends after the diagnostic conclusion.  It does not authorize a
  follow-up architecture.

## Stage Gates

| Stage | Purpose | Status / exit condition |
|---|---|---|
| 0 | Import source and freeze provenance | PASS |
| 1 | Reuse the local environment, reproduce the RGB baseline, and diagnose pose aggregation | COMPLETE — FAIL with an unstable positive RANSAC signal |
| 2 | Causally separate GDRNPP mask, XYZ, reliability, and aggregation bottlenecks | COMPLETE — PASS, XYZ geometry primary |
| 3A | Build a scene-disjoint PBR validation protocol and run leakage-marked calibration | COMPLETE — CALIBRATION_MISMATCH |
| 3B | Test whether frozen Patch-PnP converts controlled XYZ improvements into pose gains | COMPLETE — PATCH_PNP_UNDERUTILIZATION |
| 3C-0 | Patch-PnP-only same-budget control, required only if C2 unfreezes Patch-PnP | LOCAL PILOT PASS — CONDITIONAL |
| 3C-1 | Freeze all official parameters and train one identity-initialized quality/coverage module | FORMAL COMPLETE — C1_SCREEN_FAIL |
| 3C-2 | Jointly adapt Patch-PnP and the new module only if C1 cannot adapt | TRIGGERED — RUNTIME GATE PENDING |

Only the active stage may be executed.  A completed Stage 1 does not authorize
Stage 2.

## Asset and Reproducibility Policy

- The repository is `/home/wsluser/GDRNPP-RGBD`.
- Each formal experiment uses one pre-registered seed only.  Additional
  random-seed repetitions are not part of the research protocol; compute is
  reserved for problem-driven controls, key ablations, and cross-dataset
  validation.
- Datasets, weights, the BOP renderer, and compatible native extensions are
  reused through ignored symbolic links.
- Old RDPN6D and RRF projects are read-only references.
- Dense prediction tensors are shared in memory and are not dumped to disk.
- Git tracks source, compact metrics, protocol, decisions, and conclusions;
  datasets, weights, caches, full logs, and machine-local links remain ignored.
- RDPN6D observations may motivate an internal hypothesis, but every public
  GDRNPP claim must be supported by a GDRNPP experiment.

## Current Status

```text
Stage 0: PASS
Stage 1: COMPLETE — FAIL
Stage 2: COMPLETE — PASS (XYZ GEOMETRY)
Stage 3A: COMPLETE — CALIBRATION_MISMATCH
Stage 3B: COMPLETE — PATCH_PNP_UNDERUTILIZATION
Stage 3C-0: LOCAL PILOT PASS — CONDITIONAL B CONTROL
Stage 3C-1: FORMAL COMPLETE — C1_SCREEN_FAIL
Stage 3C-2: TRIGGERED — RUNTIME GATE PENDING
```

## 2026-07-31 — Stage 3B result

- The full frozen-head diagnostic completed all 1,445 LM-O targets and all 10
  BOP19 evaluations; both alpha-zero ADD(-S) baselines reproduced exactly.
- As XYZ is corrected from alpha 0 to 1, diagnostic RANSAC improves
  monotonically from 53.841% to 99.377% ADD(-S), gains 30.122 BOP AR points,
  and is non-negative on 8/8 objects.
- Official direct Patch-PnP instead changes from 50.242% to 49.550% ADD(-S),
  loses 0.697 BOP AR points, is non-negative on only 5/8 objects, and is not
  monotonic.
- The frozen decision is `PATCH_PNP_UNDERUTILIZATION`. RANSAC remains a
  diagnostic reference, not the deployment path.
- The next method hypothesis is one lightweight, region-balanced
  quality/coverage attention feeding the direct Patch-PnP `R,t` head.
- Stage 3C-0 was initially specified as a pre-architecture control. The final
  C1 protocol freezes Patch-PnP, so its formal run is deferred unless C2 later
  unfreezes Patch-PnP.

## 2026-07-31 — Stage 3C-0 local pilot

- All 50 PBR scenes and VOC backgrounds passed deep integrity checks.
- The completed local pilot used its frozen 8,192-instance subset. The final
  formal protocol instead trains on all 50 scenes and evaluates LM-O.
- The RTX 4060 pilot used 8,192 balanced instances, batch 4 with 12-step
  accumulation, and completed 2,048 micro-batches / 171 optimizer updates.
- Total-loss first/last-quarter median decreased 5.41%; all losses remained
  finite. This is feasibility evidence, not pose-accuracy evidence.
- Strict checkpoint reload passed. All 17 changed/trainable tensors were
  `pnp_net` tensors and zero frozen tensors changed.
- The earlier five-epoch/47-scene gate was superseded by the final protocol:
  C1 starts from the official checkpoint, freezes every official component,
  and trains only the identity-initialized residual attention on all 50 PBR
  scenes.
- RTX 4060 runs only the one-epoch architecture gate. The L40 runs the formal
  40-epoch experiment with LM-O evaluation after every five completed epochs.
- The PnP-only formal control is required only if C2 later unfreezes PnP.
