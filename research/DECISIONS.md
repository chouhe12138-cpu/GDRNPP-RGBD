# Research Decision Log

This file records decisions that change the experiment protocol. Each future
change must include the date, evidence, affected comparisons, and whether old
results remain comparable.

## 2026-07-30 — Project and publication direction

- Graduation and a submission-ready English paper take priority over edge
  deployment.
- The new project is independent of RDPN6D and the historical RRF directory.
- The old projects are read-only references; their methods do not automatically
  constrain this project.
- The publication style is an applied JCR Q2/Q3-oriented paper with a clean
  mechanism, controlled experiments, and efficiency reporting.

## 2026-07-30 — Task and baseline

- Use the known-CAD, seen-object RGB-D route.
- Use GDRNPP/ConvNeXt-Base as the formal parent baseline.
- Use LM-O as the main occlusion benchmark and YCB-Video as the transfer
  benchmark.
- Keep GT-bbox ablations separate from detector-bbox system results.
- Keep ADD(-S), BOP AR, and their component metrics separate.

## 2026-07-30 — Method boundary

- Test Camera XYZ plus validity as the initial depth representation.
- Compare RGB-only, late, sparse mid/late, and full-stage one-way residual
  topologies under a controlled protocol.
- Preserve the original geometry head, losses, and Patch-PnP in version 1.
- Allow at most one fallback to output-coordinate residual correction.
- Do not add attention, reliability gating, normals, extra loss terms, or
  deployment-specific compression unless a later decision explicitly replaces
  this boundary.

## 2026-07-30 — Execution policy

- Work stage by stage; no automatic progression.
- First establish local positive evidence, then use the L40 server.
- Use Gitee for source and compact experiment metadata only.
- Do not commit datasets, model weights, caches, secrets, or full logs.
- UR3e + D455 is optional after the public-dataset manuscript is complete.

## 2026-07-31 — Evidence-first pose-aggregation diagnostic

- The Camera-XYZ RGB-D proposal is deferred rather than deleted.
- Stage 1 now uses the official RGB-only GDRNPP/ConvNeXt-Base checkpoint on
  LM-O to test whether Patch-PnP underuses dense geometric predictions.
- GT boxes isolate pose aggregation from detection error.
- Six pre-registered solvers share one network forward; no method is tuned on
  LM-O test results.
- Stage 1 contains no training and stops immediately after a
  PASS/AXIS_PASS/PARTIAL/FAIL conclusion.
- The clean `GDRNPP-RGBD` repository is the sole writable experiment project.
  The historical RRF project supplies read-only assets through symbolic links.
- Symbolic links are preferred over copies.  Unique source assets are not
  moved because doing so would break historical reproducibility.

## 2026-07-31 — Stage 1 result

- Stage 1 completed on all 1,445 LM-O BOP19 targets and received a strict
  `FAIL` decision.
- RANSAC-EPnP improved BOP AR by 0.573 points but was non-negative on only 5/8
  objects, below the frozen 6/8 stability gate.
- The pre-registered mask × region-confidence filter performed worse than
  unfiltered RANSAC and is rejected as a reliability proxy.
- No architecture, training, detector-bbox, or RGB-D experiment is authorized
  by this result.
- Any continuation requires a new Stage 2 decision; results from this stage
  remain valid as a diagnostic and negative ablation.

## 2026-07-31 — Stage 2 causal-oracle authorization

- Stage 2 is authorized only as a frozen, no-training causal diagnostic.
- It independently tests GDRNPP visible support, XYZ, reliability, double-mask
  usage, aggregation complementarity, and rotation/translation coupling.
- LM-O test depth, masks, and poses may construct diagnostic oracles, but no
  oracle result may be presented as deployable RGB performance.
- The solver remains fixed at RANSAC-EPnP with a 3 px threshold, 100
  iterations, seed `20260730`, and a fixed top-50% reliability intervention.
- The stage stops after its pre-registered PASS/FAIL conclusion.  Training or
  architecture changes require a separate Stage 3 authorization.

## 2026-07-31 — Stage 2 result

- Stage 2 completed all 1,445 LM-O targets and all 12 official BOP19
  evaluations; Stage 1 Patch-PnP and RANSAC baselines reproduced exactly.
- Predicted XYZ geometry is the only factor that passes the frozen primary
  gate.  GT XYZ closes 97.49% of the complete ADD(-S) oracle gap, improves BOP
  AR by 30.858 points and ADD(-S) by 45.744 points, and is non-negative on 8/8
  objects.
- GT visible support, predicted full-mask support, and the current reliability
  score do not provide a stable causal improvement.
- Selecting the 50% lowest true-XYZ-error correspondences also hurts pose,
  rejecting the simple claim that better scalar pixel reliability alone is
  sufficient.  Coverage and correspondence conditioning remain relevant.
- Axis-wise Patch-PnP/RANSAC oracle selection gives a strong secondary gain
  (+3.020 BOP AR, +8.028 ADD(-S), 8/8 objects) but closes only 17.11% of the
  full oracle gap and fails the frozen 30% primary-factor threshold.
- Stage 2 is closed.  No training or network change is authorized until a new
  plan defines a held-out PBR validation protocol.

## 2026-07-31 — Stage 3A PBR validation infrastructure

- The LM-PBR archive is complete at 50 scenes / 50,000 images, but only scenes
  00–14 are currently extracted.
- Reserve scenes 12–14 as a scene-disjoint 3,000-image validation set.  Future
  training uses scenes 00–11 and 15–49 only.
- Track a deterministic 1,500-image diagnostic subset; use 300 images locally
  for the current calibration to limit random reads on the Windows-mounted
  drive.
- The official checkpoint may have trained on validation scenes.  Its current
  PBR result is calibration only and cannot support model selection.
- Do not extract the remaining 35,000 images, generate `xyz_crop`, retrain, or
  modify the network in Stage 3A.

## 2026-07-31 — Stage 3A result

- The 300-image calibration processed 2,087 valid instances and completed all
  10 frozen pose methods.
- The formal calibration result is `CALIBRATION_MISMATCH`; thresholds are not
  adjusted post hoc.
- GT XYZ is still the largest factor (+19.454 ADD(-S), 8/8 objects), but misses
  the frozen +20-point threshold by 0.546 points.
- GT mask is neutral (+0.192 points), matching the Stage 2 conclusion that
  mask support is not primary.
- True-error Top-50% improves PBR by +3.833 points but hurt LM-O by -1.799
  points.  Reliability-only selection is not cross-domain stable.
- PBR has much lower XYZ error and higher mask IoU than LM-O.  Boundary error
  is about 2.15 times interior error, and lowest-error selection reduces point
  set spatial diversity.
- The next defensible hypothesis is precision/coverage-aware correspondence
  aggregation, not a scalar reliability head alone.
- Stage 3A closes without extracting data, retraining, or changing the network.

## 2026-07-31 — Stage 3B pose-head utilization authorization

- The deployable pose remains the official direct Patch-PnP `R,t`; RANSAC is
  retained only as a diagnostic reference.
- Stage 2 proved that XYZ is causal for RANSAC but did not test whether the
  frozen official Patch-PnP can use improved XYZ.
- Stage 3B therefore interpolates predicted XYZ toward GT XYZ at five fixed
  levels while holding mask, region, 2D coordinates, support, bbox, and all
  weights fixed.
- GT depth, masks, poses, and interpolated coordinates remain oracle-only.
- The stage contains no training and stops after its pre-registered
  `PATCH_PNP_USES_IMPROVED_XYZ`, `PATCH_PNP_UNDERUTILIZATION`, or
  `MIXED_OR_INCONCLUSIVE` decision.

## 2026-07-31 — Stage 3B result

- Stage 3B completed 1,445 LM-O targets and 10 official BOP19 evaluations.
- RANSAC converts progressive XYZ correction into a monotonic +45.536-point
  ADD(-S) gain and +30.122-point BOP AR gain on 8/8 objects.
- Frozen Patch-PnP converts the same intervention into -0.692 ADD(-S) points
  and -0.697 BOP AR points and is non-negative on only 5/8 objects.
- Decision: `PATCH_PNP_UNDERUTILIZATION`.
- RANSAC is rejected as the final deployment route because this experiment
  uses it only to establish information availability.
- The next proposed intervention is direct region-balanced quality/coverage
  attention. Training is necessary because official weights define the
  diagnosed frozen mapping and contain no learned parameters for that new
  mechanism.

## 2026-07-31 — Stage 3C-0 Patch-PnP adaptation control (superseded protocol)

- Do not add the proposed quality/coverage mechanism immediately. First test
  the simpler hypothesis that the existing official Patch-PnP architecture
  has sufficient capacity after task-specific adaptation.
- Initialize from the official checkpoint, freeze the ConvNeXt backbone and
  complete geometry head, and update only the original direct `pnp_net`.
- Use PBR scenes `00–11, 15–49` for training and reserve scenes `12–14` for
  PnP-stage validation. Because the official checkpoint may have seen all PBR
  scenes, this is not a fully held-out geometry evaluation.
- Freeze seed `20260731`, five epochs, effective batch 48, Ranger `8e-5`,
  weight decay `0.01`, and FP32 for the formal L40 control.
- The RTX 4060 local pilot passed pipeline and parameter-isolation gates:
  2,048/2,048 micro-batches completed, all losses were finite, strict reload
  passed, 17 Patch-PnP tensors changed, and zero frozen tensors changed.
- Local loss is not a model-selection metric. The formal architecture decision
  remains pending PBR-stage pose/utilization validation.
- Only `ARCHITECTURE_CHANGE_JUSTIFIED` authorizes implementing the single
  lightweight quality/coverage aggregation variant. Thresholds are frozen in
  `research/stages/STAGE_03C0_PNP_ADAPTATION_CONTROL.md`.

## 2026-07-31 — Final Stage 3C training protocol

- The user selected all 50 LM-PBR scenes for training and LM-O evaluation
  after every five completed epochs; no PBR validation split is used for this
  experiment.
- C1 loads the official checkpoint and freezes backbone, geometry head, and
  Patch-PnP. Only one identity-initialized quality/coverage residual-attention
  module is trained, so an additionally fine-tuned original head is not needed
  for the primary A-versus-C1 comparison.
- The new module reweights the existing 64 region-attention maps without
  changing XYZ values, Patch-PnP input dimensions, or the direct `R,t` output.
- If C1 later requires unfreezing Patch-PnP, the 40-epoch PnP-only B control
  becomes mandatory and is compared with same-budget C2.
- Formal training uses the L40. RTX 4060 is limited to a one-epoch,
  8,192-instance architecture gate.
- Save at epochs 5, 10, ..., 40 and retain at most three complete weights:
  the best BOP-AR checkpoint and the latest two. ADD(-S)@0.1d breaks BOP-AR
  ties within 0.001.
