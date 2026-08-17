# GDRNPP Pose-Aggregation Research Plan

Current verified experiment state: `research/STATUS_CN.md`.

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
| 3C-0 | Patch-PnP-only same-budget control, required once C2 unfreezes Patch-PnP | EXP005 FORMAL COMPLETE |
| 3C-1 | Freeze all official parameters and train one identity-initialized quality/coverage module | FORMAL COMPLETE — C1_SCREEN_FAIL |
| 3C-2 | Jointly adapt Patch-PnP and the new module only if C1 cannot adapt | FORMAL COMPLETE — C2_SCREEN_FAIL |
| 4 | Test whether Region-conditioned low-order 2D–3D joint moments improve direct pose-head geometry consumption | EXP009 FIXED EPOCH 40 COMPLETE — CPM_SCREEN_FAIL |
| 4C | Test whether EXP009 is optimization-limited by its fine-tuning-scale learning rate | EXP010 AUTHORIZED; ACCESS PASS; NO FORMAL RUN RECORDED |

Historical stage completion does not by itself authorize a new architecture.
Each experiment still requires its own frozen protocol and managed run
identity.

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
Stage 3C-0: EXP005/B FORMAL COMPLETE
Stage 3C-1: FORMAL COMPLETE — C1_SCREEN_FAIL
Stage 3C-2: FORMAL COMPLETE — C2_SCREEN_FAIL
Stage 4: EXP009/CPM FIXED EPOCH 40 COMPLETE — CPM_SCREEN_FAIL
Stage 4C: EXP010/CPM-LR CONTROL AUTHORIZED — ACCESS PASS ONLY
```

EXP010 is a matched optimization control, not a second pose-head proposal. It
inherits EXP009 and changes only the experiment identity/output path and Ranger
learning rate (`8e-5` to `8e-4`). It starts fresh from the official checkpoint
and is authorized after EXP005 fixed Epoch 40 completion. EXP009 fixed Epoch 40
is now complete; the matched learning-rate comparison waits for EXP010 fixed
Epoch 40 and same-protocol diagnostics. Its result can test whether EXP009 was
optimization-limited; it cannot by itself validate or reject low-order joint
moments or 2D–3D correspondence in general.

## 2026-08-18 — EXP012 correspondence-preserving pose head

- EXP007、EXP009 与 EXP011 支持下一步优先检验 correspondence 的局部建模与
  渐进聚合，而不是继续在输入端增加固定低阶压缩；这仍是研究动机，不是方法结论。
- EXP012 保留 dense metric XYZ↔absolute ROI2D 配对，在全局压缩前使用轻量局部
  residual blocks，并以 fine/mid summary 和 high-level 4×4 grid 形成 pose 表示。
- Pred Region 只作为零启动辅助残差，不参与分组；rot6d、translation、loss、
  frozen geometry 与 official 初始化保持不变。
- 当前只完成本地工程 preflight；固定 Epoch 40 的 BOP/ADD(-S)/per-object 结果
  产生前，不对结构有效性下结论。
- 用户于 2026-08-18 授权 EXP012 在服务器 access/create 与 CUDA gate 通过后直接
  formal，不执行 smoke/audit48；这是运行路径授权，不是服务器运行完成证据。

## 2026-08-11 — Stage 3C closure and Stage 4 execution

- C2 completed 40 epochs. Its best BOP AR was Epoch 35 `0.6935201845`; fixed
  Epoch 40 was `0.6930057670`. Both missed the pre-registered `+0.50 pp` BOP
  gate. Historical ADD(-S) macro/per-object evidence was not produced and
  remains missing, so the frozen conclusion is `C2_SCREEN_FAIL`.
- Mandatory matched control EXP005/B and EXP009/CPM passed managed gate,
  smoke, and audit under source commit
  `652d7fd9d38f8ea5cea0c5a98cc9477b66623180`, stable environment image
  `sha256:f3055cb660032bbb4c1b7cfd9b1840a6c98359d0562a3a4f0601f7238f7291ee`,
  and seed `42`.
- EXP005 formal later completed as `RUN-20260811-063606-formal-s42-a01`.
  EXP009 formal was `RUN-20260811-063626-formal-s42-a01`. After earlier
  interrupted resume attempts, an external fixed Epoch 40 checkpoint and its
  BOP evaluation were recovered and verified: BOP AR `0.5983921569`, ADD(-S)
  target-micro `0.3806228374`, macro-object `0.3768665461` derived from the
  logged per-object recalls, and `2/8` nonnegative objects. All three frozen
  gates fail, so the result is `CPM_SCREEN_FAIL`. EXP011 later completed the
  fixed-E40 mechanism diagnostic with decision `MISMATCH_IMPORTANT`.
- CPM tests only whether Region-conditioned low-order joint moments are
  sufficient for the proposed mechanism. A negative CPM result would not show
  that 2D–3D correspondence itself is unimportant.

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
