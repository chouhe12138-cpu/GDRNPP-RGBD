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

