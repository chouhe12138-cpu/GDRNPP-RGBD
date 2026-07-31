# EXP-20260731-006 — Quality/Coverage Residual Attention

Status: `IMPLEMENTED — LOCAL/FORWARD VALIDATION PENDING`

The experiment implements one identity-initialized, lightweight,
region-balanced quality/coverage mechanism feeding the official direct
Patch-PnP `R,t` head.

Frozen protocol:

```text
research/stages/STAGE_03C1_QUALITY_COVERAGE_ATTENTION.md
```

The official components are frozen in C1.  Therefore the primary comparison
is the official checkpoint A versus C1.  Patch-PnP-only B becomes mandatory
only if C2 unfreezes Patch-PnP.

No pose-accuracy result exists yet.  Unit/preflight success is only
implementation evidence.
