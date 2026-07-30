# Upstream GDRNPP Snapshot

## Source

- Archive: `/mnt/d/6D姿态估计/gdrnpp_bop2022-main.zip`
- Archive size: `4,325,134` bytes
- SHA-256: `9cfc699f3be4b4dce2b79c2ac1df7950c026c642913ddc3a4c045831edfdfaa0`
- Archive timestamp: `2026-07-07 14:06:52 +08:00`
- Migration date: `2026-07-30`
- Zip integrity: PASS
- Zip entries: `1,656`
- Extracted entries below the top-level directory: `1,655`

This project was imported from a GitHub source archive, not cloned with the
upstream Git history. The first local commit is therefore an immutable snapshot
boundary rather than an upstream commit reference.

## Inventory Check

The archive contains the expected top-level entries:

```text
LICENSE
README.md
configs/
core/
det/
docs/
lib/
ref/
requirements/
scripts/
tools/
```

The core GDRNPP model entry
`core/gdrn_modeling/models/GDRN_double_mask.py` is present.

## Known Incomplete External Component

`bop_renderer` is preserved exactly as supplied by the archive, but it is a
broken symbolic link:

```text
bop_renderer -> /data/lxy/bop_renderer
```

That path belongs to the upstream author's machine. Stage 0 does not replace,
download, or compile it. Resolving `bop_renderer` is a Stage 1 environment task
and must be recorded with its own source/version.

## Provenance Rule

The original snapshot commit must not be rewritten. All research code,
configuration, documentation, and compatibility fixes are added in later
commits so that the upstream boundary stays auditable.

