# Quality/Coverage experiment

This directory contains Stage 3C-1 preflight, launchers, tests, and plotting.
The authoritative protocol is
`research/stages/STAGE_03C1_QUALITY_COVERAGE_ATTENTION.md`.

Run source/tests on the workstation before transferring to the server:

```bash
PYTHONPATH="$PWD" conda run -n pytorch22 python -m pytest -q \
  -o cache_dir=/tmp/gdrnpp-pytest-cache \
  research/quality_coverage/tests

PYTHONPATH="$PWD" conda run -n pytorch22 \
  python -m research.quality_coverage.preflight
```

For the offline L40:

1. push tracked source to the private Gitee repository;
2. transfer the already-tested Docker image through the laboratory-approved
   channel rather than building from GitHub on the server;
3. mount datasets, official weights, output, and cache directories into the
   same paths used by the config;
4. run the preflight inside the container;
5. run `research/quality_coverage/run_local.sh` once inside the container as
   the server-environment smoke test;
6. start `research/quality_coverage/run_l40.sh` only after the smoke test.

Datasets, full checkpoints, Docker archives, caches, and logs must not be
committed to Gitee.

After the screening seed finishes:

```bash
python -m research.quality_coverage.summarize \
  output/EXP-20260731-006/quality_coverage_full \
  output/EXP-20260731-006/official_gt
```

Only after `C1_SCREEN_PASS`, run the two confirmation seeds:

```bash
CONFIG_PATH=configs/gdrn/lmo_pbr/convnext_stage3c1_quality_coverage_lmo_seed_20260801.py \
  research/quality_coverage/run_l40.sh
CONFIG_PATH=configs/gdrn/lmo_pbr/convnext_stage3c1_quality_coverage_lmo_seed_20260802.py \
  research/quality_coverage/run_l40.sh
```
