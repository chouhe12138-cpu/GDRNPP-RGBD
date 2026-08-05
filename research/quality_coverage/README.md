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

The host-side controller keeps routine Docker commands short and refuses to
overwrite an existing smoke output:

```bash
docker/l40/stage3c1.sh smoke
docker/l40/stage3c1.sh status
docker/l40/stage3c1.sh watch
docker/l40/stage3c1.sh validate
```

`validate` checks completion, finite losses, strict checkpoint loading,
bit-identical frozen official tensors, and updates isolated to the nine new
quality/coverage tensors.

For a Python interface similar to `model.train(...)`, open
`train_stage3c1.py`, review the centralized settings, and change its
`launch` value only when the run should actually start. The API automatically:

- reruns the smoke checkpoint gate;
- creates a non-overwriting output directory and run manifest;
- computes the official GT-box baseline when it is missing;
- trains and evaluates LM-O at the configured epoch interval;
- retains the configured best/recent checkpoints through the existing engine;
- draws loss/LM-O/per-object figures and writes the screening summary.

Formal mode locks the frozen paper protocol. Parameter changes require
`protocol="exploratory"` and a different run name, and the manifest labels
that run as unsuitable for replacing the formal paper comparison.

Datasets, full checkpoints, Docker archives, caches, and logs must not be
committed to Gitee.

After the single formal run finishes:

```bash
python -m research.quality_coverage.summarize \
  output/EXP-20260731-006/quality_coverage_full \
  output/EXP-20260731-006/official_gt
```

Do not run additional experiments that differ only by random seed.  The
pre-created alternate-seed configuration files are retained for provenance
but are not scheduled.  After `C1_SCREEN_PASS`, use available GPUs for
problem-driven controls, key ablations, or cross-dataset validation.

The formal C1 run has now completed with `C1_SCREEN_FAIL`.  Its legacy output
directory is preserved unchanged.  Subsequent B/C2 runs use the structured
layout and compact logging documented in `research/stage3c_runtime/README.md`.
The ADD(-S) reader now accepts both historical `error=ad_ntop=*` and actual
`error:ad_ntop:*` evaluator directory names.
