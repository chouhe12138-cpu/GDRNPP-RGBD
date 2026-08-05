# Stage 3C B/C2 runtime

This package provides the matched B and C2 preflight, launch, artifact, and
checkpoint-isolation controls.  It does not start either formal experiment by
itself.

## Output and logging

B and C2 opt into a structured layout:

```text
output/stage3c/<role>/
  meta/          resolved config and environment
  train/         metrics.jsonl
  checkpoints/   best/recent checkpoints and checkpoint state
  evaluations/   epoch_005 through epoch_040
  summary/       compact tables and figures
```

TensorBoard is disabled.  Compact mode logs selected setup facts, progress,
evaluation summaries, checkpoints, and failures without printing the full
model/config or successful BOP subprocess output.  A final evaluation is not
repeated when epoch 40 was already evaluated.

## Local checks

Activate Conda `pytorch22` before running Python:

```bash
conda activate pytorch22
python -m pytest -q research/stage3c_runtime/tests
python -m research.stage3c_runtime.preflight B
python -m research.stage3c_runtime.preflight C2
```

`--build-model` adds strict checkpoint loading, trainable-tensor, and
optimizer-group learning-rate checks.

## Server workflow

The user runs all server commands.  No SSH action is performed by repository
tools.

```text
docker/l40/stage3bc2.sh B access
docker/l40/stage3bc2.sh B create
docker/l40/stage3bc2.sh B gate
docker/l40/stage3bc2.sh B smoke
docker/l40/stage3bc2.sh B validate

docker/l40/stage3bc2.sh C2 create
docker/l40/stage3bc2.sh C2 gate
docker/l40/stage3bc2.sh C2 smoke
docker/l40/stage3bc2.sh C2 validate
```

Only after both smoke validations pass:

```text
docker/l40/stage3bc2.sh B formal
docker/l40/stage3bc2.sh C2 formal
```

The `formal` command reruns smoke validation and refuses to launch while the
smoke is running, failed, or missing its checkpoint and metrics.

After both fixed epoch-40 outputs are available in one filesystem, run
`research.stage3c_runtime.compare_formal` to measure C2's added value directly
against B using the same three gates.

The checked-in defaults follow the laboratory account/container convention:

```text
B:  lab0 / physical GPU 0 / container lab0_chx
C2: lab1 / physical GPU 1 / container lab1_chx
```

Lab0 uses its account-owned root without environment overrides:

```text
assets:   /data/labs/lab0/docker_data/chx
baseline: /data/labs/lab0/docker_data/chx/baselines/official_gt
outputs:  /data/labs/lab0/docker_data/chx/outputs
logs:     /data/labs/lab0/docker_data/chx/logs
```

Therefore a new SSH/VS Code terminal can directly run
`docker/l40/stage3bc2.sh B status` without repeating `export`.  Environment
variables remain available only as optional overrides.

For lab0, `docker/l40/lab0_b.sh` is the preferred one-command entry point. It
explicitly discards stale shell overrides, can terminate only the known
invalid early-formal process, validates the completed smoke, and runs the
batch-48 NUM_WORKERS benchmark in the background:

```text
docker/l40/lab0_b.sh recover
docker/l40/lab0_b.sh benchmark-workers
docker/l40/lab0_b.sh benchmark-status
docker/l40/lab0_b.sh benchmark-watch
```

If GPU0 is occupied by another user's process, the benchmark records
`WAITING_GPU` and waits in the background for up to 24 hours. It never signals
or modifies the other process.
