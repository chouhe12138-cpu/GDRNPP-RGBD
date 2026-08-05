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

After both fixed epoch-40 outputs are available in one filesystem, run
`research.stage3c_runtime.compare_formal` to measure C2's added value directly
against B using the same three gates.

For lab0, first test the existing lab1 assets without writing to them:

```bash
ASSET_ROOT=/data/labs/lab1/docker_data/chx \
BASELINE_ROOT=/data/labs/lab1/docker_data/chx/outputs/EXP-20260731-006/official_gt \
docker/l40/stage3bc2.sh B access
```

`ACCESS=PASS` confirms Docker-daemon access, physical GPU 0 visibility,
directory traversal/read access, and official-checkpoint readability.  It
does not create a container or any directory.  Reuse the same two environment
variables for `B create`, `B gate`, `B smoke`, and `B validate`.

The image is host-wide when lab0 and lab1 use the same Docker daemon.  Runtime
containers use the invoking account's numeric UID/GID, so the shared image
does not require rebuilding merely because the accounts differ.  Build the
image once from the exact Gitee commit used by both roles.  If the access
check fails, do not change lab1 ownership or permissions; arrange an
administrator-approved shared path or a lab0-owned asset copy.
