# Workspace Operating Notes

Keep this file lightweight. It routes work; it does not freeze future research
choices.

## Start and context

- Run `git status` before changing anything and preserve all uncommitted work.
- The required local Python environment is Conda `pytorch22`. Before any
  Python, PyTorch, CUDA, test, or experiment command, run
  `conda activate pytorch22` (source Conda's shell hook first in a
  non-interactive shell). Do not diagnose a missing system `python`, package,
  CUDA runtime, or GPU before checking again inside this activated environment.
- Read `research/HANDOFF.md` for experiment status.
- Read `research/SERVER_MIGRATION_HANDOFF.md` only for server, Docker, GPU, or
  migration tasks.
- Read `research/RESEARCH_PLAN.md`, `research/DECISIONS.md`, stage protocols,
  and external research notes only when the task actually requires them.
- New research hypotheses and architecture ideas belong under
  `E:\6D姿态估计的研究` (`/mnt/e/6D姿态估计的研究` in WSL), not in this
  repository. Do not load that directory automatically.
- Objective experiment results, reproducibility settings, and current runtime
  state may be recorded in the repository.

## Safety

- Do not reset, clean, overwrite, delete, commit, or push unless the user
  explicitly requests that exact action.
- Do not generate SSH keys, store access tokens, or initiate SSH sessions.
  The user runs server commands and returns their output.
- Preserve other users' accounts, directories, containers, images, volumes,
  and GPU allocations.

## Server workflow

- Prefer short repository scripts for repeated or multi-step server work.
  Short read-only one-line diagnostics are acceptable; avoid long pasted
  command blocks.
- Use `/usr/bin/docker`, not the `sudo docker` alias. The user has no
  administrator password.
- `lab1` is assigned physical GPU 1 only. It appears as logical CUDA device 0
  inside `lab1_chx_stage3c1`.
- Keep detailed and changeable server facts in
  `research/SERVER_MIGRATION_HANDOFF.md` rather than duplicating them here.
