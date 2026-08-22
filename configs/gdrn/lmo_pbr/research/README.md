# Future research configs

`exp013/` contains three preregistered pose-head variants. A/B formal configs
are authorized only after their local gates are recorded; C remains executable
for local engineering validation while its experiment metadata stays `PLANNED`
until both A and B pass their fixed E40 gates.

This opt-in hierarchy is for experiments created after the current B/C2 runs.
Existing Stage 3C configs remain in their original locations and are not
re-parented.

Layers:

1. `_base_/lmo_gt_eval.py` defines the official checkpoint and LM-O GT-box evaluation.
2. `_base_/pbr40_screening.py` adds the frozen 40-epoch PBR screening budget.
3. `templates/pose_head/` shows train/smoke/eval leaf responsibilities.

Create a new experiment-specific directory from the template, replace its
identity and model intervention, register `EXPERIMENT.json`, and validate the
fully resolved config before running. Seed-only configs are not created.
