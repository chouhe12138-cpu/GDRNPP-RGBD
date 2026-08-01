#!/usr/bin/env python3
"""Edit the settings below, then run this file from an IDE or Python."""

from research.quality_coverage.trainer import Stage3C1


model = Stage3C1()

# Keep protocol="official" for the paper screening run. In this mode the
# publication-critical settings are locked. To try different values, use a
# new name and protocol="exploratory"; exploratory results are labelled and
# must not replace the formal comparison.
model.train(
    epochs=40,
    batch=48,
    effective_batch=48,
    lr=8e-4,
    workers=8,
    evaluate_every=5,
    save_every=5,
    keep_checkpoints=3,
    seed=20260731,
    name="quality_coverage_full",
    protocol="official",
    baseline=True,
    plots=True,
    resume=False,
    launch=False,  # Change only this line to True when ready to start.
)

# In a Python console or notebook:
# model.status("quality_coverage_full")
# model.watch("quality_coverage_full")
