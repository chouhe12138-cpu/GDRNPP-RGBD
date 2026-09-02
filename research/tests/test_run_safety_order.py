from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "core/gdrn_modeling/engine/engine.py"
BUNDLE_SCRIPT = ROOT / "docker/l40/create_bundle.sh"


def test_epoch_checkpoint_is_saved_before_periodic_evaluation():
    source = ENGINE.read_text(encoding="utf-8")
    checkpoint = source.index("# Persist the completed epoch before periodic evaluation")
    evaluation = source.index("should_evaluate_epoch(", checkpoint)
    assert source.index("checkpointer.save(", checkpoint, evaluation) < evaluation


def test_bundle_entrypoint_checks_clean_main_before_creation():
    source = BUNDLE_SCRIPT.read_text(encoding="utf-8")
    assert '[[ "${branch}" == "main" ]]' in source
    clean_check = source.index("status --porcelain --untracked-files=all")
    create = source.index('bundle create "${bundle}" main')
    assert clean_check < create
    assert '[[ ! -e "${bundle}" ]]' in source
    subprocess.run(["bash", "-n", str(BUNDLE_SCRIPT)], check=True)
