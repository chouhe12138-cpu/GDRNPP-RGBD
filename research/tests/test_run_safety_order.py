from __future__ import annotations

import shutil
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


def test_bundle_entrypoint_checks_clean_attached_branch_before_creation():
    source = BUNDLE_SCRIPT.read_text(encoding="utf-8")
    attached_check = source.index("symbolic-ref --quiet --short HEAD")
    clean_check = source.index("status --porcelain --untracked-files=all")
    create = source.index('bundle create "${bundle}" "${branch}"')
    assert attached_check < clean_check < create
    assert '[[ "${branch}" == "main" ]]' not in source
    assert '[[ ! -e "${bundle}" ]]' in source
    subprocess.run(["bash", "-n", str(BUNDLE_SCRIPT)], check=True)


def test_bundle_entrypoint_accepts_clean_research_branch(tmp_path):
    repo = tmp_path / "repo"
    script = repo / "docker/l40/create_bundle.sh"
    script.parent.mkdir(parents=True)
    shutil.copy2(BUNDLE_SCRIPT, script)
    subprocess.run(["git", "init", "-b", "exp-test", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "test"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
    (repo / "tracked.txt").write_text("bundle contract\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.txt", "docker/l40/create_bundle.sh"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "fixture"], check=True, capture_output=True)

    output = repo / ".bundles"
    result = subprocess.run(
        [str(script), ".bundles"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "branch=exp-test" in result.stdout
    assert len(list(output.glob("GDRNPP-RGBD-*.bundle"))) == 1
