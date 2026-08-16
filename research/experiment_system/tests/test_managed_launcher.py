from pathlib import Path

import json


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = PROJECT_ROOT / "docker/l40/managed_experiment.sh"


def test_managed_launcher_uses_writable_account_local_home_and_dataset_cache():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'home_dir="${root}/home"' in source
    assert '--mount "type=bind,src=${home_dir},dst=/home/gdrn"' in source
    assert (
        '--env "GDRN_DATASET_CACHE_DIR=/home/gdrn/.cache/gdrnpp_datasets"'
        in source
    )
    assert '"${root}/cache/gdrnpp_datasets"' in source
    assert '"${repo_root}/.cache"' in source
    assert 'test -w \\${GDRN_DATASET_CACHE_DIR}' in source
    assert (
        '--mount "type=bind,src=${root}/cache/gdrnpp_datasets,'
        'dst=/workspace/gdrnpp/.cache"'
        in source
    )
    assert "test -w /workspace/gdrnpp/.cache" in source


def test_managed_launcher_registers_exp010_and_enforces_metadata_authorization():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "EXP005|EXP009|EXP010" in source
    assert 'EXP010)' in source
    assert 'experiment_id="EXP-20260816-010-cpm-official-lr-control"' in source
    assert (
        'config_root="configs/gdrn/lmo_pbr/research/'
        'exp010_cpm_official_lr_control"' in source
    )
    assert "require_run_authorization" in source
    assert "AUTHORIZED|RUNNING" in source

    metadata = json.loads(
        (
            PROJECT_ROOT
            / "research/experiments/EXP-20260816-010-cpm-official-lr-control/"
            "EXPERIMENT.json"
        ).read_text(encoding="utf-8")
    )
    assert metadata["status"] == "AUTHORIZED"
    assert metadata["decision"].startswith("AUTHORIZED_AFTER_EXP005")
