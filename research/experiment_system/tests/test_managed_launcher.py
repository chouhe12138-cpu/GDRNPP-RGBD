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


def test_managed_launcher_registers_exp012_with_its_own_preflight():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "EXP005|EXP009|EXP010|EXP012" in source
    assert 'EXP012)' in source
    assert (
        'experiment_id="EXP-20260817-012-hierarchical-correspondence-head"'
        in source
    )
    assert 'isolation_role="PNP_REPLACEMENT"' in source
    assert "python -m research.next_pose_head.preflight" in source
    metadata = json.loads(
        (
            PROJECT_ROOT
            / "research/experiments/EXP-20260817-012-hierarchical-correspondence-head/"
            "EXPERIMENT.json"
        ).read_text(encoding="utf-8")
    )
    assert metadata["status"] == "AUTHORIZED"
    assert metadata["protocol"]["run_order"] == ["gate", "formal"]
    assert metadata["evidence"]["server_access_create_gate_formal"] == "NOT_RUN"


def test_exp012_formal_skips_smoke_and_audit_without_changing_other_experiments():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'if [[ "${experiment}" != "EXP012" ]]; then' in source
    assert "require_complete smoke\n        require_complete audit" in source
