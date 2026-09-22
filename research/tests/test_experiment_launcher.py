from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / 'docker/l40/experiment.sh'


def shell(body, check=True):
    result = subprocess.run(['bash', '-c', f'source {LAUNCHER!s}\n{body}'],
                            cwd=ROOT, text=True, capture_output=True)
    if check and result.returncode:
        raise AssertionError(result.stderr)
    return result


def test_launcher_is_valid_shell_and_retired_protocols_stay_disabled():
    subprocess.run(['bash', '-n', str(LAUNCHER)], check=True)
    source = LAUNCHER.read_text()
    assert 'exp025_lmo)' in source
    for retired in ('lm13_gdrn)', 'lm13_pbr)', 'lmo_full_imagenet)', 'legacy_lmo)'):
        assert retired not in source
    assert 'datasets/lm_imgn' not in source


def test_profile_mapping_accepts_exp025_and_rejects_unknown_protocols():
    ok = shell("container_config_value() { echo exp025_lmo; }\nresolve_resource_profile train.py")
    assert ok.stdout.strip() == 'exp025_lmo'
    for value in ('', 'legacy_lmo', 'lm13_gdrn', 'exp025_lm13', 'typo'):
        bad = shell(f"container_config_value() {{ echo {value!r}; }}\nresolve_resource_profile train.py", False)
        assert bad.returncode != 0 and 'unknown TRAIN_PROTOCOL.NAME' in bad.stderr


def test_exp026_server_profile_is_explicitly_blocked_until_release():
    blocked = shell('''container_config_value() {
  case "$2" in
    TRAIN_PROTOCOL.NAME) echo exp026_lmo ;;
    RESEARCH_PROTOCOL.SERVER_RELEASE_ALLOWED) echo False ;;
  esac
}
resolve_resource_profile train.py''', False)
    assert blocked.returncode != 0 and 'EXP026 SERVER_BLOCKED' in blocked.stderr
    released = shell('''container_config_value() {
  case "$2" in
    TRAIN_PROTOCOL.NAME) echo exp026_lmo ;;
    RESEARCH_PROTOCOL.SERVER_RELEASE_ALLOWED) echo True ;;
  esac
}
resolve_resource_profile train.py''')
    assert released.stdout.strip() == 'exp026_lmo'


@pytest.mark.parametrize('arm,machine,initialization', [
    ('official_frozen', 'lab0', 'official_lmo'),
    ('imagenet_full', 'lab1', 'imagenet'),
])
def test_exp025_resource_gate_enforces_arm_machine(arm, machine, initialization):
    body = f'''machine={machine}
container=test
docker_bin=/bin/true
require_container_path() {{ :; }}
require_container_glob() {{ :; }}
require_voc_data() {{ :; }}
require_convnext_weights() {{ :; }}
container_config_value() {{
  case "$2" in
    BACKBONE_INIT) echo {initialization} ;;
    EXP025_ARM) echo {arm} ;;
    MODEL.POSE_NET.CAD_ATTENTION_HEAD.HIERARCHY_PATH) echo /home/gdrn/.cache/gdrnpp_datasets/exp025/consistent_v3.npz ;;
  esac
}}
require_exp025_resources train.py'''
    assert shell(body).returncode == 0
    wrong = 'lab1' if machine == 'lab0' else 'lab0'
    bad = shell(body.replace(f'machine={machine}', f'machine={wrong}', 1), False)
    assert bad.returncode != 0 and 'must run on' in bad.stderr


def test_gate_command_is_real_batch48_egl_and_unique_subdirectory():
    result = shell('build_gate_command cfg.py /workspace/gdrnpp/output/run 32768')
    command = result.stdout
    assert 'research.exp025.real_smoke' in command
    assert '--renderer egl' in command and '--batch-size 48' in command and '--steps 8' in command
    assert '--amp-scale 32768' in command and '/gate' in command


def test_resource_gate_precedes_output_creation():
    source = LAUNCHER.read_text()
    assert source.index('runtime_gate "${mode}" "${config}"') < source.index('mkdir -p "${run_host}"')
