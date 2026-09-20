"""Use the existing fake-container harness; no Docker/SSH is performed."""
import pytest
from research.tests.test_experiment_launcher import _resource_gate, EXP025_HIERARCHY, CONVNEXT_CHECKPOINT


@pytest.mark.parametrize('source', ['official_lmo', 'imagenet'])
def test_exp025_profile(source):
    unused = (CONVNEXT_CHECKPOINT,) if source == 'official_lmo' else (
        '/workspace/gdrnpp/pretrained_models/lmo_pbr/model_final_wo_optim.pth',)
    result = _resource_gate(f'fake_backbone_init={source}\n'
        'profile=$(resolve_resource_profile train.py)\nrequire_profile_resources "$profile" train.py',
        hierarchy=EXP025_HIERARCHY, train_protocol='exp025_lmo', missing=unused)
    assert 'RESOURCE_PROFILE exp025_lmo' in result.stdout


@pytest.mark.parametrize('failure', ['hierarchy', 'weights', 'preflight', 'unknown'])
def test_exp025_fail_closed(failure):
    missing = (EXP025_HIERARCHY,) if failure == 'hierarchy' else (
        ('/workspace/gdrnpp/pretrained_models/lmo_pbr/model_final_wo_optim.pth',) if failure == 'weights' else ())
    preamble = 'fake_server_preflight=1\n' if failure == 'preflight' else (
        'fake_backbone_init=invalid\n' if failure == 'unknown' else '')
    result = _resource_gate(preamble+'require_exp025_resources train.py',
        hierarchy=EXP025_HIERARCHY, train_protocol='exp025_lmo', missing=missing, check=False)
    assert result.returncode != 0
