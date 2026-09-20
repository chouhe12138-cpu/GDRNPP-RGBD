from pathlib import Path

import pytest
from mmcv import Config

from research.run_contract import validate_research_run_config

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / 'configs/gdrn/lmo_pbr/research/exp025_hierarchical_cad_attention'


@pytest.mark.parametrize('name', ['train_official_frozen.py', 'train_imagenet_full.py'])
def test_exp025_formal_stays_locked_before_server_gate(name):
    cfg = Config.fromfile(str(CFG / name))
    validate_research_run_config(cfg, mode='prepare')
    with pytest.raises(ValueError, match='not ready'):
        validate_research_run_config(cfg, mode='formal')


@pytest.mark.parametrize('name', ['train_official_frozen.py', 'train_imagenet_full.py'])
def test_exp025_formal_contract_after_unlock(name):
    cfg = Config.fromfile(str(CFG / name))
    cfg.RESEARCH_PROTOCOL.FORMAL_READY = True
    cfg.SOLVER.AMP.INIT_SCALE = 16384
    result = validate_research_run_config(cfg, mode='formal')
    assert result['batch_size'] == 48 and result['total_epochs'] == 40
    assert result['evaluation_period'] == 5 and result['amp_enabled']


def test_smoke_contract_remains_small_and_non_evaluating():
    cfg = Config.fromfile(str(CFG / 'smoke.py'))
    result = validate_research_run_config(cfg, mode='smoke')
    assert result['batch_size'] == 4 and result['total_epochs'] == 1
    assert result['evaluation_period'] == 0
