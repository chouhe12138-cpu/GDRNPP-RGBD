from pathlib import Path
import sys

from mmcv import Config

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / 'configs/gdrn/lmo_pbr/research/exp025_hierarchical_cad_attention'


def test_two_exp025_arms_are_explicit_and_matched():
    frozen = Config.fromfile(str(CFG / 'train_official_frozen.py'))
    full = Config.fromfile(str(CFG / 'train_imagenet_full.py'))
    assert (frozen.EXP025_ARM, frozen.BACKBONE_INIT, frozen.TRAIN_BACKBONE) == (
        'official_frozen', 'official_lmo', False)
    assert (full.EXP025_ARM, full.BACKBONE_INIT, full.TRAIN_BACKBONE) == (
        'imagenet_full', 'imagenet', True)
    for key in ('EXPERIMENT_ID', 'SEED', 'DATASETS', 'TEST'):
        assert frozen[key] == full[key]
    for key in ('IMS_PER_BATCH', 'REFERENCE_BS', 'TOTAL_EPOCHS'):
        assert frozen.SOLVER[key] == full.SOLVER[key]
    assert frozen.SOLVER.IMS_PER_BATCH == frozen.SOLVER.REFERENCE_BS == 48
    assert not frozen.RESEARCH_PROTOCOL.FORMAL_READY
    assert not full.RESEARCH_PROTOCOL.FORMAL_READY
    assert 'INIT_SCALE' not in frozen.SOLVER.AMP and 'INIT_SCALE' not in full.SOLVER.AMP
    assert '/exp025/consistent_v3.npz' in '/' + frozen.MODEL.POSE_NET.CAD_ATTENTION_HEAD.HIERARCHY_PATH


def test_exp025_model_import_does_not_load_old_experiment_modules():
    for name in list(sys.modules):
        if name.startswith('core.gdrn_modeling.models.GDRN_') or name.startswith('research.exp022'):
            sys.modules.pop(name, None)
    __import__('core.gdrn_modeling.models.GDRN_CAD')
    assert 'research.exp022' not in sys.modules
    assert 'core.gdrn_modeling.models.GDRN_PCC' not in sys.modules
    assert 'core.gdrn_modeling.models.GDRN_double_mask' not in sys.modules


def test_retained_pcc_model_has_no_deleted_experiment_import():
    __import__('core.gdrn_modeling.models.GDRN_PCC')
    assert 'research.exp022' not in sys.modules


def test_historical_execution_directories_are_absent():
    retired = [
        ROOT / 'research/exp022',
        ROOT / 'research/diagnostics',
        ROOT / 'configs/gdrn/research/exp022_progressive_pcc',
        ROOT / 'configs/gdrn/lmo_pbr/research/exp021_global_hierarchical_cad',
    ]
    remaining = [file for path in retired if path.exists() for file in path.rglob('*')
                 if file.is_file() and '__pycache__' not in file.parts]
    assert remaining == []
    assert (ROOT / 'research/experiments/EXP-20260916-022-progressive-pcc/RECORD.md').is_file()
