"""LM candidate isolation and fail-closed contract regression."""
from pathlib import Path
import subprocess
import sys

import pytest
from mmcv import Config

from core.gdrn_modeling.models.GDRN_CAD import dataset_context
from research.run_contract import validate_research_run_config

ROOT = Path(__file__).resolve().parents[3]
LM = ROOT / 'configs/gdrn/lm/research/candidate_cad/train_imagenet_full.py'
LM_SMOKE = LM.with_name('smoke.py')
LMO = ROOT / 'configs/gdrn/lmo_pbr/research/exp025_hierarchical_cad_attention/train_imagenet_full.py'
OLD_LM = LMO.with_name('train_lm13_imagenet_full.py')


@pytest.mark.parametrize('path,identity,count', [(LM, 'lm13', 13), (LMO, 'lmo', 8)])
def test_config_switch_in_independent_process(path, identity, count):
    code = ('from mmcv import Config; from core.gdrn_modeling.models.GDRN_CAD import dataset_context; '
            'import sys; cfg=Config.fromfile(sys.argv[1]); ctx=dataset_context(cfg); '
            'print(ctx.key, len(ctx.object_ids), cfg.MODEL.POSE_NET.NUM_CLASSES)')
    result = subprocess.run([sys.executable, '-c', code, str(path)], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip().endswith(f'{identity} {count} {count}')


def test_candidate_keeps_historical_lm_protocol_and_is_never_formal():
    cfg = Config.fromfile(str(LM))
    old = Config.fromfile(str(OLD_LM))
    assert cfg.EXPERIMENT_ID == 'LM13-CAD-CANDIDATE'
    assert tuple(cfg.DATASETS.TRAIN) == tuple(old.DATASETS.TRAIN)
    assert tuple(cfg.DATASETS.TEST) == tuple(old.DATASETS.TEST)
    for key in ('IMS_PER_BATCH', 'REFERENCE_BS', 'TOTAL_EPOCHS', 'CHECKPOINT_PERIOD',
                'OPTIMIZER_CFG', 'WARMUP_ITERS', 'ANNEAL_POINT'):
        assert cfg.SOLVER[key] == old.SOLVER[key]
    assert cfg.VAL == old.VAL
    assert cfg.DATASET_CONTEXT == old.DATASET_CONTEXT
    assert cfg.MODEL.POSE_NET.NUM_CLASSES == old.MODEL.POSE_NET.NUM_CLASSES == 13
    assert cfg.BACKBONE_LR_MULT == old.BACKBONE_LR_MULT == .1
    assert cfg.MODEL.POSE_NET.CAD_ATTENTION_HEAD.INIT_CFG.residual_target_mode == 'gt_route'
    assert cfg.RESEARCH_PROTOCOL.STAGE == 'candidate'
    assert cfg.RESEARCH_PROTOCOL.SERVER_RELEASE_ALLOWED is False
    assert 'EXP-' not in cfg.EXPERIMENT_ID and 'experiments' not in cfg.OUTPUT_DIR
    validate_research_run_config(cfg, mode='prepare')
    cfg.RESEARCH_PROTOCOL.FORMAL_READY = True
    with pytest.raises(ValueError, match='Candidate configuration cannot enter formal'):
        validate_research_run_config(cfg, mode='formal')


def test_smoke_entry_is_local_and_short():
    cfg = Config.fromfile(str(LM_SMOKE))
    validate_research_run_config(cfg, mode='smoke')
    assert tuple(cfg.DATASETS.TRAIN) == ('lm_13_train_smoke', 'lm_imgn_13_train_1k_per_obj_smoke')
    assert cfg.SOLVER.IMS_PER_BATCH == 4 and cfg.SOLVER.AMP.ENABLED


def test_real_render_and_test_smoke_samples_share_class_order():
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from core.gdrn_modeling.datasets.dataset_factory import register_datasets_in_cfg

    cfg = Config.fromfile(str(LM_SMOKE))
    cfg.DATASETS.TEST = ('lm_13_test_smoke',)
    register_datasets_in_cfg(cfg)
    orders = []
    for name in (*cfg.DATASETS.TRAIN, *cfg.DATASETS.TEST):
        records = DatasetCatalog.get(name)
        assert len(records) == 8
        first = records[0]
        annotation = first['annotations'][0]
        assert {'file_name', 'cam', 'annotations', 'scene_im_id'} <= first.keys()
        assert {'category_id', 'bbox', 'pose', 'trans', 'segmentation',
                'bbox3d_and_center', 'xyz_path'} <= annotation.keys()
        assert 0 <= annotation['category_id'] < 13
        orders.append(tuple(MetadataCatalog.get(name).objs))
    assert orders[0] == orders[1] == orders[2]
    # Online rendering supplies XYZ; pre-generated real XYZ crops are not installed here.
    assert cfg.MODEL.POSE_NET.XYZ_ONLINE


def test_lm_hierarchy_identity_and_data_order_fail_closed():
    cfg = Config.fromfile(str(LM))
    assert dataset_context(cfg).object_ids == (1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15)
    bad = Config.fromfile(str(LM))
    bad.MODEL.POSE_NET.NUM_CLASSES = 8
    with pytest.raises(ValueError, match='NUM_CLASSES'):
        dataset_context(bad)
    bad = Config.fromfile(str(LM))
    bad.DATASETS.TRAIN = ('lm_13_train_online', 'lmo_pbr_train')
    with pytest.raises(ValueError, match='Training split object order mismatch'):
        dataset_context(bad)
    bad = Config.fromfile(str(LM))
    bad.DATASETS.TEST = ('lmo_bop_test',)
    with pytest.raises(ValueError, match='Train/test object order mismatch'):
        dataset_context(bad)
    bad = Config.fromfile(str(LM))
    bad.CAD_HIERARCHY_CONTRACT.SHA256 = '0' * 64
    with pytest.raises(ValueError, match='SHA mismatch'):
        dataset_context(bad)
    bad = Config.fromfile(str(LM))
    bad.CAD_HIERARCHY_CONTRACT.DATASET_KEY = 'lmo'
    with pytest.raises(ValueError, match='contract dataset key mismatch'):
        dataset_context(bad)
    bad = Config.fromfile(str(LM))
    bad.CAD_HIERARCHY_CONTRACT.MODE = 'geometry_adaptive'
    with pytest.raises(ValueError, match='mode mismatch'):
        dataset_context(bad)
    bad = Config.fromfile(str(LM))
    bad.MODEL.POSE_NET.CAD_ATTENTION_HEAD.HIERARCHY_PATH = \
        str(ROOT / '.local/dataset_cache/exp025/consistent_v3.npz')
    with pytest.raises(ValueError, match='Hierarchy object order mismatch|SHA mismatch'):
        dataset_context(bad)
    bad = Config.fromfile(str(LM))
    bad.MODEL.POSE_NET.CAD_ATTENTION_HEAD.HIERARCHY_PATH = '$MISSING_LM_HIERARCHY/tree.npz'
    with pytest.raises(ValueError, match='Unresolved hierarchy path'):
        dataset_context(bad)


def test_immutable_experiments_and_launcher_block_unchanged():
    exp025 = Config.fromfile(str(LMO))
    exp026 = Config.fromfile(str(ROOT / 'configs/gdrn/lmo_pbr/research/'
                                   'exp026_residual_aligned_sampling_ablation/train_uniform_full.py'))
    exp025.MODEL.POSE_NET.CAD_ATTENTION_HEAD.HIERARCHY_PATH = \
        exp026.MODEL.POSE_NET.CAD_ATTENTION_HEAD.HIERARCHY_PATH
    exp025.CAD_HIERARCHY_CONTRACT = exp026.CAD_HIERARCHY_CONTRACT
    with pytest.raises(ValueError):
        dataset_context(exp025)
    exp026.RESEARCH_PROTOCOL.FORMAL_READY = True
    with pytest.raises(ValueError, match='SERVER_BLOCKED'):
        validate_research_run_config(exp026, mode='formal')


def test_exp025_builder_compatibility_default_is_cad_hierarchy():
    from research.exp025.dataset_context import resolve_dataset_context
    cfg = Config.fromfile(str(LMO))
    context = resolve_dataset_context(cfg, require_hierarchy=False)
    assert context.hierarchy_path == Path(cfg.MODEL.POSE_NET.CAD_ATTENTION_HEAD.HIERARCHY_PATH)
