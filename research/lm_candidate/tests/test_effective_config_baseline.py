"""Exact effective-config regression against clean d0bd434 (2026-09-22).

The pinned environment makes the path-bearing config fields deterministic.  Hash
the complete resolved dictionary, including inherited fields that a refactor
could otherwise silently drop.  Scientific fields are also asserted explicitly
in the candidate/EXP025/EXP026 contract tests.
"""
import hashlib
import json
from pathlib import Path

import pytest
from mmcv import Config


ROOT = Path(__file__).resolve().parents[3]
CONFIGS = {
    'lm': ('lm/research/candidate_cad/train_imagenet_full.py',
           '2f66d339fc53617359f699b81cc956e1be0ba36c526c174543ca6e361e6642a8'),
    'exp025_frozen': ('lmo_pbr/research/exp025_hierarchical_cad_attention/train_official_frozen.py',
                      '278633f10105d33d33746e2c46105a009abcdaadd3a71be8992ee6a127dc06c6'),
    'exp025_full': ('lmo_pbr/research/exp025_hierarchical_cad_attention/train_imagenet_full.py',
                    'd89e0bd1d65521a9f45b6fd2b01ee05030e0dafd3c9865e1cab227da64e76bac'),
    'exp026_uniform': ('lmo_pbr/research/exp026_residual_aligned_sampling_ablation/train_uniform_full.py',
                       'c94f35ff932be8e07cf3f52deb0363c85fba897ea4a70e8acf2499f583eb5239'),
    'exp026_adaptive': ('lmo_pbr/research/exp026_residual_aligned_sampling_ablation/train_adaptive_full.py',
                        '6a57f3d7ff9dbf20850385fb629abc2b10dd9a4753ed5165822538e9cddf4247'),
    'old_lm': ('lmo_pbr/research/exp025_hierarchical_cad_attention/train_lm13_imagenet_full.py',
               '91aaa807bb173442f4db65de704c44ed425f89e370d28d05db79e9c69cba5e2f'),
}


@pytest.mark.parametrize('name', tuple(CONFIGS))
def test_effective_config_matches_d0bd434(name, monkeypatch):
    monkeypatch.setenv('GDRN_DATASET_CACHE_DIR', '.local/dataset_cache')
    monkeypatch.setenv('GDRN_CONVNEXT_BASE_WEIGHTS', '/__baseline__/convnext.pth')
    path, expected = CONFIGS[name]
    cfg = Config.fromfile(str(ROOT / 'configs/gdrn' / path))
    payload = json.dumps(cfg._cfg_dict.to_dict(), sort_keys=True,
                         separators=(',', ':'), default=str)
    assert hashlib.sha256(payload.encode()).hexdigest() == expected
