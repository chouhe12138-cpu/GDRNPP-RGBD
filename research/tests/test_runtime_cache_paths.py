from __future__ import annotations

from core.base_data_loader import resolve_bg_cache_path
from core.gdrn_modeling.datasets.lmo_bop_test import resolve_dataset_cache_root


def test_bg_cache_uses_xdg_cache_home_when_set(tmp_path):
    assert resolve_bg_cache_path(
        "VOC", "digest", {"XDG_CACHE_HOME": str(tmp_path)}
    ) == str(tmp_path / "bg_paths_VOC_digest.pkl")


def test_bg_cache_falls_back_to_original_relative_directory():
    assert resolve_bg_cache_path("VOC", "digest", {}) == ".cache/bg_paths_VOC_digest.pkl"


def test_lmo_dataset_cache_keeps_separate_gdrn_contract(tmp_path):
    assert resolve_dataset_cache_root(
        {"GDRN_DATASET_CACHE_DIR": str(tmp_path)}
    ) == str(tmp_path)
