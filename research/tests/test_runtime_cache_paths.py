from __future__ import annotations

from pathlib import Path

import pytest

from core.base_data_loader import resolve_bg_cache_path
from core.gdrn_modeling.datasets.lmo_bop_test import resolve_dataset_cache_root
from core.gdrn_modeling.engine.engine_utils import (
    egl_mesh_cache_working_directory,
    resolve_egl_mesh_cache_dir,
)


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


def test_egl_mesh_cache_uses_xdg_cache_home_for_read_only_checkout(tmp_path):
    assert resolve_egl_mesh_cache_dir(environ={"XDG_CACHE_HOME": str(tmp_path)}) == str(
        tmp_path / "gdrnpp_egl_meshes"
    )


def test_egl_mesh_cache_redirects_nested_relative_caches_and_restores_cwd(tmp_path):
    original_dir = Path.cwd()
    with egl_mesh_cache_working_directory(
        environ={"XDG_CACHE_HOME": str(tmp_path)}
    ) as cache_dir:
        assert Path.cwd() == tmp_path / "gdrnpp_egl_meshes"
        assert Path(".cache").resolve() == Path(cache_dir) / ".cache"
    assert Path.cwd() == original_dir


def test_egl_mesh_cache_restores_cwd_after_loader_error(tmp_path):
    original_dir = Path.cwd()
    with pytest.raises(RuntimeError, match="loader failed"):
        with egl_mesh_cache_working_directory(
            environ={"XDG_CACHE_HOME": str(tmp_path)}
        ):
            raise RuntimeError("loader failed")
    assert Path.cwd() == original_dir
