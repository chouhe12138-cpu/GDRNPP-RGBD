from pathlib import Path


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
