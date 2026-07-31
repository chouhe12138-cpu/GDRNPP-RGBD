from __future__ import annotations

from research.pbr_validation.split_protocol import (
    TRAIN_SCENES,
    VALIDATION_SCENES,
    select_diagnostic_images,
)


def test_scene_split_is_disjoint_and_complete() -> None:
    assert not set(TRAIN_SCENES) & set(VALIDATION_SCENES)
    assert set(TRAIN_SCENES) | set(VALIDATION_SCENES) == set(range(50))
    assert len(TRAIN_SCENES) == 47


def test_diagnostic_sampling_is_deterministic() -> None:
    first = select_diagnostic_images(range(1000), scene_id=12)
    second = select_diagnostic_images(reversed(range(1000)), scene_id=12)
    assert first == second
    assert len(first) == 500
    assert len(set(first)) == 500


def test_scenes_get_distinct_samples() -> None:
    assert select_diagnostic_images(range(1000), 12) != select_diagnostic_images(range(1000), 13)
