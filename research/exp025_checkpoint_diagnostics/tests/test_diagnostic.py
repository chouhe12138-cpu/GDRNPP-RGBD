import numpy as np

from research.exp025_checkpoint_diagnostics.interventions import linear_cka
from research.exp025_checkpoint_diagnostics.oracle import pose_error, select_records, summary


def test_linear_cka_identity_and_orthogonal_samples():
    samples = np.eye(8, dtype=np.float32)
    assert np.isclose(linear_cka(samples, samples), 1.)
    assert np.isclose(linear_cka(samples, samples @ np.eye(8)[::-1]), 1.)


def test_sample_selection_fixed_seed_and_identity():
    records = [dict(scene_im_id=f'1/{i}', annotations=[dict(category_id=0), dict(category_id=1)])
               for i in range(6)]
    first = select_records(records, [1, 5], 3, 42)
    second = select_records(records, [1, 5], 3, 42)
    assert [(obj, sid, index) for obj, sid, index, _ in first] == [
        (obj, sid, index) for obj, sid, index, _ in second]
    assert len(first) == 6
    assert all(len(item['annotations']) == 1 for _, _, _, item in first)
    assert all(len(record['annotations']) == 2 for record in records)


def test_failed_pose_not_silently_scored():
    assert pose_error(None, np.eye(3, 4)) == dict(rotation_deg=None, translation_m=None)
    assert summary([]) == dict(count=0)
