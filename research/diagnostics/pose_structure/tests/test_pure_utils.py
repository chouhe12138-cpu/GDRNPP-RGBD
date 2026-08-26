from __future__ import annotations

import math
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from research.diagnostics.pose_structure.common import cosine_scalar
from research.diagnostics.pose_structure.metrics import monotonicity, rotation_error_deg, translation_error_cm


def test_rotation_error_identity():
    r = torch.eye(3).unsqueeze(0)
    e = rotation_error_deg(r, r)
    assert torch.allclose(e, torch.zeros_like(e), atol=1e-5)


def test_translation_error_cm():
    a = torch.tensor([[0.0, 0.0, 0.0]])
    b = torch.tensor([[0.01, 0.0, 0.0]])
    e = translation_error_cm(a, b)
    assert torch.allclose(e, torch.tensor([1.0]))


def test_cosine_scalar():
    assert abs(cosine_scalar(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])) - 1.0) < 1e-6
    assert abs(cosine_scalar(torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])) + 1.0) < 1e-6


def test_monotonicity():
    out = monotonicity([3, 2, 1], increasing=False)
    assert out["strict"] is True
    out2 = monotonicity([1, 3, 2], increasing=True)
    assert out2["strict"] is False
