"""EXP025 failure telemetry, accumulation boundaries and the residual-probe contract."""
import json

import pytest
import torch
from torch import nn

from research.exp025.runtime import (NonFiniteTrainingError, amp_step, grad_norm_stats,
                                     head_telemetry, raw_residual_stats, token_norm_stats)
from research.exp025.residual_probe import GTT3ConditionedResidual, probe_prediction
from core.utils import solver_utils


def test_raw_residual_and_token_statistics():
    stats = raw_residual_stats(torch.tensor([-0.5, 6.0, 10.0]))
    assert stats['abs_max'] == 10.0
    assert stats['fraction_abs_gt_5'] == pytest.approx(2/3)
    assert stats['fraction_abs_gt_9'] == pytest.approx(1/3)
    assert stats['tanh_exact_saturation'] == pytest.approx(1/3)  # only tanh(10) saturates in fp32

    tokens = torch.tensor([[[3., 4.]], [[0., 0.]]])  # norms 5 and 0
    norms = token_norm_stats(tokens)
    assert norms == dict(mean=2.5, p95=pytest.approx(4.75), max=5.0)


def test_head_telemetry_is_json_safe(head):
    feature = torch.randn(2, 1024, 8, 8)
    diagnostics = {}
    head(feature, torch.tensor([0, 1]), diagnostics=diagnostics)
    values = head_telemetry(head, diagnostics)
    json.dumps(values)  # the report writer must accept it verbatim
    for key in ('image_tokens_token_norm', 'cad_tokens_token_norm', 't3_logits', 'raw_residual',
                'bounded_residual_norm_max', 't3_classifier_weight_abs_max'):
        assert key in values
    assert values['bounded_residual_norm_max'] <= 1.00001
    assert set(values['raw_residual']) == {'abs_max', 'fraction_abs_gt_5', 'fraction_abs_gt_9',
                                           'tanh_exact_saturation'}


class _NanGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return gradient * float('nan')


class _NanGradientModel(nn.Module):
    """The NaN must enter through the parameter's own gradient path."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, value):
        return _NanGradient.apply(value * self.scale)


def test_amp_step_reports_the_failing_step_and_parameter():
    """The FP32 path needs no CUDA, so the failure telemetry is covered without a GPU."""
    model = _NanGradientModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    with pytest.raises(NonFiniteTrainingError, match='Non-finite gradient') as caught:
        amp_step(model, optimizer, None, model(torch.ones(2, 2)).sum(), step=7)
    telemetry = caught.value.telemetry
    assert telemetry['step'] == 7 and telemetry['amp_scale'] is None
    assert telemetry['non_finite_parameters'] == ['scale']
    assert telemetry['grad']['non_finite_parameters'] == ['scale']
    assert telemetry['grad']['grad_norm']['scale'] is None  # non-finite norms stay JSON-safe
    json.dumps(telemetry)


def test_amp_step_reports_a_non_finite_loss_before_backward():
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    with pytest.raises(NonFiniteTrainingError, match='Non-finite loss') as caught:
        amp_step(model, optimizer, None, torch.tensor(float('nan'), requires_grad=True), step=3)
    assert caught.value.telemetry['step'] == 3
    assert caught.value.telemetry['loss_finite'] is False


def test_grad_norm_groups_head_submodules():
    model = nn.Sequential(nn.Linear(4, 2), nn.Linear(2, 1))
    model(torch.ones(1, 4)).sum().backward()
    stats = grad_norm_stats(model)
    assert set(stats['grad_norm']) == {'0', '1'}
    assert stats['non_finite_parameters'] == []


def test_accumulation_boundaries_at_epoch_tail():
    """The boundary set the accumulation smoke must observe with 25 iterations and window 12."""
    iterations_per_epoch, accumulate = 25, 12
    boundaries = {iteration for iteration in range(iterations_per_epoch)
                  if solver_utils.should_optimizer_step(iteration, accumulate, iterations_per_epoch, 25)}
    assert boundaries == {11, 23, 24}
    assert [solver_utils.accumulation_window_size(iteration, accumulate, iterations_per_epoch)
            for iteration in (0, 11, 12, 23, 24)] == [12, 12, 12, 12, 1]


def test_probe_loss_equals_the_head_residual_term(head):
    """Single-symmetry objects only: with two branches the probe selects by its own
    conditioned fit while the head re-scores the returned residual, and those can differ."""
    torch.manual_seed(0)
    probe = GTT3ConditionedResidual(token_dim=16)
    feature = torch.randn(2, 1024, 8, 8)
    classes = torch.tensor([0, 0])  # object 0 has one symmetry branch in the fixture
    xyz = torch.rand(2, 3, 64, 64)
    mask = torch.ones(2, 1, 64, 64)
    prediction, probe_loss = probe_prediction(head, feature, classes, probe, xyz, mask)
    losses, stats, _, valid = head.loss(prediction, classes, xyz, mask, route_weight=0., return_targets=True)
    torch.testing.assert_close(probe_loss, losses['loss_cad_residual'], rtol=1e-5, atol=1e-6)
    assert prediction['residual'].norm(dim=1).max() <= 1.00001
    assert torch.isfinite(probe_loss)
    assert stats['cad_valid_points'] > 0


def test_probe_conditioning_uses_the_selected_t3_token():
    torch.manual_seed(0)
    probe = GTT3ConditionedResidual(token_dim=8)
    image_tokens = torch.zeros(1, 4, 8)
    bank = torch.randn(1, 512, 8)
    first = probe(image_tokens, bank, torch.full((1, 2, 2), 3, dtype=torch.long))
    second = probe(image_tokens, bank, torch.full((1, 2, 2), 400, dtype=torch.long))
    assert first.shape == (1, 4, 3)
    assert not torch.allclose(first, second)
    # the image token is concatenated unchanged: zeroing it must not change the gather
    assert torch.equal(probe(image_tokens, bank, torch.full((1, 2, 2), 3, dtype=torch.long)), first)
    with pytest.raises(RuntimeError, match='out of bounds'):
        probe(image_tokens, bank, torch.full((1, 2, 2), 512, dtype=torch.long))
