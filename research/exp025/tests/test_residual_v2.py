"""Residual V2: predicted soft-T3 conditioning, detached route, zero-initialized output.

The contract this module pins down is what makes the conditioning legitimate: no
ground-truth id reaches the predictor, the route probability is an input rather than a
gradient path, and the initial residual is exactly the T3 anchor.
"""
import pytest
import torch

from core.gdrn_modeling.models.heads.hierarchical_cad_attention_head import (SoftT3ResidualPredictor,
                                                                            bounded_residual)


def prediction(head, feature, classes, with_grad=False):
    diagnostics = {}
    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context:
        return head(feature, classes, diagnostics=diagnostics), diagnostics


def targets(feature, classes, head, route_weight):
    prediction_dict, _ = prediction(head, feature, classes, with_grad=True)
    xyz = torch.rand(len(classes), 3, 64, 64)
    mask = torch.ones(len(classes), 1, 64, 64)
    losses, _ = head.loss(prediction_dict, classes, xyz, mask, route_weight=route_weight)
    return prediction_dict, losses


def test_v2_output_shape_probability_and_bound(head):
    torch.manual_seed(0)
    feature = torch.randn(2, 1024, 8, 8)
    classes = torch.tensor([0, 1])
    out, diagnostics = prediction(head, feature, classes)
    assert out['t3_logits'].shape == (2, 512, 64, 64)
    assert out['residual'].shape == (2, 3, 64, 64)
    assert out['mask_logit'].shape == (2, 1, 64, 64)
    assert out['residual'].norm(dim=1).max() <= 1.00001

    probabilities = diagnostics['t3_probabilities']
    assert probabilities.shape == (2, 64 * 64, 512)
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(2, 64 * 64), atol=1e-5, rtol=1e-5)
    assert not probabilities.requires_grad and probabilities.grad_fn is None  # detached


def test_v2_context_is_projected_to_the_configured_width(head):
    torch.manual_seed(0)
    _, diagnostics = prediction(head, torch.randn(2, 1024, 8, 8), torch.tensor([0, 1]))
    context = diagnostics['soft_t3_context']
    assert context.shape == (2, 64 * 64, 64)  # [B,P,context_dim]
    assert torch.isfinite(context).all()
    expected = head.residual_predictor.context_projection.out_features
    assert context.shape[-1] == expected == 64


def test_v2_zero_initialization_starts_at_the_t3_anchor(head):
    torch.manual_seed(0)
    out, diagnostics = prediction(head, torch.randn(2, 1024, 8, 8), torch.tensor([0, 1]))
    torch.testing.assert_close(diagnostics['raw_residual'],
                               torch.zeros_like(diagnostics['raw_residual']), rtol=0, atol=0)
    assert torch.equal(out['residual'], torch.zeros_like(out['residual']))
    assert head.residual_predictor.final.weight.abs().sum() == 0
    assert head.residual_predictor.final.bias.abs().sum() == 0


def test_v2_conditioning_uses_the_predicted_distribution(head):
    """The context is the expectation under the classifier's own belief, not an id lookup."""
    torch.manual_seed(0)
    image_tokens = torch.randn(1, 4, 16)
    bank = torch.randn(1, 512, 16)
    logits = torch.randn(1, 4, 512)
    predictor = head.residual_predictor
    context, probabilities = predictor.context(bank, logits)
    expected = torch.einsum('bpk,bkd->bpd', torch.softmax(logits.float(), -1), predictor.context_projection(bank.float()))
    torch.testing.assert_close(context, expected, atol=1e-5, rtol=1e-5)
    # A different belief must give a different context: the predictor reads the logits.
    other, _ = predictor.context(bank, torch.randn(1, 4, 512))
    assert not torch.allclose(context, other)
    with pytest.raises(ValueError, match='512 T3 classes'):
        predictor.context(bank, torch.randn(1, 4, 64))


def test_v2_zero_init_raw_and_bound_helpers_agree():
    raw = torch.zeros(2, 3, 4, 4)
    assert torch.equal(bounded_residual(raw), raw)


def test_residual_loss_does_not_backprop_into_the_t3_classifier(head):
    torch.manual_seed(0)
    feature = torch.randn(2, 1024, 8, 8)
    classes = torch.tensor([0, 1])
    _, losses = targets(feature, classes, head, route_weight=0.)
    head.zero_grad(set_to_none=True)
    losses['loss_cad_residual'].backward()
    gradient = head.t3_classifier.weight.grad
    assert gradient is None or torch.count_nonzero(gradient) == 0
    # The residual branch itself is still trained end to end.
    assert head.residual_predictor.final.weight.grad is not None
    assert torch.count_nonzero(head.residual_predictor.final.weight.grad) > 0


def test_route_loss_still_trains_the_t3_classifier(head):
    torch.manual_seed(0)
    feature = torch.randn(2, 1024, 8, 8)
    classes = torch.tensor([0, 1])
    _, losses = targets(feature, classes, head, route_weight=1.)
    head.zero_grad(set_to_none=True)
    (losses['loss_cad_t1'] + losses['loss_cad_t2'] + losses['loss_cad_t3']).backward()
    gradient = head.t3_classifier.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_soft_t3_residual_predictor_is_zero_initialized_and_shaped():
    predictor = SoftT3ResidualPredictor(token_dim=16, context_dim=8)
    image_tokens = torch.randn(2, 5, 16)
    bank = torch.randn(2, 512, 16)
    logits = torch.randn(2, 5, 512)
    raw, context, probabilities = predictor(image_tokens, bank, logits)
    assert raw.shape == (2, 5, 3) and context.shape == (2, 5, 8)
    assert torch.count_nonzero(raw) == 0
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(2, 5), atol=1e-5, rtol=1e-5)
