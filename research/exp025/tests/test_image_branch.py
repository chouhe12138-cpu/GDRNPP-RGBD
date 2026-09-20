"""EXP025 image branch: four self-attention stages that rewrite the feature they hand on."""
import torch

from core.gdrn_modeling.models.heads.hierarchical_cad_attention_head import IMAGE_STAGES

# The `head` fixture lives in conftest.py, shared with the other diagnostic tests.
STAGE_FEATURES = ((512, 8), (256, 16), (128, 32), (64, 64))


def stage_inputs(head, feature, classes):
    """Inputs each stage and transition actually receives, in call order."""
    seen = {'stages': [], 'transitions': []}
    handles = [module.register_forward_pre_hook(
        lambda _module, inputs, key=key: seen[key].append(inputs[0].detach().clone()))
        for key, modules in (('stages', head.stages), ('transitions', head.transitions))
        for module in modules]
    try:
        head.encode(feature, classes)
    finally:
        for handle in handles:
            handle.remove()
    return seen


def test_image_stage_shapes(head):
    feature = head.input_adapter(torch.randn(2, 1024, 8, 8))
    assert feature.shape == (2, 512, 8, 8)
    for index, (stage, (_, resolution)) in enumerate(zip(head.stages, STAGE_FEATURES)):
        feature, tokens = stage(feature)
        assert tokens.shape == (2, resolution ** 2, head.token_dim)
        if index == len(head.stages) - 1:
            # Nothing consumes the last stage's feature, so it is not projected back.
            assert feature is None and stage.to_feature is None
        else:
            feature = head.transitions[index](feature)
            channels, resolution = STAGE_FEATURES[index + 1]
            assert feature.shape == (2, channels, resolution, resolution)


def test_stage_attention_is_written_back_into_the_feature(head):
    feature = head.input_adapter(torch.randn(1, 1024, 8, 8))
    for index, stage in enumerate(head.stages):
        updated, tokens = stage(feature)
        # Attention must change the tokens, not just recompute the projection.
        assert not torch.allclose(tokens, stage.to_tokens(feature).flatten(2).transpose(1, 2))
        if index < len(head.transitions):
            # Every stage that hands a feature on writes its attention back into it.
            assert not torch.allclose(updated, feature)
            feature = head.transitions[index](updated)


def test_perturbing_a_stage_attention_changes_the_next_stage_input(head):
    classes = torch.tensor([0])
    feature = torch.randn(1, 1024, 8, 8)
    reference = stage_inputs(head, feature, classes)
    assert [len(reference[key]) for key in ('stages', 'transitions')] == [4, 3]
    for index in range(len(head.stages) - 1):
        with torch.no_grad():
            for parameter in head.stages[index].attention.parameters():
                parameter.add_(0.05)
        perturbed = stage_inputs(head, feature, classes)
        with torch.no_grad():
            for parameter in head.stages[index].attention.parameters():
                parameter.sub_(0.05)
        assert not torch.allclose(reference['stages'][index + 1], perturbed['stages'][index + 1]), \
            f'stage {index} self-attention never reached stage {index + 1}'
        assert not torch.allclose(reference['transitions'][index], perturbed['transitions'][index])
    # The rollback above left the parameters as they were (float add/sub is not exact).
    torch.testing.assert_close(reference['stages'][3], stage_inputs(head, feature, classes)['stages'][3])


def test_cross_attention_reads_t0_t1_t2_t3_in_order(head):
    classes = torch.tensor([0, 1])
    banks = head.token_banks(classes)
    assert [bank.shape[1] for bank in banks] == [1, 8, 64, 512]
    calls = []
    handles = [block.register_forward_hook(
        lambda _module, inputs, output: calls.append((inputs[0].detach(), inputs[1].detach(),
                                                      output.detach())))
        for block in head.cross_attention]
    try:
        tokens, _ = head.encode(torch.randn(2, 1024, 8, 8), classes)
    finally:
        for handle in handles:
            handle.remove()
    assert [bank.shape[1] for _, bank, _ in calls] == [1, 8, 64, 512]
    for index, (query, bank, _) in enumerate(calls):
        torch.testing.assert_close(bank, banks[index])
        if index:
            # Each block consumes the previous block's output: one ladder, not four taps.
            torch.testing.assert_close(query, calls[index - 1][2])
    torch.testing.assert_close(tokens, calls[-1][2])


def test_prediction_shapes_and_per_stage_gradients(head):
    head.zero_grad(set_to_none=True)
    feature = torch.randn(2, 1024, 8, 8, requires_grad=True)
    classes = torch.tensor([0, 1])
    prediction = head(feature, classes)
    assert prediction['t3_logits'].shape == (2, 512, 64, 64)
    assert prediction['residual'].shape == (2, 3, 64, 64)
    assert prediction['mask_logit'].shape == (2, 1, 64, 64)
    losses, _ = head.loss(prediction, classes, torch.rand(2, 3, 64, 64),
                          torch.ones(2, 1, 64, 64))
    sum(losses.values()).backward()
    for index, stage in enumerate(head.stages):
        for name, parameter in stage.named_parameters():
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0, \
                f'stage {index}.{name} received no gradient'
    assert feature.grad is not None and torch.isfinite(feature.grad).all()


def test_stage_geometry(head):
    assert IMAGE_STAGES[0][0] == 512 and IMAGE_STAGES[-1][1] == 64
    assert [attention for _, _, attention in IMAGE_STAGES] == ['global', 'global', 'window', 'window']
    assert [stage.resolution for stage in head.stages] == [8, 16, 32, 64]
    assert [bool(stage.global_attention) for stage in head.stages] == [True, True, False, False]
