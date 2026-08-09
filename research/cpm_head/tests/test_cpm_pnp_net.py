from __future__ import annotations

import torch

from core.gdrn_modeling.models.heads.cpm_pnp_net import (
    CorrespondenceAwareMomentPnPNet,
    compute_effective_support_qc,
    compute_region_moment_descriptor,
    compute_region_weighting,
)


def make_inputs(
    *, batch: int = 2, regions: int = 64, height: int = 4, width: int = 4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    coor = torch.rand(batch, 5, height, width)
    region = torch.softmax(torch.randn(batch, regions, height, width), dim=1)
    mask = torch.rand(batch, 1, height, width)
    extents = torch.rand(batch, 3) + 0.1
    return coor, region, mask, extents


def test_descriptor_shape_coverage_and_parameter_budget() -> None:
    coor, region, mask, extents = make_inputs()
    model = CorrespondenceAwareMomentPnPNet()
    encoding = model.encode_moments(
        coor, region=region, extents=extents, mask_attention=mask
    )
    assert encoding.raw_descriptor.shape == (2, 64, 21)
    assert encoding.scaled_descriptor.shape == (2, 64, 21)
    assert torch.allclose(
        encoding.weighting.coverage.sum(dim=1), torch.ones(2), atol=1e-6
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 822_281


def test_all_zero_mask_produces_zero_descriptor_and_qc() -> None:
    coor, region, mask, extents = make_inputs(batch=1)
    mask.zero_()
    model = CorrespondenceAwareMomentPnPNet()
    encoding = model.encode_moments(
        coor, region=region, extents=extents, mask_attention=mask
    )
    qc = compute_effective_support_qc(encoding.weighting)
    assert torch.count_nonzero(encoding.raw_descriptor) == 0
    assert torch.count_nonzero(qc.effective_sample_size) == 0
    assert torch.count_nonzero(qc.max_normalized_weight) == 0


def test_empty_regions_and_single_pixel_covariance() -> None:
    xyz = torch.tensor([[[[0.2, 0.7]], [[0.3, 0.1]], [[0.4, 0.8]]]])
    roi = torch.tensor([[[[0.1, 0.9]], [[0.2, 0.8]]]])
    region = torch.zeros(1, 3, 1, 2)
    region[:, 0] = 1.0
    mask = torch.tensor([[[[1.0, 0.0]]]])
    weighting = compute_region_weighting(region, mask)
    descriptor = compute_region_moment_descriptor(xyz, roi, weighting)
    assert weighting.valid.tolist() == [[True, False, False]]
    assert torch.allclose(weighting.coverage, torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.allclose(descriptor[0, 0, 6:], torch.zeros(15), atol=1e-7)
    assert torch.count_nonzero(descriptor[0, 1:]) == 0


def test_effective_sample_size_known_cases() -> None:
    region = torch.ones(1, 1, 1, 4)
    uniform = compute_region_weighting(region, torch.ones(1, 1, 1, 4))
    uniform_qc = compute_effective_support_qc(uniform)
    assert torch.allclose(
        uniform_qc.effective_sample_size, torch.tensor([[4.0]], dtype=torch.float64)
    )
    assert torch.allclose(
        uniform_qc.max_normalized_weight, torch.tensor([[0.25]], dtype=torch.float64)
    )

    concentrated_mask = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
    concentrated = compute_region_weighting(region, concentrated_mask)
    concentrated_qc = compute_effective_support_qc(concentrated)
    assert torch.allclose(
        concentrated_qc.effective_sample_size,
        torch.tensor([[1.0]], dtype=torch.float64),
    )
    assert torch.allclose(
        concentrated_qc.max_normalized_weight,
        torch.tensor([[1.0]], dtype=torch.float64),
    )


def test_softmax_tail_can_be_valid_with_tiny_coverage() -> None:
    pixels = 100
    dominant = 0.9999
    tail = 1.0 - dominant
    region = torch.empty(1, 2, 1, pixels)
    region[:, 0] = dominant
    region[:, 1] = tail
    weighting = compute_region_weighting(region, torch.ones(1, 1, 1, pixels))
    qc = compute_effective_support_qc(weighting)
    assert weighting.valid.tolist() == [[True, True]]
    assert 0.0 < weighting.coverage[0, 1] < 1e-3
    assert torch.allclose(
        qc.effective_sample_size[0],
        torch.tensor([100.0, 100.0], dtype=torch.float64),
        atol=1e-5,
    )


def test_concentrated_softmax_tail_has_effective_sample_size_near_one() -> None:
    pixels = 100
    tail = torch.full((pixels,), 1e-12)
    tail[0] = 1e-4
    region = torch.stack([1.0 - tail, tail], dim=0).reshape(1, 2, 1, pixels)
    weighting = compute_region_weighting(region, torch.ones(1, 1, 1, pixels))
    qc = compute_effective_support_qc(weighting)
    assert weighting.valid.tolist() == [[True, True]]
    assert 0.0 < weighting.coverage[0, 1] < 1e-5
    assert 1.0 <= qc.effective_sample_size[0, 1] < 1.001
    assert qc.max_normalized_weight[0, 1] > 0.999


def test_complete_tuple_permutation_is_invariant() -> None:
    coor, region, mask, extents = make_inputs(batch=1, height=3, width=3)
    model = CorrespondenceAwareMomentPnPNet()
    baseline = model.encode_moments(
        coor, region=region, extents=extents, mask_attention=mask
    ).raw_descriptor
    permutation = torch.tensor([8, 1, 6, 3, 4, 5, 2, 7, 0])

    def permute(tensor: torch.Tensor) -> torch.Tensor:
        shape = tensor.shape
        return tensor.flatten(2)[..., permutation].reshape(shape)

    changed = model.encode_moments(
        permute(coor),
        region=permute(region),
        extents=extents,
        mask_attention=permute(mask),
    ).raw_descriptor
    assert torch.allclose(baseline, changed, atol=2e-6, rtol=1e-5)


def test_roi_only_permutation_changes_cross_covariance() -> None:
    xyz = torch.tensor(
        [[[[0.0, 1.0, 2.0, 3.0]], [[0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0]]]]
    )
    roi = torch.tensor(
        [[[[0.0, 0.2, 0.7, 1.0]], [[0.0, 0.1, 0.4, 0.9]]]]
    )
    region = torch.ones(1, 1, 1, 4)
    mask = torch.ones(1, 1, 1, 4)
    weighting = compute_region_weighting(region, mask)
    baseline = compute_region_moment_descriptor(xyz, roi, weighting)
    changed = compute_region_moment_descriptor(xyz, roi.flip(-1), weighting)
    assert not torch.allclose(baseline[..., 15:21], changed[..., 15:21])


def test_no_cross_keeps_first_order_region_pairing() -> None:
    coor, region, mask, extents = make_inputs(batch=1)
    full = CorrespondenceAwareMomentPnPNet(use_cross_covariance=True)
    no_cross = CorrespondenceAwareMomentPnPNet(use_cross_covariance=False)
    full_descriptor = full.encode_moments(
        coor, region=region, extents=extents, mask_attention=mask
    ).raw_descriptor
    no_cross_descriptor = no_cross.encode_moments(
        coor, region=region, extents=extents, mask_attention=mask
    ).raw_descriptor
    assert torch.allclose(full_descriptor[..., :15], no_cross_descriptor[..., :15])
    assert torch.count_nonzero(no_cross_descriptor[..., 15:21]) == 0


def test_forward_backward_is_finite_and_does_not_mutate_input() -> None:
    coor, region, mask, extents = make_inputs()
    original = coor.clone()
    model = CorrespondenceAwareMomentPnPNet()
    rotation, translation = model(coor, region, extents, mask)
    assert rotation.shape == (2, 6)
    assert translation.shape == (2, 3)
    assert bool(torch.isfinite(rotation).all())
    assert bool(torch.isfinite(translation).all())
    (rotation.square().mean() + translation.square().mean()).backward()
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    assert torch.equal(coor, original)


def test_full_resolution_forward_smoke() -> None:
    coor, region, mask, extents = make_inputs(batch=1, height=64, width=64)
    model = CorrespondenceAwareMomentPnPNet()
    with torch.no_grad():
        rotation, translation = model(coor, region, extents, mask)
    assert rotation.shape == (1, 6)
    assert translation.shape == (1, 3)
    assert bool(torch.isfinite(rotation).all())
    assert bool(torch.isfinite(translation).all())


def test_nonfinite_region_is_a_hard_failure() -> None:
    region = torch.ones(1, 1, 1, 1)
    region[0, 0, 0, 0] = float("nan")
    try:
        compute_region_weighting(region, torch.ones(1, 1, 1, 1))
    except ValueError as error:
        assert "non-finite" in str(error)
    else:
        raise AssertionError("non-finite region posterior was silently accepted")


def test_legacy_conv_pnp_keys_are_filtered_but_unknown_keys_are_reported() -> None:
    model = CorrespondenceAwareMomentPnPNet()
    incompatible = model.load_state_dict(
        {
            "fc1.weight": torch.randn(4, 4),
            "fc_r.weight": torch.randn(6, 256),
            "unknown.weight": torch.randn(1),
        },
        strict=False,
    )
    assert "fc1.weight" not in incompatible.unexpected_keys
    assert "fc_r.weight" not in incompatible.unexpected_keys
    assert incompatible.unexpected_keys == ["unknown.weight"]
    assert "moment_fc1.weight" in incompatible.missing_keys
