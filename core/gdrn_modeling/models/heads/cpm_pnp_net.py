"""Correspondence-aware low-order moment pose head.

This module intentionally contains only the pose-head mathematics.  Factory
registration and full-model wiring live in the existing GDRN integration
modules so this head remains independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class RegionWeighting:
    """Soft region weights and their visible-support mass."""

    weights: torch.Tensor
    region_mass: torch.Tensor
    coverage: torch.Tensor
    valid: torch.Tensor
    total_effective_mass: torch.Tensor


@dataclass(frozen=True)
class EffectiveSupportQC:
    """Non-learned diagnostics for softmax-tail region support."""

    effective_sample_size: torch.Tensor
    max_normalized_weight: torch.Tensor


@dataclass(frozen=True)
class MomentEncoding:
    """Raw and deterministically rescaled CPM descriptors."""

    raw_descriptor: torch.Tensor
    scaled_descriptor: torch.Tensor
    weighting: RegionWeighting


def compute_region_weighting(
    region_posterior: torch.Tensor,
    visible_mask: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> RegionWeighting:
    """Build visible-support-normalized weights for each soft region.

    The region posterior is expected to be a foreground-channel softmax.  The
    visible mask is sanitized because the legacy L1 mask conversion can emit
    NaN for a spatially constant prediction.
    """

    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    if region_posterior.ndim != 4:
        raise ValueError(
            "region_posterior must have shape BxKxHxW, got "
            f"{tuple(region_posterior.shape)}"
        )
    if visible_mask.ndim != 4 or visible_mask.shape[1] != 1:
        raise ValueError(
            "visible_mask must have shape Bx1xHxW, got "
            f"{tuple(visible_mask.shape)}"
        )
    if (
        region_posterior.shape[0] != visible_mask.shape[0]
        or region_posterior.shape[2:] != visible_mask.shape[2:]
    ):
        raise ValueError("region_posterior and visible_mask spatial shapes differ")
    if not bool(torch.isfinite(region_posterior).all()):
        raise ValueError("region_posterior contains non-finite values")
    if bool((region_posterior < -1e-7).any()):
        raise ValueError("region_posterior contains negative probabilities")

    posterior_sum = region_posterior.sum(dim=1)
    if not bool(
        torch.allclose(
            posterior_sum,
            torch.ones_like(posterior_sum),
            atol=1e-3,
            rtol=1e-3,
        )
    ):
        raise ValueError("region_posterior must sum to one across regions")

    mask = torch.nan_to_num(visible_mask, nan=0.0, posinf=1.0, neginf=0.0)
    mask = mask.clamp(min=0.0, max=1.0).flatten(2).squeeze(1)
    posterior = region_posterior.flatten(2)

    effective_contribution = posterior * mask.unsqueeze(1)
    region_mass = effective_contribution.sum(dim=-1)
    valid = region_mass > eps
    effective_mass = torch.where(valid, region_mass, torch.zeros_like(region_mass))
    total_effective_mass = effective_mass.sum(dim=1, keepdim=True)

    coverage = torch.where(
        total_effective_mass > eps,
        effective_mass / total_effective_mass.clamp_min(eps),
        torch.zeros_like(effective_mass),
    )
    weights = torch.where(
        valid.unsqueeze(-1),
        effective_contribution / region_mass.clamp_min(eps).unsqueeze(-1),
        torch.zeros_like(effective_contribution),
    )
    return RegionWeighting(
        weights=weights,
        region_mass=region_mass,
        coverage=coverage,
        valid=valid,
        total_effective_mass=total_effective_mass,
    )


def compute_effective_support_qc(weighting: RegionWeighting) -> EffectiveSupportQC:
    """Compute FP64 Kish effective sample size and maximum pixel weight.

    These values are audit-only.  They are not fed to the decoder and do not
    change the scientific variables of the pose head.
    """

    weights = weighting.weights.to(dtype=torch.float64)
    valid = weighting.valid
    squared_weight_sum = weights.square().sum(dim=-1)
    effective_sample_size = torch.where(
        valid,
        squared_weight_sum.clamp_min(torch.finfo(torch.float64).tiny).reciprocal(),
        torch.zeros_like(squared_weight_sum),
    )
    effective_sample_size = effective_sample_size.clamp(
        min=0.0, max=float(weights.shape[-1])
    )
    max_normalized_weight = torch.where(
        valid,
        weights.max(dim=-1).values,
        torch.zeros_like(squared_weight_sum),
    )
    return EffectiveSupportQC(
        effective_sample_size=effective_sample_size,
        max_normalized_weight=max_normalized_weight,
    )


def compute_region_moment_descriptor(
    xyz: torch.Tensor,
    roi_2d: torch.Tensor,
    weighting: RegionWeighting,
    *,
    use_cross_covariance: bool = True,
) -> torch.Tensor:
    """Return BxKx21 population-moment descriptors.

    The descriptor layout is coverage, mu_X, mu_U, upper(C_XX),
    upper(C_UU), and row-major(C_XU).
    """

    if xyz.ndim != 4 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape Bx3xHxW, got {tuple(xyz.shape)}")
    if roi_2d.ndim != 4 or roi_2d.shape[1] != 2:
        raise ValueError(
            f"roi_2d must have shape Bx2xHxW, got {tuple(roi_2d.shape)}"
        )
    if xyz.shape[0] != roi_2d.shape[0] or xyz.shape[2:] != roi_2d.shape[2:]:
        raise ValueError("xyz and roi_2d shapes differ")
    if weighting.weights.shape[0] != xyz.shape[0]:
        raise ValueError("weighting and coordinate batch sizes differ")
    if weighting.weights.shape[-1] != xyz.shape[2] * xyz.shape[3]:
        raise ValueError("weighting and coordinate spatial sizes differ")
    if not bool(torch.isfinite(xyz).all()) or not bool(torch.isfinite(roi_2d).all()):
        raise ValueError("xyz or roi_2d contains non-finite values")

    xyz_points = xyz.flatten(2).transpose(1, 2)
    roi_points = roi_2d.flatten(2).transpose(1, 2)
    weights = weighting.weights

    mean_xyz = torch.einsum("bkn,bnd->bkd", weights, xyz_points)
    mean_roi = torch.einsum("bkn,bnd->bkd", weights, roi_points)

    second_xx = torch.einsum(
        "bkn,bni,bnj->bkij", weights, xyz_points, xyz_points
    )
    second_uu = torch.einsum(
        "bkn,bni,bnj->bkij", weights, roi_points, roi_points
    )
    second_xu = torch.einsum(
        "bkn,bni,bnj->bkij", weights, xyz_points, roi_points
    )

    covariance_xx = second_xx - mean_xyz.unsqueeze(-1) * mean_xyz.unsqueeze(-2)
    covariance_uu = second_uu - mean_roi.unsqueeze(-1) * mean_roi.unsqueeze(-2)
    covariance_xu = second_xu - mean_xyz.unsqueeze(-1) * mean_roi.unsqueeze(-2)
    covariance_xx = 0.5 * (covariance_xx + covariance_xx.transpose(-1, -2))
    covariance_uu = 0.5 * (covariance_uu + covariance_uu.transpose(-1, -2))

    valid_matrix = weighting.valid.unsqueeze(-1).unsqueeze(-1)
    covariance_xx = torch.where(
        valid_matrix, covariance_xx, torch.zeros_like(covariance_xx)
    )
    covariance_uu = torch.where(
        valid_matrix, covariance_uu, torch.zeros_like(covariance_uu)
    )
    covariance_xu = torch.where(
        valid_matrix, covariance_xu, torch.zeros_like(covariance_xu)
    )

    xx_upper = covariance_xx[..., (0, 0, 0, 1, 1, 2), (0, 1, 2, 1, 2, 2)]
    uu_upper = covariance_uu[..., (0, 0, 1), (0, 1, 1)]
    xu_flat = covariance_xu.flatten(start_dim=-2)
    if not use_cross_covariance:
        xu_flat = torch.zeros_like(xu_flat)

    descriptor = torch.cat(
        [
            weighting.coverage.unsqueeze(-1),
            mean_xyz,
            mean_roi,
            xx_upper,
            uu_upper,
            xu_flat,
        ],
        dim=-1,
    )
    if descriptor.shape[-1] != 21:
        raise RuntimeError(f"unexpected moment descriptor size {descriptor.shape[-1]}")
    return descriptor


class CorrespondenceAwareMomentPnPNet(nn.Module):
    """Compact direct pose regressor over fixed low-order joint moments."""

    _LEGACY_CONV_PNP_PREFIXES = ("features.", "fc1.", "fc2.", "fc_r.", "fc_t.")

    def __init__(
        self,
        *,
        num_regions: int = 64,
        rot_dim: int = 6,
        hidden_dim: int = 512,
        latent_dim: int = 256,
        mask_attention_type: str = "mul",
        denormalize_by_extent: bool = True,
        eps: float = 1e-6,
        moment_scales: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0),
        use_cross_covariance: bool = True,
    ) -> None:
        super().__init__()
        if num_regions <= 0:
            raise ValueError("num_regions must be positive")
        if mask_attention_type not in {"mul", "support"}:
            raise ValueError(
                "CPM requires visible-mask support; expected mask_attention_type "
                f"'mul' or 'support', got {mask_attention_type!r}"
            )
        if eps <= 0:
            raise ValueError("eps must be positive")
        scales = torch.as_tensor(tuple(moment_scales), dtype=torch.float32)
        if scales.shape != (5,) or not bool(torch.isfinite(scales).all()) or bool(
            (scales <= 0).any()
        ):
            raise ValueError("moment_scales must contain five finite positive values")

        self.num_regions = int(num_regions)
        self.mask_attention_type = mask_attention_type
        self.denormalize_by_extent = bool(denormalize_by_extent)
        self.eps = float(eps)
        self.use_cross_covariance = bool(use_cross_covariance)
        self.register_buffer("moment_scales", scales, persistent=True)

        # Keep names disjoint from legacy ConvPnPNet.  The official checkpoint
        # must initialize no part of the new pose head by accidental key match.
        self.moment_fc1 = nn.Linear(self.num_regions * 21, hidden_dim)
        self.moment_fc2 = nn.Linear(hidden_dim, latent_dim)
        self.act = nn.GELU()
        self.rotation_head = nn.Linear(latent_dim, rot_dim)
        self.translation_head = nn.Linear(latent_dim, 3)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for layer in (self.moment_fc1, self.moment_fc2):
            nn.init.normal_(layer.weight, std=0.001)
            nn.init.zeros_(layer.bias)
        for layer in (self.rotation_head, self.translation_head):
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Discard only explicitly recognized official ConvPnP tensors.

        New CPM checkpoints use disjoint parameter names and load normally.
        Unrecognized keys are deliberately left for PyTorch to report.
        """

        for key in list(state_dict):
            if not key.startswith(prefix):
                continue
            local_key = key[len(prefix) :]
            if local_key.startswith(self._LEGACY_CONV_PNP_PREFIXES):
                state_dict.pop(key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _apply_moment_scales(self, raw_descriptor: torch.Tensor) -> torch.Tensor:
        coverage = raw_descriptor[..., 0:1]
        mean_xyz = raw_descriptor[..., 1:4] / self.moment_scales[0]
        mean_roi = raw_descriptor[..., 4:6] / self.moment_scales[1]
        covariance_xx = raw_descriptor[..., 6:12] / self.moment_scales[2]
        covariance_uu = raw_descriptor[..., 12:15] / self.moment_scales[3]
        covariance_xu = raw_descriptor[..., 15:21] / self.moment_scales[4]
        return torch.cat(
            [
                coverage,
                mean_xyz,
                mean_roi,
                covariance_xx,
                covariance_uu,
                covariance_xu,
            ],
            dim=-1,
        )

    def encode_moments(
        self,
        coor_feat: torch.Tensor,
        *,
        region: torch.Tensor,
        extents: torch.Tensor | None,
        mask_attention: torch.Tensor,
    ) -> MomentEncoding:
        """Encode the effective CPM input without running the MLP decoder."""

        if coor_feat.ndim != 4 or coor_feat.shape[1] != 5:
            raise ValueError(
                "CPM expects XYZ3 + absolute ROI2D2, got "
                f"{tuple(coor_feat.shape)}"
            )
        if region.ndim != 4 or region.shape[1] != self.num_regions:
            raise ValueError(
                f"CPM expects {self.num_regions} region channels, got "
                f"{tuple(region.shape)}"
            )
        xyz = coor_feat[:, :3]
        roi_2d = coor_feat[:, 3:5]
        if self.denormalize_by_extent:
            if extents is None or extents.shape != (coor_feat.shape[0], 3):
                raise ValueError("extents must have shape Bx3 when XYZ denormalization is enabled")
            xyz = (xyz - 0.5) * extents.to(dtype=xyz.dtype).view(-1, 3, 1, 1)

        weighting = compute_region_weighting(region, mask_attention, eps=self.eps)
        raw_descriptor = compute_region_moment_descriptor(
            xyz,
            roi_2d,
            weighting,
            use_cross_covariance=self.use_cross_covariance,
        )
        scaled_descriptor = self._apply_moment_scales(raw_descriptor)
        return MomentEncoding(
            raw_descriptor=raw_descriptor,
            scaled_descriptor=scaled_descriptor,
            weighting=weighting,
        )

    def forward(
        self,
        coor_feat: torch.Tensor,
        region: torch.Tensor | None = None,
        extents: torch.Tensor | None = None,
        mask_attention: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if region is None:
            raise ValueError("CPM requires region posterior input")
        if mask_attention is None:
            raise ValueError("CPM requires visible-mask support")
        encoding = self.encode_moments(
            coor_feat,
            region=region,
            extents=extents,
            mask_attention=mask_attention,
        )
        latent = encoding.scaled_descriptor.flatten(start_dim=1)
        latent = self.act(self.moment_fc1(latent))
        latent = self.act(self.moment_fc2(latent))
        return self.rotation_head(latent), self.translation_head(latent)
