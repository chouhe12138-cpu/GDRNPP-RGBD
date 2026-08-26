"""EXP013 pose heads with an independent XYZ/ROI2D geometry path.

The three public classes implement one controlled progression:

* EXP013A adds a Region-free geometry residual to the EXP012 main stream.
* EXP013B adds masked 3x3 local geometry attention inside that residual.
* EXP013C reuses A's correspondence/geometry encoders and gives rotation and
  translation independent aggregation, geometry fusion scales, and latents.
"""

from __future__ import annotations

import torch
from mmcv.cnn import normal_init
from torch import nn
from torch.nn import functional as F

from .hierarchical_corr_pnp_net import (
    HierarchicalCorrespondencePnPNet,
    _ConvNormAct,
    _LocalRelationBlock,
)


class _LocalGeometryAttention(nn.Module):
    """Nine-neighbour attention driven only by relative 2D/3D geometry."""

    def __init__(self, channels: int, relation_channels: int = 16) -> None:
        super().__init__()
        self.relation_mlp = nn.Sequential(
            nn.Linear(6, relation_channels),
            nn.GELU(),
            nn.Linear(relation_channels, 1),
        )
        self.value = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.output = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.last_weights: torch.Tensor | None = None

    @staticmethod
    def _unfold_neighbours(x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        return F.unfold(x, kernel_size=3, padding=1).view(
            batch, channels, 9, height, width
        )

    def forward(
        self,
        feature: torch.Tensor,
        metric_xyz_roi2d: torch.Tensor,
        support: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if feature.shape[2:] != metric_xyz_roi2d.shape[2:]:
            raise ValueError("feature and geometry grids must have equal size")
        if support.shape != feature[:, :1].shape:
            raise ValueError("support must have shape Bx1xHxW")

        neighbours = self._unfold_neighbours(metric_xyz_roi2d)
        center = metric_xyz_roi2d.unsqueeze(2)
        delta = neighbours - center
        # Public relation order: du, dv, dx, dy, dz, ||dXYZ||.
        relation = torch.cat(
            [
                delta[:, 3:5],
                delta[:, 0:3],
                torch.linalg.vector_norm(delta[:, 0:3], dim=1, keepdim=True),
            ],
            dim=1,
        )
        relation = relation.permute(0, 3, 4, 2, 1)
        logits = self.relation_mlp(relation).squeeze(-1)

        valid_neighbour = self._unfold_neighbours(support).squeeze(1) > 0
        valid_center = support[:, 0] > 0
        valid = valid_neighbour & valid_center.unsqueeze(1)
        logits = logits.masked_fill(~valid.permute(0, 2, 3, 1), -1.0e4)
        weights = torch.softmax(logits, dim=-1).permute(0, 3, 1, 2)
        weights = weights * valid.to(weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-12)

        values = self._unfold_neighbours(self.value(feature))
        update = (values * weights.unsqueeze(1)).sum(dim=2)
        update = self.output(update) * valid_center.unsqueeze(1).to(update.dtype)
        self.last_weights = weights.detach()
        return update, weights


class XYZResidualBypassPnPNet(HierarchicalCorrespondencePnPNet):
    """EXP013A: EXP012 main path plus an independent geometry residual."""

    def __init__(
        self,
        *,
        geometry_channels: tuple[int, int, int] = (32, 48, 32),
        geometry_grid_size: int = 8,
        geometry_scale_init: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if len(geometry_channels) != 3 or min(geometry_channels) <= 0:
            raise ValueError("geometry_channels must contain three positive widths")
        if geometry_grid_size <= 0:
            raise ValueError("geometry_grid_size must be positive")
        c_fine, c_mid, c_high = geometry_channels
        self.geometry_grid_size = int(geometry_grid_size)
        self.geometry_input_projection = _ConvNormAct(5, c_fine)
        self.geometry_local_fine = _LocalRelationBlock(c_fine)
        self.geometry_downsample_mid = _ConvNormAct(c_fine, c_mid, stride=2)
        self.geometry_local_mid = _LocalRelationBlock(c_mid)
        self.geometry_downsample_high = _ConvNormAct(c_mid, c_high, stride=2)
        self.geometry_local_high = _LocalRelationBlock(c_high)
        self.geometry_projection = nn.Linear(c_high * self.geometry_grid_size**2, 256)
        self.geometry_scale = nn.Parameter(
            torch.tensor([float(geometry_scale_init)], dtype=torch.float32)
        )
        self._initialize_exp013_weights()

    def _initialize_exp013_weights(self) -> None:
        modules = (
            self.geometry_input_projection,
            self.geometry_local_fine,
            self.geometry_downsample_mid,
            self.geometry_local_mid,
            self.geometry_downsample_high,
            self.geometry_local_high,
            self.geometry_projection,
        )
        for root in modules:
            for module in root.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    normal_init(module, std=0.001)

    def _validate_and_mask_inputs(
        self,
        coor_feat: torch.Tensor,
        region: torch.Tensor | None,
        extents: torch.Tensor | None,
        mask_attention: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        metric = self._prepare_inputs(coor_feat, extents=extents)
        if region is None:
            raise ValueError("region is required for the EXP012-compatible main path")
        if region.ndim != 4 or region.shape[1] != self.num_regions:
            raise ValueError(
                f"region must have shape Bx{self.num_regions}xHxW, got {tuple(region.shape)}"
            )
        if region.shape[0] != metric.shape[0] or region.shape[2:] != metric.shape[2:]:
            raise ValueError("region and coor_feat spatial shapes differ")
        if not bool(torch.isfinite(region).all()):
            raise ValueError("region contains non-finite values")
        if (
            mask_attention is None
            or mask_attention.ndim != 4
            or mask_attention.shape[1] != 1
        ):
            raise ValueError("mask_attention must have shape Bx1xHxW")
        if (
            mask_attention.shape[0] != metric.shape[0]
            or mask_attention.shape[2:] != metric.shape[2:]
        ):
            raise ValueError("mask_attention and coor_feat spatial shapes differ")
        support = self._sanitize_mask(mask_attention)
        # Both streams are masked before their first learned spatial operation.
        return metric * support, region * support, support

    def _encode_main_features(
        self, metric: torch.Tensor, region: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fine = self.input_projection(metric)
        fine = fine + self.region_scale.to(fine.dtype) * self.region_projection(region)
        fine = self.local_fine(fine)
        mid = self.local_mid(self.downsample_mid(fine))
        high = self.local_high(self.downsample_high(mid))
        return fine, mid, high

    def _encode_main(
        self, metric: torch.Tensor, region: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fine, mid, high = self._encode_main_features(metric, region)
        main_descriptor = torch.cat(
            [
                fine.mean(dim=(2, 3)),
                mid.mean(dim=(2, 3)),
                F.adaptive_avg_pool2d(
                    high, (self.coarse_grid_size, self.coarse_grid_size)
                ).flatten(start_dim=1),
            ],
            dim=1,
        )
        latent = self.pose_act(self.pose_fc1(main_descriptor))
        latent = self.pose_act(self.pose_fc2(latent))
        return self.pose_dropout(latent), fine, mid, high

    def _refine_geometry(
        self,
        high: torch.Tensor,
        metric_16: torch.Tensor,
        support_16: torch.Tensor,
    ) -> torch.Tensor:
        del metric_16, support_16
        return high

    def _encode_geometry_grid(
        self, metric: torch.Tensor, support: torch.Tensor
    ) -> torch.Tensor:
        # This method has no Region argument by design; tests enforce that the
        # bypass remains usable and invariant when Region changes.
        fine = self.geometry_local_fine(self.geometry_input_projection(metric))
        mid = self.geometry_local_mid(self.geometry_downsample_mid(fine))
        high = self.geometry_local_high(self.geometry_downsample_high(mid))
        metric_16 = F.interpolate(metric, size=high.shape[2:], mode="nearest")
        support_16 = F.interpolate(support, size=high.shape[2:], mode="nearest")
        high = self._refine_geometry(high, metric_16, support_16)
        grid = F.adaptive_avg_pool2d(
            high, (self.geometry_grid_size, self.geometry_grid_size)
        )
        return grid

    def _encode_geometry(
        self, metric: torch.Tensor, support: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid = self._encode_geometry_grid(metric, support)
        latent = self.pose_act(self.geometry_projection(grid.flatten(start_dim=1)))
        return latent, grid

    def forward(
        self,
        coor_feat: torch.Tensor,
        region: torch.Tensor | None = None,
        extents: torch.Tensor | None = None,
        mask_attention: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        metric, masked_region, support = self._validate_and_mask_inputs(
            coor_feat, region, extents, mask_attention
        )
        main_latent, _, _, _ = self._encode_main(metric, masked_region)
        geometry_latent, _ = self._encode_geometry(metric, support)
        latent = (
            main_latent + self.geometry_scale.to(main_latent.dtype) * geometry_latent
        )
        return self.pose_rotation(latent), self.pose_translation(latent)


class GeometryAttentionResidualPnPNet(XYZResidualBypassPnPNet):
    """EXP013B: A plus masked local geometric attention at 16x16."""

    def __init__(self, *, attention_scale_init: float = 0.1, **kwargs) -> None:
        super().__init__(**kwargs)
        c_high = self.geometry_projection.in_features // self.geometry_grid_size**2
        self.geometry_attention = _LocalGeometryAttention(c_high)
        self.attention_scale = nn.Parameter(
            torch.tensor([float(attention_scale_init)], dtype=torch.float32)
        )
        for module in self.geometry_attention.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                normal_init(module, std=0.001)

    def _refine_geometry(
        self,
        high: torch.Tensor,
        metric_16: torch.Tensor,
        support_16: torch.Tensor,
    ) -> torch.Tensor:
        update, _ = self.geometry_attention(high, metric_16, support_16)
        return high + self.attention_scale.to(high.dtype) * update


class RTDecoupledGeometryPnPNet(XYZResidualBypassPnPNet):
    """EXP013C: A frontend with independent rotation/translation aggregation."""

    def __init__(
        self,
        *,
        extent_dim: int = 3,
        geometry_scale_r_init: float = 0.1,
        geometry_scale_t_init: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if extent_dim != 3:
            raise ValueError("EXP013C currently requires three object extents")
        main_high_channels = self.downsample_high.conv.out_channels
        geometry_high_channels = self.geometry_downsample_high.conv.out_channels
        rotation_in = (
            main_high_channels * self.coarse_grid_size**2
            + geometry_high_channels * self.geometry_grid_size**2
        )
        translation_in = (
            self.input_projection.conv.out_channels
            + self.downsample_mid.conv.out_channels
            + main_high_channels
            + 2 * geometry_high_channels
            + 4
            + extent_dim
        )

        # Remove the shared late decoder inherited only for frontend reuse.
        del self.pose_fc1
        del self.pose_fc2
        del self.pose_rotation
        del self.pose_translation
        del self.geometry_projection
        del self.geometry_scale
        self.geometry_scale_r = nn.Parameter(
            torch.tensor([float(geometry_scale_r_init)], dtype=torch.float32)
        )
        self.geometry_scale_t = nn.Parameter(
            torch.tensor([float(geometry_scale_t_init)], dtype=torch.float32)
        )
        self.rotation_fc1 = nn.Linear(rotation_in, 256)
        self.rotation_fc2 = nn.Linear(256, 256)
        self.rotation_output = nn.Linear(256, self.rot_dim)
        self.translation_fc1 = nn.Linear(translation_in, 256)
        self.translation_fc2 = nn.Linear(256, 256)
        self.translation_output = nn.Linear(256, 3)
        for module in (
            self.rotation_fc1,
            self.rotation_fc2,
            self.translation_fc1,
            self.translation_fc2,
        ):
            normal_init(module, std=0.001)
        normal_init(self.rotation_output, std=0.01)
        normal_init(self.translation_output, std=0.01)

    @staticmethod
    def _roi_support_stats(metric: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        roi = metric[:, 3:5]
        weight = support
        denom = weight.sum(dim=(2, 3)).clamp_min(1.0e-6)
        center = (roi * weight).sum(dim=(2, 3)) / denom
        variance = ((roi - center[:, :, None, None]) ** 2 * weight).sum(
            dim=(2, 3)
        ) / denom
        return torch.cat([center, variance.clamp_min(0).sqrt()], dim=1)

    def forward(
        self,
        coor_feat: torch.Tensor,
        region: torch.Tensor | None = None,
        extents: torch.Tensor | None = None,
        mask_attention: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        metric, masked_region, support = self._validate_and_mask_inputs(
            coor_feat, region, extents, mask_attention
        )
        fine, mid, high = self._encode_main_features(metric, masked_region)
        geometry_grid = self._encode_geometry_grid(metric, support)

        rotation_geometry = self.geometry_scale_r.to(geometry_grid.dtype) * geometry_grid
        rotation_descriptor = torch.cat(
            [
                F.adaptive_avg_pool2d(
                    high, (self.coarse_grid_size, self.coarse_grid_size)
                ).flatten(start_dim=1),
                rotation_geometry.flatten(start_dim=1),
            ],
            dim=1,
        )
        geometry_mean = geometry_grid.mean(dim=(2, 3))
        geometry_std = geometry_grid.var(dim=(2, 3), unbiased=False).clamp_min(0).sqrt()
        geometry_stats = torch.cat([geometry_mean, geometry_std], dim=1)
        geometry_stats = self.geometry_scale_t.to(geometry_grid.dtype) * geometry_stats
        translation_descriptor = torch.cat(
            [
                fine.mean(dim=(2, 3)),
                mid.mean(dim=(2, 3)),
                high.mean(dim=(2, 3)),
                geometry_stats,
                self._roi_support_stats(metric, support),
                extents.to(device=metric.device, dtype=metric.dtype),
            ],
            dim=1,
        )

        rotation_latent = self.pose_act(self.rotation_fc1(rotation_descriptor))
        rotation_latent = self.pose_dropout(
            self.pose_act(self.rotation_fc2(rotation_latent))
        )
        translation_latent = self.pose_act(self.translation_fc1(translation_descriptor))
        translation_latent = self.pose_dropout(
            self.pose_act(self.translation_fc2(translation_latent))
        )
        return self.rotation_output(rotation_latent), self.translation_output(
            translation_latent
        )
