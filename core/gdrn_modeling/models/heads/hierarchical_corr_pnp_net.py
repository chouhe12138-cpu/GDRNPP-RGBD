"""Hierarchical dense-correspondence pose head.

The head keeps XYZ and ROI-2D values paired at every spatial location, applies
small local relation blocks, and only then performs two levels of spatial
aggregation.  Region predictions are an optional, zero-start auxiliary signal;
they do not define the pooling groups.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from mmcv.cnn import normal_init



def _group_count(channels: int, requested: int = 8) -> int:
    """Return a GroupNorm group count that divides ``channels``."""

    for groups in range(min(requested, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class _LocalRelationBlock(nn.Module):
    """Residual local mixer without global attention or token flattening."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pointwise_in = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm_in = nn.GroupNorm(_group_count(channels), channels)
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.norm_depthwise = nn.GroupNorm(_group_count(channels), channels)
        self.pointwise_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm_out = nn.GroupNorm(_group_count(channels), channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.norm_in(self.pointwise_in(x)))
        x = self.act(self.norm_depthwise(self.depthwise(x)))
        x = self.norm_out(self.pointwise_out(x))
        return self.act(x + residual)


class HierarchicalCorrespondencePnPNet(nn.Module):
    """Dense XYZ/ROI correspondence head with local and hierarchical pooling.

    The public forward signature intentionally matches the existing GDRN pose
    heads: ``coor_feat`` is XYZ+ROI2D (B,5,H,W), ``region`` is an optional
    foreground Region posterior, ``extents`` denormalizes XYZ, and
    ``mask_attention`` supplies visible support.
    """

    _LEGACY_PNP_PREFIXES = (
        "features.",
        "fc1.",
        "fc2.",
        "fc_r.",
        "fc_t.",
        "moment_fc1.",
        "moment_fc2.",
        "rotation_head.",
        "translation_head.",
    )

    def __init__(
        self,
        *,
        num_regions: int = 64,
        rot_dim: int = 6,
        base_channels: int = 64,
        mid_channels: int = 96,
        high_channels: int = 128,
        mask_attention_type: str = "mul",
        denormalize_by_extent: bool = True,
        use_region_aux: bool = True,
        region_aux_dim: int = 16,
        coarse_grid_size: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_regions <= 0:
            raise ValueError("num_regions must be positive")
        if rot_dim <= 0:
            raise ValueError("rot_dim must be positive")
        if mask_attention_type not in {"none", "mul", "support"}:
            raise ValueError(
                "mask_attention_type must be one of 'none', 'mul', or 'support'"
            )
        if min(base_channels, mid_channels, high_channels) <= 0:
            raise ValueError("all channel widths must be positive")
        if use_region_aux and region_aux_dim <= 0:
            raise ValueError("region_aux_dim must be positive when Region is enabled")
        if coarse_grid_size <= 0:
            raise ValueError("coarse_grid_size must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.num_regions = int(num_regions)
        self.rot_dim = int(rot_dim)
        self.mask_attention_type = mask_attention_type
        self.denormalize_by_extent = bool(denormalize_by_extent)
        self.use_region_aux = bool(use_region_aux)
        self.region_aux_dim = int(region_aux_dim)
        self.coarse_grid_size = int(coarse_grid_size)

        self.input_projection = _ConvNormAct(5, base_channels)
        self.local_fine = _LocalRelationBlock(base_channels)

        if self.use_region_aux:
            self.region_projection = nn.Sequential(
                nn.Conv2d(num_regions, region_aux_dim, kernel_size=1, bias=False),
                nn.GroupNorm(_group_count(region_aux_dim), region_aux_dim),
                nn.GELU(),
                nn.Conv2d(region_aux_dim, base_channels, kernel_size=1, bias=False),
            )
            # A scalar zero-start residual makes Region auxiliary at init but
            # keeps the dense XYZ/ROI stream as the only initial signal.
            self.region_scale = nn.Parameter(torch.zeros(1))

        self.downsample_mid = _ConvNormAct(base_channels, mid_channels, stride=2)
        self.local_mid = _LocalRelationBlock(mid_channels)
        self.downsample_high = _ConvNormAct(mid_channels, high_channels, stride=2)
        self.local_high = _LocalRelationBlock(high_channels)

        # Fine/mid means summarize learned local relations.  The high-level
        # grid deliberately retains coarse image-space ordering for the pose
        # decoder instead of immediately collapsing every scale to moments.
        pooled_dim = base_channels + mid_channels + high_channels * coarse_grid_size**2
        self.pose_fc1 = nn.Linear(pooled_dim, 256)
        self.pose_fc2 = nn.Linear(256, 256)
        self.pose_act = nn.GELU()
        self.pose_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Keep names disjoint from both official ConvPnP and CPM checkpoints.
        # This lets old heads be discarded without deleting EXP012's own
        # trained output tensors during resume/evaluation.
        self.pose_rotation = nn.Linear(256, rot_dim)
        self.pose_translation = nn.Linear(256, 3)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                normal_init(module, std=0.001)
        normal_init(self.pose_rotation, std=0.01)
        normal_init(self.pose_translation, std=0.01)

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
        """Discard only known old pose-head keys during official warm-start."""

        for key in list(state_dict):
            if key.startswith(prefix):
                local_key = key[len(prefix) :]
                if local_key.startswith(self._LEGACY_PNP_PREFIXES):
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

    @staticmethod
    def _sanitize_mask(mask_attention: torch.Tensor) -> torch.Tensor:
        mask = torch.nan_to_num(mask_attention, nan=0.0, posinf=0.0, neginf=0.0)
        return mask.clamp(min=0.0, max=1.0)

    def _prepare_inputs(
        self,
        coor_feat: torch.Tensor,
        *,
        extents: torch.Tensor | None,
    ) -> torch.Tensor:
        if coor_feat.ndim != 4 or coor_feat.shape[1] != 5:
            raise ValueError(
                "HierarchicalCorrespondencePnPNet expects XYZ3 + ROI2D2, got "
                f"{tuple(coor_feat.shape)}"
            )
        if not bool(torch.isfinite(coor_feat).all()):
            raise ValueError("coor_feat contains non-finite values")
        xyz = coor_feat[:, :3]
        if self.denormalize_by_extent:
            if extents is None or extents.shape != (coor_feat.shape[0], 3):
                raise ValueError(
                    "extents must have shape Bx3 when XYZ denormalization is enabled"
                )
            xyz = (xyz - 0.5) * extents.to(
                device=xyz.device, dtype=xyz.dtype
            ).view(-1, 3, 1, 1)
        return torch.cat([xyz, coor_feat[:, 3:5]], dim=1)

    def forward(
        self,
        coor_feat: torch.Tensor,
        region: torch.Tensor | None = None,
        extents: torch.Tensor | None = None,
        mask_attention: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._prepare_inputs(coor_feat, extents=extents)
        if self.use_region_aux:
            if region is None:
                raise ValueError("region is required when use_region_aux=True")
            if region.ndim != 4 or region.shape[1] != self.num_regions:
                raise ValueError(
                    f"region must have shape Bx{self.num_regions}xHxW, got "
                    f"{tuple(region.shape)}"
                )
            if region.shape[0] != x.shape[0] or region.shape[2:] != x.shape[2:]:
                raise ValueError("region and coor_feat spatial shapes differ")
            if not bool(torch.isfinite(region).all()):
                raise ValueError("region contains non-finite values")

        if self.mask_attention_type != "none":
            if mask_attention is None:
                raise ValueError("mask_attention is required for the configured mask mode")
            if mask_attention.ndim != 4 or mask_attention.shape[1] != 1:
                raise ValueError(
                    "mask_attention must have shape Bx1xHxW, got "
                    f"{tuple(mask_attention.shape)}"
                )
            if mask_attention.shape[0] != x.shape[0] or mask_attention.shape[2:] != x.shape[2:]:
                raise ValueError("mask_attention and coor_feat spatial shapes differ")
            support = self._sanitize_mask(mask_attention)
        else:
            support = None

        # Apply visibility before any convolution or normalization so values
        # outside zero support cannot leak into visible local relations.
        if support is not None:
            x = x * support
            if self.use_region_aux:
                region = region * support

        x = self.input_projection(x)
        if self.use_region_aux:
            region_aux = self.region_projection(region)
            x = x + self.region_scale.to(dtype=x.dtype) * region_aux
        x = self.local_fine(x)
        fine = x

        mid = self.local_mid(self.downsample_mid(fine))
        high = self.local_high(self.downsample_high(mid))

        fine_summary = torch.mean(fine, dim=(2, 3))
        mid_summary = torch.mean(mid, dim=(2, 3))
        high_grid = F.adaptive_avg_pool2d(
            high, (self.coarse_grid_size, self.coarse_grid_size)
        ).flatten(start_dim=1)
        latent = torch.cat([fine_summary, mid_summary, high_grid], dim=1)
        latent = self.pose_act(self.pose_fc1(latent))
        latent = self.pose_act(self.pose_fc2(latent))
        latent = self.pose_dropout(latent)
        return self.pose_rotation(latent), self.pose_translation(latent)
