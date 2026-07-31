import torch
import torch.nn as nn


class QualityCoverageAttention(nn.Module):
    """Identity-initialized residual reweighting for GDRNPP region attention.

    The spatial branch learns local correspondence quality from the frozen
    coordinate, mask, and region predictions.  The coverage branch observes
    how much image area (and visible area) each region occupies.  Both
    branches end in zero-initialized layers, so the module is exactly an
    identity mapping before training.
    """

    def __init__(
        self,
        coor_channels,
        num_regions,
        hidden_dim=32,
        max_residual=0.25,
    ):
        super().__init__()
        if coor_channels <= 0 or num_regions <= 1 or hidden_dim <= 0:
            raise ValueError("coor_channels, num_regions, and hidden_dim must be positive")
        if not 0.0 < max_residual < 0.5:
            raise ValueError("max_residual must be in (0, 0.5)")

        self.coor_channels = int(coor_channels)
        self.num_regions = int(num_regions)
        self.max_residual = float(max_residual)

        local_channels = self.coor_channels + self.num_regions + 1
        self.quality_net = nn.Sequential(
            nn.Conv2d(local_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(self._group_count(hidden_dim), hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True),
        )
        self.coverage_net = nn.Sequential(
            nn.Linear(2 * self.num_regions, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_regions),
        )
        self.reset_identity()

    @staticmethod
    def _group_count(channels):
        for groups in (8, 4, 2, 1):
            if channels % groups == 0:
                return groups
        return 1

    def reset_identity(self):
        """Make both residual branches output zero at initialization."""

        nn.init.zeros_(self.quality_net[-1].weight)
        nn.init.zeros_(self.quality_net[-1].bias)
        nn.init.zeros_(self.coverage_net[-1].weight)
        nn.init.zeros_(self.coverage_net[-1].bias)

    def forward(self, coor_feat, region_attention, mask_attention):
        if coor_feat.ndim != 4 or region_attention.ndim != 4:
            raise ValueError("coor_feat and region_attention must be BCHW tensors")
        if region_attention.shape[1] != self.num_regions:
            raise ValueError(
                f"expected {self.num_regions} regions, got {region_attention.shape[1]}"
            )
        if coor_feat.shape[1] != self.coor_channels:
            raise ValueError(
                f"expected {self.coor_channels} coordinate channels, got {coor_feat.shape[1]}"
            )
        if mask_attention is None:
            mask_attention = region_attention.new_ones(
                region_attention.shape[0],
                1,
                region_attention.shape[2],
                region_attention.shape[3],
            )
        elif mask_attention.ndim == 3:
            mask_attention = mask_attention.unsqueeze(1)
        if mask_attention.shape[1] != 1:
            raise ValueError("mask_attention must have one channel")

        local_input = torch.cat([coor_feat, region_attention, mask_attention], dim=1)
        quality_delta = torch.tanh(self.quality_net(local_input))

        occupancy = region_attention.mean(dim=(2, 3))
        visible_occupancy = (region_attention * mask_attention).mean(dim=(2, 3))
        coverage_input = torch.cat([occupancy, visible_occupancy], dim=1)
        coverage_delta = torch.tanh(self.coverage_net(coverage_input)).unsqueeze(-1).unsqueeze(-1)

        residual = self.max_residual * (quality_delta + coverage_delta)
        return region_attention * (1.0 + residual)
