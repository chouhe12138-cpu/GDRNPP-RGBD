"""EXP013F: GLM-Pose-L head (M1 official input path, M2 attention pooling,
M3 depth-statistic injection for translation).

Design source: ``GLM-Pose.md`` §2. The head reuses the EXP013A main stream
(denormalized XYZ + absolute ROI2D + Region zero-start residual + visible-mask
multiplicative gating) and replaces every late flatten/FC aggregation with a
single lightweight self-attention encoder followed by learned attention
pooling, then decouples rotation and translation outputs. Four normalized
depth statistics (mask-free anchor-band method, see
``core/gdrn_modeling/datasets/roi_depth_stats.py``) are concatenated into the
translation input; with ``depth_stats=None`` the branch receives zeros so the
head stays runnable in diagnostics and smoke tests.
"""

from __future__ import annotations

import torch
from mmcv.cnn import normal_init
from torch import nn

from .exp013_geometry_pnp_net import XYZResidualBypassPnPNet


class GLMPoseLNet(XYZResidualBypassPnPNet):
    """GLM-Pose-L: shared spatial body, late R/T decoupling, depth priors."""

    def __init__(
        self,
        *,
        embed_channels: int = 256,
        attn_heads: int = 8,
        ffn_channels: int = 512,
        shared_channels: int = 256,
        depth_stats_dim: int = 4,
        depth_stats_eps: float = 1.0e-6,
        use_depth_stats: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        del use_depth_stats  # consumed by GDRN_double_mask via the config; the
        # head itself simply tolerates depth_stats=None with zero padding.
        main_channels = self.local_high.pointwise_out.out_channels
        self.embed_channels = int(embed_channels)
        self.depth_stats_dim = int(depth_stats_dim)
        self.depth_stats_eps = float(depth_stats_eps)
        if embed_channels % attn_heads != 0:
            raise ValueError("embed_channels must be divisible by attn_heads")

        # Drop the inherited A late decoder and the geometry-residual branch:
        # GLM-Pose-L keeps only the main correspondence stream (M1).
        del self.geometry_input_projection
        del self.geometry_local_fine
        del self.geometry_downsample_mid
        del self.geometry_local_mid
        del self.geometry_downsample_high
        del self.geometry_local_high
        del self.geometry_projection
        del self.geometry_scale
        del self.pose_fc1
        del self.pose_fc2
        del self.pose_dropout

        # M2: spatial-preserving self-attention over the 16x16 token grid.
        self.token_projection = nn.Linear(main_channels, embed_channels)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, 256, embed_channels)
        )
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_channels,
            nhead=attn_heads,
            dim_feedforward=ffn_channels,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.pool_score = nn.Linear(embed_channels, 1)
        self.shared_fc = nn.Linear(embed_channels, shared_channels)
        # Late decoupled outputs; translation consumes the M3 depth statistics.
        self.pose_rotation = nn.Linear(shared_channels, self.rot_dim)
        self.pose_translation = nn.Linear(shared_channels + self.depth_stats_dim, 3)

        self._initialize_glm_weights()

    def _initialize_glm_weights(self) -> None:
        for module in (
            self.token_projection,
            self.encoder_layer,
            self.pool_score,
            self.shared_fc,
        ):
            for sub in module.modules():
                if isinstance(sub, nn.Linear):
                    normal_init(sub, std=0.001)
        normal_init(self.position_embedding, std=0.02)
        normal_init(self.pose_rotation, std=0.01)
        normal_init(self.pose_translation, std=0.01)

    def forward(
        self,
        coor_feat: torch.Tensor,
        region: torch.Tensor | None = None,
        extents: torch.Tensor | None = None,
        mask_attention: torch.Tensor | None = None,
        depth_stats: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        metric, masked_region, support = self._validate_and_mask_inputs(
            coor_feat, region, extents, mask_attention
        )
        _, _, high = self._encode_main_features(metric, masked_region)

        tokens = high.flatten(2).transpose(1, 2)  # [B, H*W, C]
        if tokens.shape[1] != self.position_embedding.shape[1]:
            raise ValueError(
                f"GLMPoseLNet expects a 16x16 token grid, got {tokens.shape}"
            )
        tokens = self.token_projection(tokens) + self.position_embedding.to(
            dtype=tokens.dtype
        )
        tokens = self.encoder_layer(tokens)
        scores = self.pool_score(torch.tanh(tokens))  # [B, H*W, 1]
        weights = torch.softmax(scores, dim=1)
        pooled = (tokens * weights).sum(dim=1)  # [B, embed]
        latent = self.pose_act(self.shared_fc(pooled))

        if depth_stats is None:
            depth_features = latent.new_zeros(latent.shape[0], self.depth_stats_dim)
        else:
            if depth_stats.ndim != 2 or depth_stats.shape != (
                latent.shape[0],
                self.depth_stats_dim,
            ):
                raise ValueError(
                    "depth_stats must have shape Bx"
                    f"{self.depth_stats_dim}, got {tuple(depth_stats.shape)}"
                )
            if not bool(torch.isfinite(depth_stats).all()):
                raise ValueError("depth_stats contains non-finite values")
            depth_features = depth_stats.to(device=latent.device, dtype=latent.dtype)
        translation_input = torch.cat([latent, depth_features], dim=1)
        return self.pose_rotation(latent), self.pose_translation(translation_input)
