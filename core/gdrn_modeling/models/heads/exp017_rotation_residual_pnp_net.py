"""EXP017 support-aware rotation-only spatial residual adapter.

The head is EXP013A unchanged plus one Region-free adapter over A's existing
8x8 geometry grid.  The adapter can affect only raw rotation; raw translation
is returned directly from the original A path.
"""

from __future__ import annotations

from typing import Any

import torch
from mmcv.cnn import normal_init
from torch import nn
from torch.nn import functional as F

from .exp013_geometry_pnp_net import XYZResidualBypassPnPNet


def _fixed_permutation(size: int, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randperm(size, generator=generator).to(device=device)


class SupportAwareRotationAdapter(nn.Module):
    """Learned masked pooling over Region-free geometry tokens.

    Support is binary here: it only determines which tokens may participate in
    the softmax.  No occupancy/confidence value is added to the logits.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        grid_size: int = 8,
        token_channels: int = 64,
        score_channels: int = 32,
        rot_dim: int = 6,
    ) -> None:
        super().__init__()
        if min(in_channels, grid_size, token_channels, score_channels, rot_dim) <= 0:
            raise ValueError("EXP017 adapter dimensions must be positive")
        self.in_channels = int(in_channels)
        self.grid_size = int(grid_size)
        self.token_channels = int(token_channels)
        self.rot_dim = int(rot_dim)
        token_count = self.grid_size**2

        self.token_projection = nn.Linear(self.in_channels, self.token_channels)
        self.position_embedding = nn.Parameter(
            torch.empty(1, token_count, self.token_channels)
        )
        self.pool_score = nn.Sequential(
            nn.Linear(self.token_channels, score_channels),
            nn.GELU(),
            nn.Linear(score_channels, 1),
        )
        self.descriptor_norm = nn.LayerNorm(self.token_channels)
        self.delta_hidden = nn.Linear(self.token_channels, self.token_channels)
        self.delta_act = nn.GELU()
        self.delta_output = nn.Linear(self.token_channels, self.rot_dim)

        self.last_pool_weights: torch.Tensor | None = None
        self.last_valid_tokens: torch.Tensor | None = None
        self.last_descriptor: torch.Tensor | None = None
        self.last_delta_r: torch.Tensor | None = None
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        normal_init(self.token_projection, std=0.001)
        for module in self.pool_score.modules():
            if isinstance(module, nn.Linear):
                normal_init(module, std=0.001)
        normal_init(self.delta_hidden, std=0.001)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        # Identity-preserving initialization with alpha_r=1: the residual is
        # exactly zero while its output layer receives a first-step gradient.
        nn.init.zeros_(self.delta_output.weight)
        nn.init.zeros_(self.delta_output.bias)

    def _valid_tokens(
        self, support: torch.Tensor, batch: int, device: torch.device
    ) -> torch.Tensor:
        if support.ndim != 4 or support.shape[0] != batch or support.shape[1] != 1:
            raise ValueError("support must have shape Bx1xHxW")
        if not bool(torch.isfinite(support).all()):
            raise ValueError("support contains non-finite values")
        # Binary validity only.  adaptive max means a partially supported cell
        # remains valid, without turning occupancy into a soft confidence.
        valid = F.adaptive_max_pool2d(
            (support > 0).to(dtype=torch.float32),
            (self.grid_size, self.grid_size),
        ) > 0
        valid = valid[:, 0].flatten(1).to(device=device)
        if bool((valid.sum(dim=1) == 0).any()):
            raise ValueError("EXP017 rotation adapter requires non-zero visible support")
        return valid

    def forward(
        self,
        geometry_grid: torch.Tensor,
        support: torch.Tensor,
        *,
        pooling: str = "learned",
        position_mode: str = "normal",
        token_shuffle: bool = False,
        seed: int = 20260902,
        return_info: bool = False,
    ):
        expected = (self.in_channels, self.grid_size, self.grid_size)
        if geometry_grid.ndim != 4 or tuple(geometry_grid.shape[1:]) != expected:
            raise ValueError(
                f"geometry_grid must have shape Bx{expected[0]}x{expected[1]}x{expected[2]}, "
                f"got {tuple(geometry_grid.shape)}"
            )
        if not bool(torch.isfinite(geometry_grid).all()):
            raise ValueError("geometry_grid contains non-finite values")

        batch = geometry_grid.shape[0]
        tokens = geometry_grid.flatten(2).transpose(1, 2)
        valid = self._valid_tokens(support, batch, tokens.device)
        if token_shuffle:
            permutation = _fixed_permutation(tokens.shape[1], tokens.device, seed + 11)
            tokens = tokens[:, permutation]

        tokens = self.token_projection(tokens)
        position = self.position_embedding.to(device=tokens.device, dtype=tokens.dtype)
        if position_mode == "normal":
            pass
        elif position_mode == "shuffle":
            permutation = _fixed_permutation(tokens.shape[1], tokens.device, seed + 23)
            position = position[:, permutation]
        elif position_mode == "zero":
            position = torch.zeros_like(position)
        else:
            raise ValueError(f"Unknown position_mode={position_mode!r}")
        tokens = tokens + position

        if pooling == "learned":
            logits = self.pool_score(tokens).squeeze(-1)
            logits = logits.masked_fill(~valid, -torch.inf)
            weights = torch.softmax(logits, dim=1)
            # Make exact zero and exact re-normalization explicit even though
            # masked softmax already has those semantics.
            weights = weights * valid.to(weights.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True)
        elif pooling == "uniform":
            weights = valid.to(tokens.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unknown pooling={pooling!r}")

        pooled = (tokens * weights.unsqueeze(-1)).sum(dim=1)
        descriptor = self.descriptor_norm(pooled)
        hidden = self.delta_act(self.delta_hidden(descriptor))
        delta_r = self.delta_output(hidden)

        self.last_pool_weights = weights.detach()
        self.last_valid_tokens = valid.detach()
        self.last_descriptor = descriptor.detach()
        self.last_delta_r = delta_r.detach()
        if not return_info:
            return delta_r
        return delta_r, {
            "weights": weights,
            "valid": valid,
            "descriptor": descriptor,
            "pooling": pooling,
            "position_mode": position_mode,
            "token_shuffle": bool(token_shuffle),
        }


class SupportAwareRotationResidualPnPNet(XYZResidualBypassPnPNet):
    """EXP013A plus a support-aware Region-free raw-rotation residual."""

    def __init__(
        self,
        *,
        adapter_token_channels: int = 64,
        adapter_score_channels: int = 32,
        alpha_r_init: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if self.geometry_grid_size != 8:
            raise ValueError("EXP017 requires the existing EXP013A 8x8 geometry grid")
        geometry_channels = (
            self.geometry_projection.in_features // self.geometry_grid_size**2
        )
        self.rotation_adapter = SupportAwareRotationAdapter(
            in_channels=geometry_channels,
            grid_size=self.geometry_grid_size,
            token_channels=adapter_token_channels,
            score_channels=adapter_score_channels,
            rot_dim=self.rot_dim,
        )
        self.alpha_r = nn.Parameter(
            torch.tensor([float(alpha_r_init)], dtype=torch.float32)
        )

    def adapter_parameters(self):
        yield from self.rotation_adapter.parameters()
        yield self.alpha_r

    def adapter_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.adapter_parameters())

    def _adapter_geometry_grid(self, geometry_grid: torch.Tensor) -> torch.Tensor:
        """Select the adapter input without changing EXP017's default graph."""
        return geometry_grid

    def forward_with_adapter_intervention(
        self,
        coor_feat: torch.Tensor,
        region: torch.Tensor | None = None,
        extents: torch.Tensor | None = None,
        mask_attention: torch.Tensor | None = None,
        *,
        pooling: str = "learned",
        position_mode: str = "normal",
        token_shuffle: bool = False,
        seed: int = 20260902,
        adapter_enabled: bool = True,
        detach_adapter_geometry: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | str | bool]]:
        metric, masked_region, support = self._validate_and_mask_inputs(
            coor_feat, region, extents, mask_attention
        )
        main_latent, _, _, _ = self._encode_main(metric, masked_region)
        geometry_latent, geometry_grid = self._encode_geometry(metric, support)
        latent_a = (
            main_latent
            + self.geometry_scale.to(main_latent.dtype) * geometry_latent
        )
        raw_r_a = self.pose_rotation(latent_a)
        raw_t_a = self.pose_translation(latent_a)
        adapter_grid = self._adapter_geometry_grid(geometry_grid)
        if detach_adapter_geometry is True:
            adapter_grid = geometry_grid.detach()
        elif detach_adapter_geometry is False:
            adapter_grid = geometry_grid
        if adapter_enabled:
            delta_r, info = self.rotation_adapter(
                adapter_grid,
                support,
                pooling=pooling,
                position_mode=position_mode,
                token_shuffle=token_shuffle,
                seed=seed,
                return_info=True,
            )
        else:
            delta_r = torch.zeros_like(raw_r_a)
            info = {
                "pooling": pooling,
                "position_mode": position_mode,
                "token_shuffle": bool(token_shuffle),
            }
        raw_r = raw_r_a + self.alpha_r.to(raw_r_a.dtype) * delta_r
        info.update(
            {
                "raw_r_a": raw_r_a,
                "raw_t_a": raw_t_a,
                "delta_r": delta_r,
                "geometry_grid": geometry_grid,
                "support": support,
                "adapter_enabled": bool(adapter_enabled),
                "adapter_geometry_detached": adapter_grid is not geometry_grid,
            }
        )
        return raw_r, raw_t_a, info

    def forward(
        self,
        coor_feat: torch.Tensor,
        region: torch.Tensor | None = None,
        extents: torch.Tensor | None = None,
        mask_attention: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_r, raw_t, _ = self.forward_with_adapter_intervention(
            coor_feat,
            region=region,
            extents=extents,
            mask_attention=mask_attention,
        )
        return raw_r, raw_t


class DetachedSupportAwareRotationResidualPnPNet(
    SupportAwareRotationResidualPnPNet
):
    """EXP017-B: block only the adapter-to-A geometry-encoder gradient."""

    def _adapter_geometry_grid(self, geometry_grid: torch.Tensor) -> torch.Tensor:
        return geometry_grid.detach()
