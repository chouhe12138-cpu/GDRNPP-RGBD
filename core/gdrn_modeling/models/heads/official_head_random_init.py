"""EXP013E: the official ConvPnPNet rebuilt with random initialization.

The rebuilt head nests the official module under a ``head.`` submodule so its
state-dict keys (``pnp_net.head.*``) never collide with the official
checkpoint's pose-head keys (``pnp_net.*``). During official-checkpoint warm
start the known official pose-head keys are discarded explicitly (same
mechanism as ``HierarchicalCorrespondencePnPNet``), which guarantees the
random initialization survives while backbone/geometry tensors load exactly.
"""

from __future__ import annotations

import torch
from torch import nn

from .conv_pnp_net import ConvPnPNet


class OfficialConvPnPNetRandomInit(nn.Module):
    """Official ConvPnPNet topology, randomly initialized, key-safe."""

    _OFFICIAL_PNP_PREFIXES = (
        "features.",
        "fc1.",
        "fc2.",
        "fc_r.",
        "fc_t.",
    )

    def __init__(
        self,
        nIn: int,
        num_regions: int,
        mask_attention_type: str,
        rot_dim: int,
        **head_kwargs,
    ) -> None:
        super().__init__()
        self.head = ConvPnPNet(
            nIn=nIn,
            num_regions=num_regions,
            mask_attention_type=mask_attention_type,
            rot_dim=rot_dim,
            **head_kwargs,
        )

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
        """Discard known official pose-head keys during official warm start."""

        for key in list(state_dict):
            if key.startswith(prefix):
                local_key = key[len(prefix) :]
                if local_key.startswith(self._OFFICIAL_PNP_PREFIXES):
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

    def forward(
        self,
        coor_feat: torch.Tensor,
        region: torch.Tensor | None = None,
        extents: torch.Tensor | None = None,
        mask_attention: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.head(coor_feat, region=region, extents=extents, mask_attention=mask_attention)
