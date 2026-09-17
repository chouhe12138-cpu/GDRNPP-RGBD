"""Image attention and local CAD matching blocks for EXP022."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


MATCH_BLOCK_SIZE = 16


class ImageAttentionBlock(nn.Module):
    def __init__(self, token_dim: int, num_heads: int, resolution: int,
                 window_size: int | None = None, shift_size: int = 0,
                 dropout: float = 0.0):
        super().__init__()
        if window_size is not None and (resolution % window_size or not 0 <= shift_size < window_size):
            raise ValueError("Window size must divide resolution; shift must be inside a window")
        self.resolution = resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(token_dim)
        self.attn = nn.MultiheadAttention(token_dim, num_heads, dropout=dropout, batch_first=True)
        if window_size is not None and shift_size:
            self.register_buffer("shift_mask", self._make_shift_mask(resolution, window_size, shift_size),
                                 persistent=False)
        else:
            self.shift_mask = None

    @staticmethod
    def _partition(grid: torch.Tensor, window_size: int) -> torch.Tensor:
        batch, height, width, channels = grid.shape
        return (grid.reshape(batch, height // window_size, window_size,
                             width // window_size, window_size, channels)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(batch * (height // window_size) * (width // window_size),
                         window_size * window_size, channels))

    @staticmethod
    def _reverse(windows: torch.Tensor, batch: int, resolution: int,
                 window_size: int) -> torch.Tensor:
        channels = windows.shape[-1]
        return (windows.reshape(batch, resolution // window_size, resolution // window_size,
                                window_size, window_size, channels)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(batch, resolution, resolution, channels))

    @classmethod
    def _make_shift_mask(cls, resolution: int, window_size: int, shift_size: int) -> torch.Tensor:
        regions = torch.zeros(1, resolution, resolution, 1, dtype=torch.long)
        sections = (slice(0, -window_size), slice(-window_size, -shift_size),
                    slice(-shift_size, None))
        label = 0
        for rows in sections:
            for columns in sections:
                regions[:, rows, columns] = label
                label += 1
        windows = cls._partition(regions, window_size).squeeze(-1)
        return windows[:, :, None] != windows[:, None, :]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, pixels, channels = tokens.shape
        if pixels != self.resolution ** 2:
            raise ValueError(f"Expected {self.resolution ** 2} image tokens, got {pixels}")
        normalized = self.norm(tokens)
        if self.window_size is None:
            update, _ = self.attn(normalized, normalized, normalized, need_weights=False)
        else:
            grid = normalized.reshape(batch, self.resolution, self.resolution, channels)
            if self.shift_size:
                grid = torch.roll(grid, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            windows = self._partition(grid, self.window_size)
            mask = None
            if self.shift_mask is not None:
                mask = self.shift_mask.repeat(batch, 1, 1).repeat_interleave(self.num_heads, dim=0)
            attended, _ = self.attn(windows, windows, windows, attn_mask=mask, need_weights=False)
            grid = self._reverse(attended, batch, self.resolution, self.window_size)
            if self.shift_size:
                grid = torch.roll(grid, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            update = grid.reshape(batch, pixels, channels)
        return tokens + update


class HierarchicalCADMatcher(nn.Module):
    """Explicit eight-child Q/K/V attention with raw route logits."""

    def __init__(self, token_dim: int):
        super().__init__()
        self.token_dim = token_dim
        self.q_proj = nn.Linear(token_dim, token_dim, bias=False)
        self.k_proj = nn.Linear(token_dim, token_dim, bias=False)
        self.v_proj = nn.Linear(token_dim, token_dim, bias=False)
        self.out_proj = nn.Linear(token_dim, token_dim, bias=False)

    def project_bank(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k_proj(tokens), self.v_proj(tokens)

    def _attention(self, queries: torch.Tensor, keys: torch.Tensor,
                   values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(self.token_dim)
        probability = F.softmax(logits.float(), dim=-1).to(values.dtype)
        return logits, torch.bmm(probability, values)

    def match_root(self, image_tokens: torch.Tensor, object_inverse: torch.Tensor,
                   bank: tuple[torch.Tensor, torch.Tensor]):
        keys, values = bank
        selected_keys = keys.index_select(0, object_inverse)
        selected_values = values.index_select(0, object_inverse)
        logits, context = self._attention(self.q_proj(image_tokens), selected_keys, selected_values)
        return logits, self.out_proj(context)

    def match_packed(self, image_tokens: torch.Tensor, object_inverse: torch.Tensor,
                     parent: torch.Tensor, bank: tuple[torch.Tensor, torch.Tensor],
                     target: torch.Tensor | None = None, anchors: torch.Tensor | None = None):
        """Reuse each (object,parent) child set across bounded 16-query blocks."""
        keys_bank, values_bank = bank
        num_parents = keys_bank.shape[1] // 8
        route_keys = object_inverse.long() * num_parents + parent.long()
        count = image_tokens.shape[0]
        num_keys = keys_bank.shape[0] * num_parents
        max_blocks = (count + MATCH_BLOCK_SIZE - 1) // MATCH_BLOCK_SIZE + num_keys
        order = torch.argsort(route_keys)
        sorted_keys = route_keys.index_select(0, order)
        group_counts = torch.bincount(route_keys, minlength=num_keys)
        group_starts = group_counts.cumsum(0) - group_counts
        blocks_per_group = torch.div(group_counts + MATCH_BLOCK_SIZE - 1,
                                     MATCH_BLOCK_SIZE, rounding_mode="floor")
        block_starts = blocks_per_group.cumsum(0) - blocks_per_group
        within = torch.arange(count, device=image_tokens.device) - group_starts[sorted_keys]
        block_ids = block_starts[sorted_keys] + torch.div(
            within, MATCH_BLOCK_SIZE, rounding_mode="floor"
        )
        slots = within.remainder(MATCH_BLOCK_SIZE)
        packed = torch.full((max_blocks, MATCH_BLOCK_SIZE), -1,
                            device=image_tokens.device, dtype=torch.long)
        packed[block_ids, slots] = torch.arange(count, device=image_tokens.device)
        valid = packed >= 0
        safe = packed.clamp_min(0)
        block_keys = torch.searchsorted(block_starts.contiguous(),
                                         torch.arange(max_blocks, device=image_tokens.device),
                                         right=True).sub(1).clamp_(0, num_keys - 1)
        child_keys = keys_bank.reshape(-1, 8, self.token_dim).index_select(0, block_keys)
        child_values = values_bank.reshape(-1, 8, self.token_dim).index_select(0, block_keys)
        projected = self.q_proj(image_tokens)
        block_query = projected.index_select(0, order)[safe]
        logits, context = self._attention(block_query, child_keys, child_values)
        positions = torch.where(valid.flatten(), order[safe.flatten()], count)
        out_logits = torch.zeros((count + 1, 8), device=image_tokens.device, dtype=logits.dtype).index_copy(
            0, positions, logits.flatten(0, 1)
        )[:count]
        out_context = torch.zeros((count + 1, self.token_dim), device=image_tokens.device,
                                  dtype=context.dtype).index_copy(
            0, positions, context.flatten(0, 1)
        )[:count]
        out_context = self.out_proj(out_context)
        if target is None:
            return out_logits, out_context, None
        if anchors is None:
            raise ValueError("Packed target labels require child anchors")
        block_target = target.index_select(0, order)[safe].float()
        block_anchors = anchors.reshape(-1, 8, 3).index_select(0, block_keys).float()
        with torch.no_grad(), torch.autocast(device_type=image_tokens.device.type, enabled=False):
            label = (block_target[:, :, None] - block_anchors[:, None]).square().sum(-1).argmin(-1)
        out_label = torch.zeros(count + 1, device=image_tokens.device, dtype=torch.long).index_copy(
            0, positions, label.flatten()
        )[:count]
        return out_logits, out_context, out_label


class PCCStage(nn.Module):
    def __init__(self, channels: int, token_dim: int, resolution: int, attention: str,
                 num_heads: int, window_size: int, shift_size: int, dropout: float):
        super().__init__()
        self.image_proj = nn.Conv2d(channels, token_dim, 1)
        self.image_norm = nn.LayerNorm(token_dim)
        if attention == "global":
            self.image_attention = nn.ModuleList((
                ImageAttentionBlock(token_dim, num_heads, resolution, dropout=dropout),
            ))
        elif attention == "window":
            self.image_attention = nn.ModuleList((
                ImageAttentionBlock(token_dim, num_heads, resolution, window_size,
                                    dropout=dropout),
                ImageAttentionBlock(token_dim, num_heads, resolution, window_size,
                                    shift_size, dropout=dropout),
            ))
        else:
            raise ValueError(f"Unsupported EXP022 image attention: {attention}")
        self.matcher = HierarchicalCADMatcher(token_dim)
        self.context_proj = nn.Linear(token_dim, channels)
        self.gate_logit = nn.Parameter(torch.tensor(-4.0))

    def project_image(self, feature: torch.Tensor) -> torch.Tensor:
        return self.image_norm(self.image_proj(feature).flatten(2).transpose(1, 2))

    def image_tokens(self, feature: torch.Tensor) -> torch.Tensor:
        tokens = self.project_image(feature)
        for block in self.image_attention:
            tokens = block(tokens)
        return tokens

    def fuse(self, feature: torch.Tensor, context: torch.Tensor,
             collect_stats: bool = False):
        batch, channels, height, width = feature.shape
        update = self.context_proj(context).transpose(1, 2).reshape(batch, channels, height, width)
        update = torch.sigmoid(self.gate_logit) * update
        ratio = None
        if collect_stats:
            ratio = (update.detach().flatten(1).norm(dim=-1) /
                     feature.detach().flatten(1).norm(dim=-1).clamp_min(1e-8))
        return feature + update, ratio


class SpatialRefinement(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        feature = F.interpolate(feature, scale_factor=2, mode="bilinear", align_corners=False)
        return self.main(feature) + self.skip(feature)
