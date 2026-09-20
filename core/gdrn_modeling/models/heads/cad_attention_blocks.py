"""EXP025 dense attention primitives, independent of legacy PCC routing."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AttentionBlock(nn.Module):
    def __init__(self, dim=256, heads=8):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, query, context=None, mask=None):
        q = self.query_norm(query)
        kv = q if context is None else self.context_norm(context)
        query = query + self.attention(q, kv, kv, attn_mask=mask, need_weights=False)[0]
        return query + self.ffn(self.ffn_norm(query))


class CADStageTransition(nn.Module):
    def __init__(self, in_channels, out_channels, gn_groups=32):
        super().__init__()
        groups = min(gn_groups, out_channels)
        while out_channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.GroupNorm(groups, out_channels),
            nn.GELU(), nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels), nn.GELU())

    def forward(self, x):
        return self.block(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))


class WindowAttentionBlock(nn.Module):
    """Window attention with broadcast SDPA mask (no cyclic boundary leakage)."""
    def __init__(self, dim=256, heads=8, resolution=64, window=8, shift=0):
        super().__init__()
        if resolution % window or not 0 <= shift < window or dim % heads:
            raise ValueError('Invalid window/shift/head geometry')
        self.resolution, self.window, self.shift = resolution, window, shift
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        mask = None
        if shift:
            grid = torch.zeros(1, resolution, resolution, 1, dtype=torch.long)
            slices = (slice(0, -window), slice(-window, -shift), slice(-shift, None))
            for i, rows in enumerate(slices):
                for j, cols in enumerate(slices):
                    grid[:, rows, cols] = 3 * i + j
            labels = self.partition(grid).squeeze(-1)
            mask = labels[:, :, None] == labels[:, None, :]
        self.register_buffer('allowed', mask, persistent=False)

    def partition(self, grid):
        b, h, w, d = grid.shape
        s = self.window
        return grid.reshape(b, h//s, s, w//s, s, d).permute(0, 1, 3, 2, 4, 5).reshape(-1, s*s, d)

    def forward(self, x):
        b, p, d = x.shape
        r, s = self.resolution, self.window
        if p != r*r:
            raise ValueError('Unexpected image token count')
        grid = self.norm(x).reshape(b, r, r, d)
        if self.shift:
            grid = grid.roll((-self.shift, -self.shift), (1, 2))
        windows = self.partition(grid)
        nw, length = (r//s)**2, s*s
        q, k, v = [t.reshape(b, nw, length, self.heads, d//self.heads).permute(0, 1, 3, 2, 4)
                   for t in self.qkv(windows).chunk(3, -1)]
        mask = None if self.allowed is None else self.allowed[None, :, None]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.)
        y = self.proj(y.permute(0, 1, 3, 2, 4).reshape(-1, length, d))
        grid = y.reshape(b, r//s, r//s, s, s, d).permute(0, 1, 3, 2, 4, 5).reshape(b, r, r, d)
        if self.shift:
            grid = grid.roll((self.shift, self.shift), (1, 2))
        x = x + grid.reshape(b, p, d)
        return x + self.ffn(self.ffn_norm(x))


class ImageSAStage(nn.Module):
    """One image-branch stage whose self-attention is written back into the feature.

    The returned feature is what the next spatial transition consumes, so the attention
    cannot degrade into a read-only side branch.  The last stage returns `None` as its
    feature: nothing downstream consumes it, and a back projection no loss reaches would
    only add parameters without gradients.
    """

    def __init__(self, channels, resolution, attention, token_dim=256, num_heads=8,
                 window=8, shift=4, write_back=True):
        super().__init__()
        if attention not in ('global', 'window') or token_dim % num_heads:
            raise ValueError('Invalid image self-attention geometry')
        self.resolution = int(resolution)
        self.global_attention = attention == 'global'
        self.to_tokens = nn.Conv2d(channels, token_dim, 1)
        if self.global_attention:
            self.attention = nn.ModuleList((AttentionBlock(token_dim, num_heads),))
        else:
            self.attention = nn.ModuleList((
                WindowAttentionBlock(token_dim, num_heads, self.resolution, window),
                WindowAttentionBlock(token_dim, num_heads, self.resolution, window, shift)))
        self.to_feature = nn.Linear(token_dim, channels) if write_back else None

    def forward(self, feature):
        batch, channels, height, width = feature.shape
        tokens = self.to_tokens(feature).flatten(2).transpose(1, 2)
        if tokens.shape[1] != self.resolution ** 2 or (height, width) != (self.resolution,) * 2:
            raise ValueError(f'Expected {self.resolution}x{self.resolution} image features')
        for block in self.attention:
            # Self-attention reads its own tokens as context: with `context=None` the global
            # block would skip its context norm and leave those parameters without gradient.
            tokens = block(tokens, tokens) if self.global_attention else block(tokens)
        if self.to_feature is None:  # final stage: the tokens are the cross-attention query
            return None, tokens
        update = self.to_feature(tokens).transpose(1, 2).reshape(batch, channels, height, width)
        return feature + update, tokens


class CADGeometryEncoder(nn.Module):
    def __init__(self, dim=256, heads=8):
        super().__init__()
        self.descriptor = nn.Sequential(nn.Linear(10, 128), nn.GELU(), nn.Linear(128, dim), nn.LayerNorm(dim))
        self.blocks = nn.ModuleList(AttentionBlock(dim, heads) for _ in range(3))
        self.parents = nn.ModuleList(nn.Linear(dim, dim) for _ in range(2))
        self.global_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.global_token, std=.02)
        self.global_readers = nn.ModuleList(AttentionBlock(dim, heads) for _ in range(3))

    def forward(self, descriptors):
        banks = []
        for depth, descriptor in enumerate(descriptors):
            tokens = self.descriptor(descriptor)
            if depth:
                tokens = tokens + self.parents[depth-1](banks[-1]).repeat_interleave(8, dim=1)
                b, n, d = tokens.shape
                tokens = self.blocks[depth](tokens.reshape(-1, 8, d)).reshape(b, n, d)
            else:
                tokens = self.blocks[depth](tokens)
            banks.append(tokens)
        t0 = self.global_token.expand(banks[0].shape[0], -1, -1)
        for reader, bank in zip(self.global_readers, banks):
            t0 = reader(t0, bank)
        return (t0, *banks)
