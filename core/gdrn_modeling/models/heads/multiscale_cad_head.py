"""EXP027 granularity-aligned image/CAD interactions."""
from __future__ import annotations

import math

import torch
from torch import nn

from .hierarchical_cad_attention_head import HierarchicalCADAttentionHead


class _MultiscaleCADHead(HierarchicalCADAttentionHead):
    feature_shapes = ((128, 64), (256, 32), (512, 16), (1024, 8))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.laterals = nn.ModuleList((nn.Conv2d(512, 256, 1),
                                       nn.Conv2d(256, 128, 1),
                                       nn.Conv2d(128, 64, 1)))
        self.lateral_alpha = nn.Parameter(torch.full((3,), .01))

    def _check_features(self, features, classes):
        if not isinstance(features, (tuple, list)) or len(features) != 4:
            raise ValueError('EXP027 requires four ConvNeXt feature maps')
        batch = features[0].shape[0]
        for feature, (channels, size) in zip(features, self.feature_shapes):
            if feature.ndim != 4 or feature.shape != (batch, channels, size, size):
                raise ValueError(f'EXP027 feature shape mismatch: expected [B,{channels},{size},{size}]')
        if classes.shape != (batch,) or torch.any((classes < 0) | (classes >= self.num_objects)):
            raise ValueError('Invalid EXP027 ROI classes')

    def _next_feature(self, index, feature, features):
        up = self.transitions[index](feature)
        lateral = self.laterals[index](features[2-index])
        if up.shape != lateral.shape:
            raise ValueError('EXP027 lateral/transition shape mismatch')
        # The scalar gate accumulates gradients over every pixel; keep that reduction
        # in FP32 so AMP loss scaling does not overflow an FP16 intermediate.
        fused = up.float() + self.lateral_alpha[index].float() * lateral.float()
        return fused.to(up.dtype)


class MultiscaleImageQueryHead(_MultiscaleCADHead):
    """A: image tokens query CAD at each scale and write changes downstream."""

    def forward(self, features, classes, diagnostics=None):
        self._check_features(features, classes)
        banks = self.token_banks(classes)
        feature = self.input_adapter(features[3])
        for index, (stage, block, bank) in enumerate(zip(self.stages, self.cross_attention, banks)):
            updated_feature, image_tokens = stage(feature)
            cad_tokens = block(image_tokens, bank)
            if index < 3:
                # The self-attention update has already been written by the stage.
                feature = stage.write_back(updated_feature, cad_tokens - image_tokens,
                                           include_bias=False)
                feature = self._next_feature(index, feature, features)
        logits = self.t3_classifier(cad_tokens)
        return self.prediction_from_tokens(cad_tokens, banks, logits, diagnostics)


class HierarchicalCADRegionQueryHead(_MultiscaleCADHead):
    """B: fixed-identity CAD queries read a multiscale image pyramid."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.t3_classifier
        self.query_parents = nn.ModuleList(nn.Linear(self.token_dim, self.token_dim) for _ in range(3))
        self.pixel_projection = nn.Linear(self.token_dim, self.token_dim)
        self.query_projection = nn.Linear(self.token_dim, self.token_dim)

    def forward(self, features, classes, diagnostics=None):
        self._check_features(features, classes)
        banks = self.token_banks(classes)
        feature = self.input_adapter(features[3])
        image_tokens = []
        for index, stage in enumerate(self.stages):
            feature, tokens = stage(feature)
            image_tokens.append(tokens)
            if index < 3:
                feature = self._next_feature(index, feature, features)
        query = None
        for index, (bank, tokens, block) in enumerate(zip(banks, image_tokens, self.cross_attention)):
            if index:
                query = bank + self.query_parents[index-1](query).repeat_interleave(8, dim=1)
            else:
                query = bank
            query = block(query, tokens)
        pixel = self.pixel_projection(image_tokens[-1])
        region = self.query_projection(query)
        logits = torch.bmm(pixel, region.transpose(1, 2)) / math.sqrt(self.token_dim)
        return self.prediction_from_tokens(image_tokens[-1], banks, logits, diagnostics)
