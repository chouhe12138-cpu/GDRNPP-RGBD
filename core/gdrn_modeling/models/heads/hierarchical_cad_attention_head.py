"""EXP025: one T3 distribution, differentiable T2/T1 marginals, bounded residual."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from core.gdrn_modeling.cad.hierarchy import load_cad_hierarchy
from .cad_attention_blocks import AttentionBlock, CADGeometryEncoder, CADStageTransition, WindowAttentionBlock


def hierarchy_log_probabilities(logits):
    """Return normalized T1/T2/T3 log probabilities, without detaching gradients."""
    if logits.ndim != 4 or logits.shape[1] != 512:
        raise ValueError('T3 logits must have shape [B,512,H,W]')
    with torch.autocast(device_type=logits.device.type, enabled=False):
        b, _, h, w = logits.shape
        t3 = F.log_softmax(logits.float(), dim=1)
        t2 = torch.logsumexp(t3.reshape(b, 64, 8, h, w), dim=2)
        t1 = torch.logsumexp(t2.reshape(b, 8, 8, h, w), dim=2)
    return t1, t2, t3


@torch.no_grad()
def nested_targets(points, classes, anchors):
    """FP32 nested nearest-child traversal; [B,P,3] -> three global [B,P] IDs."""
    with torch.autocast(device_type=points.device.type, enabled=False):
        parent = torch.zeros(points.shape[:2], device=points.device, dtype=torch.long)
        path = []
        for level in anchors:
            children = level.float().reshape(level.shape[0], -1, 8, 3)[classes[:, None], parent]
            child = (points.float()[:, :, None] - children).square().sum(-1).argmin(-1)
            parent = parent * 8 + child
            path.append(parent)
    return tuple(path)


def bounded_residual(raw):
    # FP32 avoids AMP rounding outside the unit sphere.
    value = raw.float().tanh()
    return value / value.norm(dim=1, keepdim=True).clamp_min(1.)


def masked_mean(values, valid):
    return torch.where(valid, values, torch.zeros_like(values)).sum(-1) / valid.sum(-1).clamp_min(1)


class HierarchicalCADAttentionHead(nn.Module):
    def __init__(self, hierarchy_path, token_dim=256, num_heads=8,
                 expected_object_ids=None, dataset_key='lmo', route_weight=1.,
                 residual_weight=1., mask_weight=1., residual_beta=.1):
        super().__init__()
        hierarchy = load_cad_hierarchy(hierarchy_path, expected_object_ids=expected_object_ids,
                                       dataset_key=dataset_key)
        if hierarchy.metadata.get('mode') != 'consistent' or hierarchy.metadata.get('generator_version') != 3:
            raise ValueError('EXP025 requires consistent_v3')
        if hierarchy.level_counts[:3] != (8, 64, 512) or int(hierarchy.symmetry_counts.max()) > 2:
            raise ValueError('EXP025 requires T1/T2/T3=8/64/512 and at most two symmetries')
        self.num_objects = len(hierarchy.object_ids)
        for name in ('object_ids', 'extents', 'diameters', 'symmetry_counts', 'symmetry_transforms'):
            self.register_buffer(name, getattr(hierarchy, name), persistent=False)
        for depth in (1, 2, 3):
            for field in ('anchors', 'normals', 'radii'):
                self.register_buffer(f'level{depth}_{field}', getattr(hierarchy.level(depth), field), persistent=False)
        self.route_weight, self.residual_weight, self.mask_weight = route_weight, residual_weight, mask_weight
        self.residual_beta = residual_beta
        self.geometry = CADGeometryEncoder(token_dim, num_heads)
        self.decoder = nn.Sequential(nn.Conv2d(1024, 512, 1), CADStageTransition(512, 256),
                                     CADStageTransition(256, 128), CADStageTransition(128, 64))
        self.image_projection = nn.Conv2d(64, token_dim, 1)
        self.image_attention = nn.Sequential(WindowAttentionBlock(token_dim, num_heads),
                                            WindowAttentionBlock(token_dim, num_heads, shift=4))
        self.cross_attention = nn.ModuleList(AttentionBlock(token_dim, num_heads) for _ in range(4))
        self.t3_classifier = nn.Linear(token_dim, 512)
        self.residual_predictor = nn.Linear(token_dim, 3)
        self.mask_predictor = nn.Linear(token_dim, 1)

    def token_banks(self, classes):
        unique, inverse = torch.unique(classes, sorted=True, return_inverse=True)
        extent = self.extents[unique, None]
        diameter = self.diameters[unique, None, None]
        descriptors = []
        for depth in (1, 2, 3):
            anchor = getattr(self, f'level{depth}_anchors')[unique]
            normal = getattr(self, f'level{depth}_normals')[unique]
            radius = getattr(self, f'level{depth}_radii')[unique]
            relative = (torch.zeros_like(anchor) if depth == 1 else
                        anchor - getattr(self, f'level{depth-1}_anchors')[unique].repeat_interleave(8, 1))
            descriptors.append(torch.cat((anchor/extent, normal, relative/extent, radius[..., None]/diameter), -1))
        return tuple(bank[inverse] for bank in self.geometry(descriptors))

    def encode(self, feature, classes):
        """Image tokens after self-attention and the T0/T1/T2/T3 cross-attention ladder."""
        tokens = self.image_projection(self.decoder(feature)).flatten(2).transpose(1, 2)
        tokens = self.image_attention(tokens)
        banks = self.token_banks(classes)
        for block, bank in zip(self.cross_attention, banks):
            tokens = block(tokens, bank)
        return tokens, banks

    def forward(self, feature, classes, diagnostics=None):
        if feature.ndim != 4 or feature.shape[1:] != (1024, 8, 8):
            raise ValueError('EXP025 requires [B,1024,8,8] backbone features')
        if classes.shape != (len(feature),) or torch.any((classes < 0) | (classes >= self.num_objects)):
            raise ValueError('Invalid EXP025 ROI classes')
        tokens, banks = self.encode(feature, classes)
        b = len(feature)
        dense = lambda x: x.transpose(1, 2).reshape(b, -1, 64, 64)
        raw = dense(self.residual_predictor(tokens))
        prediction = dict(t3_logits=dense(self.t3_classifier(tokens)),
                          residual=bounded_residual(raw),
                          mask_logit=dense(self.mask_predictor(tokens)))
        if diagnostics is not None:
            # Detached tensors only; statistics live in the diagnostic runners.
            diagnostics.update(image_tokens=tokens.detach(), cad_tokens=banks[3].detach(),
                               t3_logits=prediction['t3_logits'].detach(),
                               mask_logit=prediction['mask_logit'].detach(),
                               raw_residual=raw.detach(), residual=prediction['residual'].detach())
        return prediction

    def decode(self, prediction, classes, ids=None):
        """Optional IDs are for isolated diagnostics only; forward never consumes GT."""
        ids = prediction['t3_logits'].argmax(1) if ids is None else ids
        shape = ids.shape
        ids = ids.reshape(len(classes), -1)
        anchor = self.level3_anchors[classes[:, None], ids].float()
        radius = self.level3_radii[classes[:, None], ids].float()
        residual = prediction['residual'].flatten(2).transpose(1, 2)
        xyz = anchor + radius[..., None] * residual
        return (xyz / self.extents[classes, None] + .5).transpose(1, 2).reshape(len(classes), 3, *shape[-2:])

    @torch.no_grad()
    def targets(self, xyz_norm, mask, classes, branch):
        # Geometry and symmetry transforms must not inherit training AMP.
        with torch.autocast(device_type=xyz_norm.device.type, enabled=False):
            return self._targets_fp32(xyz_norm, mask, classes, branch)

    def _targets_fp32(self, xyz_norm, mask, classes, branch):
        xyz = xyz_norm.float().flatten(2).transpose(1, 2)
        valid = (mask.reshape(len(classes), -1) > .5) & torch.isfinite(xyz).all(-1)
        xyz = torch.where(valid[..., None], xyz, torch.full_like(xyz, .5))
        xyz = (xyz - .5) * self.extents[classes, None]
        transforms = self.symmetry_transforms[classes, branch].float()
        points = torch.bmm(xyz - transforms[:, None, :3, 3], transforms[:, :3, :3])
        path = nested_targets(points, classes, [getattr(self, f'level{d}_anchors') for d in (1, 2, 3)])
        anchor = self.level3_anchors[classes[:, None], path[2]]
        radius = self.level3_radii[classes[:, None], path[2]]
        target = (points - anchor) / radius[..., None]
        return path, target, valid

    def loss(self, prediction, classes, xyz_norm, mask, route_weight=None, return_targets=False):
        if mask.ndim == 3:
            mask = mask[:, None]
        route_weight = self.route_weight if route_weight is None else route_weight
        log_probs = hierarchy_log_probabilities(prediction['t3_logits'])
        residual = prediction['residual'].flatten(2).transpose(1, 2)
        per_branch, targets = [], []
        # Padded transform entries are never selected for objects with only one symmetry.
        for branch in range(self.symmetry_transforms.shape[1]):
            if branch >= 2:
                break
            path, target, valid = self.targets(xyz_norm, mask, classes, branch)
            levels = [masked_mean(F.nll_loss(lp.flatten(2), ids, reduction='none'), valid)
                      for lp, ids in zip(log_probs, path)]
            res = masked_mean(F.smooth_l1_loss(residual, target, beta=self.residual_beta,
                                              reduction='none').mean(-1), valid)
            per_branch.append(torch.stack((*levels, res), -1))
            targets.append((path, target, valid))
        values = torch.stack(per_branch, 1)
        scores = route_weight * values[:, :, :3].sum(-1) + self.residual_weight * values[:, :, 3]
        allowed = torch.arange(values.shape[1], device=classes.device)[None] < self.symmetry_counts[classes, None]
        branch = scores.detach().masked_fill(~allowed, float('inf')).argmin(1)
        selected = values[torch.arange(len(classes), device=classes.device), branch]
        losses = {f'loss_cad_t{d+1}': selected[:, d].mean() * route_weight for d in range(3)}
        losses['loss_cad_residual'] = selected[:, 3].mean() * self.residual_weight
        losses['loss_cad_mask'] = F.binary_cross_entropy_with_logits(prediction['mask_logit'].float(), mask.float()) * self.mask_weight
        valid = targets[0][2]
        norms = torch.stack([t[1].norm(dim=-1) for t in targets], 1)
        chosen_norms = norms[torch.arange(len(classes), device=classes.device), branch]
        stats = dict(cad_route_sum=selected[:, :3].sum(-1).mean().detach(),
                     cad_symmetry_branch=branch.float().mean(), cad_valid_points=valid.sum().float(),
                     cad_target_outside=masked_mean((chosen_norms > 1).float(), valid).mean())
        if return_targets:
            paths = torch.stack([t[0][2] for t in targets], 1)
            chosen_ids = paths[torch.arange(len(classes), device=classes.device), branch]
            return losses, stats, chosen_ids.reshape(len(classes), *xyz_norm.shape[-2:]), valid
        return losses, stats
