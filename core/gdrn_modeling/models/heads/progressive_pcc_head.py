"""EXP022 four-stage, GPU-blocked progressive CAD correspondence head."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .global_hierarchical_cad_head import GlobalGuidedHierarchicalCADHead


OBJECT_IDS = (1, 5, 6, 8, 9, 10, 11, 12)
LEVEL_SIZES = (8, 64, 512, 4096)
MATCH_BLOCK_SIZE = 16


def load_pcc_hierarchy(path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"EXP022 hierarchy missing: {path}")
    with np.load(path, allow_pickle=False) as data:
        expected = {"object_ids", "extents", "diameters", "symmetry_counts", "symmetry_transforms",
                    "generator_version", "mode", "source_leaf_indices"}
        for depth in range(1, 5):
            expected.update({f"level{depth}_{field}" for field in ("anchors", "normals", "radii")})
        missing = expected.difference(data.files)
        if missing:
            raise ValueError(f"EXP022 hierarchy missing arrays: {sorted(missing)}")
        version, mode = int(data["generator_version"]), str(data["mode"])
        if (mode, version) not in {("reused", 1), ("independent", 2)}:
            raise ValueError("EXP022 hierarchy version or mode mismatch")
        arrays = {name: torch.from_numpy(np.asarray(data[name]).copy()) for name in expected
                  if name not in {"generator_version", "mode"}}
    if tuple(arrays["object_ids"].tolist()) != OBJECT_IDS:
        raise ValueError("EXP022 object order mismatch")
    for depth, count in enumerate(LEVEL_SIZES, start=1):
        for field, shape in (("anchors", (8, count, 3)), ("normals", (8, count, 3)),
                             ("radii", (8, count))):
            value = arrays[f"level{depth}_{field}"]
            if tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid EXP022 level{depth}_{field}: {tuple(value.shape)}")
            if field == "radii" and torch.any(value <= 0):
                raise ValueError(f"Non-positive EXP022 level{depth} radius")
    if tuple(arrays["source_leaf_indices"].shape) != (8, 4096):
        raise ValueError("EXP022 source_leaf_indices shape mismatch")
    return arrays


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


class PCCStage(nn.Module):
    def __init__(self, channels: int, token_dim: int):
        super().__init__()
        self.query = nn.Conv2d(channels, token_dim, 1)
        self.query_norm = nn.LayerNorm(token_dim)
        self.context = nn.Linear(token_dim, channels)
        self.gate_logit = nn.Parameter(torch.tensor(-4.0))

    def queries(self, feature: torch.Tensor) -> torch.Tensor:
        return self.query_norm(self.query(feature).flatten(2).transpose(1, 2))

    def fuse(self, feature: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = feature.shape
        update = self.context(context).transpose(1, 2).reshape(batch, channels, height, width)
        return feature + torch.sigmoid(self.gate_logit) * update


class ProgressivePCCHead(nn.Module):
    def __init__(self, hierarchy_path: str, token_dim: int = 256, beam_k: int = 2,
                 route_weight: float = 1.0, residual_weight: float = 1.0,
                 mask_weight: float = 1.0, residual_beta: float = 0.1):
        super().__init__()
        if beam_k != 2:
            raise ValueError("EXP022 V1 uses beam K=2")
        self.token_dim = token_dim
        self.beam_k = beam_k
        self.route_weight = float(route_weight)
        self.residual_weight = float(residual_weight)
        self.mask_weight = float(mask_weight)
        self.residual_beta = float(residual_beta)
        hierarchy = load_pcc_hierarchy(hierarchy_path)
        for name, value in hierarchy.items():
            self.register_buffer(name, value, persistent=False)
        self.input_adapter = nn.Conv2d(1024, 512, 1)
        self.token_encoder = nn.Sequential(
            nn.Linear(10, 128), nn.GELU(), nn.Linear(128, token_dim), nn.LayerNorm(token_dim)
        )
        self.stages = nn.ModuleList(PCCStage(ch, token_dim) for ch in (512, 256, 128, 64))
        self.refinements = nn.ModuleList(
            SpatialRefinement(before, after) for before, after in ((512, 256), (256, 128), (128, 64))
        )
        self.residual_head = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim), nn.GELU(), nn.Linear(token_dim, 3)
        )
        self.mask_head = nn.Conv2d(64, 1, 1)

    def _tokens(self, roi_classes: torch.Tensor):
        unique, inverse = torch.unique(roi_classes.long(), sorted=True, return_inverse=True)
        extent = self.extents.index_select(0, unique).clamp_min(1e-8)
        diameter = self.diameters.index_select(0, unique).clamp_min(1e-8)
        encoded = []
        for depth, count in enumerate(LEVEL_SIZES, start=1):
            anchors = getattr(self, f"level{depth}_anchors").index_select(0, unique)
            normals = getattr(self, f"level{depth}_normals").index_select(0, unique)
            radii = getattr(self, f"level{depth}_radii").index_select(0, unique)
            if depth == 1:
                relative = torch.zeros_like(anchors)
            else:
                parent = getattr(self, f"level{depth-1}_anchors").index_select(0, unique)
                relative = anchors - parent.repeat_interleave(8, dim=1)
            descriptor = torch.cat((anchors / extent[:, None], normals,
                                    relative / extent[:, None],
                                    radii[..., None] / diameter[:, None, None]), dim=-1)
            encoded.append(self.token_encoder(descriptor))
        return unique, inverse, encoded

    @staticmethod
    def _squared_distances(points: torch.Tensor, anchors: torch.Tensor):
        return GlobalGuidedHierarchicalCADHead._squared_distances(points, anchors)

    def _match(self, query: torch.Tensor, object_inverse: torch.Tensor,
               parent: torch.Tensor, token_bank: torch.Tensor,
               target: torch.Tensor | None = None, anchors: torch.Tensor | None = None):
        """Pack equal routes into statically bounded blocks without a CUDA scalar read."""
        num_parents = token_bank.shape[1] // 8
        keys = object_inverse.long() * num_parents + parent.long()
        count = query.shape[0]
        num_keys = token_bank.shape[0] * num_parents
        # Sum(ceil(group_count / block)) is bounded by ceil(count / block) + num_keys.
        max_blocks = (count + MATCH_BLOCK_SIZE - 1) // MATCH_BLOCK_SIZE + num_keys
        order = torch.argsort(keys)
        sorted_keys = keys.index_select(0, order)
        group_counts = torch.bincount(keys, minlength=num_keys)
        group_starts = group_counts.cumsum(0) - group_counts
        blocks_per_group = torch.div(group_counts + MATCH_BLOCK_SIZE - 1,
                                     MATCH_BLOCK_SIZE, rounding_mode="floor")
        block_starts = blocks_per_group.cumsum(0) - blocks_per_group
        within = torch.arange(count, device=query.device) - group_starts[sorted_keys]
        block_ids = block_starts[sorted_keys] + torch.div(
            within, MATCH_BLOCK_SIZE, rounding_mode="floor"
        )
        slots = within.remainder(MATCH_BLOCK_SIZE)
        packed = torch.full((max_blocks, MATCH_BLOCK_SIZE), -1,
                            device=query.device, dtype=torch.long)
        packed[block_ids, slots] = torch.arange(count, device=query.device)
        valid = packed >= 0
        safe = packed.clamp_min(0)
        block_keys = torch.searchsorted(block_starts.contiguous(),
                                         torch.arange(max_blocks, device=query.device),
                                         right=True).sub(1).clamp_(0, num_keys - 1)
        token = token_bank.reshape(-1, 8, self.token_dim).index_select(0, block_keys)
        block_query = query.index_select(0, order)[safe]
        logits = torch.bmm(block_query, token.transpose(1, 2)) / math.sqrt(self.token_dim)
        probability = torch.softmax(logits.float(), dim=-1).to(token.dtype)
        context = torch.bmm(probability, token)
        # All padded slots write to a discarded sentinel, leaving one scatter per output.
        positions = torch.where(valid.flatten(), order[safe.flatten()], count)
        out_logits = torch.zeros((count + 1, 8), device=query.device, dtype=logits.dtype).index_copy(
            0, positions, logits.flatten(0, 1)
        )[:count]
        out_context = torch.zeros((count + 1, self.token_dim), device=query.device,
                                  dtype=context.dtype).index_copy(
            0, positions, context.flatten(0, 1)
        )[:count]
        if target is None:
            return out_logits, out_context, None
        block_target = target.index_select(0, order)[safe].float()
        block_anchors = anchors.reshape(-1, 8, 3).index_select(0, block_keys).float()
        with torch.no_grad(), torch.autocast(device_type=query.device.type, enabled=False):
            label = self._squared_distances(block_target, block_anchors).argmin(-1)
        out_label = torch.zeros(count + 1, device=query.device, dtype=torch.long).index_copy(
            0, positions, label.flatten()
        )[:count]
        return out_logits, out_context, out_label

    @staticmethod
    def _targets_at_resolution(xyz: torch.Tensor, mask: torch.Tensor, size: int):
        support = F.interpolate(mask.float(), size=(size, size), mode="area")
        metric = F.interpolate(xyz * mask, size=(size, size), mode="area") / support.clamp_min(1e-6)
        return metric.flatten(2).transpose(1, 2), (support[:, 0].flatten(1) > 0.5)

    @staticmethod
    def _nearest_child(points: torch.Tensor, object_inverse: torch.Tensor,
                       parent: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        grouped = anchors.reshape(anchors.shape[0], -1, 8, 3)
        candidates = grouped[object_inverse, parent].float()
        with torch.no_grad(), torch.autocast(device_type=points.device.type, enabled=False):
            return (points.float()[:, None] - candidates).square().sum(-1).argmin(-1)

    def _target_paths(self, targets, inverse: torch.Tensor, level_anchors):
        """Prepare all resolution-specific teacher paths outside the stage forward."""
        paths = []
        batch = inverse.shape[0]
        with torch.no_grad():
            for depth, (points, valid) in enumerate(targets, start=1):
                pixels = points.shape[1]
                object_index = inverse[:, None].expand(-1, pixels).reshape(-1)
                parent = torch.zeros(batch * pixels, device=inverse.device, dtype=torch.long)
                flat_points = points.reshape(-1, 3)
                for ancestor in range(depth - 1):
                    parent = parent * 8 + self._nearest_child(
                        flat_points, object_index, parent, level_anchors[ancestor]
                    )
                child = self._nearest_child(flat_points, object_index, parent,
                                            level_anchors[depth - 1])
                paths.append((points, valid, parent, child))
        return paths

    @staticmethod
    def _transform_targets(targets, rotation: torch.Tensor, translation: torch.Tensor):
        # Rigid transforms commute with mask-weighted area averaging at each resolution.
        return [(torch.bmm(points - translation[:, None], rotation), valid)
                for points, valid in targets]

    def _stage1(self, backbone: torch.Tensor, inverse: torch.Tensor,
                tokens: list[torch.Tensor]):
        feature = self.input_adapter(backbone)
        batch = backbone.shape[0]
        query = self.stages[0].queries(feature)
        object_index = inverse[:, None].expand(-1, 64).reshape(-1)
        parent = torch.zeros(batch * 64, device=inverse.device, dtype=torch.long)
        logits, context, _ = self._match(query.reshape(-1, self.token_dim), object_index,
                                         parent, tokens[0])
        feature = self.stages[0].fuse(feature, context.view(batch, 64, self.token_dim))
        return self.refinements[0](feature), logits.view(batch, 64, 8)

    def _train_branch(self, feature: torch.Tensor, stage1_logits: torch.Tensor,
                      paths, classes: torch.Tensor, inverse: torch.Tensor,
                      tokens: list[torch.Tensor], mask: torch.Tensor):
        points, valid, _, child = paths[0]
        first_loss = F.cross_entropy(stage1_logits.float().transpose(1, 2),
                                     child.view(classes.shape[0], -1), reduction="none")
        stage_losses = [(first_loss * valid).sum(-1) / valid.sum(-1).clamp_min(1)]
        final_query = final_parent = final_child = final_target = final_valid = None
        batch = classes.shape[0]
        for depth in range(2, 5):
            stage = self.stages[depth - 1]
            size = 4 * (2 ** depth)
            query = stage.queries(feature)
            points, valid, parent, child = paths[depth - 1]
            point_classes = inverse[:, None].expand(-1, size * size).reshape(-1)
            logits, context, _ = self._match(query.reshape(-1, self.token_dim),
                                             point_classes, parent, tokens[depth - 1])
            pixel_loss = F.cross_entropy(logits.float(), child, reduction="none").view(batch, -1)
            stage_losses.append((pixel_loss * valid).sum(-1) / valid.sum(-1).clamp_min(1))
            feature = stage.fuse(feature, context.view(batch, size * size, self.token_dim))
            if depth < 4:
                feature = self.refinements[depth-1](feature)
            else:
                final_query, final_parent, final_child = stage.queries(feature), parent, child
                final_target, final_valid = points, valid
        leaf = final_parent * 8 + final_child
        leaf_anchor = self.level4_anchors[classes[:, None], leaf.view(batch, -1)]
        leaf_radius = self.level4_radii[classes[:, None], leaf.view(batch, -1)]
        leaf_token = tokens[3][inverse[:, None], leaf.view(batch, -1)]
        raw = torch.tanh(self.residual_head(torch.cat((final_query, leaf_token), dim=-1)))
        residual = raw / raw.norm(dim=-1, keepdim=True).clamp_min(1.0)
        target_residual = (final_target - leaf_anchor) / leaf_radius[..., None]
        residual_loss = F.smooth_l1_loss(
            residual.float(), target_residual.float(), beta=self.residual_beta,
            reduction="none",
        ).mean(-1)
        residual_loss = (residual_loss * final_valid).sum(-1) / final_valid.sum(-1).clamp_min(1)
        mask_logit = self.mask_head(feature)
        mask_loss = F.binary_cross_entropy_with_logits(
            mask_logit.float(), mask.float(), reduction="none"
        ).flatten(1).mean(-1)
        route_loss = torch.stack(stage_losses).mean(0)
        return route_loss, residual_loss, mask_loss, mask_logit

    def _select_symmetry_losses(self, canonical: torch.Tensor,
                                alternate: torch.Tensor):
        """Select a CAD-equivalent branch by geometry, then keep all its losses."""
        weights = canonical.new_tensor([self.route_weight, self.residual_weight])
        use_alternate = (((alternate[:, :2] - canonical[:, :2]) * weights)
                         .sum(-1).detach() < 0)
        return torch.where(use_alternate[:, None], alternate, canonical), use_alternate

    @staticmethod
    def _resize_sparse_paths(indices: torch.Tensor, log_scores: torch.Tensor):
        """Bilinearly mix four spatial neighbours of a sparse categorical path state."""
        batch, height, width, k = indices.shape
        new_h, new_w = height * 2, width * 2
        yy = (torch.arange(new_h, device=indices.device, dtype=torch.float32) + 0.5) / 2 - 0.5
        xx = (torch.arange(new_w, device=indices.device, dtype=torch.float32) + 0.5) / 2 - 0.5
        y0, x0 = yy.floor(), xx.floor()
        wy, wx = yy - y0, xx - x0
        y_indices = torch.stack((y0, y0 + 1), -1).long().clamp(0, height - 1)
        x_indices = torch.stack((x0, x0 + 1), -1).long().clamp(0, width - 1)
        y_weights = torch.stack((1 - wy, wy), -1)
        x_weights = torch.stack((1 - wx, wx), -1)
        flat_ids = indices.reshape(batch, height * width, k)
        flat_prob = log_scores.exp().reshape(batch, height * width, k)
        candidate_ids, candidate_prob = [], []
        for yi in range(2):
            for xi in range(2):
                position = (y_indices[:, yi, None] * width + x_indices[None, :, xi]).reshape(-1)
                weight = (y_weights[:, yi, None] * x_weights[None, :, xi]).reshape(1, -1, 1)
                candidate_ids.append(flat_ids.index_select(1, position))
                candidate_prob.append(flat_prob.index_select(1, position) * weight)
        ids = torch.cat(candidate_ids, dim=-1)
        probability = torch.cat(candidate_prob, dim=-1)
        sorted_ids, order = torch.sort(ids, dim=-1)
        sorted_prob = probability.gather(-1, order)
        first = torch.ones_like(sorted_ids, dtype=torch.bool)
        first[..., 1:] = sorted_ids[..., 1:] != sorted_ids[..., :-1]
        group = first.long().cumsum(-1) - 1
        merged = torch.zeros_like(sorted_prob).scatter_add(-1, group, sorted_prob)
        group_ids = torch.zeros_like(sorted_ids).scatter(-1, group, sorted_ids)
        values, slot = merged.topk(k, dim=-1)
        result_ids = group_ids.gather(-1, slot).reshape(batch, new_h, new_w, k)
        scores = (values / values.sum(-1, keepdim=True).clamp_min(1e-12)).clamp_min(1e-12).log()
        return result_ids, scores.reshape(batch, new_h, new_w, k)

    def _infer(self, backbone: torch.Tensor, classes: torch.Tensor,
               inverse: torch.Tensor, tokens: list[torch.Tensor]):
        batch = classes.shape[0]
        feature = self.input_adapter(backbone)
        beam_ids = beam_scores = None
        level_predictions = []
        level_beams = []
        for depth, stage in enumerate(self.stages, start=1):
            size = 4 * (2 ** depth)
            query = stage.queries(feature)
            pixels = size * size
            object_index = inverse[:, None].expand(-1, pixels).reshape(-1)
            if depth == 1:
                parent = torch.zeros(batch * pixels, device=classes.device, dtype=torch.long)
                logits, context, _ = self._match(query.reshape(-1, self.token_dim), object_index,
                                                  parent, tokens[0])
                all_scores = F.log_softmax(logits.float(), dim=-1)
                best_scores, best_ids = all_scores.topk(2, dim=-1)
                context = context.view(batch, pixels, self.token_dim)
            else:
                beam_ids, beam_scores = self._resize_sparse_paths(beam_ids, beam_scores)
                parent = beam_ids.reshape(batch * pixels, 2)
                prior = beam_scores.reshape(batch * pixels, 2)
                q = query.reshape(-1, self.token_dim).repeat_interleave(2, dim=0)
                object_twice = object_index.repeat_interleave(2)
                logits, local_context, _ = self._match(q, object_twice, parent.reshape(-1), tokens[depth-1])
                conditional = F.log_softmax(logits.float().reshape(-1, 2, 8), dim=-1)
                all_scores = (prior[..., None] + conditional).reshape(-1, 16)
                best_scores, best_slot = all_scores.topk(2, dim=-1)
                source_parent = parent.gather(1, torch.div(best_slot, 8, rounding_mode="floor"))
                best_ids = source_parent * 8 + best_slot.remainder(8)
                parent_prob = torch.softmax(prior.float(), dim=-1).to(local_context.dtype)
                context = (local_context.reshape(-1, 2, self.token_dim) * parent_prob[..., None]).sum(1)
                context = context.reshape(batch, pixels, self.token_dim)
            beam_ids = best_ids.reshape(batch, size, size, 2)
            beam_scores = (best_scores - torch.logsumexp(best_scores, dim=-1, keepdim=True)).reshape(
                batch, size, size, 2
            )
            level_predictions.append(beam_ids[..., 0])
            level_beams.append(beam_ids)
            feature = stage.fuse(feature, context)
            if depth < 4:
                feature = self.refinements[depth-1](feature)
        leaf = beam_ids[..., 0].reshape(batch, 4096)
        anchor = self.level4_anchors[classes[:, None], leaf]
        radius = self.level4_radii[classes[:, None], leaf]
        leaf_token = tokens[3][inverse[:, None], leaf]
        final_query = self.stages[3].queries(feature)
        raw = torch.tanh(self.residual_head(torch.cat((final_query, leaf_token), dim=-1)))
        residual = raw / raw.norm(dim=-1, keepdim=True).clamp_min(1.0)
        xyz_metric = anchor + radius[..., None] * residual
        xyz = xyz_metric / self.extents.index_select(0, classes)[:, None] + 0.5
        return {
            "xyz_norm": xyz.transpose(1, 2).reshape(batch, 3, 64, 64),
            "mask_logit": self.mask_head(feature),
            "route_ids": level_predictions,
            "route_beams": level_beams,
            "leaf_ids": leaf.reshape(batch, 64, 64),
            "beam_scores": beam_scores,
            "residual_norm": residual.norm(dim=-1).reshape(batch, 64, 64),
        }

    def forward(self, backbone: torch.Tensor, roi_classes: torch.Tensor,
                gt_xyz_norm: torch.Tensor | None = None, gt_mask: torch.Tensor | None = None):
        if backbone.ndim != 4 or tuple(backbone.shape[1:]) != (1024, 8, 8):
            raise ValueError(f"EXP022 expects [B,1024,8,8], got {tuple(backbone.shape)}")
        unique, inverse, tokens = self._tokens(roi_classes)
        if gt_xyz_norm is None:
            return self._infer(backbone, roi_classes, inverse, tokens)
        if gt_mask is None:
            raise ValueError("EXP022 requires visible mask with XYZ targets")
        if gt_mask.ndim == 3:
            gt_mask = gt_mask[:, None]
        metric = (gt_xyz_norm.float() - 0.5) * self.extents.index_select(0, roi_classes)[:, :, None, None]
        target_pyramid = [self._targets_at_resolution(metric, gt_mask, 4 * (2 ** depth))
                          for depth in range(1, 5)]
        level_anchors = [getattr(self, f"level{depth}_anchors").index_select(0, unique)
                         for depth in range(1, 5)]
        transforms = self.symmetry_transforms.index_select(0, roi_classes).float()
        counts = self.symmetry_counts.index_select(0, roi_classes)
        rotation = transforms[:, 0, :3, :3]
        translation = transforms[:, 0, :3, 3]
        canonical = self._transform_targets(target_pyramid, rotation, translation)
        canonical_paths = self._target_paths(canonical, inverse, level_anchors)
        stage1_feature, stage1_logits = self._stage1(backbone, inverse, tokens)
        base = torch.stack(self._train_branch(
            stage1_feature, stage1_logits, canonical_paths, roi_classes, inverse, tokens, gt_mask
        )[:3], dim=-1)
        selected = torch.zeros(len(roi_classes), device=roi_classes.device, dtype=torch.long)
        symmetric = torch.nonzero(counts > 1, as_tuple=False).flatten()
        if symmetric.numel():
            sym_classes = roi_classes.index_select(0, symmetric)
            sym_inverse = inverse.index_select(0, symmetric)
            sym_transforms = transforms.index_select(0, symmetric)
            rotation = sym_transforms[:, 1, :3, :3]
            translation = sym_transforms[:, 1, :3, 3]
            alternate_targets = self._transform_targets(
                [(points.index_select(0, symmetric), valid.index_select(0, symmetric))
                 for points, valid in target_pyramid], rotation, translation
            )
            alternate_paths = self._target_paths(alternate_targets, sym_inverse, level_anchors)
            alternate = torch.stack(self._train_branch(
                stage1_feature.index_select(0, symmetric),
                stage1_logits.index_select(0, symmetric), alternate_paths,
                sym_classes, sym_inverse, tokens, gt_mask.index_select(0, symmetric)
            )[:3], dim=-1)
            replacement, use_alternate = self._select_symmetry_losses(
                base.index_select(0, symmetric), alternate
            )
            base = base.index_copy(0, symmetric, replacement)
            selected = selected.index_copy(0, symmetric, use_alternate.long())
        chosen = base.mean(0)
        return {
            "loss_pcc_route": chosen[0] * self.route_weight,
            "loss_pcc_residual": chosen[1] * self.residual_weight,
            "loss_pcc_mask": chosen[2] * self.mask_weight,
        }, {"selected_symmetry_branch_mean": selected.float().mean(),
            "fusion_gates": torch.stack([torch.sigmoid(stage.gate_logit) for stage in self.stages]).detach()}
