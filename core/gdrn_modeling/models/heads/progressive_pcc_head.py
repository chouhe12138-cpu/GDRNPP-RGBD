"""EXP022 four-stage, GPU-blocked progressive CAD correspondence head."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pcc_blocks import PCCStage, SpatialRefinement


OBJECT_IDS = (1, 5, 6, 8, 9, 10, 11, 12)
LEVEL_SIZES = (8, 64, 512, 4096)


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


class ProgressivePCCHead(nn.Module):
    def __init__(self, hierarchy_path: str, token_dim: int = 256, beam_k: int = 2,
                 route_weight: float = 1.0, residual_weight: float = 1.0,
                 mask_weight: float = 1.0, residual_beta: float = 0.1,
                 num_heads: int = 8,
                 stage_attention: tuple[str, ...] = ("global", "global", "window", "window"),
                 window_size: int = 8, shift_size: int = 4,
                 attention_dropout: float = 0.0):
        super().__init__()
        if beam_k != 2:
            raise ValueError("EXP022 V1 uses beam K=2")
        if tuple(stage_attention) != ("global", "global", "window", "window"):
            raise ValueError("EXP022 V1 requires global/global/window/window image attention")
        if token_dim % num_heads:
            raise ValueError("EXP022 token_dim must be divisible by num_heads")
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
        self.cad_token_encoder = nn.Sequential(
            nn.Linear(10, 128), nn.GELU(), nn.Linear(128, token_dim), nn.LayerNorm(token_dim)
        )
        self.stages = nn.ModuleList(
            PCCStage(ch, token_dim, resolution, attention, num_heads, window_size,
                     shift_size, attention_dropout)
            for ch, resolution, attention in zip((512, 256, 128, 64), (8, 16, 32, 64),
                                                  stage_attention)
        )
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
            encoded.append(self.cad_token_encoder(descriptor))
        return unique, inverse, encoded

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

    @staticmethod
    def _route_diagnostics(logits: torch.Tensor, valid: torch.Tensor):
        probability = F.softmax(logits.detach().float(), dim=-1)
        entropy = -(probability * probability.clamp_min(1e-12).log()).sum(-1)
        top_two = probability.topk(2, dim=-1).values
        denominator = valid.sum(-1).clamp_min(1)
        return tuple((value * valid).sum(-1) / denominator for value in
                     (entropy, top_two[..., 0], top_two.sum(-1)))

    def _stage1(self, backbone: torch.Tensor, inverse: torch.Tensor,
                banks: list[tuple[torch.Tensor, torch.Tensor]], valid: torch.Tensor):
        feature = self.input_adapter(backbone)
        batch = backbone.shape[0]
        stage = self.stages[0]
        query = stage.image_tokens(feature)
        logits, context = stage.matcher.match_root(query, inverse, banks[0])
        feature, update_ratio = stage.fuse(feature, context, collect_stats=True)
        diagnostics = (update_ratio, *self._route_diagnostics(logits, valid))
        return self.refinements[0](feature), logits.view(batch, 64, 8), diagnostics

    def _train_branch(self, feature: torch.Tensor, stage1_logits: torch.Tensor,
                      paths, classes: torch.Tensor, inverse: torch.Tensor,
                      tokens: list[torch.Tensor], banks: list[tuple[torch.Tensor, torch.Tensor]],
                      mask: torch.Tensor, stage1_diagnostics):
        points, valid, _, child = paths[0]
        first_loss = F.cross_entropy(stage1_logits.float().transpose(1, 2),
                                     child.view(classes.shape[0], -1), reduction="none")
        stage_losses = [(first_loss * valid).sum(-1) / valid.sum(-1).clamp_min(1)]
        stage_diagnostics = [stage1_diagnostics]
        final_query = final_parent = final_child = final_target = final_valid = None
        batch = classes.shape[0]
        for depth in range(2, 5):
            stage = self.stages[depth - 1]
            size = 4 * (2 ** depth)
            query = stage.image_tokens(feature)
            points, valid, parent, child = paths[depth - 1]
            point_classes = inverse[:, None].expand(-1, size * size).reshape(-1)
            logits, context, _ = stage.matcher.match_packed(
                query.reshape(-1, self.token_dim), point_classes, parent, banks[depth - 1]
            )
            pixel_loss = F.cross_entropy(logits.float(), child, reduction="none").view(batch, -1)
            stage_losses.append((pixel_loss * valid).sum(-1) / valid.sum(-1).clamp_min(1))
            logits = logits.view(batch, size * size, 8)
            feature, update_ratio = stage.fuse(
                feature, context.view(batch, size * size, self.token_dim), collect_stats=True
            )
            stage_diagnostics.append((update_ratio, *self._route_diagnostics(logits, valid)))
            if depth < 4:
                feature = self.refinements[depth-1](feature)
            else:
                final_query, final_parent, final_child = stage.project_image(feature), parent, child
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
        diagnostics = {
            name: torch.stack([stage_values[index] for stage_values in stage_diagnostics], dim=-1)
            for index, name in enumerate(("fusion_update_ratio", "route_entropy",
                                          "top1_route_prob", "top2_route_prob_mass"))
        }
        return route_loss, residual_loss, mask_loss, diagnostics

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

    @staticmethod
    def _advance_beam(parent: torch.Tensor, prior: torch.Tensor,
                      conditional: torch.Tensor):
        """Select the best two of the retained parents' sixteen child paths."""
        scores = (prior[..., None] + conditional).reshape(-1, 16)
        best_scores, best_slot = scores.topk(2, dim=-1)
        source_parent = parent.gather(1, torch.div(best_slot, 8, rounding_mode="floor"))
        best_ids = source_parent * 8 + best_slot.remainder(8)
        return best_scores, best_ids

    def _infer(self, backbone: torch.Tensor, classes: torch.Tensor,
               inverse: torch.Tensor, tokens: list[torch.Tensor],
               banks: list[tuple[torch.Tensor, torch.Tensor]]):
        batch = classes.shape[0]
        feature = self.input_adapter(backbone)
        beam_ids = beam_scores = None
        level_predictions = []
        level_beams = []
        for depth, stage in enumerate(self.stages, start=1):
            size = 4 * (2 ** depth)
            query = stage.image_tokens(feature)
            pixels = size * size
            object_index = inverse[:, None].expand(-1, pixels).reshape(-1)
            if depth == 1:
                logits, context = stage.matcher.match_root(query, inverse, banks[0])
                all_scores = F.log_softmax(logits.float(), dim=-1)
                best_scores, best_ids = all_scores.reshape(-1, 8).topk(2, dim=-1)
            else:
                beam_ids, beam_scores = self._resize_sparse_paths(beam_ids, beam_scores)
                parent = beam_ids.reshape(batch * pixels, 2)
                prior = beam_scores.reshape(batch * pixels, 2)
                q = query.reshape(-1, self.token_dim).repeat_interleave(2, dim=0)
                object_twice = object_index.repeat_interleave(2)
                logits, local_context, _ = stage.matcher.match_packed(
                    q, object_twice, parent.reshape(-1), banks[depth-1]
                )
                conditional = F.log_softmax(logits.float().reshape(-1, 2, 8), dim=-1)
                best_scores, best_ids = self._advance_beam(parent, prior, conditional)
                parent_prob = torch.softmax(prior.float(), dim=-1).to(local_context.dtype)
                context = (local_context.reshape(-1, 2, self.token_dim) * parent_prob[..., None]).sum(1)
                context = context.reshape(batch, pixels, self.token_dim)
            beam_ids = best_ids.reshape(batch, size, size, 2)
            beam_scores = (best_scores - torch.logsumexp(best_scores, dim=-1, keepdim=True)).reshape(
                batch, size, size, 2
            )
            level_predictions.append(beam_ids[..., 0])
            level_beams.append(beam_ids)
            feature, _ = stage.fuse(feature, context)
            if depth < 4:
                feature = self.refinements[depth-1](feature)
        leaf = beam_ids[..., 0].reshape(batch, 4096)
        anchor = self.level4_anchors[classes[:, None], leaf]
        radius = self.level4_radii[classes[:, None], leaf]
        leaf_token = tokens[3][inverse[:, None], leaf]
        final_query = self.stages[3].project_image(feature)
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
        banks = [stage.matcher.project_bank(token)
                 for stage, token in zip(self.stages, tokens)]
        if gt_xyz_norm is None:
            return self._infer(backbone, roi_classes, inverse, tokens, banks)
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
        stage1_feature, stage1_logits, stage1_diagnostics = self._stage1(
            backbone, inverse, banks, canonical_paths[0][1]
        )
        canonical_output = self._train_branch(
            stage1_feature, stage1_logits, canonical_paths, roi_classes, inverse,
            tokens, banks, gt_mask, stage1_diagnostics
        )
        base = torch.stack(canonical_output[:3], dim=-1)
        diagnostics = canonical_output[3]
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
            alternate_stage1 = tuple(value.index_select(0, symmetric)
                                     for value in stage1_diagnostics)
            alternate_output = self._train_branch(
                stage1_feature.index_select(0, symmetric),
                stage1_logits.index_select(0, symmetric), alternate_paths,
                sym_classes, sym_inverse, tokens, banks, gt_mask.index_select(0, symmetric),
                alternate_stage1
            )
            alternate = torch.stack(alternate_output[:3], dim=-1)
            replacement, use_alternate = self._select_symmetry_losses(
                base.index_select(0, symmetric), alternate
            )
            base = base.index_copy(0, symmetric, replacement)
            selected = selected.index_copy(0, symmetric, use_alternate.long())
            diagnostics = {
                name: values.index_copy(
                    0, symmetric,
                    torch.where(use_alternate[:, None], alternate_output[3][name],
                                values.index_select(0, symmetric))
                ) for name, values in diagnostics.items()
            }
        chosen = base.mean(0)
        return {
            "loss_pcc_route": chosen[0] * self.route_weight,
            "loss_pcc_residual": chosen[1] * self.residual_weight,
            "loss_pcc_mask": chosen[2] * self.mask_weight,
        }, {"selected_symmetry_branch_mean": selected.float().mean(),
            "fusion_gates": torch.stack([torch.sigmoid(stage.gate_logit) for stage in self.stages]).detach(),
            **{name: values.detach().mean(0) for name, values in diagnostics.items()}}
