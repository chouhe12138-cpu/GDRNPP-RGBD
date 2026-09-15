"""EXP021 global-guided hierarchical CAD correspondence decoder."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LMO_OBJECT_IDS = (1, 5, 6, 8, 9, 10, 11, 12)
HIERARCHY_VERSION = 1
HIERARCHY_SAMPLE_COUNT = 200_000
HIERARCHY_SEED = 20260914
_ROUTE_BLOCK_SIZE = 256
_COARSE_LABEL_CHUNK_SIZE = 32768


def load_hierarchy(path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"EXP021 CAD hierarchy is missing: {path}. "
            "Run python -m research.exp021.build_cad_hierarchy first."
        )
    with np.load(path, allow_pickle=False) as data:
        required = {
            "object_ids",
            "extents",
            "diameters",
            "coarse_anchors",
            "coarse_normals",
            "coarse_radii",
            "fine_anchors",
            "fine_normals",
            "fine_radii",
            "symmetry_transforms",
            "symmetry_counts",
            "generator_version",
            "sample_count",
            "seed",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"EXP021 hierarchy missing arrays: {sorted(missing)}")
        metadata = {
            name: int(np.asarray(data[name]))
            for name in ("generator_version", "sample_count", "seed")
        }
        runtime_names = required.difference(metadata)
        arrays = {
            name: torch.from_numpy(np.asarray(data[name]).copy())
            for name in runtime_names
        }
    expected_metadata = {
        "generator_version": HIERARCHY_VERSION,
        "sample_count": HIERARCHY_SAMPLE_COUNT,
        "seed": HIERARCHY_SEED,
    }
    if metadata != expected_metadata:
        raise ValueError(
            f"EXP021 hierarchy metadata mismatch: got {metadata}, expected {expected_metadata}"
        )
    if tuple(arrays["object_ids"].tolist()) != LMO_OBJECT_IDS:
        raise ValueError(
            f"EXP021 hierarchy object order must be {LMO_OBJECT_IDS}, "
            f"got {tuple(arrays['object_ids'].tolist())}"
        )
    expected_shapes = {
        "extents": (8, 3),
        "diameters": (8,),
        "coarse_anchors": (8, 64, 3),
        "coarse_normals": (8, 64, 3),
        "coarse_radii": (8, 64),
        "fine_anchors": (8, 64, 64, 3),
        "fine_normals": (8, 64, 64, 3),
        "fine_radii": (8, 64, 64),
        "symmetry_counts": (8,),
    }
    for name, shape in expected_shapes.items():
        if tuple(arrays[name].shape) != shape:
            raise ValueError(
                f"EXP021 hierarchy {name} must have shape {shape}, "
                f"got {tuple(arrays[name].shape)}"
            )
    transforms = arrays["symmetry_transforms"]
    if transforms.ndim != 4 or transforms.shape[0] != 8 or tuple(transforms.shape[2:]) != (4, 4):
        raise ValueError(
            "EXP021 hierarchy symmetry_transforms must have shape [8,S,4,4]"
        )
    counts = arrays["symmetry_counts"]
    if torch.any(counts < 1) or torch.any(counts > transforms.shape[1]):
        raise ValueError("EXP021 hierarchy symmetry_counts are outside stored transform bounds")
    for name, value in arrays.items():
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            raise ValueError(f"EXP021 hierarchy contains non-finite {name}")
    if torch.any(arrays["fine_radii"] <= 0):
        raise ValueError("EXP021 hierarchy leaf radii must be positive")
    return arrays


class _GlobalCADBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=0.0, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=0.0, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim)
        )

    def forward(self, tokens: torch.Tensor, cad_tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.self_norm(tokens)
        tokens = tokens + self.self_attn(normalized, normalized, normalized, need_weights=False)[0]
        normalized = self.cross_norm(tokens)
        tokens = tokens + self.cross_attn(
            normalized, cad_tokens, cad_tokens, need_weights=False
        )[0]
        return tokens + self.ffn(self.ffn_norm(tokens))


class GlobalGuidedHierarchicalCADHead(nn.Module):
    """Shared hierarchical CAD decoder used by EXP021 arms B and C.

    CAD geometry is held in non-persistent buffers. Checkpoints therefore store
    only learned parameters and always require the deterministic hierarchy
    artifact declared by the config.
    """

    def __init__(
        self,
        hierarchy_path: str,
        use_global_guidance: bool = False,
        token_dim: int = 256,
        num_heads: int = 4,
        num_transformer_layers: int = 2,
        ffn_dim: int = 1024,
        coarse_loss_weight: float = 1.0,
        fine_loss_weight: float = 1.0,
        xyz_loss_weight: float = 1.0,
        xyz_smooth_l1_beta: float = 0.01,
        default_beam_k: int = 4,
    ):
        super().__init__()
        if token_dim != 256:
            raise ValueError("EXP021 V1 fixes token_dim=256")
        if default_beam_k not in (1, 2, 4, 8):
            raise ValueError("default_beam_k must be one of 1/2/4/8")
        hierarchy = load_hierarchy(hierarchy_path)
        for name, value in hierarchy.items():
            dtype = torch.long if name in {"object_ids", "symmetry_counts"} else torch.float32
            self.register_buffer(name, value.to(dtype=dtype), persistent=False)

        self.use_global_guidance = bool(use_global_guidance)
        self.token_dim = int(token_dim)
        self.default_beam_k = int(default_beam_k)
        self.coarse_loss_weight = float(coarse_loss_weight)
        self.fine_loss_weight = float(fine_loss_weight)
        self.xyz_loss_weight = float(xyz_loss_weight)
        self.xyz_smooth_l1_beta = float(xyz_smooth_l1_beta)

        self.cad_token_mlp = nn.Sequential(
            nn.Linear(10, 128), nn.GELU(), nn.Linear(128, token_dim), nn.LayerNorm(token_dim)
        )
        self.query_proj = nn.Conv2d(256, token_dim, 1)
        self.query_norm = nn.LayerNorm(token_dim)
        self.residual_head = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim), nn.GELU(), nn.Linear(token_dim, 3)
        )

        if self.use_global_guidance:
            self.image_projection = nn.Conv2d(1024, token_dim, 1)
            self.image_positions = nn.Parameter(torch.zeros(1, 64, token_dim))
            self.global_token = nn.Parameter(torch.zeros(1, 1, token_dim))
            self.global_blocks = nn.ModuleList(
                [_GlobalCADBlock(token_dim, num_heads, ffn_dim) for _ in range(num_transformer_layers)]
            )
            self.backbone_residual = nn.Conv2d(token_dim, 1024, 1)
            nn.init.zeros_(self.backbone_residual.weight)
            nn.init.zeros_(self.backbone_residual.bias)
            self.global_bias = nn.Sequential(
                nn.Linear(token_dim, 128), nn.GELU(), nn.Linear(128, 64)
            )

    def _select(self, tensor: torch.Tensor, roi_classes: torch.Tensor) -> torch.Tensor:
        return tensor.index_select(0, roi_classes.long())

    def _unique_descriptors(
        self, roi_classes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        descriptor_classes, descriptor_inverse = torch.unique(
            roi_classes.long(), sorted=True, return_inverse=True
        )
        extents = self._select(self.extents, descriptor_classes).clamp_min(1e-8)
        diameters = self._select(self.diameters, descriptor_classes).view(-1, 1, 1).clamp_min(1e-8)
        coarse = self._select(self.coarse_anchors, descriptor_classes)
        coarse_desc = torch.cat(
            [
                coarse / extents[:, None],
                self._select(self.coarse_normals, descriptor_classes),
                torch.zeros_like(coarse),
                self._select(self.coarse_radii, descriptor_classes)[..., None] / diameters,
            ],
            dim=-1,
        )
        fine = self._select(self.fine_anchors, descriptor_classes)
        fine_desc = torch.cat(
            [
                fine / extents[:, None, None],
                self._select(self.fine_normals, descriptor_classes),
                (fine - coarse[:, :, None]) / extents[:, None, None],
                self._select(self.fine_radii, descriptor_classes)[..., None]
                / diameters[:, :, None],
            ],
            dim=-1,
        )
        return (
            descriptor_classes,
            descriptor_inverse,
            self.cad_token_mlp(coarse_desc),
            self.cad_token_mlp(fine_desc),
        )

    def _descriptor_state(self, roi_classes: torch.Tensor) -> dict[str, torch.Tensor]:
        classes, inverse, coarse_tokens, fine_tokens = self._unique_descriptors(
            roi_classes
        )
        return {
            "classes": classes,
            "inverse": inverse,
            "coarse_tokens": coarse_tokens,
            "fine_tokens": fine_tokens,
        }

    def _descriptors(self, roi_classes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, inverse, coarse_tokens, fine_tokens = self._unique_descriptors(roi_classes)
        return coarse_tokens.index_select(0, inverse), fine_tokens.index_select(0, inverse)

    def enhance_backbone(
        self,
        backbone_feature: torch.Tensor,
        roi_classes: torch.Tensor,
        descriptor_state: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.use_global_guidance:
            return backbone_feature, None
        if descriptor_state is None:
            descriptor_state = self._descriptor_state(roi_classes)
        coarse_tokens = descriptor_state["coarse_tokens"].index_select(
            0, descriptor_state["inverse"]
        )
        image = self.image_projection(backbone_feature).flatten(2).transpose(1, 2)
        if image.shape[1] != 64:
            raise ValueError(f"EXP021 expects an 8x8 backbone feature, got {image.shape[1]} tokens")
        image = image + self.image_positions
        global_token = self.global_token.expand(image.shape[0], -1, -1)
        tokens = torch.cat([global_token, image], dim=1)
        for block in self.global_blocks:
            tokens = block(tokens, coarse_tokens)
        global_bias = self.global_bias(tokens[:, 0])
        image_update = tokens[:, 1:].transpose(1, 2).reshape_as(self.image_projection(backbone_feature))
        return backbone_feature + self.backbone_residual(image_update), global_bias

    def _queries(self, decoder_feature: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        query = self.query_proj(decoder_feature)
        b, _c, h, w = query.shape
        query = self.query_norm(query.flatten(2).transpose(1, 2))
        return query, h, w

    @staticmethod
    def bounded_residual(raw: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        value = torch.tanh(raw)
        scale = torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1.0)
        return radius[..., None] * value / scale

    def _decode_one(
        self,
        query: torch.Tensor,
        coarse_logits: torch.Tensor,
        fine_tokens: torch.Tensor,
        fine_anchors: torch.Tensor,
        fine_radii: torch.Tensor,
        extents: torch.Tensor,
        beam_k: int,
        h: int,
        w: int,
    ) -> dict[str, torch.Tensor]:
        b, n, _ = query.shape
        top_logp, top_parent = F.log_softmax(coarse_logits, dim=-1).topk(beam_k, dim=-1)
        best_score = query.new_full((b, n), -torch.inf)
        best_parent = torch.zeros((b, n), dtype=torch.long, device=query.device)
        best_child = torch.zeros_like(best_parent)
        for batch_index in range(b):
            for parent in range(64):
                locations = (top_parent[batch_index] == parent).nonzero(as_tuple=False)
                if locations.numel() == 0:
                    continue
                pixel_ids = locations[:, 0]
                beam_slots = locations[:, 1]
                logits = query[batch_index, pixel_ids] @ fine_tokens[batch_index, parent].T
                fine_logp, child = F.log_softmax(logits / math.sqrt(self.token_dim), dim=-1).max(-1)
                score = top_logp[batch_index, pixel_ids, beam_slots] + fine_logp
                replace = score > best_score[batch_index, pixel_ids]
                chosen_pixels = pixel_ids[replace]
                best_score[batch_index, chosen_pixels] = score[replace]
                best_parent[batch_index, chosen_pixels] = parent
                best_child[batch_index, chosen_pixels] = child[replace]
        batch = torch.arange(b, device=query.device)[:, None]
        leaf_token = fine_tokens[batch, best_parent, best_child]
        anchor = fine_anchors[batch, best_parent, best_child]
        radius = fine_radii[batch, best_parent, best_child]
        residual = self.bounded_residual(
            self.residual_head(torch.cat([query, leaf_token], dim=-1)), radius
        )
        xyz_metric = anchor + residual
        xyz_norm = xyz_metric / extents[:, None] + 0.5
        return {
            "xyz_norm": xyz_norm.transpose(1, 2).reshape(b, 3, h, w),
            "parent": best_parent.reshape(b, h, w),
            "child": best_child.reshape(b, h, w),
            "joint_log_probability": best_score.reshape(b, h, w),
        }

    def _encoded_state(
        self,
        decoder_feature: torch.Tensor,
        roi_classes: torch.Tensor,
        global_bias: torch.Tensor | None = None,
        descriptor_state: dict[str, torch.Tensor] | None = None,
    ) -> tuple[dict, int, int]:
        query, h, w = self._queries(decoder_feature)
        if descriptor_state is None:
            descriptor_state = self._descriptor_state(roi_classes)
        descriptor_classes = descriptor_state["classes"]
        descriptor_inverse = descriptor_state["inverse"]
        coarse_unique = descriptor_state["coarse_tokens"]
        fine_unique = descriptor_state["fine_tokens"]
        coarse_tokens = coarse_unique.index_select(0, descriptor_inverse)
        coarse_logits = torch.einsum("bnd,bkd->bnk", query, coarse_tokens) / math.sqrt(self.token_dim)
        if global_bias is not None:
            coarse_logits = coarse_logits + global_bias[:, None]
        return {
            "query": query,
            "coarse_logits": coarse_logits,
            "descriptor_classes": descriptor_classes,
            "descriptor_inverse": descriptor_inverse,
            "fine_tokens_unique": fine_unique,
        }, h, w

    def decode(
        self,
        decoder_feature: torch.Tensor,
        roi_classes: torch.Tensor,
        global_bias: torch.Tensor | None = None,
        beam_ks: Iterable[int] = (4,),
        descriptor_state: dict[str, torch.Tensor] | None = None,
    ) -> dict:
        state, h, w = self._encoded_state(
            decoder_feature, roi_classes, global_bias, descriptor_state
        )
        query = state["query"]
        coarse_logits = state["coarse_logits"]
        fine_tokens = state["fine_tokens_unique"].index_select(
            0, state["descriptor_inverse"]
        )
        state["fine_tokens"] = fine_tokens
        fine_anchors = self._select(self.fine_anchors, roi_classes)
        fine_radii = self._select(self.fine_radii, roi_classes)
        extents = self._select(self.extents, roi_classes)
        decoded = {
            int(k): self._decode_one(
                query,
                coarse_logits,
                fine_tokens,
                fine_anchors,
                fine_radii,
                extents,
                int(k),
                h,
                w,
            )
            for k in beam_ks
        }
        state["decoded"] = decoded
        return state

    def _symmetry_targets(
        self, xyz_metric: torch.Tensor, class_index: int
    ) -> list[torch.Tensor]:
        count = int(self.symmetry_counts[class_index].item())
        transforms = self.symmetry_transforms[class_index, :count]
        flat = xyz_metric.reshape(-1, 3)
        targets = []
        for transform in transforms:
            rotation = transform[:3, :3]
            translation = transform[:3, 3]
            targets.append((flat - translation) @ rotation)
        return targets

    @staticmethod
    def _squared_distances(
        points: torch.Tensor, anchors: torch.Tensor
    ) -> torch.Tensor:
        """Pairwise squared distance for [..., P, 3] and [..., A, 3]."""
        point_norm = points.square().sum(dim=-1, keepdim=True)
        anchor_norm = anchors.square().sum(dim=-1).unsqueeze(-2)
        product = torch.matmul(points, anchors.transpose(-1, -2))
        return (point_norm + anchor_norm - 2.0 * product).clamp_min_(0.0)

    @staticmethod
    def _pack_route_blocks(
        route_key: torch.Tensor, block_size: int = _ROUTE_BLOCK_SIZE
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack sorted variable-size routes into fixed-size GPU blocks.

        Returns the sort order, padded indices into that sorted order, a padding
        mask, and the route key for every block. Only the final allocation shape
        is materialized as a host scalar; route membership never leaves CUDA.
        """
        order = torch.argsort(route_key)
        sorted_key = route_key.index_select(0, order)
        unique_keys, group_counts = torch.unique_consecutive(
            sorted_key, return_counts=True
        )
        group_ids = torch.repeat_interleave(
            torch.arange(unique_keys.numel(), device=route_key.device), group_counts
        )
        group_starts = torch.cumsum(group_counts, dim=0) - group_counts
        within_group = torch.arange(
            route_key.numel(), device=route_key.device
        ) - group_starts.index_select(0, group_ids)
        blocks_per_group = torch.div(
            group_counts + block_size - 1, block_size, rounding_mode="floor"
        )
        block_starts = torch.cumsum(blocks_per_group, dim=0) - blocks_per_group
        block_ids = block_starts.index_select(0, group_ids) + torch.div(
            within_group, block_size, rounding_mode="floor"
        )
        slots = torch.remainder(within_group, block_size)
        num_blocks = int(blocks_per_group.sum().item())
        packed = torch.full(
            (num_blocks, block_size),
            -1,
            dtype=torch.long,
            device=route_key.device,
        )
        packed[block_ids, slots] = torch.arange(
            route_key.numel(), device=route_key.device
        )
        valid = packed >= 0
        block_keys = torch.repeat_interleave(unique_keys, blocks_per_group)
        return order, packed, valid, block_keys

    def loss(
        self,
        decoded_state: dict,
        roi_classes: torch.Tensor,
        gt_xyz_norm: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        query = decoded_state["query"]
        coarse_logits = decoded_state["coarse_logits"]
        descriptor_classes = decoded_state["descriptor_classes"]
        descriptor_inverse = decoded_state["descriptor_inverse"]
        fine_tokens = decoded_state["fine_tokens_unique"]
        extents = self._select(self.extents, roi_classes)
        b, n, _ = query.shape
        symmetry_slots = self.symmetry_transforms.shape[1]

        # Geometry targets and nearest-anchor labels are precision-sensitive
        # supervision, not learned tensor-core work.  Keep them in FP32 when
        # the surrounding formal training step uses FP16 autocast.
        with torch.no_grad(), torch.autocast(
            device_type=query.device.type, enabled=False
        ):
            gt_metric = (
                gt_xyz_norm.float().permute(0, 2, 3, 1).reshape(b, n, 3) - 0.5
            ) * extents.float()[:, None]
            transforms = self._select(self.symmetry_transforms, roi_classes).float()
            rotations = transforms[:, :, :3, :3]
            translations = transforms[:, :, :3, 3]
            symmetry_targets = torch.einsum(
                "bsni,bsij->bsnj",
                gt_metric[:, None] - translations[:, :, None],
                rotations,
            )
            valid_symmetry = (
                torch.arange(symmetry_slots, device=query.device)[None]
                < self._select(self.symmetry_counts, roi_classes)[:, None]
            )
            foreground = gt_mask.reshape(b, n) > 0.5
            valid = valid_symmetry[:, :, None] & foreground[:, None]
            locations = valid.nonzero(as_tuple=False)

        zero = query.sum() * 0.0
        if locations.numel() == 0:
            components = torch.stack([zero, zero, zero])
            selected = torch.zeros(b, dtype=torch.long, device=query.device)
        else:
            batch_index, symmetry_index, pixel_index = locations.unbind(1)
            target = symmetry_targets[batch_index, symmetry_index, pixel_index]
            q = query[batch_index, pixel_index]
            branch_index = batch_index * symmetry_slots + symmetry_index

            with torch.no_grad(), torch.autocast(
                device_type=query.device.type, enabled=False
            ):
                coarse_label = torch.empty(
                    target.shape[0], dtype=torch.long, device=query.device
                )
                for start in range(0, target.shape[0], _COARSE_LABEL_CHUNK_SIZE):
                    stop = min(start + _COARSE_LABEL_CHUNK_SIZE, target.shape[0])
                    anchors = self.coarse_anchors.index_select(
                        0, roi_classes[batch_index[start:stop]].long()
                    )
                    coarse_label[start:stop] = self._squared_distances(
                        target[start:stop, None], anchors
                    ).squeeze(1).argmin(-1)

            coarse_point_loss = F.cross_entropy(
                coarse_logits[batch_index, pixel_index], coarse_label, reduction="none"
            )
            route_key = descriptor_inverse[batch_index] * 64 + coarse_label
            order, packed, packed_valid, block_keys = self._pack_route_blocks(
                route_key
            )
            sorted_q = q.index_select(0, order)
            sorted_target = target.index_select(0, order)
            sorted_branch = branch_index.index_select(0, order)
            sorted_batch = batch_index.index_select(0, order)
            sorted_coarse_loss = coarse_point_loss.index_select(0, order)

            safe_packed = packed.clamp_min(0)
            packed_q = sorted_q[safe_packed]
            packed_target = sorted_target[safe_packed]
            packed_batch = sorted_batch[safe_packed]
            descriptor_index = torch.div(block_keys, 64, rounding_mode="floor")
            parent = torch.remainder(block_keys, 64)
            object_class = descriptor_classes.index_select(0, descriptor_index)
            anchors = self.fine_anchors[object_class, parent]
            token_bank = fine_tokens[descriptor_index, parent]

            with torch.no_grad(), torch.autocast(
                device_type=query.device.type, enabled=False
            ):
                fine_label = self._squared_distances(
                    packed_target.float(), anchors.float()
                ).argmin(-1)
            fine_logits = torch.matmul(
                packed_q, token_bank.transpose(1, 2)
            ) / math.sqrt(self.token_dim)
            fine_loss_grid = F.cross_entropy(
                fine_logits.flatten(0, 1),
                fine_label.flatten(),
                reduction="none",
            ).view_as(fine_label)
            anchor = torch.gather(
                anchors,
                1,
                fine_label[..., None].expand(-1, -1, 3),
            )
            radius = torch.gather(
                self.fine_radii[object_class, parent], 1, fine_label
            )
            leaf_token = torch.gather(
                token_bank,
                1,
                fine_label[..., None].expand(-1, -1, self.token_dim),
            )
            residual = self.bounded_residual(
                self.residual_head(torch.cat([packed_q, leaf_token], dim=-1)),
                radius,
            )
            packed_extents = extents[packed_batch]
            pred_norm = (anchor + residual) / packed_extents + 0.5
            target_norm = packed_target / packed_extents + 0.5
            xyz_loss_grid = F.smooth_l1_loss(
                pred_norm,
                target_norm,
                beta=self.xyz_smooth_l1_beta,
                reduction="none",
            ).mean(-1)

            packed_branch = sorted_branch[safe_packed]
            packed_coarse_loss = sorted_coarse_loss[safe_packed]
            flat_valid = packed_valid.flatten()
            branch_index = packed_branch.flatten()[flat_valid]

            point_components = torch.stack(
                [
                    packed_coarse_loss.flatten()[flat_valid],
                    fine_loss_grid.flatten()[flat_valid],
                    xyz_loss_grid.flatten()[flat_valid],
                ]
            ).float()
            branch_count = b * symmetry_slots
            branch_sums = query.new_zeros((3, branch_count)).scatter_add(
                1, branch_index[None].expand(3, -1), point_components
            )
            counts = torch.bincount(branch_index, minlength=branch_count)
            branch_components = branch_sums / counts.clamp_min(1).to(query.dtype)[None]
            branch_components = branch_components.view(3, b, symmetry_slots)
            branch_components = branch_components + zero
            weighted = (
                branch_components[0] * self.coarse_loss_weight
                + branch_components[1] * self.fine_loss_weight
                + branch_components[2] * self.xyz_loss_weight
            )
            branch_valid = valid_symmetry & (counts.view(b, symmetry_slots) > 0)
            empty = ~foreground.any(-1)
            branch_valid[empty, 0] = True
            weighted = weighted.masked_fill(~branch_valid, torch.inf)
            selected = weighted.detach().argmin(-1)
            selected_components = branch_components.permute(1, 2, 0)[
                torch.arange(b, device=query.device), selected
            ]
            components = selected_components.mean(0)
        losses = {
            "loss_cad_coarse": components[0] * self.coarse_loss_weight,
            "loss_cad_fine": components[1] * self.fine_loss_weight,
            "loss_cad_xyz": components[2] * self.xyz_loss_weight,
        }
        stats = {
            "selected_symmetry_branch_mean": selected.to(torch.float32).mean(),
            "cad_foreground_pixels": (gt_mask > 0.5).sum().detach(),
        }
        return losses, stats

    def forward(
        self,
        decoder_feature: torch.Tensor,
        roi_classes: torch.Tensor,
        global_bias: torch.Tensor | None = None,
        beam_ks: Iterable[int] | None = None,
        gt_xyz_norm: torch.Tensor | None = None,
        gt_mask: torch.Tensor | None = None,
        descriptor_state: dict[str, torch.Tensor] | None = None,
    ) -> tuple[dict, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        beam_ks = tuple(beam_ks or (self.default_beam_k,))
        losses: dict[str, torch.Tensor] = {}
        stats: dict[str, torch.Tensor] = {}
        if gt_xyz_norm is not None:
            if gt_mask is None:
                raise ValueError("gt_mask is required with gt_xyz_norm")
            state, _h, _w = self._encoded_state(
                decoder_feature, roi_classes, global_bias, descriptor_state
            )
            losses, stats = self.loss(state, roi_classes, gt_xyz_norm, gt_mask)
        else:
            state = self.decode(
                decoder_feature,
                roi_classes,
                global_bias,
                beam_ks,
                descriptor_state,
            )
        return state, losses, stats
