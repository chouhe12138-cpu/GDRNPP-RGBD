from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .common import CapturedPoseCall, DiagnosticBatch
from .metrics import (
    PoseMetricAccumulator,
    ScalarAccumulator,
    rotation_error_deg,
    translation_error_cm,
)
from .model_access import decode_raw_pose, unwrap_model


def _fixed_permutation(size: int, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randperm(size, generator=generator).to(device=device)


def _token_support(support: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Map the visible-support mask to the GLM 16x16 token grid.

    A token is considered valid when any source support falls in its pooled cell.
    This is intentionally conservative for the diagnostic: partially visible cells
    stay available rather than being removed by nearest-neighbour sampling.
    """
    valid = F.adaptive_max_pool2d(support.float(), (height, width)) > 0.0
    valid = valid[:, 0].flatten(1)
    if bool((valid.sum(dim=1) == 0).any()):
        raise RuntimeError("At least one sample has no valid support on the GLM token grid")
    return valid


def _depth_features(head, latent: torch.Tensor, depth_stats: Optional[torch.Tensor]) -> torch.Tensor:
    if depth_stats is None:
        return latent.new_zeros(latent.shape[0], head.depth_stats_dim)
    expected = (latent.shape[0], head.depth_stats_dim)
    if depth_stats.ndim != 2 or tuple(depth_stats.shape) != expected:
        raise ValueError(f"depth_stats must have shape {expected}, got {tuple(depth_stats.shape)}")
    if not bool(torch.isfinite(depth_stats).all()):
        raise ValueError("depth_stats contains non-finite values")
    return depth_stats.to(device=latent.device, dtype=latent.dtype)


def probe_glm_forward(
    head,
    call: CapturedPoseCall,
    *,
    depth_stats: Optional[torch.Tensor],
    pooling: str = "learned",
    support_masked: bool = False,
    position_mode: str = "normal",
    token_shuffle: bool = False,
    seed: int = 20260902,
):
    """Reproduce GLMPoseLNet forward with one controlled intervention.

    This function does not change model parameters and does not run an optimizer.
    The normal branch is checked against the captured formal F forward before any
    diagnostic result is accepted.
    """
    required = (
        "_validate_and_mask_inputs",
        "_encode_main_features",
        "token_projection",
        "position_embedding",
        "encoder_layer",
        "pool_score",
        "shared_fc",
        "pose_rotation",
        "pose_translation",
        "depth_stats_dim",
    )
    missing = [name for name in required if not hasattr(head, name)]
    if missing:
        raise RuntimeError(f"F diagnostic requires GLMPoseLNet-like head; missing {missing}")

    metric, masked_region, support = head._validate_and_mask_inputs(
        call.coor_feat, call.region, call.extents, call.mask_attention
    )
    _, _, high = head._encode_main_features(metric, masked_region)
    batch, _channels, height, width = high.shape
    tokens = high.flatten(2).transpose(1, 2)
    token_count = tokens.shape[1]
    if token_count != head.position_embedding.shape[1]:
        raise RuntimeError(
            f"Position table/token mismatch: {token_count} vs {head.position_embedding.shape[1]}"
        )

    valid = _token_support(support, height, width)
    if token_shuffle:
        perm = _fixed_permutation(token_count, tokens.device, seed + 11)
        tokens = tokens[:, perm]

    tokens = head.token_projection(tokens)
    position = head.position_embedding.to(device=tokens.device, dtype=tokens.dtype)
    if position_mode == "normal":
        pass
    elif position_mode == "shuffle":
        perm = _fixed_permutation(token_count, tokens.device, seed + 23)
        position = position[:, perm]
    elif position_mode == "zero":
        position = torch.zeros_like(position)
    else:
        raise ValueError(f"Unknown position_mode={position_mode}")
    tokens = tokens + position

    key_padding_mask = ~valid if support_masked else None
    tokens = head.encoder_layer(tokens, src_key_padding_mask=key_padding_mask)
    scores = head.pool_score(torch.tanh(tokens)).squeeze(-1)

    if pooling == "learned":
        if support_masked:
            scores = scores.masked_fill(~valid, -1.0e4)
        weights = torch.softmax(scores, dim=1)
        if support_masked:
            weights = weights * valid.to(weights.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    elif pooling == "uniform":
        if support_masked:
            weights = valid.to(tokens.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        else:
            weights = torch.full_like(scores, 1.0 / float(token_count))
    else:
        raise ValueError(f"Unknown pooling={pooling}")

    pooled = (tokens * weights.unsqueeze(-1)).sum(dim=1)
    latent = head.pose_act(head.shared_fc(pooled))
    depth = _depth_features(head, latent, depth_stats)
    translation_input = torch.cat([latent, depth], dim=1)
    raw_rot = head.pose_rotation(latent)
    raw_t = head.pose_translation(translation_input)
    return raw_rot, raw_t, {
        "weights": weights.detach(),
        "valid": valid.detach(),
        "support_fraction": valid.float().mean(dim=1).detach(),
        "pooling_valid_mass": (weights * valid.to(weights.dtype)).sum(dim=1).detach(),
    }


def _metric_mean(summary: Dict[str, Any], key: str) -> float:
    value = summary.get(key, {}).get("mean", float("nan"))
    return float(value)


def _safe_delta(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return float(a - b)


class FGLMPoseDiagnostic:
    """Targeted local diagnostic for the completed EXP013F checkpoint.

    The screen answers four implementation questions only:
    1) does unmasked pooling allocate material mass to invalid support tokens?
    2) does learned pooling beat a matched uniform pooling intervention?
    3) does rotation actually depend on token/position spatial organization?
    4) do the four ROI depth statistics materially help translation?

    It is mechanism evidence, not a BOP evaluation and not a model-selection run.
    """

    name = "exp013f_targeted_diagnostic"

    def __init__(
        self,
        *,
        seed: int = 20260902,
        reproduction_tol: float = 1.0e-6,
        invalid_mass_band: float = 0.10,
        rotation_effect_band_deg: float = 0.50,
        translation_effect_band_cm: float = 0.20,
        spatial_output_band_deg: float = 1.00,
    ) -> None:
        self.seed = int(seed)
        self.reproduction_tol = float(reproduction_tol)
        self.invalid_mass_band = float(invalid_mass_band)
        self.rotation_effect_band_deg = float(rotation_effect_band_deg)
        self.translation_effect_band_cm = float(translation_effect_band_cm)
        self.spatial_output_band_deg = float(spatial_output_band_deg)
        self.pose_metrics: Dict[str, PoseMetricAccumulator] = defaultdict(PoseMetricAccumulator)
        self.effects: Dict[str, ScalarAccumulator] = defaultdict(ScalarAccumulator)
        self.support_stats = ScalarAccumulator()
        self.integrity = ScalarAccumulator()
        self.skipped = []
        self.batches = 0

    def _run_variant(self, head, call, name: str, batch_index: int):
        depth = call.depth_stats
        kwargs = dict(depth_stats=depth, seed=self.seed + 1000 * batch_index)
        if name == "normal":
            return probe_glm_forward(head, call, **kwargs)
        if name == "support_masked":
            return probe_glm_forward(head, call, support_masked=True, **kwargs)
        if name == "uniform_pool":
            return probe_glm_forward(head, call, pooling="uniform", **kwargs)
        if name == "position_shuffle":
            return probe_glm_forward(head, call, position_mode="shuffle", **kwargs)
        if name == "token_shuffle":
            return probe_glm_forward(head, call, token_shuffle=True, **kwargs)
        if name == "depth_zero":
            if depth is None:
                raise RuntimeError("depth_zero requires captured formal F depth_stats")
            return probe_glm_forward(head, call, depth_stats=torch.zeros_like(depth), seed=kwargs["seed"])
        if name == "depth_shuffle":
            if depth is None:
                raise RuntimeError("depth_shuffle requires captured formal F depth_stats")
            if depth.shape[0] < 2:
                raise RuntimeError("depth_shuffle requires batch_size >= 2")
            shuffled = torch.roll(depth, shifts=1, dims=0)
            return probe_glm_forward(head, call, depth_stats=shuffled, seed=kwargs["seed"])
        raise ValueError(name)

    def update(self, db: DiagnosticBatch, runtime) -> None:
        head = unwrap_model(runtime.model).pnp_net
        call = db.pose_call
        if type(head).__name__ != "GLMPoseLNet":
            raise RuntimeError(f"Expected GLMPoseLNet, got {type(head).__name__}")
        if call.depth_stats is None:
            raise RuntimeError(
                "Formal F depth_stats were not captured. Apply the bundled common.py/model_access.py fix first."
            )
        if call.mask_attention is None:
            raise RuntimeError("F diagnostic requires visible-support mask_attention")
        batch_depth = db.batch.get("roi_depth_stats")
        if batch_depth is None:
            raise RuntimeError("F config requires roi_depth_stats in the prepared batch")
        depth_capture_max = float((call.depth_stats - batch_depth).abs().max().cpu())
        self.integrity.add(depth_capture_max_abs=depth_capture_max)
        if depth_capture_max > self.reproduction_tol:
            raise RuntimeError(
                f"Captured depth_stats differ from batch roi_depth_stats: {depth_capture_max:.3e}"
            )

        self.batches += 1
        with torch.no_grad():
            normal_r, normal_t, normal_info = self._run_variant(head, call, "normal", self.batches)

            rot_max = float((normal_r - call.raw_rot).abs().max().cpu())
            trans_max = float((normal_t - call.raw_t).abs().max().cpu())
            self.integrity.add(raw_rotation_max_abs=rot_max, raw_translation_max_abs=trans_max)
            if rot_max > self.reproduction_tol or trans_max > self.reproduction_tol:
                raise RuntimeError(
                    "Manual F reproduction does not match captured formal forward: "
                    f"rot={rot_max:.3e}, trans={trans_max:.3e}, tol={self.reproduction_tol:.3e}"
                )

            normal_pred = decode_raw_pose(runtime.cfg, normal_r, normal_t, db.batch, is_train=False)
            gt_r, gt_t = db.batch["ego_rot"], db.batch["trans"]
            classes = db.batch.get("roi_cls")
            self.pose_metrics["normal"].add(normal_pred.rot, normal_pred.trans, gt_r, gt_t, classes)

            valid = normal_info["valid"]
            weights = normal_info["weights"]
            support_fraction = normal_info["support_fraction"]
            valid_mass = normal_info["pooling_valid_mass"]
            invalid_mass = 1.0 - valid_mass
            invalid_fraction = 1.0 - support_fraction
            max_invalid_weight = torch.where(
                ~valid,
                weights,
                torch.zeros_like(weights),
            ).max(dim=1).values
            for i in range(weights.shape[0]):
                denom = float(invalid_fraction[i].cpu())
                ratio = float(invalid_mass[i].cpu()) / max(denom, 1.0e-12)
                self.support_stats.add(
                    support_fraction=float(support_fraction[i].cpu()),
                    pooling_valid_mass=float(valid_mass[i].cpu()),
                    pooling_invalid_mass=float(invalid_mass[i].cpu()),
                    invalid_mass_vs_uniform_ratio=ratio,
                    max_invalid_token_weight=float(max_invalid_weight[i].cpu()),
                )

            for name in (
                "support_masked",
                "uniform_pool",
                "position_shuffle",
                "token_shuffle",
                "depth_zero",
                "depth_shuffle",
            ):
                try:
                    raw_r, raw_t, _info = self._run_variant(head, call, name, self.batches)
                    pred = decode_raw_pose(runtime.cfg, raw_r, raw_t, db.batch, is_train=False)
                    self.pose_metrics[name].add(pred.rot, pred.trans, gt_r, gt_t, classes)
                    self.effects[name].add(
                        rotation_output_change_deg=float(
                            rotation_error_deg(pred.rot, normal_pred.rot).mean().cpu()
                        ),
                        translation_output_change_cm=float(
                            translation_error_cm(pred.trans, normal_pred.trans).mean().cpu()
                        ),
                        raw_rotation_rms_change=float(
                            (raw_r - normal_r).float().pow(2).mean().sqrt().cpu()
                        ),
                        raw_translation_rms_change=float(
                            (raw_t - normal_t).float().pow(2).mean().sqrt().cpu()
                        ),
                    )
                    if name.startswith("depth_"):
                        self.integrity.add(
                            **{
                                f"{name}_rotation_raw_max_abs": float(
                                    (raw_r - normal_r).abs().max().cpu()
                                )
                            }
                        )
                except Exception as exc:
                    self.skipped.append(f"batch={self.batches} {name}: {type(exc).__name__}: {exc}")

    def summary(self) -> Dict[str, Any]:
        pose = {k: v.summary() for k, v in self.pose_metrics.items()}
        effects = {k: v.summary() for k, v in self.effects.items()}
        support = self.support_stats.summary()
        integrity = self.integrity.summary()

        normal_rot = _metric_mean(pose.get("normal", {}), "rotation_deg")
        normal_trans = _metric_mean(pose.get("normal", {}), "translation_cm")

        comparison = {}
        for name, summary in pose.items():
            if name == "normal":
                continue
            comparison[name] = {
                "rotation_error_delta_deg_vs_normal": _safe_delta(
                    _metric_mean(summary, "rotation_deg"), normal_rot
                ),
                "translation_error_delta_cm_vs_normal": _safe_delta(
                    _metric_mean(summary, "translation_cm"), normal_trans
                ),
            }

        invalid_mass_mean = support.get("pooling_invalid_mass", {}).get("mean", float("nan"))
        uniform_rot_delta = comparison.get("uniform_pool", {}).get(
            "rotation_error_delta_deg_vs_normal", float("nan")
        )
        position_rot_delta = comparison.get("position_shuffle", {}).get(
            "rotation_error_delta_deg_vs_normal", float("nan")
        )
        token_rot_delta = comparison.get("token_shuffle", {}).get(
            "rotation_error_delta_deg_vs_normal", float("nan")
        )
        depth_zero_t_delta = comparison.get("depth_zero", {}).get(
            "translation_error_delta_cm_vs_normal", float("nan")
        )
        depth_shuffle_t_delta = comparison.get("depth_shuffle", {}).get(
            "translation_error_delta_cm_vs_normal", float("nan")
        )
        spatial_candidates = [
            effects.get("position_shuffle", {}).get("rotation_output_change_deg", {}).get("mean", float("nan")),
            effects.get("token_shuffle", {}).get("rotation_output_change_deg", {}).get("mean", float("nan")),
        ]
        spatial_candidates = [float(v) for v in spatial_candidates if np.isfinite(v)]
        spatial_output = max(spatial_candidates) if spatial_candidates else float("nan")

        learned_pooling_positive = bool(
            np.isfinite(uniform_rot_delta)
            and uniform_rot_delta >= self.rotation_effect_band_deg
        )
        spatial_position_positive = bool(
            (np.isfinite(position_rot_delta) and position_rot_delta >= self.rotation_effect_band_deg)
            or (np.isfinite(token_rot_delta) and token_rot_delta >= self.rotation_effect_band_deg)
            or (np.isfinite(spatial_output) and spatial_output >= self.spatial_output_band_deg)
        )
        depth_positive = bool(
            np.isfinite(depth_zero_t_delta)
            and np.isfinite(depth_shuffle_t_delta)
            and depth_zero_t_delta >= self.translation_effect_band_cm
            and depth_shuffle_t_delta >= self.translation_effect_band_cm
        )
        invalid_mass_material = bool(
            np.isfinite(invalid_mass_mean) and invalid_mass_mean >= self.invalid_mass_band
        )

        reproduction_max = max(
            integrity.get("raw_rotation_max_abs", {}).get("p90", float("inf")),
            integrity.get("raw_translation_max_abs", {}).get("p90", float("inf")),
        )
        diagnostic_valid = bool(np.isfinite(reproduction_max) and reproduction_max <= self.reproduction_tol)

        return {
            "scope": {
                "checkpoint_family": "EXP013F GLM-Pose-L E40",
                "formal_bop": False,
                "optimizer_steps": 0,
                "purpose": "Choose implementation constraints for EXP017; do not treat subset metrics as final accuracy.",
            },
            "settings": {
                "seed": self.seed,
                "reproduction_tol": self.reproduction_tol,
                "invalid_mass_band": self.invalid_mass_band,
                "rotation_effect_band_deg": self.rotation_effect_band_deg,
                "translation_effect_band_cm": self.translation_effect_band_cm,
                "spatial_output_band_deg": self.spatial_output_band_deg,
            },
            "batches": self.batches,
            "pose_metrics": pose,
            "effect_sizes": effects,
            "support_pooling": support,
            "integrity": integrity,
            "comparison_to_normal": comparison,
            "screen": {
                "diagnostic_valid": diagnostic_valid,
                "invalid_support_pooling_mass_material": invalid_mass_material,
                "learned_pooling_has_positive_rotation_evidence": learned_pooling_positive,
                "spatial_position_has_rotation_evidence": spatial_position_positive,
                "depth_stats_have_translation_evidence": depth_positive,
            },
            "exp017_design_implications": {
                "base": "Keep EXP013A main stream + Region-free geometry path + original translation path.",
                "support": (
                    "Mask invalid support in every spatial normalization/weighting operation; F shows material invalid-token pooling mass."
                    if invalid_mass_material
                    else "Still mask invalid support as a correctness constraint; F did not show large invalid pooling mass on this subset."
                ),
                "rotation": (
                    "Use a position-aware rotation-only spatial residual adapter."
                    if spatial_position_positive
                    else "Keep the rotation-only adapter simple/convolutional; do not justify Transformer-style positional machinery from F."
                ),
                "pooling": (
                    "Lightweight learned scale/spatial weighting may be retained, but do not transfer the full F Transformer."
                    if learned_pooling_positive
                    else "Do not transfer F learned pooling; start with support-weighted/simple multiscale fusion."
                ),
                "depth": (
                    "Depth statistics show local translation signal, but defer them to a later translation-only ablation; EXP017 remains rotation-only."
                    if depth_positive
                    else "Exclude depth statistics from EXP017; current diagnostic does not support a material translation benefit."
                ),
                "translation": "Do not redesign translation in EXP017.",
            },
            "skipped": self.skipped,
        }
