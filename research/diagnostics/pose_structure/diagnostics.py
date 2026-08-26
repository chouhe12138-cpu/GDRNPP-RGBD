from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from detectron2.utils.events import EventStorage

from core.utils.data_utils import xyz_to_region_batch

from .common import CapturedPoseCall, DiagnosticBatch, cosine_scalar, tensor_rms
from .geometry_solver import solve_pnp_from_correspondence
from .head_adapters import (
    head_family,
    run_branch_variant,
    spatial_intervention,
    supported_branch_variants,
    trace_head_features,
)
from .metrics import PoseMetricAccumulator, ScalarAccumulator, monotonicity, rotation_error_deg, translation_error_cm
from .model_access import (
    call_head,
    capture_model_pose_call,
    decode_raw_pose,
    make_model_kwargs,
    model_input_from_batch,
    unwrap_model,
)


class D1RTOracle:
    name = "d1_rt_oracle"

    def __init__(self):
        self.metrics = {k: PoseMetricAccumulator() for k in ("pred", "gtR_predt", "predR_gtt", "gt")}

    def update(self, db: DiagnosticBatch, runtime) -> None:
        b = db.batch
        gt_r, gt_t = b["ego_rot"], b["trans"]
        cls = b.get("roi_cls")
        variants = {
            "pred": (db.pred_rot, db.pred_trans),
            "gtR_predt": (gt_r, db.pred_trans),
            "predR_gtt": (db.pred_rot, gt_t),
            "gt": (gt_r, gt_t),
        }
        for name, (r, t) in variants.items():
            self.metrics[name].add(r, t, gt_r, gt_t, cls)

    def summary(self):
        out = {k: v.summary() for k, v in self.metrics.items()}
        pred_r = out["pred"]["rotation_deg"].get("mean", float("nan"))
        pred_t = out["pred"]["translation_cm"].get("mean", float("nan"))
        out["interpretation_hints"] = {
            "rotation_oracle_zeroes_rotation_error_by_definition": True,
            "translation_oracle_zeroes_translation_error_by_definition": True,
            "use_downstream_point/BOP evaluator_if_formal_combined_metric_is_needed": True,
            "baseline_rotation_deg_mean": pred_r,
            "baseline_translation_cm_mean": pred_t,
        }
        return out


class D2BranchAblation:
    name = "d2_branch_ablation"

    def __init__(self):
        self.metrics: Dict[str, PoseMetricAccumulator] = {}
        self.scalars: Dict[str, ScalarAccumulator] = defaultdict(ScalarAccumulator)
        self.skipped: Dict[str, str] = {}
        self.variants_seen: List[str] = []

    def update(self, db: DiagnosticBatch, runtime) -> None:
        base = unwrap_model(runtime.model)
        head = base.pnp_net
        b = db.batch
        gt_r, gt_t, cls = b["ego_rot"], b["trans"], b.get("roi_cls")
        variants = supported_branch_variants(head)
        self.variants_seen = variants
        normal_raw = None
        normal_pred = None
        for variant in variants:
            try:
                with torch.no_grad():
                    raw_r, raw_t = run_branch_variant(head, db.pose_call, variant)
                    pred = decode_raw_pose(runtime.cfg, raw_r, raw_t, b, is_train=False)
                self.metrics.setdefault(variant, PoseMetricAccumulator()).add(pred.rot, pred.trans, gt_r, gt_t, cls)
                if variant == "normal":
                    normal_raw = (raw_r.detach(), raw_t.detach())
                    normal_pred = pred
                elif normal_raw is not None:
                    self.scalars[variant].add(
                        raw_rot_l2=float((raw_r - normal_raw[0]).float().pow(2).mean().sqrt().cpu()),
                        raw_t_l2=float((raw_t - normal_raw[1]).float().pow(2).mean().sqrt().cpu()),
                        final_rot_delta_deg=float(rotation_error_deg(pred.rot, normal_pred.rot).mean().cpu()),
                        final_trans_delta_cm=float(translation_error_cm(pred.trans, normal_pred.trans).mean().cpu()),
                    )
            except Exception as exc:
                self.skipped[variant] = f"{type(exc).__name__}: {exc}"
        try:
            trace = trace_head_features(head, db.pose_call)
            for k, v in trace.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    self.scalars["feature_trace"].add(**{k: v})
        except Exception as exc:
            self.skipped["feature_trace"] = f"{type(exc).__name__}: {exc}"

    def summary(self):
        return {
            "head_variants": self.variants_seen,
            "pose_metrics": {k: v.summary() for k, v in self.metrics.items()},
            "effect_sizes": {k: v.summary() for k, v in self.scalars.items()},
            "skipped": self.skipped,
            "decision_rule": {
                "main_only_close_to_normal": "geometry residual contributes little to final pose",
                "attention_zero_close_to_normal": "local attention contributes little",
                "region_zero_large_drop_but_geometry_only_weak": "Region-free geometry path did not become an independently useful pose path",
            },
        }


def _region_labels_to_prob(labels: torch.Tensor, channels: int, dtype: torch.dtype) -> torch.Tensor:
    if labels.ndim == 4 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 3:
        raise ValueError(f"Expected region labels BxHxW, got {tuple(labels.shape)}")
    labels = labels.long()
    b, h, w = labels.shape
    out = torch.zeros(b, channels, h, w, device=labels.device, dtype=dtype)
    fg = labels > 0
    idx = (labels - 1).clamp(min=0, max=channels - 1)
    out.scatter_(1, idx[:, None], 1.0)
    out *= fg[:, None].to(dtype)
    return out


def _synced_region_from_xyz(mixed_xyz: torch.Tensor, batch: Dict[str, Any], pred_region: torch.Tensor) -> torch.Tensor:
    channels = pred_region.shape[1]
    if "roi_fps_points" in batch and "roi_mask_obj" in batch:
        metric = (mixed_xyz - 0.5) * batch["roi_extent"][:, :, None, None]
        labels = xyz_to_region_batch(
            metric.permute(0, 2, 3, 1).contiguous(),
            batch["roi_fps_points"],
            mask=batch["roi_mask_obj"],
        )
        return _region_labels_to_prob(labels, channels, pred_region.dtype)
    # GT roi_region describes the original GT XYZ only.  Reusing it for an
    # interpolated XYZ map would falsely label a fixed GT Region tensor as a
    # synchronized intervention.  Skip this path unless FPS anchors are
    # available and Region can actually be recomputed from mixed_xyz.
    raise RuntimeError("Synced Region requires roi_fps_points+roi_mask_obj")


class D3CorrespondenceUtilization:
    name = "d3_correspondence_utilization"

    def __init__(self, alphas: Sequence[float], run_solver: bool = True):
        self.alphas = [float(a) for a in alphas]
        self.run_solver = bool(run_solver)
        self.metrics: Dict[str, PoseMetricAccumulator] = defaultdict(PoseMetricAccumulator)
        self.solver_metrics: Dict[str, PoseMetricAccumulator] = defaultdict(PoseMetricAccumulator)
        self.skipped: Dict[str, List[str]] = defaultdict(list)

    def update(self, db: DiagnosticBatch, runtime) -> None:
        base = unwrap_model(runtime.model)
        head = base.pnp_net
        b = db.batch
        call = db.pose_call
        if call.coor_feat.shape[1] != 5:
            raise RuntimeError(f"D3 expects XYZ3+ROI2D2 pose input, got {tuple(call.coor_feat.shape)}")
        if "roi_xyz" not in b:
            raise RuntimeError("D3 requires GT roi_xyz; use the GT-enabled train/diagnostic path")
        pred_xyz = call.coor_feat[:, :3]
        gt_xyz = b["roi_xyz"].to(pred_xyz.dtype)
        roi2d = call.coor_feat[:, 3:5]
        gt_r, gt_t, cls = b["ego_rot"], b["trans"], b.get("roi_cls")

        for alpha in self.alphas:
            mixed_xyz = pred_xyz.lerp(gt_xyz, alpha)
            mixed_feat = torch.cat([mixed_xyz, roi2d], dim=1)
            paths = {"fixed_region": call.region}
            if call.region is not None:
                try:
                    paths["synced_region"] = _synced_region_from_xyz(mixed_xyz, b, call.region)
                except Exception as exc:
                    self.skipped["synced_region"].append(f"alpha={alpha}: {type(exc).__name__}: {exc}")
            for path, region in paths.items():
                key = f"{path}/alpha_{alpha:.2f}"
                with torch.no_grad():
                    raw_r, raw_t = call_head(head, call, coor_feat=mixed_feat, region=region)
                    pred = decode_raw_pose(runtime.cfg, raw_r, raw_t, b, is_train=False)
                self.metrics[key].add(pred.rot, pred.trans, gt_r, gt_t, cls)

            if self.run_solver and "roi_coord_2d" in b and call.mask_attention is not None:
                try:
                    sr, st, info = solve_pnp_from_correspondence(
                        mixed_xyz,
                        b["roi_coord_2d"],
                        b["roi_extent"],
                        b["roi_cam"],
                        call.mask_attention,
                        db.raw_data,
                    )
                    if sr is not None:
                        self.solver_metrics[f"alpha_{alpha:.2f}"].add(sr, st, gt_r, gt_t, cls)
                    else:
                        self.skipped["explicit_pnp"].append(f"alpha={alpha}: {info}")
                except Exception as exc:
                    self.skipped["explicit_pnp"].append(f"alpha={alpha}: {type(exc).__name__}: {exc}")

    def summary(self):
        summaries = {k: v.summary() for k, v in self.metrics.items()}
        curves = {}
        for path in ("fixed_region", "synced_region"):
            rot, trans = [], []
            available = True
            for a in self.alphas:
                k = f"{path}/alpha_{a:.2f}"
                if k not in summaries:
                    available = False
                    break
                rot.append(summaries[k]["rotation_deg"]["mean"])
                trans.append(summaries[k]["translation_cm"]["mean"])
            if available:
                curves[path] = {
                    "rotation_error_should_decrease": monotonicity(rot, increasing=False),
                    "translation_error_should_decrease": monotonicity(trans, increasing=False),
                    "rotation_deg_mean": rot,
                    "translation_cm_mean": trans,
                }
        return {
            "alphas": self.alphas,
            "pose_metrics": summaries,
            "curves": curves,
            "explicit_pnp": {k: v.summary() for k, v in self.solver_metrics.items()},
            "skipped": dict(self.skipped),
            "decision_rule": "If explicit PnP improves with better XYZ but learned pose does not, the main bottleneck remains correspondence-to-pose utilization.",
        }


def _loss_groups(loss_dict: Dict[str, torch.Tensor]):
    r_keys = [k for k in loss_dict if k in ("loss_PM_R", "loss_rot") or k.endswith("PM_R")]
    t_keys = [
        k for k in loss_dict
        if k in ("loss_centroid", "loss_z")
        or k.startswith("loss_trans")
        or k in ("loss_PM_T", "loss_PM_xy", "loss_PM_z", "loss_PM_xy_noP", "loss_PM_z_noP")
    ]
    if not r_keys:
        raise RuntimeError(f"No rotation loss key found in {sorted(loss_dict)}")
    if not t_keys:
        raise RuntimeError(f"No translation loss key found in {sorted(loss_dict)}")
    return r_keys, t_keys, sum(loss_dict[k] for k in r_keys), sum(loss_dict[k] for k in t_keys)


def _is_shared_pose_param(name: str) -> bool:
    blocked = (
        "pose_rotation", "pose_translation", "rotation_output", "translation_output",
        "rotation_fc", "translation_fc",
    )
    return not any(x in name for x in blocked)


def _flatten_grads(grads, params):
    chunks = []
    for g, p in zip(grads, params):
        if g is None:
            chunks.append(torch.zeros_like(p, memory_format=torch.contiguous_format).flatten())
        else:
            chunks.append(g.contiguous().flatten())
    return torch.cat(chunks) if chunks else None


class D4RTGradientConflict:
    name = "d4_rt_gradient_conflict"

    def __init__(self):
        self.stats = ScalarAccumulator()
        self.meta: Dict[str, Any] = {}
        self.skipped: List[str] = []

    def update(self, db: DiagnosticBatch, runtime) -> None:
        base = unwrap_model(runtime.model)
        head = base.pnp_net
        latent_inputs: Dict[str, torch.Tensor] = {}
        handles = []

        def _capture(name):
            def hook(_m, args):
                if args and torch.is_tensor(args[0]):
                    latent_inputs[name] = args[0]
            return hook

        r_module = getattr(head, "pose_rotation", None) or getattr(head, "rotation_output", None)
        t_module = getattr(head, "pose_translation", None) or getattr(head, "translation_output", None)
        if r_module is not None:
            handles.append(r_module.register_forward_pre_hook(_capture("r")))
        if t_module is not None:
            handles.append(t_module.register_forward_pre_hook(_capture("t")))

        try:
            inp = model_input_from_batch(runtime.cfg, db.batch)
            # Training-loss code logs scalars through Detectron2's ambient
            # EventStorage even though this diagnostic never trains the model.
            with EventStorage():
                output = base(inp, **make_model_kwargs(db.batch, do_loss=True))
            _out_dict, loss_dict = output
            r_keys, t_keys, loss_r, loss_t = _loss_groups(loss_dict)
            self.meta["rotation_loss_keys"] = r_keys
            self.meta["translation_loss_keys"] = t_keys

            if "r" in latent_inputs and "t" in latent_inputs:
                lr, lt = latent_inputs["r"], latent_inputs["t"]
                shared = lr is lt or (lr.data_ptr() == lt.data_ptr() and lr.shape == lt.shape)
                self.meta["late_latent_shared"] = bool(shared)
                if shared:
                    gr = torch.autograd.grad(loss_r, lr, retain_graph=True, allow_unused=True)[0]
                    gt = torch.autograd.grad(loss_t, lt, retain_graph=True, allow_unused=True)[0]
                    if gr is not None and gt is not None:
                        self.stats.add(
                            late_latent_grad_cosine=cosine_scalar(gr, gt),
                            late_latent_grad_r_norm=float(gr.detach().float().norm().cpu()),
                            late_latent_grad_t_norm=float(gt.detach().float().norm().cpu()),
                        )

            named = [(n, p) for n, p in head.named_parameters() if p.requires_grad and _is_shared_pose_param(n)]
            params = [p for _n, p in named]
            self.meta["shared_parameter_tensors"] = [n for n, _p in named]
            if params:
                grs = torch.autograd.grad(loss_r, params, retain_graph=True, allow_unused=True)
                gts = torch.autograd.grad(loss_t, params, retain_graph=False, allow_unused=True)
                gr = _flatten_grads(grs, params)
                gt = _flatten_grads(gts, params)
                if gr is not None and gt is not None:
                    self.stats.add(
                        shared_param_grad_cosine=cosine_scalar(gr, gt),
                        shared_param_grad_r_norm=float(gr.detach().float().norm().cpu()),
                        shared_param_grad_t_norm=float(gt.detach().float().norm().cpu()),
                    )
        except Exception as exc:
            self.skipped.append(f"{type(exc).__name__}: {exc}")
        finally:
            for h in handles:
                h.remove()
            base.zero_grad(set_to_none=True)

    def summary(self):
        s = self.stats.summary()
        vals = self.stats.values.get("shared_param_grad_cosine", [])
        negative = float(np.mean(np.asarray(vals) < 0)) if vals else float("nan")
        return {
            "stats": s,
            "negative_shared_parameter_gradient_fraction": negative,
            "meta": self.meta,
            "skipped": self.skipped,
            "decision_rule": "Frequent negative R/t gradient cosine on genuinely shared representations supports decoupling aggregation/late latent; it is mechanism evidence, not proof of final accuracy gain.",
        }


def _pose_objective(pred, batch, mode: str = "rotation_only"):
    gt_r, gt_t = batch["ego_rot"], batch["trans"]
    rot = (pred.rot - gt_r).pow(2).mean()
    trans = (pred.trans - gt_t).abs().mean()
    if mode == "rotation_only":
        return rot
    if mode == "translation_only":
        return trans
    if mode == "combined":
        return rot + 10.0 * trans
    raise ValueError(mode)


def _project_rms_(x: torch.Tensor, limit: float):
    with torch.no_grad():
        rms = x.pow(2).mean().sqrt()
        if float(rms) > limit:
            x.mul_(float(limit) / float(rms + 1e-12))


class D5GeometryInterfaceAdaptation:
    name = "d5_geometry_interface_adaptation"

    def __init__(self, steps: int = 3, lr: float = 0.5, max_xyz_rms: float = 0.02, adapt_region: bool = True):
        self.steps = int(steps)
        self.lr = float(lr)
        self.max_xyz_rms = float(max_xyz_rms)
        self.adapt_region = bool(adapt_region)
        self.stats: Dict[str, ScalarAccumulator] = defaultdict(ScalarAccumulator)
        self.skipped: List[str] = []

    def _run(self, db: DiagnosticBatch, runtime, adapt_region: bool):
        head = unwrap_model(runtime.model).pnp_net
        call = db.pose_call
        if call.coor_feat.shape[1] != 5:
            raise RuntimeError("D5 currently requires XYZ3+ROI2D2 inputs")
        base_xyz = call.coor_feat[:, :3].detach()
        roi2d = call.coor_feat[:, 3:5].detach()
        xyz_delta = torch.zeros_like(base_xyz, requires_grad=True)
        params = [xyz_delta]
        base_region = call.region.detach() if call.region is not None else None
        region_delta = None
        base_region_logits = None
        if adapt_region and base_region is not None:
            base_region_logits = (base_region.clamp_min(1e-7)).log().detach()
            region_delta = torch.zeros_like(base_region_logits, requires_grad=True)
            params.append(region_delta)

        history = []
        for _ in range(self.steps):
            mixed_xyz = (base_xyz + xyz_delta).clamp(0.0, 1.0)
            feat = torch.cat([mixed_xyz, roi2d], dim=1)
            region = base_region
            if region_delta is not None:
                region = torch.softmax(base_region_logits + region_delta, dim=1)
            raw_r, raw_t = call_head(head, call, coor_feat=feat, region=region)
            pred = decode_raw_pose(runtime.cfg, raw_r, raw_t, db.batch, is_train=True)
            obj = _pose_objective(pred, db.batch, mode="rotation_only")
            grads = torch.autograd.grad(obj, params, retain_graph=False, allow_unused=False)
            with torch.no_grad():
                for p, g in zip(params, grads):
                    p.add_(g, alpha=-self.lr)
                _project_rms_(xyz_delta, self.max_xyz_rms)
                if region_delta is not None:
                    _project_rms_(region_delta, self.max_xyz_rms)
            history.append(float(obj.detach().cpu()))

        with torch.no_grad():
            mixed_xyz = (base_xyz + xyz_delta).clamp(0.0, 1.0)
            feat = torch.cat([mixed_xyz, roi2d], dim=1)
            region = base_region if region_delta is None else torch.softmax(base_region_logits + region_delta, dim=1)
            raw_r, raw_t = call_head(head, call, coor_feat=feat, region=region)
            adapted = decode_raw_pose(runtime.cfg, raw_r, raw_t, db.batch, is_train=False)
            final_xyz_rms = float(xyz_delta.pow(2).mean().sqrt().cpu())
            final_region_rms = float(region_delta.pow(2).mean().sqrt().cpu()) if region_delta is not None else 0.0

            # Same-RMS random control.
            g = torch.Generator(device="cpu")
            g.manual_seed(20260826)
            noise = torch.randn(base_xyz.shape, generator=g, dtype=base_xyz.dtype).to(base_xyz.device)
            noise = noise / noise.pow(2).mean().sqrt().clamp_min(1e-12) * max(final_xyz_rms, 1e-12)
            random_feat = torch.cat([(base_xyz + noise).clamp(0.0, 1.0), roi2d], dim=1)
            random_region = base_region
            if region_delta is not None and base_region_logits is not None:
                rnoise = torch.randn(base_region_logits.shape, generator=g, dtype=base_region_logits.dtype).to(base_region_logits.device)
                rnoise = rnoise / rnoise.pow(2).mean().sqrt().clamp_min(1e-12) * max(final_region_rms, 1e-12)
                random_region = torch.softmax(base_region_logits + rnoise, dim=1)
            rr, rt = call_head(head, call, coor_feat=random_feat, region=random_region)
            random_pred = decode_raw_pose(runtime.cfg, rr, rt, db.batch, is_train=False)

        return adapted, random_pred, final_xyz_rms, final_region_rms, history

    def update(self, db: DiagnosticBatch, runtime) -> None:
        gt_r, gt_t = db.batch["ego_rot"], db.batch["trans"]
        base_re = float(rotation_error_deg(db.pred_rot, gt_r).mean().cpu())
        base_te = float(translation_error_cm(db.pred_trans, gt_t).mean().cpu())
        for use_region in ([False, True] if self.adapt_region and db.pose_call.region is not None else [False]):
            name = "xyz_region_adapt" if use_region else "xyz_adapt"
            try:
                adapted, random_pred, xyz_rms, region_rms, history = self._run(db, runtime, use_region)
                adap_re = float(rotation_error_deg(adapted.rot, gt_r).mean().cpu())
                adap_te = float(translation_error_cm(adapted.trans, gt_t).mean().cpu())
                rand_re = float(rotation_error_deg(random_pred.rot, gt_r).mean().cpu())
                rand_te = float(translation_error_cm(random_pred.trans, gt_t).mean().cpu())
                self.stats[name].add(
                    baseline_rotation_deg=base_re,
                    adapted_rotation_deg=adap_re,
                    rotation_improvement_deg=base_re - adap_re,
                    baseline_translation_cm=base_te,
                    adapted_translation_cm=adap_te,
                    translation_improvement_cm=base_te - adap_te,
                    random_rotation_deg=rand_re,
                    random_translation_cm=rand_te,
                    xyz_delta_rms=xyz_rms,
                    region_logit_delta_rms=region_rms,
                    first_objective=history[0] if history else float("nan"),
                    last_objective=history[-1] if history else float("nan"),
                )
            except Exception as exc:
                self.skipped.append(f"{name}: {type(exc).__name__}: {exc}")

    def summary(self):
        return {
            "settings": {"steps": self.steps, "lr": self.lr, "max_xyz_rms": self.max_xyz_rms},
            "stats": {k: v.summary() for k, v in self.stats.items()},
            "skipped": self.skipped,
            "decision_rule": "Large task-directed improvement from a very small XYZ/Region perturbation, beyond same-RMS random perturbation, supports a frozen-geometry-to-pose interface adaptation mismatch.",
        }


class D6SpatialSensitivity:
    name = "d6_spatial_sensitivity"

    def __init__(self, seed: int = 1234):
        self.seed = int(seed)
        self.metrics: Dict[str, PoseMetricAccumulator] = defaultdict(PoseMetricAccumulator)
        self.effects: Dict[str, ScalarAccumulator] = defaultdict(ScalarAccumulator)
        self.skipped: Dict[str, str] = {}

    def update(self, db: DiagnosticBatch, runtime) -> None:
        head = unwrap_model(runtime.model).pnp_net
        b = db.batch
        gt_r, gt_t, cls = b["ego_rot"], b["trans"], b.get("roi_cls")
        variants = ["normal", "main_grid_shuffle"]
        if hasattr(head, "geometry_downsample_high"):
            variants.extend(["geometry_grid_shuffle", "both_grid_shuffle"])
        base_pred = None
        for variant in variants:
            try:
                with torch.no_grad(), spatial_intervention(head, variant, seed=self.seed):
                    raw_r, raw_t = call_head(head, db.pose_call)
                    pred = decode_raw_pose(runtime.cfg, raw_r, raw_t, b, is_train=False)
                self.metrics[variant].add(pred.rot, pred.trans, gt_r, gt_t, cls)
                if variant == "normal":
                    base_pred = pred
                elif base_pred is not None:
                    self.effects[variant].add(
                        rotation_output_change_deg=float(rotation_error_deg(pred.rot, base_pred.rot).mean().cpu()),
                        translation_output_change_cm=float(translation_error_cm(pred.trans, base_pred.trans).mean().cpu()),
                    )
            except Exception as exc:
                self.skipped[variant] = f"{type(exc).__name__}: {exc}"

    def summary(self):
        return {
            "pose_metrics": {k: v.summary() for k, v in self.metrics.items()},
            "output_sensitivity": {k: v.summary() for k, v in self.effects.items()},
            "skipped": self.skipped,
            "decision_rule": "If spatial-cell shuffling barely changes rotation, the decoder is not strongly using spatial order; if rotation changes strongly while translation is stable, spatial order is specifically useful for R.",
        }
