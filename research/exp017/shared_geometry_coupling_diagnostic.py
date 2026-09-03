#!/usr/bin/env python3
"""Read-only EXP017 adapter/shared-geometry autograd diagnostic.

This tool never calls backward(), optimizer.step(), or checkpoint save.  It
uses autograd.grad on the same E10 state and same batches to separate the
normal EXP017 gradient from the graph obtained when only the adapter input is
detached.  Its subset re/te values are diagnostic metrics, not official BOP.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch

from core.gdrn_modeling.models.heads.exp017_rotation_residual_pnp_net import (
    SupportAwareRotationResidualPnPNet,
)
from research.diagnostics.pose_structure.common import cosine_scalar, tensor_rms
from research.diagnostics.pose_structure.metrics import (
    PoseMetricAccumulator,
    ScalarAccumulator,
)
from research.diagnostics.pose_structure.model_access import decode_raw_pose
from research.diagnostics.pose_structure.runtime import (
    build_runtime,
    iter_diagnostic_batches,
)


GEOMETRY_ENCODER_PREFIXES = (
    "geometry_input_projection.",
    "geometry_local_fine.",
    "geometry_downsample_mid.",
    "geometry_local_mid.",
    "geometry_downsample_high.",
    "geometry_local_high.",
)

VARIANTS = {
    "normal": dict(adapter_enabled=True, detach_adapter_geometry=False),
    "adapter_off": dict(adapter_enabled=False, detach_adapter_geometry=False),
    "adapter_detached": dict(adapter_enabled=True, detach_adapter_geometry=True),
}


def _flatten_gradients(grads, parameters) -> torch.Tensor:
    return torch.cat(
        [
            (torch.zeros_like(parameter) if grad is None else grad)
            .detach()
            .float()
            .flatten()
            for grad, parameter in zip(grads, parameters)
        ]
    )


def _forward_variant(head, call, options):
    return head.forward_with_adapter_intervention(
        call.coor_feat,
        region=call.region,
        extents=call.extents,
        mask_attention=call.mask_attention,
        **options,
    )


def diagnose_batches(runtime, max_batches: int) -> dict[str, object]:
    head = runtime.model.pnp_net
    if not isinstance(head, SupportAwareRotationResidualPnPNet):
        raise TypeError(f"Expected EXP017 head, got {type(head).__name__}")

    shared_named = [
        (name, parameter)
        for name, parameter in head.named_parameters()
        if name.startswith(GEOMETRY_ENCODER_PREFIXES)
    ]
    if not shared_named:
        raise RuntimeError("No EXP013A shared geometry encoder parameters found")
    shared_parameters = [parameter for _name, parameter in shared_named]
    adapter_parameters = list(head.adapter_parameters())

    metrics = {name: PoseMetricAccumulator() for name in VARIANTS}
    stats = ScalarAccumulator()
    exact = {
        "normal_vs_detached_raw_r": True,
        "normal_vs_detached_raw_t": True,
        "normal_vs_off_raw_t": True,
    }
    batches = 0

    for db in iter_diagnostic_batches(runtime, max_batches):
        batches += 1
        outputs = {}
        with torch.no_grad():
            for name, options in VARIANTS.items():
                raw_r, raw_t, info = _forward_variant(head, db.pose_call, options)
                pred = decode_raw_pose(
                    runtime.cfg, raw_r, raw_t, db.batch, is_train=False
                )
                outputs[name] = (raw_r, raw_t, pred, info)
                metrics[name].add(
                    pred.rot,
                    pred.trans,
                    db.batch["ego_rot"],
                    db.batch["trans"],
                    db.batch.get("roi_cls"),
                )

        normal_r, normal_t, normal_pred, normal_info = outputs["normal"]
        off_r, off_t, off_pred, _off_info = outputs["adapter_off"]
        detached_r, detached_t, detached_pred, detached_info = outputs[
            "adapter_detached"
        ]
        exact["normal_vs_detached_raw_r"] &= torch.equal(normal_r, detached_r)
        exact["normal_vs_detached_raw_t"] &= torch.equal(normal_t, detached_t)
        exact["normal_vs_off_raw_t"] &= torch.equal(normal_t, off_t)
        stats.add(
            residual_raw_r_rms=float(tensor_rms(normal_r - off_r).cpu()),
            residual_decoded_rotation_rms=float(
                tensor_rms(normal_pred.rot - off_pred.rot).cpu()
            ),
            normal_vs_off_raw_t_rms=float(tensor_rms(normal_t - off_t).cpu()),
            normal_vs_off_decoded_translation_rms=float(
                tensor_rms(normal_pred.trans - off_pred.trans).cpu()
            ),
            normal_vs_detached_raw_r_rms=float(tensor_rms(normal_r - detached_r).cpu()),
            normal_vs_detached_decoded_rotation_rms=float(
                tensor_rms(normal_pred.rot - detached_pred.rot).cpu()
            ),
            alpha_r=float(head.alpha_r.detach().cpu()),
            delta_r_rms=float(tensor_rms(normal_info["delta_r"]).cpu()),
        )
        if not detached_info["adapter_geometry_detached"]:
            raise RuntimeError("Detached intervention did not detach the adapter grid")

        gradient_vectors = {}
        adapter_gradient_vectors = {}
        for name, options in VARIANTS.items():
            raw_r, raw_t, _info = _forward_variant(head, db.pose_call, options)
            pred = decode_raw_pose(runtime.cfg, raw_r, raw_t, db.batch, is_train=True)
            rotation_objective = (pred.rot - db.batch["ego_rot"]).square().mean()
            grads = torch.autograd.grad(
                rotation_objective,
                shared_parameters + adapter_parameters,
                allow_unused=True,
            )
            split = len(shared_parameters)
            gradient_vectors[name] = _flatten_gradients(
                grads[:split], shared_parameters
            )
            adapter_gradient_vectors[name] = _flatten_gradients(
                grads[split:], adapter_parameters
            )
            stats.add(
                **{
                    f"rotation_proxy_objective_{name}": float(
                        rotation_objective.detach().cpu()
                    ),
                    f"shared_geometry_rotation_grad_norm_{name}": float(
                        gradient_vectors[name].norm().cpu()
                    ),
                    f"adapter_rotation_grad_norm_{name}": float(
                        adapter_gradient_vectors[name].norm().cpu()
                    ),
                }
            )

        injected = gradient_vectors["normal"] - gradient_vectors["adapter_detached"]
        detached_norm = gradient_vectors["adapter_detached"].norm()
        raw_r, raw_t, _info = _forward_variant(head, db.pose_call, VARIANTS["normal"])
        pred = decode_raw_pose(runtime.cfg, raw_r, raw_t, db.batch, is_train=True)
        translation_objective = (pred.trans - db.batch["trans"]).square().mean()
        translation_grads = torch.autograd.grad(
            translation_objective,
            shared_parameters,
            allow_unused=True,
        )
        translation_gradient = _flatten_gradients(
            translation_grads, shared_parameters
        )
        stats.add(
            adapter_injected_shared_geometry_grad_norm=float(injected.norm().cpu()),
            adapter_injected_to_a_rotation_grad_ratio=float(
                (injected.norm() / detached_norm.clamp_min(1.0e-12)).cpu()
            ),
            normal_vs_detached_shared_grad_cosine=cosine_scalar(
                gradient_vectors["normal"], gradient_vectors["adapter_detached"]
            ),
            translation_proxy_objective=float(translation_objective.detach().cpu()),
            shared_geometry_translation_grad_norm=float(
                translation_gradient.norm().cpu()
            ),
            adapter_injected_vs_translation_grad_cosine=cosine_scalar(
                injected, translation_gradient
            ),
            a_rotation_vs_translation_grad_cosine=cosine_scalar(
                gradient_vectors["adapter_detached"], translation_gradient
            ),
            normal_vs_detached_adapter_grad_rms=float(
                tensor_rms(
                    adapter_gradient_vectors["normal"]
                    - adapter_gradient_vectors["adapter_detached"]
                ).cpu()
            ),
        )

    return {
        "batches": batches,
        "samples": sum(len(acc.rot_deg) for acc in metrics.values()) // len(metrics),
        "shared_geometry_parameter_tensors": [
            name for name, _parameter in shared_named
        ],
        "shared_geometry_parameter_count": sum(p.numel() for p in shared_parameters),
        "variants": {
            name: accumulator.summary() for name, accumulator in metrics.items()
        },
        "output_exactness": exact,
        "gradient_and_output_stats": stats.summary(),
        "official_metrics": {
            "available": False,
            "reason": (
                "This bounded read-only diagnostic does not run the official BOP/ADD "
                "evaluator; its subset re/te values are mechanism evidence only."
            ),
        },
    }


def _write_summary(path: Path, payload: dict[str, object]) -> None:
    stats = payload["diagnostic"]["gradient_and_output_stats"]
    lines = [
        "# EXP017 shared-geometry coupling diagnostic",
        "",
        f"- checkpoint: `{payload['checkpoint']}`",
        f"- samples: `{payload['diagnostic']['samples']}`",
        "- optimizer/backward/checkpoint writes: `none`",
        "- official BOP/ADD: `not run`",
        "",
        "The decisive quantity is `adapter_injected_shared_geometry_grad_norm`: "
        "normal and detached forwards are value-identical, so a non-zero gradient "
        "difference is the adapter's extra path into EXP013A's geometry encoder.",
        "",
        "```json",
        json.dumps(stats, indent=2, sort_keys=True),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.time()
    runtime = build_runtime(
        args.config_file,
        args.checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
    )
    try:
        diagnostic = diagnose_batches(runtime, args.max_batches)
    finally:
        runtime.close()
    payload = {
        "status": "PASS",
        "config_file": args.config_file,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": (
            int(match.group(1))
            if (match := re.search(r"epoch_(\d+)", Path(args.checkpoint).name))
            else None
        ),
        "seed": args.seed,
        "elapsed_sec": time.time() - started,
        "diagnostic": diagnostic,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_summary(args.output_dir / "SUMMARY.md", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
