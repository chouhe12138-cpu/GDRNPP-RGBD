from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from .diagnostics import (
    D1RTOracle,
    D2BranchAblation,
    D3CorrespondenceUtilization,
    D4RTGradientConflict,
    D5GeometryInterfaceAdaptation,
    D6SpatialSensitivity,
)
from .model_access import unwrap_model
from .reporting import write_results
from .runtime import build_runtime, iter_diagnostic_batches



def _module_sha256(module) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        h.update(name.encode("utf-8"))
        t = tensor.detach().contiguous().cpu()
        h.update(str(t.dtype).encode("ascii"))
        h.update(str(tuple(t.shape)).encode("ascii"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def _parse_alphas(text: str):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Low-cost structural diagnostics for GDRNPP pose heads")
    p.add_argument("--config-file", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--diagnostics", default="d1,d2,d3,d4,d5,d6", help="Comma-separated: d1,d2,d3,d4,d5,d6")
    p.add_argument("--max-batches", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    p.add_argument("--no-explicit-pnp", action="store_true")
    p.add_argument("--d5-steps", type=int, default=3)
    p.add_argument("--d5-lr", type=float, default=0.5)
    p.add_argument("--d5-max-xyz-rms", type=float, default=0.02)
    p.add_argument("--d5-no-region", action="store_true")
    p.add_argument(
        "--xyz-renderer", choices=("egl", "cpp"), default=None,
        help="Override MODEL.POSE_NET.XYZ_RENDERER (use cpp for local WSL diagnostics)",
    )
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    selected = {x.strip().lower() for x in args.diagnostics.split(",") if x.strip()}
    valid = {"d1", "d2", "d3", "d4", "d5", "d6"}
    unknown = selected - valid
    if unknown:
        raise SystemExit(f"Unknown diagnostics: {sorted(unknown)}")

    diagnostics = []
    if "d1" in selected:
        diagnostics.append(D1RTOracle())
    if "d2" in selected:
        diagnostics.append(D2BranchAblation())
    if "d3" in selected:
        diagnostics.append(D3CorrespondenceUtilization(_parse_alphas(args.alphas), run_solver=not args.no_explicit_pnp))
    if "d4" in selected:
        diagnostics.append(D4RTGradientConflict())
    if "d5" in selected:
        diagnostics.append(
            D5GeometryInterfaceAdaptation(
                steps=args.d5_steps,
                lr=args.d5_lr,
                max_xyz_rms=args.d5_max_xyz_rms,
                adapt_region=not args.d5_no_region,
            )
        )
    if "d6" in selected:
        diagnostics.append(D6SpatialSensitivity(seed=args.seed + 600))

    cfg_overrides = {}
    if args.xyz_renderer is not None:
        cfg_overrides["MODEL.POSE_NET.XYZ_RENDERER"] = args.xyz_renderer
    runtime = build_runtime(
        args.config_file,
        args.checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        cfg_overrides=cfg_overrides,
    )
    started = time.time()
    pnp_before = _module_sha256(unwrap_model(runtime.model).pnp_net)
    try:
        for batch_idx, db in enumerate(iter_diagnostic_batches(runtime, args.max_batches), 1):
            print(f"[diagnostic] batch {batch_idx}/{args.max_batches}")
            for diagnostic in diagnostics:
                print(f"  - {diagnostic.name}")
                diagnostic.update(db, runtime)
    finally:
        runtime.close()

    pnp_after = _module_sha256(unwrap_model(runtime.model).pnp_net)
    if pnp_before != pnp_after:
        raise RuntimeError(
            "Diagnostic run changed pose-head state_dict. This violates the no-training invariant; "
            "inspect temporary interventions before using any result."
        )
    summaries = {d.name: d.summary() for d in diagnostics}
    metadata = {
        "config_file": args.config_file,
        "checkpoint": args.checkpoint,
        "head_type": type(runtime.model.pnp_net).__name__,
        "max_batches": args.max_batches,
        "batch_size": args.batch_size,
        "samples_requested": args.max_batches * args.batch_size,
        "seed": args.seed,
        "elapsed_sec": time.time() - started,
        "pose_head_state_sha256_before": pnp_before,
        "pose_head_state_sha256_after": pnp_after,
        "pose_head_weights_unchanged": True,
        "note": "Direct diagnostic metrics are not official BOP scores; run formal BOP evaluation only after a structural hypothesis survives these screens.",
    }
    write_results(args.output_dir, metadata, summaries)
    print(f"[diagnostic] wrote {Path(args.output_dir) / 'results.json'}")
    print(f"[diagnostic] wrote {Path(args.output_dir) / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
