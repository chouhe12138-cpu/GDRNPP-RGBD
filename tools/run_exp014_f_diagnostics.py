#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from research.diagnostics.pose_structure.f_glm_diagnostic import FGLMPoseDiagnostic
from research.diagnostics.pose_structure.reporting import write_results
from research.diagnostics.pose_structure.runtime import build_runtime, iter_diagnostic_batches


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Targeted no-training diagnostic for the completed EXP013F GLM-Pose-L checkpoint"
    )
    p.add_argument(
        "--config-file",
        default="configs/gdrn/lmo_pbr/research/exp013/f_glm_pose_l/train.py",
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-batches", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260902)
    return p


def _mean(d, *keys):
    cur = d
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def write_decision(output_dir: str, summary: dict) -> None:
    out = Path(output_dir)
    screen = summary["screen"]
    impl = summary["exp017_design_implications"]
    comparison = summary["comparison_to_normal"]
    support = summary["support_pooling"]

    lines = [
        "# EXP013F Targeted Diagnostic Decision",
        "",
        "本文件只用于决定 EXP017 的结构约束；这里的子集误差不是正式 BOP 结果。",
        "",
        "## 有效性",
        "",
        f"- diagnostic_valid: `{screen['diagnostic_valid']}`",
        f"- invalid_support_pooling_mass_material: `{screen['invalid_support_pooling_mass_material']}`",
        f"- learned_pooling_has_positive_rotation_evidence: `{screen['learned_pooling_has_positive_rotation_evidence']}`",
        f"- spatial_position_has_rotation_evidence: `{screen['spatial_position_has_rotation_evidence']}`",
        f"- depth_stats_have_translation_evidence: `{screen['depth_stats_have_translation_evidence']}`",
        "",
        "## 关键数值",
        "",
        f"- normal pooling invalid mass mean: `{_mean(support, 'pooling_invalid_mass', 'mean')}`",
    ]
    for name in (
        "support_masked",
        "uniform_pool",
        "position_shuffle",
        "token_shuffle",
        "depth_zero",
        "depth_shuffle",
    ):
        item = comparison.get(name, {})
        lines.append(
            f"- {name}: Δrotation error `{item.get('rotation_error_delta_deg_vs_normal')}` deg; "
            f"Δtranslation error `{item.get('translation_error_delta_cm_vs_normal')}` cm"
        )
    lines += [
        "",
        "## 对 EXP017 的直接约束",
        "",
        f"- Base: {impl['base']}",
        f"- Support: {impl['support']}",
        f"- Rotation: {impl['rotation']}",
        f"- Pooling: {impl['pooling']}",
        f"- Depth: {impl['depth']}",
        f"- Translation: {impl['translation']}",
        "",
        "## 下一步",
        "",
        "基于本文件与 results.json 设计 EXP017。只输出结构方案、参数预算、preflight/smoke 和预注册 gate；不要启动 formal 训练。",
    ]
    (out / "DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    runtime = build_runtime(
        args.config_file,
        args.checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
    )
    diagnostic = FGLMPoseDiagnostic(seed=args.seed)
    started = time.time()
    try:
        for batch_idx, db in enumerate(iter_diagnostic_batches(runtime, args.max_batches), 1):
            print(f"[EXP013F-DIAG] batch {batch_idx}/{args.max_batches}")
            diagnostic.update(db, runtime)
    finally:
        runtime.close()

    summary = diagnostic.summary()
    metadata = {
        "experiment": "LOCAL-DIAG-EXP013F (output folder EXP-014; not a repository experiment ID)",
        "source_model": "EXP013F GLM-Pose-L E40",
        "config_file": args.config_file,
        "checkpoint": args.checkpoint,
        "head_type": type(runtime.model.pnp_net).__name__,
        "max_batches": args.max_batches,
        "batch_size": args.batch_size,
        "samples_requested": args.max_batches * args.batch_size,
        "seed": args.seed,
        "elapsed_sec": time.time() - started,
        "formal_bop": False,
        "optimizer_steps": 0,
    }
    write_results(args.output_dir, metadata, {diagnostic.name: summary})
    write_decision(args.output_dir, summary)
    print(f"[EXP013F-DIAG] wrote {Path(args.output_dir) / 'results.json'}")
    print(f"[EXP013F-DIAG] wrote {Path(args.output_dir) / 'SUMMARY.md'}")
    print(f"[EXP013F-DIAG] wrote {Path(args.output_dir) / 'DECISION.md'}")


if __name__ == "__main__":
    main()
