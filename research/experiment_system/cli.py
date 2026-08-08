#!/usr/bin/env python3
"""Opt-in CLI for experiment metadata, artifacts, assets, and evaluation indexes."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .artifacts import (
    create_run_directory,
    initialize_run_state,
    read_run_state,
    transition_run_state,
)
from .assets import resolve_assets
from .checkpoints import verify_checkpoint_index
from .manifest import (
    build_run_manifest,
    make_run_id,
    read_json,
    validate_experiment,
    validate_run_manifest,
    write_run_manifest,
)
from .metrics import (
    index_bop_evaluation,
    metric_registry_payload,
    verify_indexed_evaluation,
)
from .registry import compare_generated_registry, registry_payload


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENTS = PROJECT_ROOT / "research" / "experiments"
DEFAULT_CATALOG = Path(__file__).with_name("assets.json")


def print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)

    registry = sub.add_parser("registry", help="validate or inspect experiment metadata")
    registry.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS)
    registry.add_argument("--check", action="store_true")
    registry.add_argument(
        "--json-index",
        type=Path,
        default=PROJECT_ROOT / "research" / "experiment_index.json",
    )
    registry.add_argument(
        "--markdown-index",
        type=Path,
        default=PROJECT_ROOT / "research" / "EXPERIMENT_INDEX.md",
    )

    assets = sub.add_parser("assets", help="resolve and validate a machine path profile")
    assets.add_argument("--profile", type=Path, required=True)
    assets.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    assets.add_argument("--asset", action="append", dest="asset_ids")

    source = sub.add_parser("source", help="show Git provenance and formal cleanliness")
    source.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)

    audit = sub.add_parser("audit", help="check Docker context and staged-file hygiene")
    audit.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)

    freeze = sub.add_parser("verify-freeze", help="verify protected active B/C2 files")
    freeze.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    freeze.add_argument(
        "--freeze-file",
        type=Path,
        default=PROJECT_ROOT / "research" / "active_run_freeze.json",
    )

    prepare = sub.add_parser("prepare", help="create a non-overwriting run directory")
    prepare.add_argument("--experiment", type=Path, required=True)
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--mode", choices=("smoke", "audit", "formal", "diagnostic"), required=True)
    prepare.add_argument("--seed", type=int, required=True)
    prepare.add_argument("--attempt", type=int, default=1)
    prepare.add_argument("--run-id")
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    prepare.add_argument("--profile", type=Path)
    prepare.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    prepare.add_argument("--asset", action="append", dest="asset_ids")
    prepare.add_argument("--image-id")
    prepare.add_argument("--image-revision")
    prepare.add_argument("--image", help="Docker image reference to inspect instead of manual identity")
    prepare.add_argument("--docker-bin", type=Path, default=Path("/usr/bin/docker"))
    prepare.add_argument("--parent-run-id")

    state = sub.add_parser("state", help="read or transition a run state")
    state.add_argument("run_dir", type=Path)
    state.add_argument("--set", dest="new_state")
    state.add_argument("--message", default="state updated by experiment CLI")

    step = sub.add_parser("step", help="register or transition train/eval/diagnostic steps")
    step.add_argument("run_dir", type=Path)
    step.add_argument("--step-id", required=True)
    step.add_argument("--kind", choices=("train", "eval", "diagnostic", "summarize", "verify"))
    step.add_argument("--command-line")
    step.add_argument("--set", dest="new_state")
    step.add_argument("--message", default="step metadata updated")
    step.add_argument("--input", action="append", dest="inputs")
    step.add_argument("--output", action="append", dest="outputs")
    step.add_argument("--complete-run", action="store_true")

    evaluation = sub.add_parser("index-evaluation", help="index raw BOP evaluator results")
    evaluation.add_argument("evaluation_root", type=Path)
    evaluation.add_argument("--dataset", required=True)
    evaluation.add_argument("--bbox-type", choices=("gt", "det"), required=True)
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--check-only", action="store_true")

    verify = sub.add_parser("verify-run", help="validate run metadata and indexed evaluations")
    verify.add_argument("run_dir", type=Path)

    compare = sub.add_parser("compare", help="compare normalized result to a frozen baseline")
    compare.add_argument("--experiment", type=Path, required=True)
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--result", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)

    sub.add_parser("metrics", help="print canonical metric definitions")
    return root


def command_registry(args: argparse.Namespace) -> int:
    if args.check:
        compare_generated_registry(
            args.experiments_root,
            args.json_index,
            args.markdown_index,
        )
    payload = registry_payload(args.experiments_root)
    payload["status"] = "PASS"
    print_json(payload)
    return 0


def command_assets(args: argparse.Namespace) -> int:
    result = resolve_assets(args.catalog, args.profile, args.asset_ids)
    result["status"] = "PASS"
    print_json(result)
    return 0


def command_source(args: argparse.Namespace) -> int:
    from .manifest import collect_git_provenance

    result = collect_git_provenance(args.repo_root.resolve())
    result["formal_clean"] = not result["git_dirty"]
    print_json(result)
    return 0


def command_audit(args: argparse.Namespace) -> int:
    from .security import audit_repository

    print_json(audit_repository(args.repo_root.resolve()))
    return 0


def command_verify_freeze(args: argparse.Namespace) -> int:
    from .freeze import verify_active_freeze

    print_json(verify_active_freeze(args.repo_root.resolve(), args.freeze_file.resolve()))
    return 0


def command_prepare(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    experiment = read_json(args.experiment.resolve())
    validate_experiment(experiment, args.experiment.parent.name)
    run_id = args.run_id or make_run_id(args.mode, args.seed, args.attempt)
    from mmcv import Config

    from .manifest import sha256_bytes

    from .config_contract import validate_config_contract

    planned_run_dir = (
        args.output_root.resolve() / experiment["experiment_id"] / run_id
    )
    resolved_config = Config.fromfile(str(args.config.resolve()))
    validate_config_contract(
        experiment,
        resolved_config,
        args.mode,
        args.seed,
        run_id,
        planned_run_dir,
    )
    resolved_text = resolved_config.pretty_text
    inputs: list[dict[str, Any]] = []
    path_profile_id = None
    if args.profile:
        resolved = resolve_assets(args.catalog, args.profile, args.asset_ids)
        inputs = resolved["assets"]
        path_profile_id = resolved["profile_id"]
    image_id = args.image_id
    image_revision = args.image_revision
    if args.image:
        if image_id or image_revision:
            raise ValueError("use --image or manual --image-id/--image-revision, not both")
        from .docker_image import inspect_docker_image

        image_identity = inspect_docker_image(args.image, args.docker_bin)
        image_id = image_identity["image_id"]
        image_revision = image_identity["revision"]
    manifest = build_run_manifest(
        experiment=experiment,
        run_id=run_id,
        mode=args.mode,
        seed=args.seed,
        repo_root=repo_root,
        config_path=args.config,
        inputs=inputs,
        image_id=image_id,
        image_revision=image_revision,
        path_profile_id=path_profile_id,
        parent_run_id=args.parent_run_id,
    )
    manifest["config"]["resolved_path"] = "meta/resolved_config.py"
    manifest["config"]["resolved_sha256"] = sha256_bytes(resolved_text.encode("utf-8"))
    run_dir = create_run_directory(
        args.output_root.resolve(),
        experiment["experiment_id"],
        run_id,
    )
    initialize_run_state(run_dir)
    snapshot = run_dir / "meta" / "config_snapshot.py"
    shutil.copyfile(args.config.resolve(), snapshot)
    resolved_path = run_dir / "meta" / "resolved_config.py"
    resolved_path.write_text(resolved_text, encoding="utf-8")
    write_run_manifest(run_dir, manifest)
    result = {
        "status": "PREPARED",
        "experiment_id": experiment["experiment_id"],
        "run_id": run_id,
        "run_dir": str(run_dir),
        "manifest": str(run_dir / "meta" / "run_manifest.json"),
        "resolved_config": str(run_dir / "meta" / "resolved_config.py"),
    }
    print_json(result)
    return 0


def command_state(args: argparse.Namespace) -> int:
    if args.new_state:
        result = transition_run_state(args.run_dir.resolve(), args.new_state, args.message)
    else:
        result = read_run_state(args.run_dir.resolve())
    print_json(result)
    return 0


def command_step(args: argparse.Namespace) -> int:
    from .steps import register_step, transition_step

    run_dir = args.run_dir.resolve()
    if args.new_state:
        result = transition_step(
            run_dir,
            args.step_id,
            args.new_state,
            args.message,
            complete_run=args.complete_run,
        )
    else:
        if not args.kind or not args.command_line:
            raise ValueError("registering a step requires --kind and --command-line")
        result = register_step(
            run_dir,
            args.step_id,
            args.kind,
            args.command_line,
            inputs=args.inputs,
            outputs=args.outputs,
        )
    print_json(result)
    return 0


def command_index_evaluation(args: argparse.Namespace) -> int:
    index, normalized = index_bop_evaluation(
        args.evaluation_root,
        dataset_id=args.dataset,
        bbox_type=args.bbox_type,
        checkpoint_id=args.checkpoint,
        write=not args.check_only,
    )
    print_json({"status": "PASS", "index": index, "normalized": normalized})
    return 0


def command_verify_run(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    manifest = read_json(run_dir / "meta" / "run_manifest.json")
    validate_run_manifest(manifest)
    state = read_run_state(run_dir)
    snapshot = run_dir / "meta" / "config_snapshot.py"
    if not snapshot.is_file():
        raise FileNotFoundError(f"config snapshot is missing: {snapshot}")
    resolved = run_dir / manifest["config"].get("resolved_path", "meta/resolved_config.py")
    if not resolved.is_file():
        raise FileNotFoundError(f"resolved config is missing: {resolved}")
    from .manifest import sha256_file

    if sha256_file(snapshot) != manifest["config"]["sha256"]:
        raise RuntimeError("config snapshot differs from the run manifest")
    if sha256_file(resolved) != manifest["config"].get("resolved_sha256"):
        raise RuntimeError("resolved config differs from the run manifest")
    checked_evaluations = 0
    for index_path in sorted((run_dir / "evaluations").glob("*/*/evaluation_index.json")):
        verify_indexed_evaluation(index_path.parent)
        checked_evaluations += 1
    checked_checkpoints = verify_checkpoint_index(run_dir)
    from .steps import load_steps

    steps = load_steps(run_dir)["steps"]
    print_json(
        {
            "status": "PASS",
            "experiment_id": manifest["experiment_id"],
            "run_id": manifest["run_id"],
            "run_state": state["status"],
            "indexed_evaluations_checked": checked_evaluations,
            "indexed_checkpoints_checked": checked_checkpoints,
            "steps_checked": len(steps),
        }
    )
    return 0


def command_compare(args: argparse.Namespace) -> int:
    from .summaries import (
        compare_screening_metrics,
        load_normalized_metrics,
        write_screening_summary,
    )

    experiment = read_json(args.experiment.resolve())
    summary = compare_screening_metrics(
        experiment,
        load_normalized_metrics(args.baseline.resolve()),
        load_normalized_metrics(args.result.resolve()),
    )
    write_screening_summary(args.output.resolve(), summary)
    print_json(summary)
    return 0


def main() -> int:
    args = parser().parse_args()
    commands = {
        "registry": command_registry,
        "assets": command_assets,
        "source": command_source,
        "audit": command_audit,
        "verify-freeze": command_verify_freeze,
        "prepare": command_prepare,
        "state": command_state,
        "step": command_step,
        "index-evaluation": command_index_evaluation,
        "verify-run": command_verify_run,
        "compare": command_compare,
        "metrics": lambda _args: (print_json(metric_registry_payload()) or 0),
    }
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
