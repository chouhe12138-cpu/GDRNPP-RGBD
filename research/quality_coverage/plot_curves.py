#!/usr/bin/env python3
"""Create paper-style loss, LM-O metric, and per-object ADD(-S) plots."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_output", type=Path)
    parser.add_argument("--baseline-output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_json_lines(path: Path):
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def scalar(value):
    """Unwrap the ``[value, iteration]`` form written by MyJSONWriter."""
    if isinstance(value, list):
        return value[0]
    return value


def find_score(eval_dir: Path, pattern: str):
    paths = sorted(glob.glob(str(eval_dir / "*" / pattern), recursive=True))
    if len(paths) != 1:
        return None
    return json.loads(Path(paths[0]).read_text(encoding="utf-8"))


def find_add_score(eval_dir: Path):
    paths = []
    for directory_pattern in ("error=ad_ntop=*", "error:ad_ntop:*"):
        for score_name in (
            "scores_th=0.100_min-visib=-1.000.json",
            "scores_th:0.100_min-visib:-1.000.json",
        ):
            pattern = f"{directory_pattern}/{score_name}"
            paths.extend(glob.glob(str(eval_dir / "**" / pattern), recursive=True))
    unique = sorted(set(paths))
    if len(unique) != 1:
        return None
    return json.loads(Path(unique[0]).read_text(encoding="utf-8"))


def load_epoch_scores(training_output: Path):
    rows = []
    eval_dirs = list(training_output.glob("inference_epoch_*"))
    eval_dirs.extend((training_output / "evaluations").glob("epoch_*"))
    for eval_dir in sorted(eval_dirs):
        if eval_dir.name.startswith("inference_epoch_"):
            epoch = int(eval_dir.name.split("_")[2])
        else:
            epoch = int(eval_dir.name.split("_")[1])
        bop = find_score(eval_dir / "lmo_bop_test", "scores_bop19.json")
        add = find_add_score(eval_dir / "lmo_bop_test")
        if bop is not None:
            rows.append(
                {
                    "epoch": epoch,
                    "bop_ar": float(bop["bop19_average_recall"]),
                    "add_s": None if add is None else float(add["recall"]),
                    "object_add": {} if add is None else add.get("obj_recalls", {}),
                }
            )
    return sorted(rows, key=lambda row: row["epoch"])


def save_loss_plot(rows, output_path: Path):
    keys = ["total_loss", "loss_PM_R", "loss_centroid", "loss_z"]
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    for key in keys:
        points = [
            (scalar(row.get("epoch")), scalar(row.get(key)))
            for row in rows
            if key in row and "epoch" in row
        ]
        if points:
            axis.plot([p[0] for p in points], [p[1] for p in points], label=key)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Training loss")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_metric_plot(rows, output_path: Path):
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    epochs = [row["epoch"] for row in rows]
    axis.plot(epochs, [100.0 * row["bop_ar"] for row in rows], marker="o", label="BOP AR")
    add_rows = [row for row in rows if row["add_s"] is not None]
    if add_rows:
        axis.plot(
            [row["epoch"] for row in add_rows],
            [100.0 * row["add_s"] for row in add_rows],
            marker="s",
            label="ADD(-S) 0.1d",
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Recall / AR (%)")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_object_plot(baseline, best, output_path: Path):
    object_ids = sorted(set(baseline) | set(best), key=int)
    x = list(range(len(object_ids)))
    width = 0.38
    fig, axis = plt.subplots(figsize=(8.0, 4.5))
    axis.bar([value - width / 2 for value in x], [100 * baseline.get(k, 0) for k in object_ids], width, label="Official")
    axis.bar([value + width / 2 for value in x], [100 * best.get(k, 0) for k in object_ids], width, label="Quality+coverage")
    axis.set_xticks(x, object_ids)
    axis.set_xlabel("LM-O object ID")
    axis.set_ylabel("ADD(-S) 0.1d recall (%)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    structured = (args.training_output / "evaluations").is_dir()
    output_dir = args.output_dir or (
        args.training_output / "summary/figures"
        if structured
        else args.training_output / "paper_figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = (
        args.training_output / "train/metrics.jsonl"
        if structured
        else args.training_output / "metrics.json"
    )
    training_rows = load_json_lines(metrics_path)
    epoch_rows = load_epoch_scores(args.training_output)
    save_loss_plot(training_rows, output_dir / "training_losses.png")
    save_metric_plot(epoch_rows, output_dir / "lmo_metrics_by_epoch.png")

    if args.baseline_output and epoch_rows:
        baseline_add = find_add_score(args.baseline_output)
        valid_add = [row for row in epoch_rows if row["add_s"] is not None]
        if baseline_add is not None and valid_add:
            best = max(valid_add, key=lambda row: (row["bop_ar"], row["add_s"]))
            save_object_plot(
                baseline_add.get("obj_recalls", {}),
                best["object_add"],
                output_dir / "per_object_add_s.png",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
