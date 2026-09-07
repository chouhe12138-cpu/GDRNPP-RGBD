"""Export a complete EXP019 run, invoke the current BOP evaluator, and summarize."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import EXPECTED_LMO_TARGETS, ExperimentConfig
from .gates import evaluate_gates


ROOT = Path(__file__).resolve().parents[3]
CONSUMERS = ("patch", "ransac", "epro")


def _alpha_token(alpha):
    return f"{int(round(float(alpha) * 100)):03d}"


def _result_name(consumer, alpha):
    return f"{consumer}alpha{_alpha_token(alpha)}_lmo-test.csv"


def _flatten(values):
    return " ".join(f"{float(value):.9g}" for value in np.asarray(values).reshape(-1))


def _load_rows(run_dir):
    rows = []
    with (run_dir / "poses.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def export_bop(rows, result_dir, cfg):
    result_dir.mkdir(parents=True, exist_ok=False)
    grouped = defaultdict(list)
    for item in rows:
        for consumer in CONSUMERS:
            pose = item[consumer]
            if pose["success"]:
                grouped[(consumer, float(item["alpha"]))].append(
                    {
                        "scene_id": int(item["scene_id"]),
                        "im_id": int(item["im_id"]),
                        "obj_id": int(item["obj_id"]),
                        "score": 1.0,
                        "R": _flatten(pose["R"]),
                        "t": _flatten(np.asarray(pose["t"], dtype=np.float64) * 1000.0),
                        "time": -1.0,
                    }
                )
    names = []
    fields = ["scene_id", "im_id", "obj_id", "score", "R", "t", "time"]
    for consumer in CONSUMERS:
        for alpha in cfg.alphas:
            name = _result_name(consumer, alpha)
            names.append(name)
            with (result_dir / name).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(grouped[(consumer, float(alpha))])
    return names


def run_bop(result_dir, eval_dir, names):
    toolkit = ROOT / ".local" / "bop_toolkit"
    renderer = ROOT / ".local" / "bop_renderer" / "build"
    required = [toolkit / "scripts" / "eval_bop19_pose.py", renderer]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing BOP evaluation dependencies:\n" + "\n".join(missing))
    environment = os.environ.copy()
    pythonpath = [str(ROOT), str(toolkit), str(renderer)]
    if environment.get("PYTHONPATH"):
        pythonpath.append(environment["PYTHONPATH"])
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(pythonpath),
            "BOP_PATH": str((ROOT / "datasets" / "BOP_DATASETS").resolve()),
            "BOP_RESULTS_PATH": str(result_dir),
            "BOP_EVAL_PATH": str(eval_dir),
            "BOP_RENDERER_PATH": str(renderer),
            "BOP_NUM_WORKERS": "1",
        }
    )
    command = [
        sys.executable,
        str(ROOT / "lib" / "pysixd" / "scripts" / "eval_pose_results_more.py"),
        f"--results_path={result_dir}",
        f"--eval_path={eval_dir}",
        f"--result_filenames={','.join(names)}",
        "--renderer_type=cpp",
        "--error_types=mspd,mssd,vsd,ad,reS,teS",
        "--targets_filename=test_targets_bop19.json",
        "--n_top=1",
        "--dataset=lmo",
    ]
    subprocess.run(command, check=True, cwd=ROOT, env=environment)


def _load_add_score(result_root):
    candidates = []
    for pattern in (
        "error=ad_ntop=*/scores_th=0.100_min-visib=-1.000.json",
        "error:ad_ntop:*/scores_th:0.100_min-visib:-1.000.json",
    ):
        candidates.extend(result_root.glob(pattern))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one ADD(-S) score under {result_root}, got {candidates}")
    return json.loads(candidates[0].read_text(encoding="utf-8"))


def summarize(rows, eval_dir, cfg):
    summary = {consumer: {} for consumer in CONSUMERS}
    for consumer in CONSUMERS:
        for alpha in cfg.alphas:
            key = f"{alpha:.2f}"
            selected = [row for row in rows if float(row["alpha"]) == alpha]
            result_root = eval_dir / Path(_result_name(consumer, alpha)).stem
            bop = json.loads((result_root / "scores_bop19.json").read_text(encoding="utf-8"))
            add = _load_add_score(result_root)
            summary[consumer][key] = {
                "add": float(add["recall"]),
                "bop": float(bop["bop19_average_recall"]),
                "reS": float(bop["bop19_average_recall_reS"]),
                "teS": float(bop["bop19_average_recall_teS"]),
                "per_object_add": {str(k): float(v) for k, v in add["obj_recalls"].items()},
                "solve_failures": int(sum(not row[consumer]["success"] for row in selected)),
                "mean_support_points": float(np.mean([row["num_support"] for row in selected])),
            }
    return summary


def reproduction_report(cfg, summary):
    observed = {
        "patch_alpha0_add": summary["patch"]["0.00"]["add"],
        "patch_alpha0_bop": summary["patch"]["0.00"]["bop"],
        "ransac_alpha0_add": summary["ransac"]["0.00"]["add"],
        "ransac_alpha0_bop": summary["ransac"]["0.00"]["bop"],
        "ransac_alpha1_add": summary["ransac"]["1.00"]["add"],
        "ransac_alpha1_bop": summary["ransac"]["1.00"]["bop"],
    }
    expected = {
        "patch_alpha0_add": cfg.historical_patch_alpha0_add,
        "patch_alpha0_bop": cfg.historical_patch_alpha0_bop,
        "ransac_alpha0_add": cfg.historical_ransac_alpha0_add,
        "ransac_alpha0_bop": cfg.historical_ransac_alpha0_bop,
        "ransac_alpha1_add": cfg.historical_ransac_alpha1_add,
        "ransac_alpha1_bop": cfg.historical_ransac_alpha1_bop,
    }
    deltas = {key: observed[key] - expected[key] for key in expected}
    return {
        "status": "PASS" if all(abs(value) <= 0.001 for value in deltas.values()) else "FAIL",
        "absolute_tolerance": 0.001,
        "observed": observed,
        "historical": expected,
        "deltas": deltas,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "COMPLETE" or metadata.get("num_targets") != EXPECTED_LMO_TARGETS:
        raise RuntimeError("Official BOP aggregation requires a complete 1,445-target EXP019 run")
    cfg = ExperimentConfig()
    rows = _load_rows(run_dir)
    expected_rows = EXPECTED_LMO_TARGETS * len(cfg.alphas)
    if len(rows) != expected_rows:
        raise RuntimeError(f"poses.jsonl contains {len(rows)} rows, expected {expected_rows}")
    result_dir, eval_dir = run_dir / "bop_results", run_dir / "bop_eval"
    names = export_bop(rows, result_dir, cfg)
    run_bop(result_dir, eval_dir, names)
    summary = summarize(rows, eval_dir, cfg)
    reproduction = reproduction_report(cfg, summary)
    gate_report = evaluate_gates(cfg, summary)
    if reproduction["status"] != "PASS":
        gate_report["decision"] = "PROTOCOL_REPRODUCTION_FAILED_STOP"
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "reproduction_report.json").write_text(
        json.dumps(reproduction, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "gate_report.json").write_text(
        json.dumps(gate_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"reproduction": reproduction, "gates": gate_report}, indent=2))


if __name__ == "__main__":
    main()
