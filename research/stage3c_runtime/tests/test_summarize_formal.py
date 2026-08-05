import json

import research.stage3c_runtime.summarize_formal as summarize_module


def write_scores(root, bop, add, objects):
    result = root / "method_lmo-test"
    result.mkdir(parents=True)
    (result / "scores_bop19.json").write_text(
        json.dumps({"bop19_average_recall": bop}),
        encoding="utf-8",
    )
    add_dir = result / "error:ad_ntop:1"
    add_dir.mkdir()
    (add_dir / "scores_th=0.100_min-visib=-1.000.json").write_text(
        json.dumps({"recall": add, "obj_recalls": objects}),
        encoding="utf-8",
    )


def test_fixed_epoch_summary_writes_compact_outputs(tmp_path, monkeypatch):
    training = tmp_path / "B"
    baseline = tmp_path / "A"
    write_scores(
        training / "evaluations/epoch_040/lmo_bop_test",
        0.70,
        0.52,
        {str(index): 0.5 for index in range(1, 9)},
    )
    write_scores(
        baseline / "inference/lmo_bop_test",
        0.69,
        0.50,
        {str(index): 0.4 for index in range(1, 9)},
    )
    checkpoints = training / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "model_epoch_040.pth").write_bytes(b"checkpoint")
    monkeypatch.setattr(
        "sys.argv",
        ["summarize", "B", str(training), str(baseline)],
    )
    assert summarize_module.main() == 0
    result = json.loads((training / "summary/results.json").read_text())
    assert result["fixed_epoch"] == 40
    assert result["status"] == "SCREEN_PASS"
    assert (training / "summary/results.csv").is_file()
    assert (training / "summary/per_object.csv").is_file()
    assert (checkpoints / "SHA256SUMS").is_file()
    assert (checkpoints / "checkpoint_index.json").is_file()
