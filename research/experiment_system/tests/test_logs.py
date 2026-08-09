import json

from research.experiment_system.logs import compact_and_write_warning_summary


def test_warning_summary_deduplicates_repeated_messages(tmp_path):
    run = tmp_path / "run"
    (run / "train").mkdir(parents=True)
    (run / "summary").mkdir()
    (run / "train/console.log").write_text(
        "20260810_120000|WRN|core.test@1: repeated warning\n"
        "ordinary progress\n"
        "20260810_120100|WRN|core.test@1: repeated warning\n"
        "RuntimeWarning: separate warning\n",
        encoding="utf-8",
    )

    payload = compact_and_write_warning_summary(run)

    assert payload["unique_warnings"] == 2
    assert payload["total_warning_occurrences"] == 3
    repeated = next(
        record for record in payload["warnings"] if "repeated warning" in record["message"]
    )
    assert repeated["count"] == 2
    on_disk = json.loads((run / "summary/warnings.json").read_text())
    assert on_disk["source"] == "train/console.log"
    compacted = (run / "train/console.log").read_text()
    assert compacted.count("repeated warning") == 1
