import json

import research.stage3c_runtime.write_overview as overview_module


def test_overview_preserves_legacy_c1_as_an_index(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "write_overview",
            "--output-root",
            str(tmp_path),
            "--legacy-c1",
            "/legacy/c1",
            "--c1-checkpoint-sha256",
            "abc123",
        ],
    )
    assert overview_module.main() == 0
    overview = json.loads((tmp_path / "stage3c/overview.json").read_text())
    c1 = overview["experiments"]["C1"]
    assert c1["legacy_output"] == "/legacy/c1"
    assert c1["legacy_output_preserved"] is True
    assert c1["checkpoint_sha256"] == "abc123"
