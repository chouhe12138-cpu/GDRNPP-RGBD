import json

import pytest

from research.pnp_control.summarize_local import summarize


def test_summarize_reports_finite_downward_loss(tmp_path):
    metrics = tmp_path / "metrics.json"
    rows = [
        {"iteration": index, "total_loss": [value, index], "loss_PM_R": value,
         "loss_centroid": value, "loss_z": value}
        for index, value in enumerate((4.0, 3.0, 2.0, 1.0))
    ]
    metrics.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    result = summarize(metrics)
    assert result["status"] == "PASS"
    assert result["all_losses_finite"] is True
    assert result["losses"]["total_loss"]["relative_change_percent"] == pytest.approx(-75.0)


def test_summarize_rejects_nonfinite_loss(tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "iteration": 0,
                "total_loss": float("nan"),
                "loss_PM_R": 1.0,
                "loss_centroid": 1.0,
                "loss_z": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="non-finite total_loss"):
        summarize(metrics)
