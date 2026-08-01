import json

import pytest

import research.quality_coverage.trainer as trainer_module
from research.quality_coverage.plot_curves import scalar
from research.quality_coverage.trainer import Stage3C1, TrainingOptions


def test_official_training_options_are_locked():
    TrainingOptions().validate()
    with pytest.raises(ValueError, match="official protocol is locked"):
        TrainingOptions(epochs=5).validate()
    with pytest.raises(ValueError, match="official protocol is locked"):
        TrainingOptions(learning_rate=1e-3).validate()


def test_exploratory_options_allow_explicit_parameter_changes():
    TrainingOptions(
        epochs=10,
        batch_size=12,
        effective_batch_size=48,
        learning_rate=1e-3,
        evaluate_every=5,
        save_every=5,
        name="debug_batch12",
        protocol="exploratory",
    ).validate()


def test_effective_batch_must_be_an_exact_multiple():
    with pytest.raises(ValueError, match="positive multiple"):
        TrainingOptions(
            batch_size=10,
            effective_batch_size=48,
            protocol="exploratory",
            name="invalid_batch",
        ).validate()


def test_preview_does_not_create_output(tmp_path, capsys):
    model = Stage3C1(workspace=tmp_path)
    job = model.train(launch=False)
    assert not job.launched
    assert not job.output_dir.exists()
    assert "PREVIEW_ONLY" in capsys.readouterr().out


def test_generated_config_and_runner_use_mounted_paths(tmp_path):
    model = Stage3C1(workspace=tmp_path)
    options = TrainingOptions()
    output = tmp_path / "outputs/EXP-20260731-006/quality_coverage_full"
    config = output / "stage3c1_runtime_config.py"
    plot_script = output / "plot_curves_runtime.py"
    rendered_config = model._render_config(options)
    rendered_runner = model._render_runner(
        options,
        config,
        output,
        plot_script,
    )
    assert "TOTAL_EPOCHS=40" in rendered_config
    assert "IMS_PER_BATCH=48" in rendered_config
    assert "EVAL_PERIOD=5" in rendered_config
    assert "/workspace/gdrnpp/output/EXP-20260731-006" in rendered_runner
    assert "run_baseline.sh" in rendered_runner
    assert "plot_curves_runtime.py" in rendered_runner
    assert "quality_coverage.summarize" in rendered_runner


def test_launch_materializes_a_reproducible_run_without_overwriting(
    tmp_path, monkeypatch
):
    calls = []

    class Result:
        returncode = 0

    def fake_run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return Result()

    monkeypatch.setattr(trainer_module.subprocess, "run", fake_run)
    model = Stage3C1(workspace=tmp_path)
    job = model.train(launch=True)
    assert job.launched
    assert job.output_dir.is_dir()
    assert job.output_dir.stat().st_mode & 0o777 == 0o777
    assert (job.output_dir / "stage3c1_runtime_config.py").is_file()
    assert (job.output_dir / "run_inside_container.sh").is_file()
    assert (job.output_dir / "run_manifest.json").is_file()
    assert (job.output_dir / "plot_curves_runtime.py").is_file()
    assert len(calls) == 2
    assert calls[-1][0][:3] == ["/usr/bin/docker", "exec", "-d"]


def test_resume_refuses_changed_settings(tmp_path):
    model = Stage3C1(workspace=tmp_path)
    output = tmp_path / "outputs/EXP-20260731-006/debug"
    output.mkdir(parents=True)
    (output / "last_checkpoint").write_text("model_0000001.pth", encoding="utf-8")
    (output / "stage3c1_runtime_config.py").write_text("", encoding="utf-8")
    (output / "plot_curves_runtime.py").write_text("", encoding="utf-8")
    original = TrainingOptions(
        epochs=5,
        evaluate_every=5,
        save_every=5,
        name="debug",
        protocol="exploratory",
    )
    (output / "run_manifest.json").write_text(
        json.dumps({"options": {**original.__dict__, "resume": False}}),
        encoding="utf-8",
    )
    changed = TrainingOptions(
        epochs=10,
        evaluate_every=5,
        save_every=5,
        name="debug",
        protocol="exploratory",
        resume=True,
    )
    with pytest.raises(ValueError, match="settings differ"):
        model._prepare_output(changed, output)


def test_plot_scalar_unwraps_metric_writer_values():
    assert scalar([0.125, 99]) == pytest.approx(0.125)
    assert scalar(0.25) == pytest.approx(0.25)
