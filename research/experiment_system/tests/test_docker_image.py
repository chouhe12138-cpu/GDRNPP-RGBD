from pathlib import Path

from research.experiment_system.docker_image import inspect_docker_image


def test_image_inspection_reads_id_and_revision(monkeypatch):
    monkeypatch.setattr(
        "subprocess.check_output",
        lambda *args, **kwargs: "sha256:image\t" + "a" * 40 + "\n",
    )
    result = inspect_docker_image("gdrnpp:test", Path("/usr/bin/docker"))
    assert result["image_id"] == "sha256:image"
    assert result["revision"] == "a" * 40
