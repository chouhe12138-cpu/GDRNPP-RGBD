from research.managed_runtime.run import checkpoint_for_mode, detect_failure_kind


def test_checkpoint_mapping_is_fixed_to_epoch_40_for_formal(tmp_path):
    path, checkpoint_id, epoch, kind = checkpoint_for_mode(tmp_path, "formal")
    assert path == tmp_path / "checkpoints/model_epoch_040.pth"
    assert checkpoint_id == "epoch_040"
    assert epoch == 40
    assert kind == "fixed_final"


def test_failure_classification_distinguishes_cuda_oom(tmp_path):
    log = tmp_path / "console.log"
    log.write_text("RuntimeError: CUDA out of memory\n", encoding="utf-8")
    assert detect_failure_kind(log) == "CUDA_OOM"
    log.write_text("RuntimeError: data loader failed\n", encoding="utf-8")
    assert detect_failure_kind(log) == "RUNTIME_ERROR"
