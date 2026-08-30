import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "package_e001_artifacts",
    Path(__file__).parents[1] / "scripts" / "package_e001_artifacts.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
PackagingError = _MODULE.PackagingError
_collect_files = _MODULE._collect_files
_sha256 = _MODULE._sha256
_write_archive = _MODULE._write_archive


TRAINING_FILES = (
    "config.snapshot.yaml",
    "environment.json",
    "run_manifest.json",
    "run.log",
    "preflight.json",
    "token_length_audit.json",
    "seed_report.json",
    "training_metrics.json",
    "training_log.jsonl",
    "training_identity.json",
)
EVALUATION_FILES = (
    "config.snapshot.yaml",
    "environment.json",
    "run_manifest.json",
    "run.log",
    "predictions.csv",
    "failures.csv",
    "metrics.json",
    "resume_identity.json",
    "comparison_e000.json",
)


def _fake_runs(tmp_path: Path) -> tuple[Path, Path]:
    training = tmp_path / "training"
    evaluation = tmp_path / "evaluation"
    training.mkdir()
    evaluation.mkdir()
    for filename in TRAINING_FILES:
        (training / filename).write_text(filename, encoding="utf-8")
    for filename in EVALUATION_FILES:
        (evaluation / filename).write_text(filename, encoding="utf-8")
    adapter = training / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    return training, evaluation


def test_e001_archive_is_deterministic(tmp_path: Path) -> None:
    training, evaluation = _fake_runs(tmp_path)
    files = _collect_files(training, evaluation, include_checkpoint=False)
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"

    _write_archive(first, files)
    _write_archive(second, files)

    assert _sha256(first) == _sha256(second)
    assert all("checkpoints/" not in member for _, member in files)


def test_packaging_rejects_base_model_weights_in_adapter(tmp_path: Path) -> None:
    training, evaluation = _fake_runs(tmp_path)
    (training / "adapter" / "model-00001-of-00002.safetensors").write_bytes(b"base")

    with pytest.raises(PackagingError, match="base-model weights"):
        _collect_files(training, evaluation, include_checkpoint=False)
