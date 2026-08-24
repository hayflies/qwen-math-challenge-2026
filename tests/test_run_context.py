import json
from pathlib import Path

import pytest

from qwen_math_challenge.config import load_config
from qwen_math_challenge.run_context import start_run


def _project_with_config(tmp_path: Path, output_root: str = "outputs") -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("rules\n", encoding="utf-8")
    config_path = tmp_path / "configs" / "test.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        f"""schema_version: 1
experiment:
  experiment_id: unit_test
  phase: 0
  seed: 7
runtime:
  output_root: {output_root}
  log_level: INFO
  deterministic: true
""",
        encoding="utf-8",
    )
    return config_path


def test_run_context_writes_and_finalizes_artifacts(tmp_path: Path) -> None:
    config_path = _project_with_config(tmp_path)
    config = load_config(config_path)

    with start_run(config, project_root=tmp_path) as run:
        run.write_json_artifact("result.json", {"ok": True})
        run.record_metrics({"accuracy": 1.0})
        run_dir = run.run_dir

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["experiment_id"] == "unit_test"
    assert manifest["git_commit"] is None
    assert manifest["metrics"]["accuracy"] == 1.0
    assert (run_dir / "config.snapshot.yaml").read_bytes() == config_path.read_bytes()
    assert (run_dir / "environment.json").is_file()
    assert (run_dir / "run.log").is_file()
    assert (run_dir / "result.json").is_file()


def test_run_context_records_failure(tmp_path: Path) -> None:
    config = load_config(_project_with_config(tmp_path))
    run_dir = None

    with pytest.raises(RuntimeError, match="expected failure"):
        with start_run(config, project_root=tmp_path) as run:
            run_dir = run.run_dir
            raise RuntimeError("expected failure")

    assert run_dir is not None
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failure"]["type"] == "RuntimeError"


def test_refuses_to_write_runs_under_raw_data(tmp_path: Path) -> None:
    config = load_config(_project_with_config(tmp_path, output_root="data/raw/runs"))

    with pytest.raises(ValueError, match="data/raw"):
        start_run(config, project_root=tmp_path)
