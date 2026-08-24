from pathlib import Path

import pytest

from qwen_math_challenge.config import ConfigError, find_project_root, load_config


def _write_config(path: Path, extra: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """schema_version: 1
experiment:
  experiment_id: test_run
  phase: 0
  seed: 123
runtime:
  output_root: outputs
  log_level: INFO
  deterministic: true
"""
        + extra,
        encoding="utf-8",
    )
    return path


def test_load_config_and_resolve_relative_output(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "configs" / "test.yaml")
    config = load_config(config_path)

    assert config.experiment_id == "test_run"
    assert config.seed == 123
    assert config.resolve_output_root(tmp_path) == (tmp_path / "outputs").resolve()
    assert len(config.source_sha256) == 64


@pytest.mark.parametrize("experiment_id", ["../escape", "bad/name", "", "."])
def test_rejects_unsafe_experiment_id(tmp_path: Path, experiment_id: str) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""schema_version: 1
experiment:
  experiment_id: {experiment_id!r}
  phase: 0
  seed: 1
runtime:
  output_root: outputs
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_config(config_path)


def test_rejects_secret_like_config_key(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yaml", "hf_token: should-not-be-here\n")

    with pytest.raises(ConfigError, match="Secret-like config key"):
        load_config(config_path)


def test_find_project_root(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("rules\n", encoding="utf-8")
    nested = tmp_path / "configs" / "phase0"
    nested.mkdir(parents=True)

    assert find_project_root(nested) == tmp_path.resolve()
