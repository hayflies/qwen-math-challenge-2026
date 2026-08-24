import json
from pathlib import Path

from qwen_math_challenge.environment import collect_environment, collect_git_info


def test_non_repository_git_state_is_explicit(tmp_path: Path) -> None:
    git = collect_git_info(tmp_path)

    assert git["available"] is False
    assert git["commit"] is None
    assert git["head_state"] == "not_a_repository"


def test_environment_uses_safe_allowlist(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_TOKEN", "never-record-this-secret")
    environment = collect_environment(tmp_path)
    serialized = json.dumps(environment)

    assert environment["python"]["version"]
    assert "transformers" in environment["packages"]
    assert "torch_runtime" in environment
    assert "never-record-this-secret" not in serialized
