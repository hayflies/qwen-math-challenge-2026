"""Minimal, safe YAML configuration loading for experiment entrypoints."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CONFIG_SCHEMA_VERSION = 1
_EXPERIMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
_SENSITIVE_KEYS = {
    "access_token",
    "api_key",
    "api_token",
    "authorization",
    "hf_token",
    "huggingface_token",
    "password",
    "secret",
    "token",
}


class ConfigError(ValueError):
    """Raised when a project config is missing, unsafe, or malformed."""


@dataclass(frozen=True)
class LoadedConfig:
    """Validated config plus source metadata required for reproducibility."""

    source_path: Path
    source_sha256: str
    raw: dict[str, Any]
    experiment: dict[str, Any]
    output_root: Path
    log_level: str
    deterministic: bool

    @property
    def experiment_id(self) -> str:
        return str(self.experiment["experiment_id"])

    @property
    def phase(self) -> int | str:
        return self.experiment["phase"]

    @property
    def seed(self) -> int:
        return int(self.experiment["seed"])

    def resolve_output_root(self, project_root: Path) -> Path:
        if self.output_root.is_absolute():
            return self.output_root.resolve()
        return (project_root / self.output_root).resolve()


def _normalized_key(key: object) -> str:
    return str(key).strip().lower().replace("-", "_")


def _reject_sensitive_keys(value: object, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = _normalized_key(key)
            current_path = (*path, str(key))
            if normalized in _SENSITIVE_KEYS:
                dotted = ".".join(current_path)
                raise ConfigError(
                    f"Secret-like config key '{dotted}' is not allowed; "
                    "use an external secret store."
                )
            _reject_sensitive_keys(child, current_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_sensitive_keys(child, (*path, str(index)))


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"'{label}' must be a YAML mapping.")
    return dict(value)


def load_config(path: str | Path) -> LoadedConfig:
    """Load and validate a UTF-8 YAML config without executing custom YAML tags."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise ConfigError(f"Config file does not exist: {source_path}")

    source_bytes = source_path.read_bytes()
    try:
        parsed = yaml.safe_load(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not parse UTF-8 YAML config: {source_path}") from exc

    root = _require_mapping(parsed, "config root")
    _reject_sensitive_keys(root)
    try:
        json.dumps(root, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ConfigError("Config values must be JSON-serializable and finite.") from exc

    schema_version = root.get("schema_version")
    if schema_version != CONFIG_SCHEMA_VERSION:
        raise ConfigError(
            f"Unsupported schema_version {schema_version!r}; expected {CONFIG_SCHEMA_VERSION}."
        )

    experiment = _require_mapping(root.get("experiment"), "experiment")
    missing = [key for key in ("experiment_id", "phase", "seed") if key not in experiment]
    if missing:
        raise ConfigError(f"Missing required experiment fields: {', '.join(missing)}")

    experiment_id = experiment["experiment_id"]
    if not isinstance(experiment_id, str) or not _EXPERIMENT_ID_PATTERN.fullmatch(experiment_id):
        raise ConfigError(
            "experiment_id must start with an alphanumeric character and contain only "
            "letters, digits, '.', '_', or '-'."
        )
    if experiment_id in {".", ".."}:
        raise ConfigError("experiment_id may not be '.' or '..'.")

    phase = experiment["phase"]
    valid_phase = (isinstance(phase, int) and not isinstance(phase, bool)) or (
        isinstance(phase, str) and bool(phase.strip())
    )
    if not valid_phase:
        raise ConfigError("phase must be a non-empty string or an integer.")

    seed = experiment["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
        raise ConfigError("seed must be an integer in the inclusive range [0, 2**32 - 1].")

    runtime = _require_mapping(root.get("runtime"), "runtime")
    output_root_value = runtime.get("output_root")
    if not isinstance(output_root_value, str) or not output_root_value.strip():
        raise ConfigError("runtime.output_root must be a non-empty path string.")

    log_level = runtime.get("log_level", "INFO")
    if not isinstance(log_level, str) or log_level.upper() not in _LOG_LEVELS:
        raise ConfigError(f"runtime.log_level must be one of {sorted(_LOG_LEVELS)}.")

    deterministic = runtime.get("deterministic", True)
    if not isinstance(deterministic, bool):
        raise ConfigError("runtime.deterministic must be a boolean.")

    return LoadedConfig(
        source_path=source_path,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        raw=copy.deepcopy(root),
        experiment=copy.deepcopy(experiment),
        output_root=Path(output_root_value),
        log_level=log_level.upper(),
        deterministic=deterministic,
    )


def find_project_root(start: str | Path) -> Path:
    """Find the nearest ancestor containing both pyproject.toml and AGENTS.md."""

    candidate = Path(start).expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if (directory / "pyproject.toml").is_file() and (directory / "AGENTS.md").is_file():
            return directory
    raise ConfigError(f"Could not find project root from: {candidate}")
