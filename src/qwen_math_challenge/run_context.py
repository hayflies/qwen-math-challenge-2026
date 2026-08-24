"""Versioned run directories, logging, config snapshots, and manifests."""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qwen_math_challenge.config import LoadedConfig, find_project_root
from qwen_math_challenge.environment import collect_environment

_COMMON_EXPERIMENT_FIELDS: dict[str, Any] = {
    "experiment_id": None,
    "phase": None,
    "git_commit": None,
    "model": None,
    "checkpoint": None,
    "dataset_version": None,
    "train_samples": None,
    "validation_split": None,
    "external_datasets": [],
    "seed": None,
    "learning_rate": None,
    "epochs": None,
    "batch_size": None,
    "gradient_accumulation": None,
    "max_length": None,
    "lora_config": None,
    "prompt_template": None,
    "inference_config": None,
    "parser_version": None,
    "local_validation_score": None,
    "per_category_scores": {},
    "leaderboard_score": None,
    "notes": None,
}


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: object) -> None:
    serialized = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    _atomic_write_bytes(path, serialized + b"\n")


def _configure_logger(name: str, log_path: Path, level: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def _close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _create_unique_run_directory(output_root: Path, run_stem: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    for suffix in range(1000):
        name = run_stem if suffix == 0 else f"{run_stem}_{suffix:03d}"
        candidate = output_root / name
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"Could not allocate a unique run directory below {output_root}")


@dataclass
class RunContext:
    """Context manager that finalizes a run manifest on success or failure."""

    config: LoadedConfig
    project_root: Path
    run_id: str
    run_dir: Path
    logger: logging.Logger
    manifest: dict[str, Any]
    _closed: bool = field(default=False, init=False)

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "run_manifest.json"

    def write_json_artifact(self, filename: str, payload: object) -> Path:
        path = Path(filename)
        if path.name != filename or path.suffix.lower() != ".json":
            raise ValueError("JSON artifact filename must be a plain basename ending in '.json'.")
        artifact_path = self.run_dir / filename
        _atomic_write_json(artifact_path, payload)
        self.manifest.setdefault("artifacts", {})[filename] = filename
        _atomic_write_json(self.manifest_path, self.manifest)
        self.logger.info("Wrote artifact %s", filename)
        return artifact_path

    def register_artifact(self, filename: str) -> Path:
        """Register an existing plain-basename artifact inside this run directory."""

        path = Path(filename)
        if path.name != filename:
            raise ValueError("Artifact filename must be a plain basename.")
        artifact_path = self.run_dir / filename
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Artifact does not exist: {artifact_path}")
        self.manifest.setdefault("artifacts", {})[filename] = filename
        _atomic_write_json(self.manifest_path, self.manifest)
        self.logger.info("Registered artifact %s", filename)
        return artifact_path

    def record_metrics(self, metrics: Mapping[str, Any]) -> None:
        normalized = dict(metrics)
        json.dumps(normalized, allow_nan=False)
        self.manifest.setdefault("metrics", {}).update(normalized)
        _atomic_write_json(self.manifest_path, self.manifest)

    def __enter__(self) -> RunContext:
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, traceback: object) -> bool:
        if self._closed:
            return False
        completed_at = _utc_now().isoformat()
        self.manifest["finished_at_utc"] = completed_at
        self.manifest["completed_at"] = completed_at
        if exc is None:
            self.manifest["status"] = "completed"
            self.logger.info("Run completed successfully")
        else:
            self.manifest["status"] = "failed"
            self.manifest["failure"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            self.logger.exception("Run failed", exc_info=(exc_type, exc, traceback))
        _atomic_write_json(self.manifest_path, self.manifest)
        _close_logger(self.logger)
        self._closed = True
        return False


def start_run(
    config: LoadedConfig,
    *,
    project_root: str | Path | None = None,
    now: datetime | None = None,
) -> RunContext:
    """Create a non-overwriting run directory and its initial reproducibility artifacts."""

    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else find_project_root(config.source_path)
    )
    output_root = config.resolve_output_root(root)
    raw_root = (root / "data" / "raw").resolve()
    if _is_within(output_root, raw_root):
        raise ValueError("runtime.output_root may not be data/raw or any of its descendants.")

    timestamp = now or _utc_now()
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    timestamp = timestamp.astimezone(UTC)
    run_stem = f"{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}_{config.source_sha256[:8]}"
    experiment_root = output_root / config.experiment_id
    run_dir = _create_unique_run_directory(experiment_root, run_stem)
    run_id = run_dir.name

    snapshot_path = run_dir / "config.snapshot.yaml"
    _atomic_write_bytes(snapshot_path, config.source_path.read_bytes())

    environment = collect_environment(root)
    environment_path = run_dir / "environment.json"
    _atomic_write_json(environment_path, environment)

    log_path = run_dir / "run.log"
    logger = _configure_logger(
        f"qwen_math_challenge.{config.experiment_id}.{run_id}",
        log_path,
        config.log_level,
    )

    manifest = copy.deepcopy(_COMMON_EXPERIMENT_FIELDS)
    manifest.update(copy.deepcopy(config.experiment))
    manifest.update(
        {
            "schema_version": config.raw["schema_version"],
            "run_id": run_id,
            "status": "started",
            "started_at_utc": timestamp.isoformat(),
            "started_at": timestamp.isoformat(),
            "finished_at_utc": None,
            "completed_at": None,
            "config_sha256": config.source_sha256,
            "git_commit": environment["git"]["commit"],
            "git_branch": environment["git"]["branch"],
            "git_head_state": environment["git"]["head_state"],
            "git_dirty": environment["git"]["dirty"],
            "artifacts": {
                "config_snapshot": snapshot_path.name,
                "environment": environment_path.name,
                "log": log_path.name,
            },
            "metrics": {},
        }
    )
    manifest_path = run_dir / "run_manifest.json"
    _atomic_write_json(manifest_path, manifest)
    logger.info("Started run %s", run_id)
    logger.info("Config SHA-256: %s", config.source_sha256)

    return RunContext(
        config=config,
        project_root=root,
        run_id=run_id,
        run_dir=run_dir,
        logger=logger,
        manifest=manifest,
    )
