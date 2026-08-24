"""Safe execution-environment and Git metadata collection."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_PACKAGE_DISTRIBUTIONS = {
    "numpy": "numpy",
    "pyyaml": "PyYAML",
    "torch": "torch",
    "transformers": "transformers",
    "datasets": "datasets",
    "accelerate": "accelerate",
    "peft": "peft",
    "trl": "trl",
    "bitsandbytes": "bitsandbytes",
    "tokenizers": "tokenizers",
    "safetensors": "safetensors",
}


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for label, distribution in _PACKAGE_DISTRIBUTIONS.items():
        try:
            versions[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[label] = None
    return versions


def _command_version(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else None


def collect_git_info(project_root: str | Path) -> dict[str, Any]:
    """Collect Git state without requiring an existing commit."""

    root = Path(project_root).resolve()
    if shutil.which("git") is None:
        return {
            "available": False,
            "branch": None,
            "commit": None,
            "head_state": "git_unavailable",
            "dirty": None,
        }

    def run_git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )

    try:
        inside = run_git("rev-parse", "--is-inside-work-tree")
    except (OSError, subprocess.TimeoutExpired):
        inside = None
    if inside is None or inside.returncode != 0 or inside.stdout.strip() != "true":
        return {
            "available": False,
            "branch": None,
            "commit": None,
            "head_state": "not_a_repository",
            "dirty": None,
        }

    branch_result = run_git("symbolic-ref", "--quiet", "--short", "HEAD")
    commit_result = run_git("rev-parse", "--verify", "HEAD")
    status_result = run_git("status", "--porcelain=v1", "--untracked-files=normal")
    commit = commit_result.stdout.strip() if commit_result.returncode == 0 else None

    return {
        "available": True,
        "branch": branch_result.stdout.strip() if branch_result.returncode == 0 else None,
        "commit": commit,
        "head_state": "attached" if commit else "unborn",
        "dirty": bool(status_result.stdout.strip()) if status_result.returncode == 0 else None,
    }


def _collect_torch_runtime() -> dict[str, Any]:
    package_version = _package_versions()["torch"]
    if package_version is None:
        return {
            "installed": False,
            "importable": False,
            "version": None,
            "cuda_available": False,
            "mps_available": False,
        }

    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on an optional binary runtime
        return {
            "installed": True,
            "importable": False,
            "version": package_version,
            "import_error_type": type(exc).__name__,
            "cuda_available": False,
            "mps_available": False,
        }

    cuda_available = bool(torch.cuda.is_available())
    cuda_devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                }
            )

    mps_backend = getattr(torch.backends, "mps", None)
    mps_built = bool(mps_backend and mps_backend.is_built())
    mps_available = bool(mps_backend and mps_backend.is_available())

    return {
        "installed": True,
        "importable": True,
        "version": str(torch.__version__),
        "cuda_compiled_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": len(cuda_devices),
        "cuda_devices": cuda_devices,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        "mps_built": mps_built,
        "mps_available": mps_available,
    }


def _collect_nvidia_smi() -> dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {"available": False, "gpus": []}
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"available": False, "gpus": []}
    if completed.returncode != 0:
        return {"available": False, "gpus": []}

    gpus = []
    for index, line in enumerate(completed.stdout.splitlines()):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            gpus.append(
                {
                    "index": index,
                    "name": parts[0],
                    "memory_total_mib": parts[1],
                    "driver_version": parts[2],
                }
            )
    return {"available": True, "gpus": gpus}


def collect_environment(project_root: str | Path) -> dict[str, Any]:
    """Collect a safe allowlist of reproducibility metadata, never the full environment."""

    root = Path(project_root).resolve()
    return {
        "collected_at_utc": datetime.now(UTC).isoformat(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "byteorder": sys.byteorder,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "cpu_count": os.cpu_count(),
        },
        "packages": _package_versions(),
        "torch_runtime": _collect_torch_runtime(),
        "nvidia_smi": _collect_nvidia_smi(),
        "tools": {
            "git": _command_version(["git", "--version"]),
            "uv": _command_version(["uv", "--version"]),
        },
        "git": collect_git_info(root),
    }
