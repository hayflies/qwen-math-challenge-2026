"""Random seed and deterministic runtime helpers."""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any

import numpy as np


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("seed must be an integer in the inclusive range [0, 2**32 - 1].")


def seed_everything(seed: int, *, deterministic: bool = True) -> dict[str, Any]:
    """Seed Python, NumPy, and PyTorch when available.

    Setting PYTHONHASHSEED here affects child processes, not hash randomization that already
    happened while starting the current interpreter. Entrypoints record this limitation instead
    of claiming stronger determinism than Python can provide.
    """

    _validate_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    report: dict[str, Any] = {
        "seed": seed,
        "deterministic_requested": deterministic,
        "python_random_seeded": True,
        "numpy_seeded": True,
        "python_hash_seed_for_child_processes": str(seed),
        "python_hash_seed_applies_to_current_process": False,
        "torch_seeded": False,
        "torch_deterministic_algorithms": None,
    }

    try:
        import torch
    except ImportError:
        report["torch_status"] = "not_installed"
        return report
    except Exception as exc:  # pragma: no cover - depends on an optional binary runtime
        report["torch_status"] = "import_failed"
        report["torch_import_error_type"] = type(exc).__name__
        return report

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        if deterministic:
            torch.backends.cudnn.benchmark = False

    report["torch_status"] = "available"
    report["torch_seeded"] = True
    report["torch_deterministic_algorithms"] = bool(torch.are_deterministic_algorithms_enabled())
    return report


def deterministic_probe() -> dict[str, Any]:
    """Return a small cross-run fingerprint from the seeded Python and NumPy RNGs."""

    values = {
        "python_uint32": [random.getrandbits(32) for _ in range(8)],
        "numpy_int31": np.random.randint(0, 2**31, size=8, dtype=np.int64).tolist(),
    }
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"values": values, "sha256": hashlib.sha256(canonical).hexdigest()}
