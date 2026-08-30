"""Create a deterministic E001 artifact archive without base-model weights."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import tarfile
from pathlib import Path


class PackagingError(ValueError):
    """Raised when an E001 training/evaluation run is incomplete or incompatible."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", required=True, type=Path)
    parser.add_argument("--evaluation-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--include-latest-checkpoint",
        action="store_true",
        help="Include the latest resumable checkpoint in addition to the final adapter.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(run: Path, role: str) -> dict:
    path = run / "run_manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackagingError(f"{role} run manifest is missing or invalid: {path}") from exc
    if value.get("status") != "completed" or value.get("experiment_id") != "E001":
        raise PackagingError(f"{role} run is not a completed E001 run.")
    return value


def _required_files(run: Path, names: tuple[str, ...], role: str) -> list[Path]:
    paths = [run / name for name in names]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise PackagingError(f"{role} run is missing required files: {missing}")
    return paths


def _latest_checkpoint(run: Path) -> Path:
    candidates = []
    for path in (run / "checkpoints").glob("checkpoint-*"):
        if path.is_dir():
            try:
                step = int(path.name.removeprefix("checkpoint-"))
            except ValueError:
                continue
            candidates.append((step, path))
    if not candidates:
        raise PackagingError("No resumable checkpoint directory was found.")
    checkpoint = max(candidates)[1]
    for name in ("checkpoint_metadata.json", "trainer_state.json", "optimizer.pt", "scheduler.pt"):
        if not (checkpoint / name).is_file():
            raise PackagingError(f"Latest checkpoint is missing {name}.")
    return checkpoint


def _collect_files(
    training: Path, evaluation: Path, *, include_checkpoint: bool
) -> list[tuple[Path, str]]:
    training_files = _required_files(
        training,
        (
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
        ),
        "training",
    )
    evaluation_files = _required_files(
        evaluation,
        (
            "config.snapshot.yaml",
            "environment.json",
            "run_manifest.json",
            "run.log",
            "predictions.csv",
            "failures.csv",
            "metrics.json",
            "resume_identity.json",
            "comparison_e000.json",
        ),
        "evaluation",
    )
    adapter = training / "adapter"
    if not adapter.is_dir():
        raise PackagingError("Training run does not contain the final adapter directory.")
    forbidden_base_weights = {
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    }
    adapter_files = [path for path in sorted(adapter.rglob("*")) if path.is_file()]
    unexpected_base_weights = [
        path.name
        for path in adapter_files
        if path.name in forbidden_base_weights
        or (path.name.startswith("model-") and path.suffix == ".safetensors")
    ]
    if unexpected_base_weights:
        raise PackagingError(
            f"Adapter directory unexpectedly contains base-model weights: {unexpected_base_weights}"
        )
    files = [(path, f"training/{path.name}") for path in training_files]
    files.extend((path, f"evaluation/{path.name}") for path in evaluation_files)
    files.extend(
        (path, f"training/adapter/{path.relative_to(adapter).as_posix()}") for path in adapter_files
    )
    if include_checkpoint:
        checkpoint = _latest_checkpoint(training)
        files.extend(
            (
                path,
                f"training/checkpoints/{checkpoint.name}/{path.relative_to(checkpoint).as_posix()}",
            )
            for path in sorted(checkpoint.rglob("*"))
            if path.is_file()
        )
    members = [member for _, member in files]
    if len(members) != len(set(members)):
        raise PackagingError("Archive member paths are not unique.")
    return sorted(files, key=lambda item: item[1])


def _write_archive(output: Path, files: list[tuple[Path, str]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for source, member in files:
                    info = archive.gettarinfo(str(source), arcname=member)
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    with source.open("rb") as handle:
                        archive.addfile(info, handle)
    temporary.replace(output)


def main() -> int:
    args = parse_args()
    try:
        training = args.training_run.expanduser().resolve()
        evaluation = args.evaluation_run.expanduser().resolve()
        training_manifest = _load_manifest(training, "training")
        evaluation_manifest = _load_manifest(evaluation, "evaluation")
        if not training_manifest.get("canonical_full_run"):
            raise PackagingError("Training run is a smoke/non-canonical run.")
        if evaluation_manifest.get("limit") is not None:
            raise PackagingError("Evaluation run is a limited/non-canonical run.")
        if not evaluation_manifest.get("adapter_loaded"):
            raise PackagingError("Evaluation manifest does not prove adapter loading.")
        if training_manifest.get("adapter_sha256") != evaluation_manifest.get("adapter_sha256"):
            raise PackagingError("Training/evaluation adapter SHA-256 values differ.")
        output = args.output.expanduser().resolve()
        if output.suffixes[-2:] != [".tar", ".gz"]:
            raise PackagingError("Output must end in .tar.gz.")
        files = _collect_files(
            training,
            evaluation,
            include_checkpoint=args.include_latest_checkpoint,
        )
        _write_archive(output, files)
        manifest = {
            "schema_version": 1,
            "experiment_id": "E001",
            "archive": output.name,
            "archive_sha256": _sha256(output),
            "include_latest_checkpoint": args.include_latest_checkpoint,
            "members": {
                member: {"sha256": _sha256(source), "size_bytes": source.stat().st_size}
                for source, member in files
            },
            "base_model_weights_included": False,
            "training_run_id": training_manifest["run_id"],
            "evaluation_run_id": evaluation_manifest["run_id"],
            "adapter_sha256": training_manifest["adapter_sha256"],
        }
        manifest_path = output.with_suffix("").with_suffix(".manifest.json")
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (PackagingError, OSError, KeyError) as exc:
        print(f"E001 packaging failed: {exc}", file=sys.stderr)
        return 2
    print(output)
    print(manifest["archive_sha256"])
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
