"""Exact, CPU-only reconstruction of the deadline final-submission artifact."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qwen_math_challenge.config import LoadedConfig
from qwen_math_challenge.data.official import sha256_file
from qwen_math_challenge.evaluation.zero_shot import parse_integer_answer_v2
from qwen_math_challenge.inference.shards import (
    SHARD_PIPELINE_VERSION,
    FinalShardPlan,
    plan_final_shards,
)
from qwen_math_challenge.inference.submission import (
    FINAL_EXPERIMENT_ID,
    FINAL_SUBMISSION_PIPELINE_VERSION,
    OFFICIAL_TEST_COLUMNS,
    FinalInputBundle,
    FinalPrediction,
    FinalSubmissionError,
    FinalSubmissionSettings,
    load_final_input,
    load_final_predictions,
    validate_submission_file,
)

FINAL_EMERGENCY_CODE_COMMIT = "d7adc8961a65e0354e4089ef525bf63646f8c665"
FINAL_SUBMISSION_SHA256 = "1ff78693423e011464f3bcb6f334eb8e181b192be92919a718831e9fc292d538"
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class FrozenPredictionSource:
    """One immutable prediction prefix used to build the submitted file."""

    role: str
    relative_directory: str
    expected_rows: int
    shard_index: int | None


@dataclass(frozen=True)
class ExactReproductionExpectations:
    """Frozen facts needed to distinguish exact reconstruction from inference."""

    total_rows: int
    flag_rows: int
    real_prediction_rows: int
    fallback_rows: int
    num_shards: int
    expected_sha256: str
    emergency_code_commit: str
    sources: tuple[FrozenPredictionSource, ...]


DEFAULT_EXACT_EXPECTATIONS = ExactReproductionExpectations(
    total_rows=2000,
    flag_rows=120,
    real_prediction_rows=1703,
    fallback_rows=297,
    num_shards=2,
    expected_sha256=FINAL_SUBMISSION_SHA256,
    emergency_code_commit=FINAL_EMERGENCY_CODE_COMMIT,
    sources=(
        FrozenPredictionSource("partial_989", "partial_989", 989, None),
        FrozenPredictionSource("shard_0", "shard_0", 382, 0),
        FrozenPredictionSource("shard_1", "shard_1", 332, 1),
    ),
)


@dataclass(frozen=True)
class ExactReproductionResult:
    submission_path: Path
    checksum_path: Path
    manifest_path: Path
    sha256: str
    real_prediction_rows: int
    fallback_rows: int
    fallback_value: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "PASS",
            "submission": str(self.submission_path),
            "sha256": self.sha256,
            "real_prediction_rows": self.real_prediction_rows,
            "fallback_rows": self.fallback_rows,
            "fallback_value": self.fallback_value,
            "checksum": str(self.checksum_path),
            "manifest": str(self.manifest_path),
        }


def _read_json(path: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise FinalSubmissionError(
            f"Missing frozen {role}: {path}. Copy the exact Kaggle artifact before reproducing."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalSubmissionError(f"Frozen {role} must be valid UTF-8 JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise FinalSubmissionError(f"Frozen {role} must contain a JSON object: {path}")
    return payload


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _canonical_identity_sha256(identity: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(identity), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ordered_identity(values: Sequence[str]) -> str:
    payload = json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_common_identity(
    payload: Mapping[str, Any],
    *,
    config: LoadedConfig,
    settings: FinalSubmissionSettings,
    bundle: FinalInputBundle,
    role: str,
) -> dict[str, Any]:
    identity = payload.get("identity")
    if not isinstance(identity, dict):
        raise FinalSubmissionError(f"{role} resume identity is malformed.")
    if payload.get("identity_sha256") != _canonical_identity_sha256(identity):
        raise FinalSubmissionError(f"{role} resume identity SHA-256 is invalid.")
    expected = {
        "config_sha256": config.source_sha256,
        "input_identity": bundle.identity,
        "model_name": settings.model.name_or_path,
        "model_revision": settings.model.revision,
        "model_commit": settings.model.revision,
        "tokenizer_revision": settings.model.tokenizer_revision,
        "tokenizer_commit": settings.model.tokenizer_revision,
        "prompt": asdict(settings.prompt),
        "generation": asdict(settings.generation),
        "parser_version": settings.parser_version,
        "canonical_e000_config_sha256": settings.canonical_e000_config_sha256,
        "external_datasets": [],
        "tool_use": False,
        "adapter": None,
    }
    mismatches = sorted(key for key, value in expected.items() if identity.get(key) != value)
    if mismatches:
        raise FinalSubmissionError(f"{role} canonical identity mismatch: {mismatches!r}.")
    return identity


def _validate_run_manifest(
    manifest: Mapping[str, Any],
    *,
    config: LoadedConfig,
    settings: FinalSubmissionSettings,
    role: str,
    require_emergency_commit: bool,
    emergency_commit: str,
) -> str:
    expected = {
        "experiment_id": FINAL_EXPERIMENT_ID,
        "config_sha256": config.source_sha256,
        "checkpoint": (f"{settings.model.name_or_path}@{settings.model.revision}"),
        "parser_version": settings.parser_version,
        "lora_config": None,
    }
    mismatches = sorted(key for key, value in expected.items() if manifest.get(key) != value)
    if mismatches:
        raise FinalSubmissionError(f"{role} run manifest mismatch: {mismatches!r}.")
    commit = manifest.get("git_commit")
    if not isinstance(commit, str) or _GIT_COMMIT_PATTERN.fullmatch(commit) is None:
        raise FinalSubmissionError(f"{role} run manifest lacks a valid git commit.")
    if require_emergency_commit and commit != emergency_commit:
        raise FinalSubmissionError(
            f"{role} was not produced by frozen emergency commit {emergency_commit}."
        )
    return commit


def _validate_prediction(
    record: FinalPrediction,
    bundle: FinalInputBundle,
    *,
    role: str,
) -> None:
    if not 0 <= record.source_index < len(bundle.rows):
        raise FinalSubmissionError(f"{role} prediction has invalid source index.")
    source = bundle.rows[record.source_index]
    reparsed = parse_integer_answer_v2(record.raw_output)
    if (
        record.sample_id != source.sample_id
        or record.question != source.question
        or record.question_sha256 != source.question_sha256
        or record.is_flagged != source.is_flagged
        or record.parsed_answer != reparsed.value
        or record.parse_status != reparsed.status
    ):
        raise FinalSubmissionError(
            f"{role} prediction for {record.sample_id!r} failed source/parser integrity."
        )
    if record.parsed_answer is None:
        raise FinalSubmissionError(
            f"{role} prediction for {record.sample_id!r} has no parsed integer answer."
        )


def _validate_shard_identity(
    identity: Mapping[str, Any],
    *,
    plan: FinalShardPlan,
    source: FrozenPredictionSource,
) -> None:
    if source.shard_index is None:
        raise AssertionError("A shard source must have a shard index.")
    selected = plan.shards[source.shard_index]
    expected = {
        "index": source.shard_index,
        "num_shards": len(plan.shards),
        "selected_count": len(selected),
        "selected_ordered_id_sha256": _ordered_identity([row.sample_id for row in selected]),
        "excluded_predictions_sha256": plan.existing_predictions_sha256,
        "excluded_completed_count": len(plan.existing_predictions),
    }
    if identity.get("pipeline_version") != SHARD_PIPELINE_VERSION:
        raise FinalSubmissionError(f"{source.role} pipeline identity mismatch.")
    if identity.get("shard") != expected:
        raise FinalSubmissionError(f"{source.role} selection/exclusion identity mismatch.")


def _submission_bytes(
    bundle: FinalInputBundle,
    answers_by_id: Mapping[str, str],
    fallback_value: str,
) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(OFFICIAL_TEST_COLUMNS)
    for source in bundle.rows:
        answer = answers_by_id.get(source.sample_id, fallback_value)
        try:
            canonical = str(int(answer))
        except ValueError as exc:
            raise FinalSubmissionError(
                f"Answer for {source.sample_id!r} is not an integer string."
            ) from exc
        if canonical != answer:
            raise FinalSubmissionError(
                f"Answer for {source.sample_id!r} is not a canonical integer string."
            )
        writer.writerow([source.sample_id, source.question, answer])
    return buffer.getvalue().encode("utf-8")


def _unique_mode(values: Sequence[str]) -> tuple[str, int]:
    counts = Counter(values)
    if not counts:
        raise FinalSubmissionError("Cannot derive fallback mode without model predictions.")
    frequency = max(counts.values())
    modes = [value for value, count in counts.items() if count == frequency]
    if len(modes) != 1:
        raise FinalSubmissionError(
            "Parsed prediction mode is tied; the frozen fallback value is not uniquely defined."
        )
    return modes[0], frequency


def reproduce_exact_final_submission(
    config: LoadedConfig,
    settings: FinalSubmissionSettings,
    *,
    artifact_directory: str | Path,
    output_directory: str | Path,
    expectations: ExactReproductionExpectations = DEFAULT_EXACT_EXPECTATIONS,
) -> ExactReproductionResult:
    """Rebuild and checksum the exact deadline submission without model inference."""

    bundle = load_final_input(settings.data)
    if len(bundle.rows) != expectations.total_rows:
        raise FinalSubmissionError(
            f"Exact reproduction requires {expectations.total_rows} test rows; "
            f"got {len(bundle.rows)}."
        )
    if bundle.identity["flags"]["rows"] != expectations.flag_rows:
        raise FinalSubmissionError(
            f"Exact reproduction requires {expectations.flag_rows} flag rows."
        )
    partial_sources = [source for source in expectations.sources if source.shard_index is None]
    shard_sources = [source for source in expectations.sources if source.shard_index is not None]
    if len(partial_sources) != 1 or len(shard_sources) != expectations.num_shards:
        raise FinalSubmissionError("Frozen source specification is inconsistent.")

    artifact_root = Path(artifact_directory).expanduser().resolve()
    for source in expectations.sources:
        source_dir = artifact_root / source.relative_directory
        for filename in (
            "predictions.csv",
            "input_identity.json",
            "resume_identity.json",
            "run_manifest.json",
        ):
            path = source_dir / filename
            if not path.is_file():
                raise FinalSubmissionError(
                    f"Missing frozen {source.role} artifact: {path}. "
                    "Copy the exact Kaggle artifact before reproducing."
                )
    partial_source = partial_sources[0]
    partial_dir = artifact_root / partial_source.relative_directory
    plan = plan_final_shards(
        config,
        settings,
        existing_run=partial_dir,
        num_shards=expectations.num_shards,
    )
    if len(plan.existing_predictions) != partial_source.expected_rows:
        raise FinalSubmissionError(
            f"{partial_source.role} must contain exactly {partial_source.expected_rows} rows; "
            f"got {len(plan.existing_predictions)}."
        )

    sources_manifest: dict[str, Any] = {}
    answers_by_id: dict[str, str] = {}

    def register_records(
        source: FrozenPredictionSource,
        source_dir: Path,
        records: Sequence[FinalPrediction],
    ) -> None:
        for record in records:
            _validate_prediction(record, bundle, role=source.role)
            if record.sample_id in answers_by_id:
                raise FinalSubmissionError(
                    f"Duplicate prediction id {record.sample_id!r} across frozen sources."
                )
            answers_by_id[record.sample_id] = str(record.parsed_answer)
        input_identity = _read_json(
            source_dir / "input_identity.json", f"{source.role} input identity"
        )
        if input_identity != bundle.identity:
            raise FinalSubmissionError(
                f"{source.role} input identity does not match official files."
            )
        resume_payload = _read_json(
            source_dir / "resume_identity.json", f"{source.role} resume identity"
        )
        identity = _validate_common_identity(
            resume_payload,
            config=config,
            settings=settings,
            bundle=bundle,
            role=source.role,
        )
        run_manifest = _read_json(source_dir / "run_manifest.json", f"{source.role} run manifest")
        commit = _validate_run_manifest(
            run_manifest,
            config=config,
            settings=settings,
            role=source.role,
            require_emergency_commit=source.shard_index is not None,
            emergency_commit=expectations.emergency_code_commit,
        )
        sources_manifest[source.role] = {
            "directory": source.relative_directory,
            "rows": len(records),
            "predictions_sha256": sha256_file(source_dir / "predictions.csv"),
            "input_identity_sha256": sha256_file(source_dir / "input_identity.json"),
            "resume_identity_sha256": sha256_file(source_dir / "resume_identity.json"),
            "run_manifest_sha256": sha256_file(source_dir / "run_manifest.json"),
            "git_commit": commit,
            "pipeline_version": identity.get("pipeline_version"),
        }

    partial_identity = _read_json(
        partial_dir / "resume_identity.json", f"{partial_source.role} resume identity"
    ).get("identity", {})
    if (
        not isinstance(partial_identity, dict)
        or partial_identity.get("pipeline_version") != FINAL_SUBMISSION_PIPELINE_VERSION
    ):
        raise FinalSubmissionError(f"{partial_source.role} pipeline identity mismatch.")
    if partial_identity.get("limit") is not None:
        raise FinalSubmissionError(f"{partial_source.role} must originate from an unlimited run.")
    register_records(partial_source, partial_dir, plan.existing_predictions)

    for source in sorted(shard_sources, key=lambda item: int(item.shard_index or 0)):
        if source.shard_index is None or not 0 <= source.shard_index < expectations.num_shards:
            raise FinalSubmissionError(f"Invalid frozen shard index for {source.role}.")
        source_dir = artifact_root / source.relative_directory
        records = load_final_predictions(source_dir / "predictions.csv")
        if len(records) != source.expected_rows:
            raise FinalSubmissionError(
                f"{source.role} must contain exactly {source.expected_rows} rows; "
                f"got {len(records)}."
            )
        identity_payload = _read_json(
            source_dir / "resume_identity.json", f"{source.role} resume identity"
        )
        identity = _validate_common_identity(
            identity_payload,
            config=config,
            settings=settings,
            bundle=bundle,
            role=source.role,
        )
        _validate_shard_identity(identity, plan=plan, source=source)
        register_records(source, source_dir, records)
        expected_prefix = plan.shards[source.shard_index][: len(records)]
        if [record.sample_id for record in records] != [row.sample_id for row in expected_prefix]:
            raise FinalSubmissionError(
                f"{source.role} predictions are not the deterministic completed shard prefix."
            )

    if len(answers_by_id) != expectations.real_prediction_rows:
        raise FinalSubmissionError(
            f"Expected {expectations.real_prediction_rows} real model predictions; "
            f"got {len(answers_by_id)}."
        )
    official_ids = {row.sample_id for row in bundle.rows}
    unknown = set(answers_by_id) - official_ids
    missing = official_ids - set(answers_by_id)
    if unknown:
        raise FinalSubmissionError(
            f"Frozen predictions contain unknown IDs: {sorted(unknown)[:10]!r}."
        )
    if len(missing) != expectations.fallback_rows:
        raise FinalSubmissionError(
            f"Expected exactly {expectations.fallback_rows} fallback IDs; got {len(missing)}."
        )

    fallback_value, mode_frequency = _unique_mode(list(answers_by_id.values()))
    payload = _submission_bytes(bundle, answers_by_id, fallback_value)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expectations.expected_sha256:
        raise FinalSubmissionError(
            "Exact submission SHA-256 mismatch: "
            f"expected {expectations.expected_sha256}, got {digest}. "
            "Refusing to write a non-canonical reconstruction."
        )

    output_root = Path(output_directory).expanduser().resolve()
    submission_path = output_root / "submission.csv"
    checksum_path = output_root / "submission.csv.sha256"
    manifest_path = output_root / "reproduction_manifest.json"
    existing_targets = [
        path for path in (submission_path, checksum_path, manifest_path) if path.exists()
    ]
    if existing_targets:
        raise FinalSubmissionError(
            f"Refusing to overwrite existing reproduction outputs: {existing_targets!r}."
        )
    _atomic_write(submission_path, payload)
    validate_submission_file(submission_path, bundle.rows)
    with submission_path.open(encoding="utf-8", newline="") as handle:
        output_ids = {row["id"] for row in csv.DictReader(handle)}
    flagged_ids = {row.sample_id for row in bundle.rows if row.is_flagged}
    if len(flagged_ids) != expectations.flag_rows or not flagged_ids <= output_ids:
        raise FinalSubmissionError("Not every official flagged ID is present in submission.csv.")
    _atomic_write(checksum_path, f"{digest}  submission.csv\n".encode("ascii"))
    reproduction_manifest = {
        "schema_version": 1,
        "mode": "exact_deadline_submission_reconstruction",
        "model_inference_performed": False,
        "emergency_code_commit": expectations.emergency_code_commit,
        "total_rows": len(bundle.rows),
        "flag_rows_preserved": len(flagged_ids),
        "real_prediction_rows": len(answers_by_id),
        "fallback_rows": len(missing),
        "fallback_rule": "unique_mode_of_parsed_integer_predictions",
        "fallback_value": fallback_value,
        "fallback_mode_frequency": mode_frequency,
        "fallback_ordered_id_sha256": _ordered_identity(
            [row.sample_id for row in bundle.rows if row.sample_id in missing]
        ),
        "input_identity": bundle.identity,
        "sources": sources_manifest,
        "submission_sha256": digest,
    }
    _atomic_write(
        manifest_path,
        json.dumps(
            reproduction_manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n",
    )
    return ExactReproductionResult(
        submission_path=submission_path,
        checksum_path=checksum_path,
        manifest_path=manifest_path,
        sha256=digest,
        real_prediction_rows=len(answers_by_id),
        fallback_rows=len(missing),
        fallback_value=fallback_value,
    )
