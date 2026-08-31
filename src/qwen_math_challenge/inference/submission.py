"""Crash-safe E000 inference for the official 2026 final-test submission."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qwen_math_challenge.config import LoadedConfig, load_config
from qwen_math_challenge.data.official import sha256_file
from qwen_math_challenge.evaluation import zero_shot as zero_shot_module
from qwen_math_challenge.evaluation.zero_shot import (
    ALLOWED_MODEL_ID,
    ANSWER_PARSER_V2_VERSION,
    ANSWER_PARSER_VERSION,
    BatchGenerator,
    GenerationSettings,
    ModelSettings,
    PromptSettings,
    RuntimeSettings,
    parse_integer_answer_v2,
)

FINAL_SUBMISSION_PIPELINE_VERSION = "phase13_final_e000_v1"
FINAL_EXPERIMENT_ID = "FINAL_E000"
OFFICIAL_TEST_COLUMNS = ("id", "question", "answer")
OFFICIAL_FLAG_COLUMNS = ("id", "question")
PREDICTION_COLUMNS = (
    "source_index",
    "id",
    "question",
    "question_sha256",
    "raw_output",
    "parsed_answer",
    "parse_status",
    "parser_version",
    "input_tokens",
    "output_tokens",
    "latency_sec",
    "finish_reason",
    "truncated",
    "is_flagged",
)
CANONICAL_INTEGER_PATTERN = re.compile(r"^-?[0-9]+$")


class FinalSubmissionError(ValueError):
    """Raised when final-test identity, resume state, or submission is invalid."""


@dataclass(frozen=True)
class FinalDataSettings:
    test_path: Path
    expected_rows: int
    columns: tuple[str, ...]
    flag_path: Path
    expected_flag_rows: int
    flag_columns: tuple[str, ...]


@dataclass(frozen=True)
class FinalOutputSettings:
    predictions_filename: str
    resume_identity_filename: str
    input_identity_filename: str
    metrics_filename: str
    submission_filename: str
    submission_sha256_filename: str


@dataclass(frozen=True)
class FinalSubmissionSettings:
    model: ModelSettings
    prompt: PromptSettings
    generation: GenerationSettings
    runtime: RuntimeSettings
    data: FinalDataSettings
    output: FinalOutputSettings
    parser_version: str
    seed: int
    canonical_e000_config_sha256: str


@dataclass(frozen=True)
class FinalTestRow:
    index: int
    sample_id: str
    question: str
    question_sha256: str
    is_flagged: bool


@dataclass(frozen=True)
class FinalInputBundle:
    rows: tuple[FinalTestRow, ...]
    identity: dict[str, Any]


@dataclass(frozen=True)
class FinalPrediction:
    source_index: int
    sample_id: str
    question: str
    question_sha256: str
    raw_output: str
    parsed_answer: int | None
    parse_status: str
    input_tokens: int
    output_tokens: int
    latency_sec: float
    finish_reason: str
    truncated: bool
    is_flagged: bool

    def as_csv_row(self) -> list[str]:
        return [
            str(self.source_index),
            self.sample_id,
            self.question,
            self.question_sha256,
            self.raw_output,
            "" if self.parsed_answer is None else str(self.parsed_answer),
            self.parse_status,
            ANSWER_PARSER_V2_VERSION,
            str(self.input_tokens),
            str(self.output_tokens),
            f"{self.latency_sec:.9f}",
            self.finish_reason,
            str(self.truncated).lower(),
            str(self.is_flagged).lower(),
        ]


@dataclass(frozen=True)
class FinalSubmissionResult:
    metrics: dict[str, Any]
    input_identity: dict[str, Any]
    resume_identity: dict[str, Any]
    predictions_path: Path
    input_identity_path: Path
    resume_identity_path: Path
    metrics_path: Path
    submission_path: Path | None
    submission_sha256_path: Path | None
    artifact_sha256: dict[str, str]


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FinalSubmissionError(f"'{label}' must be a mapping.")
    return dict(value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FinalSubmissionError(f"'{label}' must be a non-empty string.")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise FinalSubmissionError(f"'{label}' must be a boolean.")
    return value


def _integer(value: object, label: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FinalSubmissionError(f"'{label}' must be an integer >= {minimum}.")
    return value


def _columns(value: object, expected: tuple[str, ...], label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise FinalSubmissionError(f"'{label}' must be a list of column names.")
    columns = tuple(value)
    if columns != expected:
        raise FinalSubmissionError(f"'{label}' must be exactly {list(expected)!r}.")
    return columns


def _filename(value: object, label: str, suffix: str) -> str:
    filename = _string(value, label)
    path = Path(filename)
    if path.name != filename or not filename.endswith(suffix):
        raise FinalSubmissionError(f"'{label}' must be a plain basename ending in {suffix!r}.")
    return filename


def _resolve_path(root: Path, value: object, label: str) -> Path:
    path = Path(_string(value, label)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_final_submission_settings(
    config: LoadedConfig,
    project_root: str | Path,
    *,
    test_path: str | Path | None = None,
    flag_path: str | Path | None = None,
) -> FinalSubmissionSettings:
    """Load a final config and prove its model/prompt/generation match E000."""

    root = Path(project_root).resolve()
    if config.experiment_id != FINAL_EXPERIMENT_ID or config.phase != 13:
        raise FinalSubmissionError("Final config must set experiment_id FINAL_E000 and phase 13.")

    canonical = _mapping(config.raw.get("canonical_e000"), "canonical_e000")
    reference_path = _resolve_path(
        root, canonical.get("reference_config_path"), "canonical_e000.reference_config_path"
    )
    expected_reference_sha = _string(
        canonical.get("reference_config_sha256"),
        "canonical_e000.reference_config_sha256",
    )
    reference = load_config(reference_path)
    if reference.source_sha256 != expected_reference_sha:
        raise FinalSubmissionError("Canonical E000 reference config SHA-256 mismatch.")
    for section in ("model", "prompt", "generation"):
        if config.raw.get(section) != reference.raw.get(section):
            raise FinalSubmissionError(
                f"Final '{section}' must exactly match the canonical E000 config."
            )

    runtime = _mapping(config.raw.get("runtime"), "runtime")
    reference_runtime = _mapping(reference.raw.get("runtime"), "canonical runtime")
    for field in ("deterministic", "device", "dtype", "batch_size"):
        if runtime.get(field) != reference_runtime.get(field):
            raise FinalSubmissionError(f"Final runtime.{field} must match canonical E000 exactly.")
    if not _boolean(runtime.get("deterministic"), "runtime.deterministic"):
        raise FinalSubmissionError("Final inference must be deterministic.")
    runtime_settings = RuntimeSettings(
        device=_string(runtime.get("device"), "runtime.device"),
        dtype=_string(runtime.get("dtype"), "runtime.dtype"),
        batch_size=_integer(runtime.get("batch_size"), "runtime.batch_size"),
    )
    if runtime_settings.batch_size != 1:
        raise FinalSubmissionError("Canonical E000 final inference requires batch_size=1.")

    model = _mapping(config.raw.get("model"), "model")
    model_settings = ModelSettings(
        name_or_path=_string(model.get("name_or_path"), "model.name_or_path"),
        revision=_string(model.get("revision"), "model.revision"),
        tokenizer_revision=_string(model.get("tokenizer_revision"), "model.tokenizer_revision"),
        local_files_only=_boolean(model.get("local_files_only"), "model.local_files_only"),
        trust_remote_code=_boolean(model.get("trust_remote_code"), "model.trust_remote_code"),
    )
    if model_settings.name_or_path != ALLOWED_MODEL_ID:
        raise FinalSubmissionError(f"Final inference permits only {ALLOWED_MODEL_ID!r}.")

    prompt = _mapping(config.raw.get("prompt"), "prompt")
    prompt_settings = PromptSettings(
        version=_string(prompt.get("version"), "prompt.version"),
        system_text=_string(prompt.get("system_text"), "prompt.system_text"),
        user_template=_string(prompt.get("user_template"), "prompt.user_template"),
    )
    generation = _mapping(config.raw.get("generation"), "generation")
    generation_settings = GenerationSettings(
        version=_string(generation.get("version"), "generation.version"),
        do_sample=_boolean(generation.get("do_sample"), "generation.do_sample"),
        num_beams=_integer(generation.get("num_beams"), "generation.num_beams"),
        max_new_tokens=_integer(generation.get("max_new_tokens"), "generation.max_new_tokens"),
        use_cache=_boolean(generation.get("use_cache"), "generation.use_cache"),
    )

    parser = _mapping(config.raw.get("parser"), "parser")
    if parser != {
        "version": ANSWER_PARSER_V2_VERSION,
        "base_version": ANSWER_PARSER_VERSION,
        "conflict_policy": "last_explicit_or_boxed_by_position",
    }:
        raise FinalSubmissionError("Final parser mapping is not the frozen v2 policy.")

    data = _mapping(config.raw.get("data"), "data")
    resolved_test = (
        Path(test_path).expanduser().resolve()
        if test_path is not None
        else _resolve_path(root, data.get("test_path"), "data.test_path")
    )
    resolved_flag = (
        Path(flag_path).expanduser().resolve()
        if flag_path is not None
        else _resolve_path(root, data.get("flag_path"), "data.flag_path")
    )
    data_settings = FinalDataSettings(
        test_path=resolved_test,
        expected_rows=_integer(data.get("expected_rows"), "data.expected_rows"),
        columns=_columns(data.get("columns"), OFFICIAL_TEST_COLUMNS, "data.columns"),
        flag_path=resolved_flag,
        expected_flag_rows=_integer(data.get("expected_flag_rows"), "data.expected_flag_rows"),
        flag_columns=_columns(data.get("flag_columns"), OFFICIAL_FLAG_COLUMNS, "data.flag_columns"),
    )
    if data_settings.expected_rows != 2000 or data_settings.expected_flag_rows != 120:
        raise FinalSubmissionError("Canonical final input requires 2000 test and 120 flag rows.")

    output = _mapping(config.raw.get("output"), "output")
    output_settings = FinalOutputSettings(
        predictions_filename=_filename(
            output.get("predictions_filename"), "output.predictions_filename", ".csv"
        ),
        resume_identity_filename=_filename(
            output.get("resume_identity_filename"),
            "output.resume_identity_filename",
            ".json",
        ),
        input_identity_filename=_filename(
            output.get("input_identity_filename"),
            "output.input_identity_filename",
            ".json",
        ),
        metrics_filename=_filename(
            output.get("metrics_filename"), "output.metrics_filename", ".json"
        ),
        submission_filename=_filename(
            output.get("submission_filename"), "output.submission_filename", ".csv"
        ),
        submission_sha256_filename=_filename(
            output.get("submission_sha256_filename"),
            "output.submission_sha256_filename",
            ".sha256",
        ),
    )
    if output_settings.submission_filename != "submission.csv":
        raise FinalSubmissionError("Official final output filename must be submission.csv.")

    return FinalSubmissionSettings(
        model=model_settings,
        prompt=prompt_settings,
        generation=generation_settings,
        runtime=runtime_settings,
        data=data_settings,
        output=output_settings,
        parser_version=ANSWER_PARSER_V2_VERSION,
        seed=config.seed,
        canonical_e000_config_sha256=reference.source_sha256,
    )


def _read_exact_csv(path: Path, columns: tuple[str, ...], role: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise FinalSubmissionError(f"{role} does not exist: {path}")
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FinalSubmissionError(f"{role} must be UTF-8 CSV.") from exc
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = tuple(next(reader))
    except StopIteration as exc:
        raise FinalSubmissionError(f"{role} is empty.") from exc
    if header != columns:
        raise FinalSubmissionError(
            f"{role} columns must be exactly {list(columns)!r}; got {list(header)!r}."
        )
    rows: list[dict[str, str]] = []
    for line_number, values in enumerate(reader, start=2):
        if len(values) != len(columns):
            raise FinalSubmissionError(
                f"{role} row {line_number} has {len(values)} fields; expected {len(columns)}."
            )
        rows.append(dict(zip(columns, values, strict=True)))
    return rows


def _ordered_identity(values: Sequence[str]) -> str:
    canonical = json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_final_input(settings: FinalDataSettings) -> FinalInputBundle:
    """Validate both official files while retaining all 2,000 rows in order."""

    test_rows = _read_exact_csv(settings.test_path, settings.columns, "final test")
    if len(test_rows) != settings.expected_rows:
        raise FinalSubmissionError(
            f"Final test must contain {settings.expected_rows} rows; got {len(test_rows)}."
        )
    seen: set[str] = set()
    for index, row in enumerate(test_rows):
        if not row["id"].strip():
            raise FinalSubmissionError(f"Final test row {index + 2} has a missing id.")
        if not row["question"].strip():
            raise FinalSubmissionError(f"Final test row {index + 2} has a missing question.")
        if row["answer"].strip():
            raise FinalSubmissionError(
                f"Final test answer for {row['id']!r} must be empty before inference."
            )
        if row["id"] in seen:
            raise FinalSubmissionError(f"Final test has duplicate id {row['id']!r}.")
        seen.add(row["id"])

    flag_rows = _read_exact_csv(settings.flag_path, settings.flag_columns, "test flags")
    if len(flag_rows) != settings.expected_flag_rows:
        raise FinalSubmissionError(
            f"Test flags must contain {settings.expected_flag_rows} rows; got {len(flag_rows)}."
        )
    question_by_id = {row["id"]: row["question"] for row in test_rows}
    flagged_ids: set[str] = set()
    for index, row in enumerate(flag_rows):
        if not row["id"].strip() or not row["question"].strip():
            raise FinalSubmissionError(f"Test flag row {index + 2} has a missing value.")
        if row["id"] in flagged_ids:
            raise FinalSubmissionError(f"Test flags have duplicate id {row['id']!r}.")
        if row["id"] not in question_by_id:
            raise FinalSubmissionError(f"Flagged id {row['id']!r} is absent from final test.")
        if question_by_id[row["id"]] != row["question"]:
            raise FinalSubmissionError(
                f"Flagged question for {row['id']!r} does not match final test exactly."
            )
        flagged_ids.add(row["id"])

    rows = tuple(
        FinalTestRow(
            index=index,
            sample_id=row["id"],
            question=row["question"],
            question_sha256=hashlib.sha256(row["question"].encode("utf-8")).hexdigest(),
            is_flagged=row["id"] in flagged_ids,
        )
        for index, row in enumerate(test_rows)
    )
    identity = {
        "schema_version": 1,
        "test": {
            "filename": settings.test_path.name,
            "sha256": sha256_file(settings.test_path),
            "rows": len(test_rows),
            "columns": list(settings.columns),
            "ordered_id_sha256": _ordered_identity([row["id"] for row in test_rows]),
        },
        "flags": {
            "filename": settings.flag_path.name,
            "sha256": sha256_file(settings.flag_path),
            "rows": len(flag_rows),
            "columns": list(settings.flag_columns),
            "ordered_id_sha256": _ordered_identity([row["id"] for row in flag_rows]),
        },
        "flagged_rows_included": True,
        "scoring_eligible_rows": len(test_rows) - len(flag_rows),
    }
    return FinalInputBundle(rows=rows, identity=identity)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: object) -> None:
    serialized = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    _atomic_write_bytes(path, serialized + b"\n")


def _write_predictions(path: Path, records: Sequence[FinalPrediction]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(PREDICTION_COLUMNS)
    writer.writerows(record.as_csv_row() for record in records)
    _atomic_write_bytes(path, buffer.getvalue().encode("utf-8"))


def _append_prediction(path: Path, record: FinalPrediction) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(record.as_csv_row())
        handle.flush()
        os.fsync(handle.fileno())


def _prediction_from_csv(row: Mapping[str, str]) -> FinalPrediction:
    if set(row) != set(PREDICTION_COLUMNS):
        raise FinalSubmissionError("Resume prediction schema is incompatible.")
    if row["parser_version"] != ANSWER_PARSER_V2_VERSION:
        raise FinalSubmissionError("Resume prediction parser version is incompatible.")
    if row["truncated"] not in {"true", "false"} or row["is_flagged"] not in {
        "true",
        "false",
    }:
        raise FinalSubmissionError("Resume prediction booleans are malformed.")
    try:
        latency = float(row["latency_sec"])
        if not math.isfinite(latency) or latency < 0:
            raise ValueError
        return FinalPrediction(
            source_index=int(row["source_index"]),
            sample_id=row["id"],
            question=row["question"],
            question_sha256=row["question_sha256"],
            raw_output=row["raw_output"],
            parsed_answer=None if row["parsed_answer"] == "" else int(row["parsed_answer"]),
            parse_status=row["parse_status"],
            input_tokens=int(row["input_tokens"]),
            output_tokens=int(row["output_tokens"]),
            latency_sec=latency,
            finish_reason=row["finish_reason"],
            truncated=row["truncated"] == "true",
            is_flagged=row["is_flagged"] == "true",
        )
    except ValueError as exc:
        raise FinalSubmissionError("Resume prediction contains malformed typed fields.") from exc


def load_final_predictions(path: Path) -> tuple[FinalPrediction, ...]:
    rows = _read_exact_csv(path, PREDICTION_COLUMNS, "resume predictions")
    return tuple(_prediction_from_csv(row) for row in rows)


def _source_hashes() -> dict[str, str]:
    zero_shot_path = Path(zero_shot_module.__file__).resolve()
    return {
        "submission.py": sha256_file(Path(__file__).resolve()),
        "zero_shot.py": sha256_file(zero_shot_path),
    }


def build_final_resume_identity(
    config: LoadedConfig,
    settings: FinalSubmissionSettings,
    generator: BatchGenerator,
    input_identity: Mapping[str, Any],
    *,
    limit: int | None,
) -> dict[str, Any]:
    identity = {
        "schema_version": 1,
        "pipeline_version": FINAL_SUBMISSION_PIPELINE_VERSION,
        "config_sha256": config.source_sha256,
        "code_sha256": _source_hashes(),
        "input_identity": dict(input_identity),
        "model_name": settings.model.name_or_path,
        "model_revision": settings.model.revision,
        "model_commit": generator.model_commit,
        "tokenizer_revision": settings.model.tokenizer_revision,
        "tokenizer_commit": generator.tokenizer_commit,
        "prompt": asdict(settings.prompt),
        "chat_template_sha256": hashlib.sha256(generator.chat_template.encode()).hexdigest(),
        "generation": asdict(settings.generation),
        "parser_version": settings.parser_version,
        "canonical_e000_config_sha256": settings.canonical_e000_config_sha256,
        "limit": limit,
        "external_datasets": [],
        "tool_use": False,
        "adapter": None,
    }
    canonical = json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return {
        "identity": identity,
        "identity_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _load_resume(
    resume_dir: Path,
    expected_identity: Mapping[str, Any],
    selected_rows: Sequence[FinalTestRow],
    output: FinalOutputSettings,
) -> tuple[FinalPrediction, ...]:
    identity_path = resume_dir / output.resume_identity_filename
    predictions_path = resume_dir / output.predictions_filename
    if not identity_path.is_file() or not predictions_path.is_file():
        raise FinalSubmissionError("Resume directory lacks identity or predictions.")
    try:
        actual_identity = json.loads(identity_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalSubmissionError("Resume identity must be valid UTF-8 JSON.") from exc
    if actual_identity != expected_identity:
        raise FinalSubmissionError("Resume identity mismatch; refusing to mix final-test runs.")
    records = load_final_predictions(predictions_path)
    if len(records) > len(selected_rows):
        raise FinalSubmissionError("Resume predictions exceed selected final-test rows.")
    for record, source in zip(records, selected_rows, strict=False):
        reparsed = parse_integer_answer_v2(record.raw_output)
        if (
            record.source_index != source.index
            or record.sample_id != source.sample_id
            or record.question != source.question
            or record.question_sha256 != source.question_sha256
            or record.is_flagged != source.is_flagged
            or record.parsed_answer != reparsed.value
            or record.parse_status != reparsed.status
        ):
            raise FinalSubmissionError(
                f"Resume prediction for {source.sample_id!r} failed integrity validation."
            )
    return records


def _metrics(
    records: Sequence[FinalPrediction],
    *,
    input_identity: Mapping[str, Any],
    total_wall_clock_sec: float,
    resumed_predictions: int,
    limit: int | None,
    generator: BatchGenerator,
) -> dict[str, Any]:
    total = len(records)
    flagged = sum(record.is_flagged for record in records)
    parse_failures = sum(record.parsed_answer is None for record in records)
    latencies = [record.latency_sec for record in records]
    output_tokens = [record.output_tokens for record in records]
    return {
        "schema_version": 1,
        "pipeline_version": FINAL_SUBMISSION_PIPELINE_VERSION,
        "experiment_id": FINAL_EXPERIMENT_ID,
        "total_rows": total,
        "flagged_rows": flagged,
        "scoring_eligible_rows": total - flagged,
        "official_total_rows": input_identity["test"]["rows"],
        "official_flagged_rows": input_identity["flags"]["rows"],
        "flagged_rows_included": True,
        "parse_failures": parse_failures,
        "max_new_tokens_hits": sum(record.finish_reason == "max_new_tokens" for record in records),
        "finish_reason_counts": dict(
            sorted(Counter(record.finish_reason for record in records).items())
        ),
        "output_tokens_total": sum(output_tokens),
        "output_tokens_mean": None if not output_tokens else sum(output_tokens) / total,
        "latency_sec_total": sum(latencies),
        "latency_sec_mean": None if not latencies else sum(latencies) / total,
        "total_wall_clock_sec": round(total_wall_clock_sec, 8),
        "resumed_predictions": resumed_predictions,
        "limit": limit,
        "canonical_full_submission": limit is None and total == input_identity["test"]["rows"],
        "parser_version": ANSWER_PARSER_V2_VERSION,
        "runtime": dict(generator.runtime_metadata()),
    }


def _write_and_validate_submission(
    path: Path,
    records: Sequence[FinalPrediction],
    source_rows: Sequence[FinalTestRow],
) -> None:
    if len(records) != len(source_rows):
        raise FinalSubmissionError("Submission coverage does not match final-test rows.")
    missing = [record.sample_id for record in records if record.parsed_answer is None]
    if missing:
        raise FinalSubmissionError(
            f"Cannot create submission with {len(missing)} parse failures: {missing[:10]!r}."
        )
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(OFFICIAL_TEST_COLUMNS)
    for record, source in zip(records, source_rows, strict=True):
        if record.sample_id != source.sample_id or record.question != source.question:
            raise FinalSubmissionError("Prediction order/content differs from final-test source.")
        writer.writerow([source.sample_id, source.question, str(record.parsed_answer)])
    _atomic_write_bytes(path, buffer.getvalue().encode("utf-8"))
    validate_submission_file(path, source_rows)


def validate_submission_file(path: Path, source_rows: Sequence[FinalTestRow]) -> None:
    if path.name != "submission.csv":
        raise FinalSubmissionError("Official output filename must be submission.csv.")
    rows = _read_exact_csv(path, OFFICIAL_TEST_COLUMNS, "submission")
    if len(rows) != len(source_rows):
        raise FinalSubmissionError("Submission row count does not match official test.")
    seen: set[str] = set()
    for index, (row, source) in enumerate(zip(rows, source_rows, strict=True)):
        if row["id"] in seen:
            raise FinalSubmissionError(f"Submission has duplicate id {row['id']!r}.")
        seen.add(row["id"])
        if row["id"] != source.sample_id or row["question"] != source.question:
            raise FinalSubmissionError(
                f"Submission row {index + 2} does not preserve official id/question/order."
            )
        if not CANONICAL_INTEGER_PATTERN.fullmatch(row["answer"]):
            raise FinalSubmissionError(
                f"Submission answer for {row['id']!r} is not a canonical integer."
            )
        if str(int(row["answer"])) != row["answer"]:
            raise FinalSubmissionError(f"Submission answer for {row['id']!r} is not canonicalized.")


def run_final_submission(
    config: LoadedConfig,
    settings: FinalSubmissionSettings,
    *,
    run_dir: str | Path,
    generator: BatchGenerator,
    limit: int | None = None,
    resume_from: str | Path | None = None,
) -> FinalSubmissionResult:
    """Infer every selected row, fsync each result, and create only a full submission."""

    bundle = load_final_input(settings.data)
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise FinalSubmissionError("limit must be a positive integer.")
        if limit > len(bundle.rows):
            raise FinalSubmissionError("limit exceeds official final-test rows.")
        selected_rows = bundle.rows[:limit]
    else:
        selected_rows = bundle.rows

    output_dir = Path(run_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_identity_path = output_dir / settings.output.input_identity_filename
    _atomic_write_json(input_identity_path, bundle.identity)
    identity = build_final_resume_identity(
        config, settings, generator, bundle.identity, limit=limit
    )
    resume_identity_path = output_dir / settings.output.resume_identity_filename
    _atomic_write_json(resume_identity_path, identity)

    existing: tuple[FinalPrediction, ...] = tuple()
    if resume_from is not None:
        existing = _load_resume(
            Path(resume_from).expanduser().resolve(),
            identity,
            selected_rows,
            settings.output,
        )
    predictions_path = output_dir / settings.output.predictions_filename
    _write_predictions(predictions_path, existing)

    started = time.perf_counter()
    for source in selected_rows[len(existing) :]:
        generated = list(generator.generate([source.question]))
        if len(generated) != 1:
            raise FinalSubmissionError("Generator must return exactly one result per test row.")
        result = generated[0]
        parsed = parse_integer_answer_v2(result.raw_output)
        record = FinalPrediction(
            source_index=source.index,
            sample_id=source.sample_id,
            question=source.question,
            question_sha256=source.question_sha256,
            raw_output=result.raw_output,
            parsed_answer=parsed.value,
            parse_status=parsed.status,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_sec=result.latency_sec,
            finish_reason=result.finish_reason,
            truncated=result.truncated,
            is_flagged=source.is_flagged,
        )
        _append_prediction(predictions_path, record)
    elapsed = time.perf_counter() - started

    records = load_final_predictions(predictions_path)
    if [record.sample_id for record in records] != [row.sample_id for row in selected_rows]:
        raise FinalSubmissionError("Final prediction ordering/coverage is incomplete.")
    metrics = _metrics(
        records,
        input_identity=bundle.identity,
        total_wall_clock_sec=elapsed,
        resumed_predictions=len(existing),
        limit=limit,
        generator=generator,
    )
    metrics_path = output_dir / settings.output.metrics_filename
    _atomic_write_json(metrics_path, metrics)

    submission_path: Path | None = None
    submission_sha256_path: Path | None = None
    if limit is None:
        submission_path = output_dir / settings.output.submission_filename
        _write_and_validate_submission(submission_path, records, bundle.rows)
        submission_digest = sha256_file(submission_path)
        submission_sha256_path = output_dir / settings.output.submission_sha256_filename
        _atomic_write_bytes(
            submission_sha256_path,
            f"{submission_digest}  {settings.output.submission_filename}\n".encode("ascii"),
        )
        metrics["submission_sha256"] = submission_digest
        _atomic_write_json(metrics_path, metrics)

    paths = [predictions_path, input_identity_path, resume_identity_path, metrics_path]
    if submission_path is not None and submission_sha256_path is not None:
        paths.extend([submission_path, submission_sha256_path])
    artifact_sha256 = {path.name: sha256_file(path) for path in paths}
    return FinalSubmissionResult(
        metrics=metrics,
        input_identity=bundle.identity,
        resume_identity=identity,
        predictions_path=predictions_path,
        input_identity_path=input_identity_path,
        resume_identity_path=resume_identity_path,
        metrics_path=metrics_path,
        submission_path=submission_path,
        submission_sha256_path=submission_sha256_path,
        artifact_sha256=artifact_sha256,
    )


def final_manifest_fields(
    settings: FinalSubmissionSettings,
    generator: BatchGenerator,
    result: FinalSubmissionResult,
    *,
    resume_from: str | None,
) -> dict[str, Any]:
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = None
    try:
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        transformers_version = None
    return {
        "pipeline_version": FINAL_SUBMISSION_PIPELINE_VERSION,
        "model_name": settings.model.name_or_path,
        "model_revision": settings.model.revision,
        "model_commit": generator.model_commit,
        "tokenizer_revision": settings.model.tokenizer_revision,
        "tokenizer_commit": generator.tokenizer_commit,
        "prompt_version": settings.prompt.version,
        "prompt_text": asdict(settings.prompt),
        "generation_config": asdict(settings.generation),
        "answer_parser_version": settings.parser_version,
        "canonical_e000_config_sha256": settings.canonical_e000_config_sha256,
        "input_identity": result.input_identity,
        "resume_identity_sha256": result.resume_identity["identity_sha256"],
        "resumed_from": resume_from,
        "flagged_rows_included": True,
        "total_samples": result.metrics["total_rows"],
        "flagged_rows": result.metrics["flagged_rows"],
        "scoring_eligible_rows": result.metrics["scoring_eligible_rows"],
        "parse_failures": result.metrics["parse_failures"],
        "artifact_sha256": result.artifact_sha256,
        "runtime": dict(generator.runtime_metadata()),
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "transformers_version": transformers_version,
        "external_datasets": [],
        "tool_use": False,
        "adapter": None,
    }
