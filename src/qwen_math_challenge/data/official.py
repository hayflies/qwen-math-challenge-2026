"""Phase 1 official CSV inspection, mandatory filtering, and provenance output."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from qwen_math_challenge.config import LoadedConfig
from qwen_math_challenge.environment import collect_git_info

PIPELINE_VERSION = "phase1_official_v001"
_INTEGER_PATTERN = re.compile(r"^[+-]?\d+$")
_URL_PATTERN = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
_MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_INSTRUCTION_RESIDUE_PATTERN = re.compile(
    r"(?:ignore\s+(?:all\s+)?previous|assistant\s*:|user\s*:|"
    r"translate\s+the\s+following|rewrite\s+the\s+following|"
    r"다음\s*(?:문장|문제)?을?\s*번역|번역해|전처리\s*지침)",
    re.IGNORECASE,
)
_CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_REPEATED_SYMBOL_PATTERN = re.compile(r"([^\w\s])\1{7,}")


class DatasetValidationError(ValueError):
    """Raised when official data violates configured schema, hash, or invariants."""


@dataclass(frozen=True)
class CsvTable:
    role: str
    path: Path
    sha256: str
    columns: tuple[str, ...]
    rows: tuple[dict[str, str], ...]

    @property
    def shape(self) -> list[int]:
        return [len(self.rows), len(self.columns)]


@dataclass(frozen=True)
class DatasetSettings:
    dataset_version: str
    raw_manifest_path: Path
    train_path: Path
    leaderboard_path: Path
    filtered_ids_path: Path
    processed_dir: Path
    schemas: dict[str, tuple[str, ...]]
    invariants: dict[str, int]
    filter_key: str
    candidate_exclusion_sources: tuple[str, ...]
    apply_candidate_exclusions: bool
    short_question_chars: int
    long_question_chars: int
    max_examples_per_reason: int


@dataclass(frozen=True)
class Phase1Result:
    dataset_version: str
    raw_train_rows: int
    mandatory_exclusion_rows: int
    clean_train_rows: int
    leaderboard_rows: int
    clean_train_sha256: str
    clean_train_path: str
    audit_report_path: str
    dataset_manifest_path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_version": self.dataset_version,
            "raw_train_rows": self.raw_train_rows,
            "mandatory_exclusion_rows": self.mandatory_exclusion_rows,
            "clean_train_rows": self.clean_train_rows,
            "leaderboard_rows": self.leaderboard_rows,
            "clean_train_sha256": self.clean_train_sha256,
            "clean_train_path": self.clean_train_path,
            "audit_report_path": self.audit_report_path,
            "dataset_manifest_path": self.dataset_manifest_path,
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetValidationError(f"'{label}' must be a mapping.")
    return dict(value)


def _require_positive_int(value: object, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise DatasetValidationError(f"'{label}' must be a {qualifier} integer.")
    return value


def _resolve_path(project_root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"'{label}' must be a non-empty path string.")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def load_dataset_settings(config: LoadedConfig, project_root: str | Path) -> DatasetSettings:
    root = Path(project_root).resolve()
    dataset = _require_mapping(config.raw.get("dataset"), "dataset")
    dataset_version = dataset.get("dataset_version")
    if not isinstance(dataset_version, str) or not dataset_version.strip():
        raise DatasetValidationError("dataset.dataset_version must be a non-empty string.")
    if dataset_version != config.experiment.get("dataset_version"):
        raise DatasetValidationError(
            "dataset.dataset_version must match experiment.dataset_version."
        )

    paths = _require_mapping(dataset.get("paths"), "dataset.paths")
    raw_manifest_path = _resolve_path(root, dataset.get("raw_manifest"), "dataset.raw_manifest")
    train_path = _resolve_path(root, paths.get("train"), "dataset.paths.train")
    leaderboard_path = _resolve_path(root, paths.get("leaderboard"), "dataset.paths.leaderboard")
    filtered_ids_path = _resolve_path(root, paths.get("filtered_ids"), "dataset.paths.filtered_ids")
    processed_dir = _resolve_path(root, paths.get("processed_dir"), "dataset.paths.processed_dir")
    raw_root = (root / "data" / "raw").resolve()
    if _is_within(processed_dir, raw_root):
        raise DatasetValidationError(
            "dataset.paths.processed_dir may not be data/raw or a descendant of it."
        )

    schema = _require_mapping(dataset.get("schema"), "dataset.schema")
    schemas: dict[str, tuple[str, ...]] = {}
    for role in ("train", "leaderboard", "filtered_ids"):
        columns = schema.get(role)
        if (
            not isinstance(columns, list)
            or not columns
            or not all(isinstance(column, str) and column for column in columns)
        ):
            raise DatasetValidationError(
                f"dataset.schema.{role} must be a non-empty list of column names."
            )
        schemas[role] = tuple(columns)
    required_columns = {
        "train": {"id", "question", "answer"},
        "leaderboard": {"id", "question"},
        "filtered_ids": {"id", "question", "answer"},
    }
    for role, required in required_columns.items():
        missing = sorted(required - set(schemas[role]))
        if missing:
            raise DatasetValidationError(
                f"dataset.schema.{role} is missing required columns: {missing}."
            )
    if "answer" in schemas["leaderboard"]:
        raise DatasetValidationError("Leaderboard schema must not contain an answer column.")

    invariant_values = _require_mapping(dataset.get("invariants"), "dataset.invariants")
    invariants = {
        key: _require_positive_int(invariant_values.get(key), f"dataset.invariants.{key}")
        for key in (
            "raw_train_rows",
            "leaderboard_rows",
            "mandatory_exclusion_rows",
            "clean_train_rows",
        )
    }

    filtering = _require_mapping(dataset.get("filtering"), "dataset.filtering")
    if filtering.get("mandatory_exclusion_role") != "filtered_ids":
        raise DatasetValidationError(
            "dataset.filtering.mandatory_exclusion_role must be 'filtered_ids'."
        )
    filter_key = filtering.get("key")
    if filter_key != "id":
        raise DatasetValidationError("Phase 1 mandatory filtering key must be exactly 'id'.")
    candidate_sources = filtering.get("candidate_exclusion_sources", [])
    if not isinstance(candidate_sources, list) or not all(
        isinstance(source, str) for source in candidate_sources
    ):
        raise DatasetValidationError(
            "dataset.filtering.candidate_exclusion_sources must be a list of paths."
        )
    apply_candidates = filtering.get("apply_candidate_exclusions", False)
    if not isinstance(apply_candidates, bool):
        raise DatasetValidationError(
            "dataset.filtering.apply_candidate_exclusions must be a boolean."
        )
    if apply_candidates:
        raise DatasetValidationError(
            "Candidate exclusions are not permitted in official_v001 Phase 1 preparation."
        )

    audit = _require_mapping(dataset.get("audit"), "dataset.audit")
    short_chars = _require_positive_int(
        audit.get("unusually_short_question_chars"),
        "dataset.audit.unusually_short_question_chars",
        allow_zero=True,
    )
    long_chars = _require_positive_int(
        audit.get("unusually_long_question_chars"),
        "dataset.audit.unusually_long_question_chars",
    )
    if long_chars <= short_chars:
        raise DatasetValidationError(
            "unusually_long_question_chars must exceed unusually_short_question_chars."
        )

    return DatasetSettings(
        dataset_version=dataset_version,
        raw_manifest_path=raw_manifest_path,
        train_path=train_path,
        leaderboard_path=leaderboard_path,
        filtered_ids_path=filtered_ids_path,
        processed_dir=processed_dir,
        schemas=schemas,
        invariants=invariants,
        filter_key=filter_key,
        candidate_exclusion_sources=tuple(candidate_sources),
        apply_candidate_exclusions=apply_candidates,
        short_question_chars=short_chars,
        long_question_chars=long_chars,
        max_examples_per_reason=_require_positive_int(
            audit.get("max_examples_per_reason"),
            "dataset.audit.max_examples_per_reason",
        ),
    )


def _load_raw_manifest(settings: DatasetSettings, project_root: Path) -> tuple[dict[str, Any], str]:
    if not settings.raw_manifest_path.is_file():
        raise DatasetValidationError(f"Raw manifest does not exist: {settings.raw_manifest_path}")
    try:
        manifest = json.loads(settings.raw_manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetValidationError("Raw manifest must be valid UTF-8 JSON.") from exc
    manifest = _require_mapping(manifest, "raw manifest")
    if manifest.get("dataset_version") != settings.dataset_version:
        raise DatasetValidationError("Raw manifest dataset_version does not match config.")
    files = _require_mapping(manifest.get("files"), "raw manifest files")
    configured_paths = {
        "train": settings.train_path,
        "leaderboard": settings.leaderboard_path,
        "filtered_ids": settings.filtered_ids_path,
    }
    for role, configured_path in configured_paths.items():
        entry = _require_mapping(files.get(role), f"raw manifest files.{role}")
        manifest_path = _resolve_path(project_root, entry.get("path"), f"manifest {role} path")
        if manifest_path != configured_path:
            raise DatasetValidationError(
                f"Raw manifest path for {role} does not match the configured path."
            )
        expected_hash = entry.get("sha256")
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise DatasetValidationError(f"Raw manifest SHA-256 for {role} is invalid.")
        if not configured_path.is_file():
            raise DatasetValidationError(f"Configured raw {role} file does not exist.")
        actual_hash = sha256_file(configured_path)
        if actual_hash != expected_hash:
            raise DatasetValidationError(
                f"Raw {role} SHA-256 mismatch: expected {expected_hash}, got {actual_hash}."
            )
    return manifest, sha256_file(settings.raw_manifest_path)


def _read_csv(role: str, path: Path, expected_columns: Sequence[str]) -> CsvTable:
    source_bytes = path.read_bytes()
    try:
        text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetValidationError(f"{role} must be UTF-8 without undecodable bytes.") from exc

    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        columns = tuple(next(reader))
    except StopIteration as exc:
        raise DatasetValidationError(f"{role} CSV is empty.") from exc
    if columns != tuple(expected_columns):
        raise DatasetValidationError(
            f"{role} columns must be {list(expected_columns)}, got {list(columns)}."
        )

    rows: list[dict[str, str]] = []
    for logical_row_number, values in enumerate(reader, start=2):
        if len(values) != len(columns):
            raise DatasetValidationError(
                f"{role} logical row {logical_row_number} has {len(values)} fields; "
                f"expected {len(columns)}."
            )
        rows.append(dict(zip(columns, values, strict=True)))

    return CsvTable(
        role=role,
        path=path,
        sha256=hashlib.sha256(source_bytes).hexdigest(),
        columns=columns,
        rows=tuple(rows),
    )


def _null_counts(table: CsvTable) -> dict[str, int]:
    return {column: sum(row[column] == "" for row in table.rows) for column in table.columns}


def _duplicate_stats(values: Sequence[str]) -> dict[str, int]:
    counts = Counter(values)
    duplicate_counts = [count for count in counts.values() if count > 1]
    return {
        "groups": len(duplicate_counts),
        "extra_rows": sum(count - 1 for count in duplicate_counts),
        "rows_in_duplicate_groups": sum(duplicate_counts),
    }


def _normalize_question(question: str) -> str:
    normalized = unicodedata.normalize("NFKC", question).casefold()
    return " ".join(normalized.split())


def _answer_values(table: CsvTable) -> list[int] | None:
    if "answer" not in table.columns:
        return None
    values: list[int] = []
    invalid_ids: list[str] = []
    for row in table.rows:
        candidate = row["answer"]
        if not _INTEGER_PATTERN.fullmatch(candidate):
            invalid_ids.append(row.get("id", ""))
            continue
        values.append(int(candidate))
    if invalid_ids:
        examples = invalid_ids[:10]
        raise DatasetValidationError(
            f"{table.role} contains non-integer answers; example IDs: {examples}."
        )
    return values


def _validate_required_values_and_ids(table: CsvTable) -> None:
    nulls = _null_counts(table)
    if any(nulls.values()):
        raise DatasetValidationError(f"{table.role} contains empty required values: {nulls}.")
    id_stats = _duplicate_stats([row["id"] for row in table.rows])
    if id_stats["groups"]:
        raise DatasetValidationError(
            f"{table.role} contains duplicate IDs: {id_stats['groups']} groups."
        )
    _answer_values(table)


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _length_statistics(questions: Sequence[str]) -> dict[str, int | float | None]:
    if not questions:
        return {
            "min": None,
            "p01": None,
            "p05": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    lengths = np.asarray([len(question) for question in questions], dtype=np.float64)
    return {
        "min": int(lengths.min()),
        "p01": _rounded(np.percentile(lengths, 1)),
        "p05": _rounded(np.percentile(lengths, 5)),
        "median": _rounded(np.percentile(lengths, 50)),
        "p95": _rounded(np.percentile(lengths, 95)),
        "p99": _rounded(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
        "mean": _rounded(lengths.mean()),
        "std": _rounded(lengths.std(ddof=0)),
    }


def _table_summary(table: CsvTable) -> dict[str, Any]:
    ids = [row["id"] for row in table.rows]
    questions = [row["question"] for row in table.rows]
    normalized_questions = [_normalize_question(question) for question in questions]
    answer_values = _answer_values(table)
    dtypes = {column: "string" for column in table.columns}
    answer_summary = None
    if answer_values is not None:
        dtypes["answer"] = "integer"
        answer_summary = {
            "all_integer": True,
            "min": min(answer_values),
            "max": max(answer_values),
            "negative_count": sum(value < 0 for value in answer_values),
            "zero_count": sum(value == 0 for value in answer_values),
        }
    null_counts = _null_counts(table)
    return {
        "path": str(table.path),
        "sha256": table.sha256,
        "shape": table.shape,
        "rows": len(table.rows),
        "columns": list(table.columns),
        "dtypes": dtypes,
        "null_counts": null_counts,
        "total_null_count": sum(null_counts.values()),
        "id_unique": len(ids) == len(set(ids)),
        "id_duplicates": _duplicate_stats(ids),
        "question_exact_duplicates": _duplicate_stats(questions),
        "question_normalized_duplicates": _duplicate_stats(normalized_questions),
        "question_character_length": _length_statistics(questions),
        "answer": answer_summary,
    }


def _has_latex_or_markup_anomaly(question: str) -> bool:
    # A single '$' is ambiguous between inline math and currency in this dataset. Only the
    # unambiguous display-math/code-fence forms and structural delimiters are checked here.
    if question.count("$$") % 2:
        return True
    if question.count("```") % 2:
        return True
    if question.count("{") != question.count("}"):
        return True
    if question.count(r"\(") != question.count(r"\)"):
        return True
    if question.count(r"\[") != question.count(r"\]"):
        return True
    begin_environments = Counter(re.findall(r"\\begin\{([^}]+)\}", question))
    end_environments = Counter(re.findall(r"\\end\{([^}]+)\}", question))
    return begin_environments != end_environments


def _question_quality_audit(table: CsvTable, settings: DatasetSettings) -> dict[str, Any]:
    questions = [row["question"] for row in table.rows]
    exact_counts = Counter(questions)
    normalized_counts = Counter(_normalize_question(question) for question in questions)
    reason_counts: Counter[str] = Counter()
    examples: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    suspect_ids: set[str] = set()

    for row in table.rows:
        question = row["question"]
        normalized = _normalize_question(question)
        reasons: list[str] = []
        if not question.strip():
            reasons.append("empty_or_whitespace_question")
        if len(question) < settings.short_question_chars:
            reasons.append("unusually_short_question")
        if len(question) > settings.long_question_chars:
            reasons.append("unusually_long_question")
        if exact_counts[question] > 1:
            reasons.append("exact_duplicate_question")
        if normalized_counts[normalized] > 1:
            reasons.append("normalized_duplicate_question")
        if _URL_PATTERN.search(question):
            reasons.append("contains_url")
        if _MARKDOWN_IMAGE_PATTERN.search(question):
            reasons.append("contains_markdown_image")
        if _has_latex_or_markup_anomaly(question):
            reasons.append("latex_or_markup_anomaly_candidate")
        if _INSTRUCTION_RESIDUE_PATTERN.search(question):
            reasons.append("translation_or_instruction_residue_candidate")
        if (
            "\ufffd" in question
            or _CONTROL_CHARACTER_PATTERN.search(question)
            or _REPEATED_SYMBOL_PATTERN.search(question)
        ):
            reasons.append("abnormal_character_pattern_candidate")

        if reasons:
            suspect_ids.add(row["id"])
        for reason in reasons:
            reason_counts[reason] += 1
            if len(examples[reason]) < settings.max_examples_per_reason:
                examples[reason].append(
                    {
                        "id": row["id"],
                        "question_length": len(question),
                    }
                )

    return {
        "suspect_row_count": len(suspect_ids),
        "reason_counts": dict(sorted(reason_counts.items())),
        "examples_by_reason": dict(sorted(examples.items())),
        "policy": "report_only_no_automatic_deletion",
    }


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return str(path)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _write_clean_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, str]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([row[column] for column in columns])
    payload = buffer.getvalue().encode("utf-8")
    _atomic_write_bytes(path, payload)
    return hashlib.sha256(payload).hexdigest()


def _overlap_summary(left: Sequence[str], right: Sequence[str]) -> dict[str, Any]:
    overlap = sorted(set(left) & set(right))
    return {"count": len(overlap), "example_values": overlap[:25]}


def _validate_expected_count(actual: int, expected: int, label: str) -> None:
    if actual != expected:
        raise DatasetValidationError(f"{label} expected {expected} rows, got {actual}.")


def run_official_data_pipeline(
    config: LoadedConfig,
    *,
    project_root: str | Path,
) -> Phase1Result:
    """Run the complete Phase 1 raw inspection and mandatory-ID clean pipeline."""

    root = Path(project_root).resolve()
    settings = load_dataset_settings(config, root)
    raw_manifest, raw_manifest_sha256 = _load_raw_manifest(settings, root)

    train = _read_csv("train", settings.train_path, settings.schemas["train"])
    leaderboard = _read_csv(
        "leaderboard", settings.leaderboard_path, settings.schemas["leaderboard"]
    )
    filtered_ids = _read_csv(
        "filtered_ids", settings.filtered_ids_path, settings.schemas["filtered_ids"]
    )
    for table in (train, leaderboard, filtered_ids):
        _validate_required_values_and_ids(table)

    _validate_expected_count(len(train.rows), settings.invariants["raw_train_rows"], "raw train")
    _validate_expected_count(
        len(leaderboard.rows), settings.invariants["leaderboard_rows"], "leaderboard"
    )
    _validate_expected_count(
        len(filtered_ids.rows),
        settings.invariants["mandatory_exclusion_rows"],
        "mandatory exclusions",
    )

    train_by_id = {row["id"]: row for row in train.rows}
    mandatory_ids = {row[settings.filter_key] for row in filtered_ids.rows}
    missing_filtered_ids = sorted(mandatory_ids - set(train_by_id))
    if missing_filtered_ids:
        raise DatasetValidationError(
            "Mandatory exclusion IDs are missing from train; examples: "
            f"{missing_filtered_ids[:25]}."
        )

    answer_mismatches = sorted(
        row["id"]
        for row in filtered_ids.rows
        if int(row["answer"]) != int(train_by_id[row["id"]]["answer"])
    )
    if answer_mismatches:
        raise DatasetValidationError(
            f"Mandatory exclusion answers differ from train; examples: {answer_mismatches[:25]}."
        )

    question_exact_matches = sum(
        row["question"] == train_by_id[row["id"]]["question"] for row in filtered_ids.rows
    )
    clean_rows = tuple(row for row in train.rows if row[settings.filter_key] not in mandatory_ids)
    clean_ids = {row["id"] for row in clean_rows}
    remaining_filtered_ids = sorted(clean_ids & mandatory_ids)
    if remaining_filtered_ids:
        raise DatasetValidationError("Mandatory exclusion IDs remain in clean train.")
    if clean_ids != set(train_by_id) - mandatory_ids:
        raise DatasetValidationError("Rows outside the mandatory exclusion set were lost.")
    _validate_expected_count(
        len(clean_rows), settings.invariants["clean_train_rows"], "clean train"
    )

    clean_path = settings.processed_dir / "train_clean.csv"
    clean_sha256 = _write_clean_csv(clean_path, settings.schemas["train"], clean_rows)
    clean = CsvTable(
        role="clean_train",
        path=clean_path,
        sha256=clean_sha256,
        columns=settings.schemas["train"],
        rows=clean_rows,
    )
    _validate_required_values_and_ids(clean)

    train_ids = [row["id"] for row in train.rows]
    leaderboard_ids = [row["id"] for row in leaderboard.rows]
    train_questions = [row["question"] for row in train.rows]
    clean_questions = [row["question"] for row in clean.rows]
    leaderboard_questions = [row["question"] for row in leaderboard.rows]
    generated_at = datetime.now(UTC).isoformat()

    table_summaries = {
        "raw_train": _table_summary(train),
        "clean_train": _table_summary(clean),
        "filtered_ids": _table_summary(filtered_ids),
        "leaderboard": _table_summary(leaderboard),
    }
    for summary in table_summaries.values():
        summary["path"] = _display_path(Path(summary["path"]), root)

    quality_audit = {
        "raw_train": _question_quality_audit(train, settings),
        "clean_train": _question_quality_audit(clean, settings),
        "filtered_ids": _question_quality_audit(filtered_ids, settings),
        "leaderboard": _question_quality_audit(leaderboard, settings),
    }
    cross_checks = {
        "train_leaderboard_id_exact_overlap": _overlap_summary(train_ids, leaderboard_ids),
        "train_leaderboard_question_exact_overlap": _overlap_summary(
            train_questions, leaderboard_questions
        ),
        "train_leaderboard_question_normalized_overlap": _overlap_summary(
            [_normalize_question(value) for value in train_questions],
            [_normalize_question(value) for value in leaderboard_questions],
        ),
        "clean_leaderboard_question_exact_overlap": _overlap_summary(
            clean_questions, leaderboard_questions
        ),
    }
    filtered_validation = {
        "filter_key": settings.filter_key,
        "mandatory_exclusion_count": len(mandatory_ids),
        "missing_from_train_count": len(missing_filtered_ids),
        "answer_mismatch_count": len(answer_mismatches),
        "question_exact_match_count": question_exact_matches,
        "question_different_count": len(filtered_ids.rows) - question_exact_matches,
        "remaining_in_clean_count": len(remaining_filtered_ids),
        "non_excluded_rows_lost_count": 0,
    }
    audit_report = {
        "schema_version": 1,
        "dataset_version": settings.dataset_version,
        "pipeline_version": PIPELINE_VERSION,
        "created_at_utc": generated_at,
        "config_sha256": config.source_sha256,
        "raw_manifest_sha256": raw_manifest_sha256,
        "tables": table_summaries,
        "filtered_id_validation": filtered_validation,
        "cross_dataset_checks": cross_checks,
        "quality_audit": quality_audit,
        "policy": {
            "automatic_deletion": "mandatory 627 IDs only",
            "candidate_exclusion_sources": list(settings.candidate_exclusion_sources),
            "candidate_exclusions_applied": False,
            "suspect_rows": "report_only",
        },
        "follow_up": {
            "tokenizer_length_analysis": (
                "Deferred until a later Phase introduces the approved Qwen tokenizer dependency."
            ),
            "near_duplicate_or_semantic_leakage": "Deferred to Phase 2/8.",
        },
    }
    audit_path = settings.processed_dir / "audit_report.json"
    _atomic_write_json(audit_path, audit_report)
    audit_sha256 = sha256_file(audit_path)

    raw_hashes_after = {
        "train": sha256_file(settings.train_path),
        "leaderboard": sha256_file(settings.leaderboard_path),
        "filtered_ids": sha256_file(settings.filtered_ids_path),
    }
    raw_hashes_before = {
        role: raw_manifest["files"][role]["sha256"]
        for role in ("train", "leaderboard", "filtered_ids")
    }
    if raw_hashes_after != raw_hashes_before:
        raise DatasetValidationError("A raw source file changed while the pipeline was running.")

    source_tables = {
        "train": train,
        "leaderboard": leaderboard,
        "filtered_ids": filtered_ids,
    }
    manifest = {
        "schema_version": 1,
        "dataset_version": settings.dataset_version,
        "created_at_utc": generated_at,
        "pipeline_version": PIPELINE_VERSION,
        "source_files": {
            role: _display_path(table.path, root) for role, table in source_tables.items()
        },
        "source_file_sha256": {role: table.sha256 for role, table in source_tables.items()},
        "source_shapes": {role: table.shape for role, table in source_tables.items()},
        "source_columns": {role: list(table.columns) for role, table in source_tables.items()},
        "raw_manifest": {
            "path": _display_path(settings.raw_manifest_path, root),
            "sha256": raw_manifest_sha256,
            "hashes_verified_before_and_after": True,
        },
        "filter_policy": {
            "mandatory_exclusion_source": _display_path(settings.filtered_ids_path, root),
            "key": settings.filter_key,
            "question_equality_used_for_filtering": False,
            "candidate_exclusion_sources": list(settings.candidate_exclusion_sources),
            "candidate_exclusions_applied": False,
        },
        "mandatory_exclusion_count": len(mandatory_ids),
        "clean_train_rows": len(clean.rows),
        "clean_train_path": _display_path(clean_path, root),
        "clean_train_sha256": clean_sha256,
        "audit_report_path": _display_path(audit_path, root),
        "audit_report_sha256": audit_sha256,
        "audit_summary": {
            "filtered_id_validation": filtered_validation,
            "cross_dataset_checks": cross_checks,
            "quality_reason_counts": {
                role: audit["reason_counts"] for role, audit in quality_audit.items()
            },
            "question_character_length": {
                role: summary["question_character_length"]
                for role, summary in table_summaries.items()
            },
        },
        "code": {
            "config_sha256": config.source_sha256,
            "module_path": _display_path(Path(__file__).resolve(), root),
            "module_sha256": sha256_file(Path(__file__).resolve()),
            "git": collect_git_info(root),
        },
    }
    manifest_path = settings.processed_dir / "dataset_manifest.json"
    _atomic_write_json(manifest_path, manifest)

    return Phase1Result(
        dataset_version=settings.dataset_version,
        raw_train_rows=len(train.rows),
        mandatory_exclusion_rows=len(filtered_ids.rows),
        clean_train_rows=len(clean.rows),
        leaderboard_rows=len(leaderboard.rows),
        clean_train_sha256=clean_sha256,
        clean_train_path=_display_path(clean_path, root),
        audit_report_path=_display_path(audit_path, root),
        dataset_manifest_path=_display_path(manifest_path, root),
    )
