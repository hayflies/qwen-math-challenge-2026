"""Integrity validation and deterministic post-analysis for the canonical E000 archive."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import shutil
import tarfile
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import yaml

from qwen_math_challenge.data import official as phase1

ANALYSIS_VERSION = "e000_error_analysis_v001"
REQUIRED_ARTIFACTS = {
    "config.snapshot.yaml",
    "environment.json",
    "failures.csv",
    "metrics.json",
    "predictions.csv",
    "resume_identity.json",
    "run.log",
    "run_manifest.json",
}
PREDICTION_COLUMNS = (
    "id",
    "question",
    "gold_answer",
    "raw_output",
    "parsed_answer",
    "correct",
    "parse_status",
    "input_tokens",
    "output_tokens",
    "latency_sec",
    "finish_reason",
    "truncated",
    "derived_category",
)


class AnalysisError(ValueError):
    """Raised when the canonical archive or analysis inputs fail integrity checks."""


@dataclass(frozen=True)
class Prediction:
    sample_id: str
    question: str
    gold_answer: int
    raw_output: str
    parsed_answer: int | None
    stored_correct: bool
    parse_status: str
    input_tokens: int
    output_tokens: int
    latency_sec: float
    finish_reason: str
    truncated: bool
    derived_category: str

    @property
    def correct(self) -> bool:
        return self.parsed_answer is not None and self.parsed_answer == self.gold_answer

    @property
    def parse_failure(self) -> bool:
        return self.parse_status.startswith("parse_failure")

    @property
    def max_token_hit(self) -> bool:
        return self.finish_reason == "max_new_tokens"

    @property
    def answer_sign(self) -> str:
        if self.gold_answer < 0:
            return "negative"
        if self.gold_answer == 0:
            return "zero"
        return "positive"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise AnalysisError(f"Invalid YAML: {path}") from exc
    if not isinstance(value, Mapping):
        raise AnalysisError(f"YAML root must be a mapping: {path}")
    return dict(value)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"Invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise AnalysisError(f"JSON root must be a mapping: {path}")
    return dict(value)


def _safe_archive_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    if not members:
        raise AnalysisError("Canonical archive is empty.")
    names = [member.name for member in members]
    if len(names) != len(set(names)):
        raise AnalysisError("Canonical archive contains duplicate member paths.")
    for member in members:
        path = PurePosixPath(member.name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or member.issym()
            or member.islnk()
            or member.isdev()
            or not (member.isdir() or member.isfile())
        ):
            raise AnalysisError(f"Unsafe archive member: {member.name!r}")
    return members


def extract_canonical_archive(archive_path: Path, destination: Path) -> tuple[Path, list[str]]:
    """Extract only validated regular files/directories and locate the single run root."""

    with tarfile.open(archive_path, "r:gz") as archive:
        members = _safe_archive_members(archive)
        for member in members:
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise AnalysisError(f"Could not read archive member: {member.name}")
            with source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle)

    file_paths = sorted(path for path in destination.rglob("*") if path.is_file())
    run_dirs = {path.parent for path in file_paths}
    if len(run_dirs) != 1:
        raise AnalysisError("Canonical archive must contain one flat run directory.")
    run_dir = next(iter(run_dirs))
    actual_names = {path.name for path in file_paths}
    missing = sorted(REQUIRED_ARTIFACTS - actual_names)
    if missing:
        raise AnalysisError(f"Canonical archive is missing required artifacts: {missing}")
    return run_dir, [path.relative_to(destination).as_posix() for path in file_paths]


def _read_csv(path: Path, expected_columns: Sequence[str]) -> list[dict[str, str]]:
    try:
        handle = path.open(encoding="utf-8", newline="")
    except OSError as exc:
        raise AnalysisError(f"Could not open CSV: {path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(expected_columns):
            raise AnalysisError(f"CSV columns mismatch for {path.name}: {reader.fieldnames!r}")
        return list(reader)


def _parse_prediction(row: Mapping[str, str]) -> Prediction:
    if row["correct"] not in {"true", "false"} or row["truncated"] not in {
        "true",
        "false",
    }:
        raise AnalysisError("Prediction booleans must be lowercase true/false.")
    try:
        parsed = None if row["parsed_answer"] == "" else int(row["parsed_answer"])
        return Prediction(
            sample_id=row["id"],
            question=row["question"],
            gold_answer=int(row["gold_answer"]),
            raw_output=row["raw_output"],
            parsed_answer=parsed,
            stored_correct=row["correct"] == "true",
            parse_status=row["parse_status"],
            input_tokens=int(row["input_tokens"]),
            output_tokens=int(row["output_tokens"]),
            latency_sec=float(row["latency_sec"]),
            finish_reason=row["finish_reason"],
            truncated=row["truncated"] == "true",
            derived_category=row["derived_category"],
        )
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"Malformed prediction row for ID {row.get('id')!r}.") from exc


def _record_discrepancy(
    discrepancies: list[dict[str, Any]], field: str, expected: Any, actual: Any
) -> None:
    if expected != actual:
        discrepancies.append({"field": field, "expected": expected, "actual": actual})


def validate_identity(run_dir: Path, freeze: Mapping[str, Any]) -> dict[str, Any]:
    config = _load_yaml(run_dir / "config.snapshot.yaml")
    metrics = _load_json(run_dir / "metrics.json")
    manifest = _load_json(run_dir / "run_manifest.json")
    resume = _load_json(run_dir / "resume_identity.json")["identity"]
    identity = freeze["identity"]
    canonical = freeze["canonical_run"]
    expected_metrics = canonical["metrics"]
    discrepancies: list[dict[str, Any]] = []

    comparisons = {
        "run_id": (canonical["run_id"], manifest.get("run_id")),
        "experiment_id": (freeze["experiment"]["experiment_id"], manifest.get("experiment_id")),
        "code_commit": (identity["code_commit"], manifest.get("git_commit")),
        "model_name": (freeze["experiment"]["model"], manifest.get("model_name")),
        "model_revision": (identity["model_revision"], manifest.get("model_revision")),
        "model_commit": (identity["model_revision"], manifest.get("model_commit")),
        "tokenizer_revision": (
            identity["tokenizer_revision"],
            manifest.get("tokenizer_revision"),
        ),
        "tokenizer_commit": (
            identity["tokenizer_revision"],
            manifest.get("tokenizer_commit"),
        ),
        "split_version": (
            freeze["experiment"]["validation_split"],
            manifest.get("split_version"),
        ),
        "validation_sha256": (identity["validation_sha256"], manifest.get("val_sha256")),
        "validation_rows": (identity["validation_rows"], config["data"]["validation_rows"]),
        "config_sha256": (identity["config_sha256"], manifest.get("config_sha256")),
        "prompt_version": (identity["prompt_version"], manifest.get("prompt_version")),
        "generation_version": (
            identity["generation_version"],
            manifest.get("generation_config", {}).get("version"),
        ),
        "parser_version": (identity["parser_version"], manifest.get("answer_parser_version")),
        "pipeline_version": (identity["pipeline_version"], manifest.get("pipeline_version")),
        "device": (canonical["device"], manifest.get("device")),
        "dtype": (canonical["dtype"], manifest.get("dtype")),
        "batch_size": (canonical["batch_size"], manifest.get("batch_size")),
        "run_status": (canonical["status"], manifest.get("status")),
        "accelerator": (canonical["accelerator"], manifest.get("device_name")),
        "total": (expected_metrics["total"], metrics.get("total")),
        "correct": (expected_metrics["correct"], metrics.get("correct")),
        "incorrect": (expected_metrics["incorrect"], metrics.get("incorrect")),
        "accuracy": (expected_metrics["accuracy"], metrics.get("accuracy")),
        "parse_failures": (expected_metrics["parse_failures"], metrics.get("parse_failures")),
        "max_new_tokens_hits": (
            expected_metrics["max_new_tokens_hits"],
            metrics.get("max_new_tokens_hits"),
        ),
        "resume_model_revision": (identity["model_revision"], resume.get("model_revision")),
        "resume_config_sha256": (identity["config_sha256"], resume.get("config_sha256")),
        "resume_validation_sha256": (
            identity["validation_sha256"],
            resume.get("validation_sha256"),
        ),
    }
    for field, (expected, actual) in comparisons.items():
        _record_discrepancy(discrepancies, field, expected, actual)

    generation = identity["generation"]
    for field in ("do_sample", "num_beams", "max_new_tokens", "use_cache"):
        _record_discrepancy(
            discrepancies,
            f"generation.{field}",
            generation[field],
            manifest.get("generation_config", {}).get(field),
        )
    _record_discrepancy(
        discrepancies,
        "resume.used",
        canonical["resume"]["used"],
        manifest.get("resumed_from") is not None,
    )
    _record_discrepancy(
        discrepancies,
        "config_snapshot_sha256",
        identity["config_sha256"],
        sha256_file(run_dir / "config.snapshot.yaml"),
    )
    if discrepancies:
        raise AnalysisError(
            "Canonical identity discrepancies: "
            + json.dumps(discrepancies, ensure_ascii=False, sort_keys=True)
        )
    return {
        "status": "PASS",
        "comparisons": sorted(comparisons),
        "discrepancies": [],
        "git_dirty_at_kaggle_runtime": bool(manifest.get("git_dirty")),
        "resumed_predictions": metrics.get("resumed_predictions"),
    }


def validate_predictions(
    run_dir: Path, validation_path: Path, freeze: Mapping[str, Any]
) -> tuple[list[Prediction], dict[str, Any]]:
    prediction_rows = _read_csv(run_dir / "predictions.csv", PREDICTION_COLUMNS)
    failure_rows = _read_csv(run_dir / "failures.csv", PREDICTION_COLUMNS)
    validation_rows = _read_csv(validation_path, ("id", "question", "answer"))
    predictions = [_parse_prediction(row) for row in prediction_rows]
    validation_by_id = {row["id"]: row for row in validation_rows}
    ids = [prediction.sample_id for prediction in predictions]
    incorrect_ids = {prediction.sample_id for prediction in predictions if not prediction.correct}
    expected = freeze["canonical_run"]["metrics"]
    checks = {
        "prediction_rows": len(predictions),
        "unique_ids": len(set(ids)),
        "duplicate_ids": len(ids) - len(set(ids)),
        "validation_id_set_equal": set(ids) == set(validation_by_id),
        "validation_order_equal": ids == [row["id"] for row in validation_rows],
        "question_mismatches": sum(
            prediction.question != validation_by_id[prediction.sample_id]["question"]
            for prediction in predictions
        ),
        "gold_mismatches": sum(
            prediction.gold_answer != int(validation_by_id[prediction.sample_id]["answer"])
            for prediction in predictions
        ),
        "stored_correct": sum(prediction.stored_correct for prediction in predictions),
        "recomputed_correct": sum(prediction.correct for prediction in predictions),
        "stored_vs_recomputed_mismatches": sum(
            prediction.stored_correct != prediction.correct for prediction in predictions
        ),
        "parse_failures": sum(prediction.parse_failure for prediction in predictions),
        "max_new_tokens_hits": sum(prediction.max_token_hit for prediction in predictions),
        "truncated": sum(prediction.truncated for prediction in predictions),
        "failure_rows": len(failure_rows),
        "failure_id_set_equals_incorrect": {row["id"] for row in failure_rows} == incorrect_ids,
    }
    required = {
        "prediction_rows": expected["total"],
        "unique_ids": expected["total"],
        "duplicate_ids": 0,
        "validation_id_set_equal": True,
        "validation_order_equal": True,
        "question_mismatches": 0,
        "gold_mismatches": 0,
        "stored_correct": expected["correct"],
        "recomputed_correct": expected["correct"],
        "stored_vs_recomputed_mismatches": 0,
        "parse_failures": expected["parse_failures"],
        "max_new_tokens_hits": expected["max_new_tokens_hits"],
        "truncated": expected["max_new_tokens_hits"],
        "failure_rows": expected["incorrect"],
        "failure_id_set_equals_incorrect": True,
    }
    if checks != required:
        raise AnalysisError(f"Prediction integrity failed: {json.dumps(checks, sort_keys=True)}")
    checks.update(
        {
            "status": "PASS",
            "recomputed_incorrect": len(predictions) - checks["recomputed_correct"],
            "recomputed_accuracy": checks["recomputed_correct"] / len(predictions),
        }
    )
    return predictions, checks


def _numeric_summary(values: Sequence[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(array.mean()), 8),
        "median": round(float(np.median(array)), 8),
        "p95": round(float(np.percentile(array, 95)), 8),
        "max": round(float(array.max()), 8),
    }


def _group_metrics(
    predictions: Sequence[Prediction], key: Callable[[Prediction], str]
) -> list[dict[str, Any]]:
    groups: defaultdict[str, list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        groups[key(prediction)].append(prediction)
    output = []
    for label, members in sorted(groups.items()):
        correct = sum(prediction.correct for prediction in members)
        parse_failures = sum(prediction.parse_failure for prediction in members)
        max_hits = sum(prediction.max_token_hit for prediction in members)
        output.append(
            {
                "group": label,
                "count": len(members),
                "correct": correct,
                "incorrect": len(members) - correct,
                "accuracy": round(correct / len(members), 10),
                "parse_failures": parse_failures,
                "parse_failure_rate": round(parse_failures / len(members), 10),
                "max_token_hits": max_hits,
                "max_token_hit_rate": round(max_hits / len(members), 10),
            }
        )
    return output


def _output_length_metrics(
    predictions: Sequence[Prediction], max_new_tokens: int
) -> tuple[list[dict[str, Any]], list[int]]:
    values = np.asarray([prediction.output_tokens for prediction in predictions])
    boundaries = [
        int(np.percentile(values, percentile, method="nearest")) for percentile in (25, 50, 75, 95)
    ]
    boundaries = sorted(set(boundaries + [max_new_tokens - 1]))

    def bucket(prediction: Prediction) -> str:
        lower = 0
        for upper in boundaries:
            if prediction.output_tokens <= upper:
                return f"{lower:04d}-{upper:04d}"
            lower = upper + 1
        return f"{max_new_tokens:04d}"

    return _group_metrics(predictions, bucket), boundaries


def _normalized_question(question: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", question).casefold().split())


def _phase1_suspects(
    clean_path: Path, phase1_config_path: Path, phase1_audit_path: Path
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    config = _load_yaml(phase1_config_path)
    dataset = config["dataset"]
    audit_config = dataset["audit"]
    rows = _read_csv(clean_path, ("id", "question", "answer"))
    exact_counts = Counter(row["question"] for row in rows)
    normalized_counts = Counter(_normalized_question(row["question"]) for row in rows)
    suspects: dict[str, list[str]] = {}
    reason_counts: Counter[str] = Counter()
    for row in rows:
        question = row["question"]
        normalized = _normalized_question(question)
        reasons: list[str] = []
        if not question.strip():
            reasons.append("empty_or_whitespace_question")
        if len(question) < audit_config["unusually_short_question_chars"]:
            reasons.append("unusually_short_question")
        if len(question) > audit_config["unusually_long_question_chars"]:
            reasons.append("unusually_long_question")
        if exact_counts[question] > 1:
            reasons.append("exact_duplicate_question")
        if normalized_counts[normalized] > 1:
            reasons.append("normalized_duplicate_question")
        if phase1._URL_PATTERN.search(question):
            reasons.append("contains_url")
        if phase1._MARKDOWN_IMAGE_PATTERN.search(question):
            reasons.append("contains_markdown_image")
        if phase1._has_latex_or_markup_anomaly(question):
            reasons.append("latex_or_markup_anomaly_candidate")
        if phase1._INSTRUCTION_RESIDUE_PATTERN.search(question):
            reasons.append("translation_or_instruction_residue_candidate")
        if (
            "\ufffd" in question
            or phase1._CONTROL_CHARACTER_PATTERN.search(question)
            or phase1._REPEATED_SYMBOL_PATTERN.search(question)
        ):
            reasons.append("abnormal_character_pattern_candidate")
        if reasons:
            suspects[row["id"]] = reasons
            reason_counts.update(reasons)

    audit = _load_json(phase1_audit_path)["quality_audit"]["clean_train"]
    verification = {
        "recomputed_suspect_rows": len(suspects),
        "reported_suspect_rows": audit["suspect_row_count"],
        "recomputed_reason_counts": dict(sorted(reason_counts.items())),
        "reported_reason_counts": audit["reason_counts"],
    }
    if (
        verification["recomputed_suspect_rows"] != verification["reported_suspect_rows"]
        or verification["recomputed_reason_counts"] != verification["reported_reason_counts"]
    ):
        raise AnalysisError("Phase 1 suspect reconstruction does not match audit_report.json.")
    verification["status"] = "PASS"
    return suspects, verification


def _comparison(predictions: Sequence[Prediction]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for label, members in (
        ("correct", [prediction for prediction in predictions if prediction.correct]),
        ("incorrect", [prediction for prediction in predictions if not prediction.correct]),
    ):
        output[label] = {
            "count": len(members),
            "question_character_length": _numeric_summary(
                [len(prediction.question) for prediction in members]
            ),
            "input_tokens": _numeric_summary([prediction.input_tokens for prediction in members]),
            "output_tokens": _numeric_summary([prediction.output_tokens for prediction in members]),
            "latency_sec": _numeric_summary([prediction.latency_sec for prediction in members]),
            "max_token_hits": sum(prediction.max_token_hit for prediction in members),
            "parse_failures": sum(prediction.parse_failure for prediction in members),
        }
    return output


def _repetitive_tail_candidate(raw_output: str) -> bool:
    tokens = re.findall(r"[\w+-]+", raw_output[-1000:].casefold())
    if len(tokens) < 20:
        return False
    most_common_count = Counter(tokens).most_common(1)[0][1]
    return most_common_count >= 10 and most_common_count / len(tokens) >= 0.35


def _max_hit_form(prediction: Prediction) -> str:
    if _repetitive_tail_candidate(prediction.raw_output):
        return "repetitive_output_candidate"
    if prediction.parse_status == "parsed_explicit":
        return "explicit_answer_present_before_limit"
    if prediction.parse_status == "parsed_boxed":
        return "boxed_answer_present_before_limit"
    if prediction.parse_status == "parsed_plain":
        return "plain_answer_present_before_limit"
    if prediction.parse_status == "parsed_fallback":
        return "reasoning_cut_off_with_fallback_integer"
    if prediction.parse_status == "parse_failure_conflict":
        return "conflicting_answer_candidates_at_limit"
    return "no_parseable_answer_at_limit"


def _parse_failure_reason(prediction: Prediction) -> str:
    if prediction.parse_status == "parse_failure_conflict":
        return "conflicting_final_answers"
    if prediction.max_token_hit:
        return "truncated_before_parseable_answer"
    if not prediction.raw_output.strip():
        return "empty_output"
    return "no_final_integer_or_other"


def _taxonomy(prediction: Prediction, p10: int, p95: int) -> str:
    if prediction.parse_failure:
        return "parse_failure"
    if prediction.max_token_hit:
        return "max_token_related"
    if prediction.output_tokens >= p95:
        return "unusually_long_reasoning"
    if prediction.output_tokens <= p10:
        return "unusually_short_reasoning"
    return "well_formed_wrong_answer"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(buffer.getvalue(), encoding="utf-8")
    temporary.replace(path)


def _sample_errors(
    predictions: Sequence[Prediction], taxonomy: Mapping[str, str], seed: int
) -> list[Prediction]:
    grouped: defaultdict[str, list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        if not prediction.correct:
            grouped[taxonomy[prediction.sample_id]].append(prediction)
    selected: list[Prediction] = []
    for label, members in sorted(grouped.items()):
        rng = random.Random(f"{seed}:{label}")
        ordered = sorted(members, key=lambda item: item.sample_id)
        selected.extend(rng.sample(ordered, min(5, len(ordered))))
    return sorted(selected, key=lambda item: item.sample_id)


def analyze_predictions(
    predictions: Sequence[Prediction], suspect_reasons: Mapping[str, Sequence[str]], seed: int
) -> dict[str, Any]:
    max_new_tokens = max(prediction.output_tokens for prediction in predictions)
    answer_sign = _group_metrics(predictions, lambda prediction: prediction.answer_sign)
    categories = _group_metrics(predictions, lambda prediction: prediction.derived_category)
    length_metrics, boundaries = _output_length_metrics(predictions, max_new_tokens)
    max_hits = [prediction for prediction in predictions if prediction.max_token_hit]
    non_hits = [prediction for prediction in predictions if not prediction.max_token_hit]
    max_hit_metrics = _group_metrics(
        predictions, lambda prediction: "max_token_hit" if prediction.max_token_hit else "non_hit"
    )
    suspect_metrics = _group_metrics(
        predictions,
        lambda prediction: (
            "phase1_suspect" if prediction.sample_id in suspect_reasons else "non_suspect"
        ),
    )
    suspect_validation_reasons = Counter(
        reason
        for prediction in predictions
        for reason in suspect_reasons.get(prediction.sample_id, ())
    )
    p10 = int(
        np.percentile(
            [prediction.output_tokens for prediction in predictions], 10, method="nearest"
        )
    )
    p95 = int(
        np.percentile(
            [prediction.output_tokens for prediction in predictions], 95, method="nearest"
        )
    )
    taxonomy = {
        prediction.sample_id: _taxonomy(prediction, p10, p95)
        for prediction in predictions
        if not prediction.correct
    }
    taxonomy_counts = dict(sorted(Counter(taxonomy.values()).items()))
    parse_failures = [prediction for prediction in predictions if prediction.parse_failure]
    parse_reason_counts = dict(
        sorted(Counter(_parse_failure_reason(prediction) for prediction in parse_failures).items())
    )
    max_form_counts = dict(
        sorted(Counter(_max_hit_form(prediction) for prediction in max_hits).items())
    )
    samples = _sample_errors(predictions, taxonomy, seed)

    metric_by_group = {row["group"]: row for row in max_hit_metrics}
    category_by_group = {row["group"]: row for row in categories}
    hypotheses = [
        {
            "id": "H1",
            "hypothesis": "Official direct-answer SFT may improve weak zero-shot categories.",
            "evidence": {
                name: category_by_group[name]["accuracy"]
                for name in ("algebra", "geometry", "number_theory", "combinatorics")
                if name in category_by_group
            },
            "caution": "Categories are heuristic and small groups require controlled validation.",
        },
        {
            "id": "H2",
            "hypothesis": "Max-token generation is a major E000 failure correlate.",
            "evidence": {
                "max_token_hit_count": len(max_hits),
                "max_token_hit_accuracy": metric_by_group["max_token_hit"]["accuracy"],
                "non_hit_accuracy": metric_by_group["non_hit"]["accuracy"],
            },
            "caution": (
                "This observational association does not prove that a larger token cap "
                "fixes reasoning."
            ),
        },
        {
            "id": "H3",
            "hypothesis": "Parser failures are not the primary overall Accuracy bottleneck.",
            "evidence": {
                "parse_failures": len(parse_failures),
                "fraction_of_all_samples": round(len(parse_failures) / len(predictions), 10),
                "maximum_possible_accuracy_point_recovery": round(
                    100 * len(parse_failures) / len(predictions), 6
                ),
            },
            "caution": (
                "Parser changes require a new version and cannot alter canonical E000 "
                "retrospectively."
            ),
        },
    ]
    return {
        "correct_vs_incorrect": _comparison(predictions),
        "answer_sign_metrics": answer_sign,
        "category_metrics": categories,
        "category_labels_derived_not_gold": True,
        "output_length_metrics": length_metrics,
        "output_length_boundaries": boundaries,
        "output_length_boundary_rationale": (
            "Empirical p25/p50/p75/p95 nearest-rank boundaries plus a separate max-token bucket."
        ),
        "max_token_metrics": max_hit_metrics,
        "max_token_form_counts": max_form_counts,
        "max_token_rows": max_hits,
        "non_hit_count": len(non_hits),
        "parse_failure_reason_counts": parse_reason_counts,
        "parse_failure_rows": parse_failures,
        "phase1_suspect_metrics": suspect_metrics,
        "phase1_suspect_validation_count": sum(
            prediction.sample_id in suspect_reasons for prediction in predictions
        ),
        "phase1_suspect_validation_reason_counts": dict(sorted(suspect_validation_reasons.items())),
        "taxonomy_counts": taxonomy_counts,
        "taxonomy_thresholds": {"output_tokens_p10": p10, "output_tokens_p95": p95},
        "taxonomy_policy": (
            "Mutually exclusive deterministic precedence: parse failure, max-token hit, "
            "unusually long, unusually short, otherwise well-formed wrong. Arithmetic, "
            "reasoning, and dataset ambiguity are manual-review candidates only."
        ),
        "taxonomy": taxonomy,
        "samples": samples,
        "phase4_hypotheses": hypotheses,
    }


def _artifact_hashes(run_dir: Path) -> dict[str, str]:
    return {path.name: sha256_file(path) for path in sorted(run_dir.iterdir()) if path.is_file()}


def analyze_canonical_archive(
    *,
    archive_path: str | Path,
    expected_archive_sha256: str,
    freeze_record_path: str | Path,
    validation_path: str | Path,
    clean_train_path: str | Path,
    phase1_config_path: str | Path,
    phase1_audit_path: str | Path,
    output_dir: str | Path,
    freeze_commit: str,
    seed: int = 2026,
) -> dict[str, Any]:
    """Validate immutable E000 artifacts and write analysis-only derivatives."""

    archive_path = Path(archive_path).resolve()
    actual_archive_sha256 = sha256_file(archive_path)
    if actual_archive_sha256 != expected_archive_sha256:
        raise AnalysisError(
            f"Archive SHA-256 mismatch: expected {expected_archive_sha256}, "
            f"got {actual_archive_sha256}."
        )
    freeze = _load_yaml(Path(freeze_record_path).resolve())
    output_dir = Path(output_dir).resolve()
    with tempfile.TemporaryDirectory(prefix="qwen-e000-analysis-") as temporary:
        run_dir, members = extract_canonical_archive(archive_path, Path(temporary))
        artifact_hashes = _artifact_hashes(run_dir)
        manifest = _load_json(run_dir / "run_manifest.json")
        for filename, expected_hash in manifest.get("artifact_sha256", {}).items():
            if artifact_hashes.get(filename) != expected_hash:
                raise AnalysisError(f"Internal artifact SHA-256 mismatch for {filename}.")
        identity = validate_identity(run_dir, freeze)
        predictions, integrity = validate_predictions(
            run_dir, Path(validation_path).resolve(), freeze
        )
        suspects, suspect_verification = _phase1_suspects(
            Path(clean_train_path).resolve(),
            Path(phase1_config_path).resolve(),
            Path(phase1_audit_path).resolve(),
        )
        analysis = analyze_predictions(predictions, suspects, seed)

        manifest_output = {
            "schema_version": 1,
            "analysis_version": ANALYSIS_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "phase3_freeze_commit": freeze_commit,
            "canonical_run_id": freeze["canonical_run"]["run_id"],
            "source_archive": archive_path.name,
            "source_archive_sha256": actual_archive_sha256,
            "source_archive_immutable": True,
            "archive_members": members,
            "internal_artifact_sha256": artifact_hashes,
            "identity_validation": identity,
            "predictions_integrity": integrity,
            "phase1_suspect_reconstruction": suspect_verification,
            "sampling_seed": seed,
            "outputs": [
                "analysis_manifest.json",
                "error_analysis.json",
                "category_metrics.csv",
                "answer_sign_metrics.csv",
                "output_length_metrics.csv",
                "max_token_hits.csv",
                "parse_failures.csv",
                "error_samples.csv",
            ],
        }

        max_hit_rows = [
            {
                "id": prediction.sample_id,
                "gold_answer": prediction.gold_answer,
                "parsed_answer": prediction.parsed_answer,
                "correct": str(prediction.correct).lower(),
                "parse_status": prediction.parse_status,
                "output_tokens": prediction.output_tokens,
                "derived_category": prediction.derived_category,
                "observed_form": _max_hit_form(prediction),
                "raw_output_tail": prediction.raw_output[-500:],
            }
            for prediction in analysis["max_token_rows"]
        ]
        parse_failure_rows = [
            {
                "id": prediction.sample_id,
                "gold_answer": prediction.gold_answer,
                "raw_output": prediction.raw_output,
                "parse_status": prediction.parse_status,
                "failure_reason": _parse_failure_reason(prediction),
                "output_tokens": prediction.output_tokens,
                "max_token_hit": str(prediction.max_token_hit).lower(),
            }
            for prediction in analysis["parse_failure_rows"]
        ]
        sample_rows = [
            {
                "id": prediction.sample_id,
                "taxonomy": analysis["taxonomy"][prediction.sample_id],
                "derived_category": prediction.derived_category,
                "gold_answer": prediction.gold_answer,
                "parsed_answer": prediction.parsed_answer,
                "parse_status": prediction.parse_status,
                "output_tokens": prediction.output_tokens,
                "question": prediction.question,
                "raw_output": prediction.raw_output,
            }
            for prediction in analysis["samples"]
        ]
        error_output = {
            key: value
            for key, value in analysis.items()
            if key
            not in {
                "max_token_rows",
                "parse_failure_rows",
                "taxonomy",
                "samples",
            }
        }

    _write_json(output_dir / "analysis_manifest.json", manifest_output)
    _write_json(output_dir / "error_analysis.json", error_output)
    common_columns = (
        "group",
        "count",
        "correct",
        "incorrect",
        "accuracy",
        "parse_failures",
        "parse_failure_rate",
        "max_token_hits",
        "max_token_hit_rate",
    )
    _write_csv(output_dir / "category_metrics.csv", analysis["category_metrics"], common_columns)
    _write_csv(
        output_dir / "answer_sign_metrics.csv", analysis["answer_sign_metrics"], common_columns
    )
    _write_csv(
        output_dir / "output_length_metrics.csv", analysis["output_length_metrics"], common_columns
    )
    _write_csv(
        output_dir / "max_token_hits.csv",
        max_hit_rows,
        (
            "id",
            "gold_answer",
            "parsed_answer",
            "correct",
            "parse_status",
            "output_tokens",
            "derived_category",
            "observed_form",
            "raw_output_tail",
        ),
    )
    _write_csv(
        output_dir / "parse_failures.csv",
        parse_failure_rows,
        (
            "id",
            "gold_answer",
            "raw_output",
            "parse_status",
            "failure_reason",
            "output_tokens",
            "max_token_hit",
        ),
    )
    _write_csv(
        output_dir / "error_samples.csv",
        sample_rows,
        (
            "id",
            "taxonomy",
            "derived_category",
            "gold_answer",
            "parsed_answer",
            "parse_status",
            "output_tokens",
            "question",
            "raw_output",
        ),
    )
    return manifest_output
