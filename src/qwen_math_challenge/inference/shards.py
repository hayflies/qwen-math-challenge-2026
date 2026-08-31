"""Emergency deterministic sharding and merge for final E000 inference."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qwen_math_challenge.config import LoadedConfig
from qwen_math_challenge.data.official import sha256_file
from qwen_math_challenge.evaluation.zero_shot import BatchGenerator, parse_integer_answer_v2
from qwen_math_challenge.inference.submission import (
    FINAL_SUBMISSION_PIPELINE_VERSION,
    FinalInputBundle,
    FinalPrediction,
    FinalSubmissionError,
    FinalSubmissionResult,
    FinalSubmissionSettings,
    FinalTestRow,
    _append_prediction,
    _atomic_write_bytes,
    _atomic_write_json,
    _load_resume,
    _metrics,
    _ordered_identity,
    _write_and_validate_submission,
    _write_predictions,
    load_final_input,
    load_final_predictions,
    validate_submission_file,
)

SHARD_PIPELINE_VERSION = "phase13_final_e000_shard_v1"


@dataclass(frozen=True)
class FinalShardPlan:
    bundle: FinalInputBundle
    existing_run: Path
    existing_predictions: tuple[FinalPrediction, ...]
    existing_predictions_sha256: str
    shards: tuple[tuple[FinalTestRow, ...], ...]
    report: dict[str, Any]


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FinalSubmissionError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalSubmissionError(f"{label} must be valid UTF-8 JSON.") from exc
    if not isinstance(value, dict):
        raise FinalSubmissionError(f"{label} must contain a JSON object.")
    return value


def _validate_prediction_records(
    records: tuple[FinalPrediction, ...],
    bundle: FinalInputBundle,
    *,
    expected_rows: tuple[FinalTestRow, ...] | None = None,
) -> dict[str, FinalPrediction]:
    source_by_id = {row.sample_id: row for row in bundle.rows}
    records_by_id: dict[str, FinalPrediction] = {}
    for record in records:
        if record.sample_id in records_by_id:
            raise FinalSubmissionError(f"Duplicate prediction id {record.sample_id!r}.")
        source = source_by_id.get(record.sample_id)
        if source is None:
            raise FinalSubmissionError(f"Unknown prediction id {record.sample_id!r}.")
        reparsed = parse_integer_answer_v2(record.raw_output)
        if (
            record.source_index != source.index
            or record.question != source.question
            or record.question_sha256 != source.question_sha256
            or record.is_flagged != source.is_flagged
            or record.parsed_answer != reparsed.value
            or record.parse_status != reparsed.status
        ):
            raise FinalSubmissionError(
                f"Prediction for {record.sample_id!r} failed source/parser integrity."
            )
        records_by_id[record.sample_id] = record
    if expected_rows is not None and [record.sample_id for record in records] != [
        row.sample_id for row in expected_rows
    ]:
        raise FinalSubmissionError("Shard prediction order/coverage is incomplete.")
    return records_by_id


def _validate_existing_run_identity(
    config: LoadedConfig,
    settings: FinalSubmissionSettings,
    bundle: FinalInputBundle,
    existing_run: Path,
) -> None:
    input_identity = _read_json(
        existing_run / settings.output.input_identity_filename, "input identity"
    )
    if input_identity != bundle.identity:
        raise FinalSubmissionError("Existing run dataset/flag identity mismatch.")
    payload = _read_json(existing_run / settings.output.resume_identity_filename, "resume identity")
    identity = payload.get("identity")
    if not isinstance(identity, dict):
        raise FinalSubmissionError("Existing resume identity payload is malformed.")
    expected = {
        "pipeline_version": FINAL_SUBMISSION_PIPELINE_VERSION,
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
        "limit": None,
        "adapter": None,
        "tool_use": False,
    }
    mismatches = [key for key, value in expected.items() if identity.get(key) != value]
    if mismatches:
        raise FinalSubmissionError(
            f"Existing run canonical identity mismatch: {sorted(mismatches)!r}."
        )


def plan_final_shards(
    config: LoadedConfig,
    settings: FinalSubmissionSettings,
    *,
    existing_run: str | Path,
    num_shards: int,
) -> FinalShardPlan:
    if isinstance(num_shards, bool) or not isinstance(num_shards, int) or num_shards < 1:
        raise FinalSubmissionError("num_shards must be a positive integer.")
    bundle = load_final_input(settings.data)
    run_path = Path(existing_run).expanduser().resolve()
    _validate_existing_run_identity(config, settings, bundle, run_path)
    predictions_path = run_path / settings.output.predictions_filename
    existing = load_final_predictions(predictions_path)
    existing_by_id = _validate_prediction_records(existing, bundle)

    remaining = tuple(row for row in bundle.rows if row.sample_id not in existing_by_id)
    shards = tuple(tuple(remaining[index::num_shards]) for index in range(num_shards))
    existing_ids = set(existing_by_id)
    shard_ids = [{row.sample_id for row in rows} for rows in shards]
    report = {
        "schema_version": 1,
        "pipeline_version": SHARD_PIPELINE_VERSION,
        "official_total": len(bundle.rows),
        "existing_completed": len(existing),
        "remaining": len(remaining),
        "num_shards": num_shards,
        "shard_counts": {str(index): len(rows) for index, rows in enumerate(shards)},
        "intersection_existing_shards": {
            str(index): len(existing_ids & ids) for index, ids in enumerate(shard_ids)
        },
        "intersection_between_shards": {
            f"{left}:{right}": len(shard_ids[left] & shard_ids[right])
            for left in range(num_shards)
            for right in range(left + 1, num_shards)
        },
        "union_count": len(existing_ids | set().union(*shard_ids)),
        "all_official_ids_covered": existing_ids | set().union(*shard_ids)
        == {row.sample_id for row in bundle.rows},
        "existing_predictions_sha256": sha256_file(predictions_path),
        "test_sha256": bundle.identity["test"]["sha256"],
        "flag_sha256": bundle.identity["flags"]["sha256"],
    }
    if (
        sum(report["intersection_existing_shards"].values()) != 0
        or sum(report["intersection_between_shards"].values()) != 0
        or report["union_count"] != len(bundle.rows)
        or not report["all_official_ids_covered"]
        or len(existing) + sum(len(rows) for rows in shards) != len(bundle.rows)
    ):
        raise FinalSubmissionError(f"Unsafe shard plan: {report!r}")
    return FinalShardPlan(
        bundle=bundle,
        existing_run=run_path,
        existing_predictions=existing,
        existing_predictions_sha256=report["existing_predictions_sha256"],
        shards=shards,
        report=report,
    )


def _shard_resume_identity(
    config: LoadedConfig,
    settings: FinalSubmissionSettings,
    generator: BatchGenerator,
    plan: FinalShardPlan,
    *,
    shard_index: int,
) -> dict[str, Any]:
    selected = plan.shards[shard_index]
    identity = {
        "schema_version": 1,
        "pipeline_version": SHARD_PIPELINE_VERSION,
        "config_sha256": config.source_sha256,
        "code_sha256": {
            "shards.py": sha256_file(Path(__file__).resolve()),
            "submission.py": sha256_file(Path(__file__).with_name("submission.py").resolve()),
        },
        "input_identity": plan.bundle.identity,
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
        "shard": {
            "index": shard_index,
            "num_shards": len(plan.shards),
            "selected_count": len(selected),
            "selected_ordered_id_sha256": _ordered_identity([row.sample_id for row in selected]),
            "excluded_predictions_sha256": plan.existing_predictions_sha256,
            "excluded_completed_count": len(plan.existing_predictions),
        },
        "external_datasets": [],
        "tool_use": False,
        "adapter": None,
    }
    canonical = json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return {
        "identity": identity,
        "identity_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def run_final_submission_shard(
    config: LoadedConfig,
    settings: FinalSubmissionSettings,
    *,
    run_dir: str | Path,
    generator: BatchGenerator,
    existing_run: str | Path,
    shard_index: int,
    num_shards: int,
    resume_from: str | Path | None = None,
) -> FinalSubmissionResult:
    plan = plan_final_shards(config, settings, existing_run=existing_run, num_shards=num_shards)
    if isinstance(shard_index, bool) or not 0 <= shard_index < num_shards:
        raise FinalSubmissionError("shard_index must satisfy 0 <= index < num_shards.")
    selected = plan.shards[shard_index]
    output_dir = Path(run_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_identity_path = output_dir / settings.output.input_identity_filename
    _atomic_write_json(input_identity_path, plan.bundle.identity)
    identity = _shard_resume_identity(config, settings, generator, plan, shard_index=shard_index)
    resume_identity_path = output_dir / settings.output.resume_identity_filename
    _atomic_write_json(resume_identity_path, identity)

    existing: tuple[FinalPrediction, ...] = tuple()
    if resume_from is not None:
        existing = _load_resume(
            Path(resume_from).expanduser().resolve(),
            identity,
            selected,
            settings.output,
        )
    predictions_path = output_dir / settings.output.predictions_filename
    _write_predictions(predictions_path, existing)
    started = time.perf_counter()
    for source in selected[len(existing) :]:
        generated = list(generator.generate([source.question]))
        if len(generated) != 1:
            raise FinalSubmissionError("Generator must return exactly one result per test row.")
        result = generated[0]
        parsed = parse_integer_answer_v2(result.raw_output)
        _append_prediction(
            predictions_path,
            FinalPrediction(
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
            ),
        )
    elapsed = time.perf_counter() - started
    records = load_final_predictions(predictions_path)
    _validate_prediction_records(records, plan.bundle, expected_rows=selected)
    metrics = _metrics(
        records,
        input_identity=plan.bundle.identity,
        total_wall_clock_sec=elapsed,
        resumed_predictions=len(existing),
        limit=None,
        generator=generator,
    )
    metrics.update(
        {
            "inference_mode": "shard",
            "shard_index": shard_index,
            "num_shards": num_shards,
            "selected_rows": len(selected),
            "excluded_completed_rows": len(plan.existing_predictions),
            "canonical_full_submission": False,
        }
    )
    metrics_path = output_dir / settings.output.metrics_filename
    _atomic_write_json(metrics_path, metrics)
    paths = [predictions_path, input_identity_path, resume_identity_path, metrics_path]
    artifact_sha256 = {path.name: sha256_file(path) for path in paths}
    return FinalSubmissionResult(
        metrics=metrics,
        input_identity=plan.bundle.identity,
        resume_identity=identity,
        predictions_path=predictions_path,
        input_identity_path=input_identity_path,
        resume_identity_path=resume_identity_path,
        metrics_path=metrics_path,
        submission_path=None,
        submission_sha256_path=None,
        artifact_sha256=artifact_sha256,
    )


def merge_final_shards(
    config: LoadedConfig,
    settings: FinalSubmissionSettings,
    *,
    existing_run: str | Path,
    shard_runs: tuple[Path, ...],
    output_dir: str | Path,
) -> dict[str, Any]:
    plan = plan_final_shards(
        config, settings, existing_run=existing_run, num_shards=len(shard_runs)
    )
    shard_by_index: dict[int, tuple[FinalPrediction, ...]] = {}
    shard_sources: dict[str, Any] = {}
    for run in shard_runs:
        run = run.expanduser().resolve()
        if (
            _read_json(run / settings.output.input_identity_filename, "shard input identity")
            != plan.bundle.identity
        ):
            raise FinalSubmissionError("Shard dataset identity mismatch.")
        payload = _read_json(
            run / settings.output.resume_identity_filename, "shard resume identity"
        )
        identity = payload.get("identity", {})
        shard = identity.get("shard", {})
        if identity.get("pipeline_version") != SHARD_PIPELINE_VERSION:
            raise FinalSubmissionError("Shard pipeline identity mismatch.")
        expected_canonical = {
            "config_sha256": config.source_sha256,
            "model_name": settings.model.name_or_path,
            "model_revision": settings.model.revision,
            "model_commit": settings.model.revision,
            "tokenizer_revision": settings.model.tokenizer_revision,
            "tokenizer_commit": settings.model.tokenizer_revision,
            "prompt": asdict(settings.prompt),
            "generation": asdict(settings.generation),
            "parser_version": settings.parser_version,
            "canonical_e000_config_sha256": settings.canonical_e000_config_sha256,
            "adapter": None,
            "tool_use": False,
        }
        mismatches = [
            key for key, value in expected_canonical.items() if identity.get(key) != value
        ]
        if mismatches:
            raise FinalSubmissionError(
                f"Shard canonical identity mismatch: {sorted(mismatches)!r}."
            )
        index = shard.get("index")
        if not isinstance(index, int) or not 0 <= index < len(shard_runs):
            raise FinalSubmissionError("Shard index is invalid.")
        if index in shard_by_index:
            raise FinalSubmissionError(f"Duplicate shard index {index}.")
        expected_rows = plan.shards[index]
        expected_shard = {
            "index": index,
            "num_shards": len(shard_runs),
            "selected_count": len(expected_rows),
            "selected_ordered_id_sha256": _ordered_identity(
                [row.sample_id for row in expected_rows]
            ),
            "excluded_predictions_sha256": plan.existing_predictions_sha256,
            "excluded_completed_count": len(plan.existing_predictions),
        }
        if shard != expected_shard or identity.get("input_identity") != plan.bundle.identity:
            raise FinalSubmissionError(f"Shard {index} selection/exclusion identity mismatch.")
        records = load_final_predictions(run / settings.output.predictions_filename)
        _validate_prediction_records(records, plan.bundle, expected_rows=expected_rows)
        shard_by_index[index] = records
        shard_sources[str(index)] = {
            "run_dir": str(run),
            "rows": len(records),
            "predictions_sha256": sha256_file(run / settings.output.predictions_filename),
            "resume_identity_sha256": payload.get("identity_sha256"),
        }
    if set(shard_by_index) != set(range(len(shard_runs))):
        raise FinalSubmissionError("Not every planned shard is present.")

    merged: dict[str, FinalPrediction] = {}
    for label, records in (
        ("existing", plan.existing_predictions),
        *((f"shard-{index}", shard_by_index[index]) for index in range(len(shard_runs))),
    ):
        for record in records:
            if record.sample_id in merged:
                raise FinalSubmissionError(
                    f"Conflicting duplicate prediction {record.sample_id!r} from {label}."
                )
            merged[record.sample_id] = record
    official_ids = {row.sample_id for row in plan.bundle.rows}
    unknown = set(merged) - official_ids
    missing = official_ids - set(merged)
    if unknown or missing:
        raise FinalSubmissionError(
            f"Merged coverage failure: unknown={sorted(unknown)[:10]!r}, "
            f"missing={sorted(missing)[:10]!r}."
        )
    ordered = tuple(merged[row.sample_id] for row in plan.bundle.rows)
    _validate_prediction_records(ordered, plan.bundle, expected_rows=plan.bundle.rows)

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FinalSubmissionError(f"Merge output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    merged_predictions = destination / "predictions.csv"
    _write_predictions(merged_predictions, ordered)
    submission = destination / "submission.csv"
    _write_and_validate_submission(submission, ordered, plan.bundle.rows)
    submission_sha = sha256_file(submission)
    sidecar = destination / "submission.csv.sha256"
    _atomic_write_bytes(sidecar, f"{submission_sha}  submission.csv\n".encode("ascii"))
    _atomic_write_json(destination / "input_identity.json", plan.bundle.identity)
    manifest = {
        "schema_version": 1,
        "pipeline_version": SHARD_PIPELINE_VERSION,
        "status": "completed",
        "total_rows": len(ordered),
        "flagged_rows": sum(record.is_flagged for record in ordered),
        "scoring_eligible_rows": len(ordered) - sum(record.is_flagged for record in ordered),
        "parse_failures": sum(record.parsed_answer is None for record in ordered),
        "existing_run": str(plan.existing_run),
        "existing_rows": len(plan.existing_predictions),
        "existing_predictions_sha256": plan.existing_predictions_sha256,
        "shards": shard_sources,
        "input_identity": plan.bundle.identity,
        "merged_predictions_sha256": sha256_file(merged_predictions),
        "submission_sha256": submission_sha,
    }
    _atomic_write_json(destination / "merge_manifest.json", manifest)
    return manifest


def validate_merged_submission(
    settings: FinalSubmissionSettings, submission_path: str | Path
) -> dict[str, Any]:
    bundle = load_final_input(settings.data)
    path = Path(submission_path).expanduser().resolve()
    validate_submission_file(path, bundle.rows)
    digest = sha256_file(path)
    sidecar = path.with_name("submission.csv.sha256")
    expected = f"{digest}  submission.csv"
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != expected:
        raise FinalSubmissionError("submission.csv.sha256 is missing or mismatched.")
    return {
        "status": "PASS",
        "submission": str(path),
        "sha256": digest,
        "total_rows": len(bundle.rows),
        "flagged_rows": bundle.identity["flags"]["rows"],
        "scoring_eligible_rows": bundle.identity["scoring_eligible_rows"],
    }
